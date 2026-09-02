//! Deterministic, host-independent PLE-Hydra logical-tier resolution.

use crate::cache_fabric::{CpuCacheFabric, CpuHotCachePlan};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// Exactness class for one isolated PLE-Hydra cache hierarchy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PleHydraMode {
    /// Byte/algorithmically exact PLE data and tier state.
    Exact,
    /// Explicitly approximate experimental PLE data that must remain isolated from exact state.
    ExperimentalApproximate {
        /// Human-readable identifier for the approximation experiment.
        label: String,
    },
}

/// Logical PLE-Hydra tier selected for one PLE/cache state key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PleHydraTier {
    /// The entry is assigned to this key's canonical private CPU owner.
    OwnerCpuHot {
        /// Linux CPU ID of the single cache owner.
        owner_cpu: u32,
    },
    /// The entry is in the shared CPU hot/warm tier.
    SharedCpuWarm,
    /// The entry is expected to be available through the mmap/Linux page-cache warm tier.
    PageCacheWarm,
    /// The entry is absent from all warm tiers and requires cold NVMe backing.
    NvmeCold,
}

/// One logical PLE-Hydra lookup observed by a host-independent simulation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PleHydraAccess {
    /// Stable PLE/cache state key that was looked up.
    pub key: u64,
    /// Highest available tier selected for the key.
    pub tier: PleHydraTier,
}

/// Aggregate logical-tier counts collected from one PLE-Hydra simulation trace.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PleHydraTierCounts {
    /// Lookups served by the canonical private CPU-owner tier.
    pub owner_cpu_hot_hits: u64,
    /// Lookups served by the shared CPU hot/warm tier.
    pub shared_cpu_warm_hits: u64,
    /// Lookups served by the mmap/Linux page-cache warm tier.
    pub page_cache_warm_hits: u64,
    /// Lookups that would require the NVMe cold tier.
    pub nvme_cold_reads: u64,
}

impl PleHydraTierCounts {
    fn record(&mut self, tier: &PleHydraTier) {
        match tier {
            PleHydraTier::OwnerCpuHot { .. } => {
                self.owner_cpu_hot_hits = self.owner_cpu_hot_hits.saturating_add(1);
            }
            PleHydraTier::SharedCpuWarm => {
                self.shared_cpu_warm_hits = self.shared_cpu_warm_hits.saturating_add(1);
            }
            PleHydraTier::PageCacheWarm => {
                self.page_cache_warm_hits = self.page_cache_warm_hits.saturating_add(1);
            }
            PleHydraTier::NvmeCold => {
                self.nvme_cold_reads = self.nvme_cold_reads.saturating_add(1);
            }
        }
    }
}

/// Results from resolving a complete logical PLE-Hydra access trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PleHydraSimulation {
    accesses: Vec<PleHydraAccess>,
    tier_counts: PleHydraTierCounts,
}

impl PleHydraSimulation {
    /// Return one tier-resolution record for each input key in trace order.
    pub fn accesses(&self) -> &[PleHydraAccess] {
        &self.accesses
    }

    /// Return aggregate counts by logical PLE-Hydra tier.
    pub fn tier_counts(&self) -> PleHydraTierCounts {
        self.tier_counts
    }
}

/// Failure while converting a CPU hot-cache plan into PLE-Hydra tier state.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PleHydraError {
    /// An experimental hierarchy lacked the identifier required to keep its evidence distinct.
    #[error("experimental PLE-Hydra mode requires a nonempty label")]
    EmptyExperimentalModeLabel,
    /// A hot-cache plan was created by a different canonical owner fabric.
    #[error(
        "PLE-Hydra hot plan owners {plan_owner_cpus:?} with {plan_hot_budget_bytes_per_owner}-byte budgets do not match active owners {hydra_owner_cpus:?} with {hydra_hot_budget_bytes_per_owner}-byte budgets"
    )]
    HotPlanFabricMismatch {
        /// Canonical owner CPUs recorded in the supplied plan.
        plan_owner_cpus: Vec<u32>,
        /// Uniform per-owner hot-cache budget recorded in the supplied plan.
        plan_hot_budget_bytes_per_owner: u64,
        /// Canonical owner CPUs in the PLE-Hydra fabric.
        hydra_owner_cpus: Vec<u32>,
        /// Uniform per-owner hot-cache budget in the active PLE-Hydra fabric.
        hydra_hot_budget_bytes_per_owner: u64,
    },
    /// A hot-cache placement was assigned to a CPU that does not own its key.
    #[error(
        "PLE-Hydra hot placement for key {key} uses CPU {actual_owner_cpu}, expected CPU {expected_owner_cpu}"
    )]
    HotPlacementOwnerMismatch {
        /// Stable PLE/cache state key.
        key: u64,
        /// CPU named by the supplied placement.
        actual_owner_cpu: u32,
        /// Canonical CPU derived from the cache fabric.
        expected_owner_cpu: u32,
    },
}

/// Read-only PLE-Hydra logical-tier state derived from a CPU cache-fabric plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PleHydra {
    mode: PleHydraMode,
    cache_fabric: CpuCacheFabric,
    owner_hot_keys: BTreeMap<u32, BTreeSet<u64>>,
    shared_cpu_warm_keys: BTreeSet<u64>,
    page_cache_warm_keys: BTreeSet<u64>,
}

impl PleHydra {
    /// Build deterministic PLE-Hydra tier state from a validated CPU hot-cache plan.
    pub fn from_hot_cache_plan(
        cache_fabric: CpuCacheFabric,
        hot_plan: &CpuHotCachePlan,
        shared_cpu_warm_keys: &[u64],
        page_cache_warm_keys: &[u64],
    ) -> Result<Self, PleHydraError> {
        Self::from_hot_cache_plan_with_mode(
            PleHydraMode::Exact,
            cache_fabric,
            hot_plan,
            shared_cpu_warm_keys,
            page_cache_warm_keys,
        )
    }

    /// Build a PLE-Hydra hierarchy in one explicit exactness mode.
    pub fn from_hot_cache_plan_with_mode(
        mode: PleHydraMode,
        cache_fabric: CpuCacheFabric,
        hot_plan: &CpuHotCachePlan,
        shared_cpu_warm_keys: &[u64],
        page_cache_warm_keys: &[u64],
    ) -> Result<Self, PleHydraError> {
        if matches!(
            &mode,
            PleHydraMode::ExperimentalApproximate { label } if label.trim().is_empty()
        ) {
            return Err(PleHydraError::EmptyExperimentalModeLabel);
        }

        let hydra_owner_cpus = cache_fabric.owner_cpus();
        let hydra_hot_budget_bytes_per_owner = cache_fabric.hot_budget_bytes_per_owner();
        if hot_plan.owner_cpus() != hydra_owner_cpus.as_slice()
            || hot_plan.hot_budget_bytes_per_owner() != hydra_hot_budget_bytes_per_owner
        {
            return Err(PleHydraError::HotPlanFabricMismatch {
                plan_owner_cpus: hot_plan.owner_cpus().to_vec(),
                plan_hot_budget_bytes_per_owner: hot_plan.hot_budget_bytes_per_owner(),
                hydra_owner_cpus,
                hydra_hot_budget_bytes_per_owner,
            });
        }

        let mut owner_hot_keys = BTreeMap::<u32, BTreeSet<u64>>::new();
        for placement in hot_plan.placements() {
            let expected_owner_cpu = cache_fabric.owner_cpu_for_key(placement.key);
            if placement.owner_cpu != expected_owner_cpu {
                return Err(PleHydraError::HotPlacementOwnerMismatch {
                    key: placement.key,
                    actual_owner_cpu: placement.owner_cpu,
                    expected_owner_cpu,
                });
            }
            owner_hot_keys
                .entry(placement.owner_cpu)
                .or_default()
                .insert(placement.key);
        }

        Ok(Self {
            mode,
            cache_fabric,
            owner_hot_keys,
            shared_cpu_warm_keys: shared_cpu_warm_keys.iter().copied().collect(),
            page_cache_warm_keys: page_cache_warm_keys.iter().copied().collect(),
        })
    }

    /// Return this hierarchy's exactness mode.
    pub fn mode(&self) -> &PleHydraMode {
        &self.mode
    }

    /// Resolve a PLE/cache state key through the owner-hot, shared-warm, page-cache and NVMe tiers.
    pub fn tier_for_key(&self, key: u64) -> PleHydraTier {
        let owner_cpu = self.cache_fabric.owner_cpu_for_key(key);
        if self
            .owner_hot_keys
            .get(&owner_cpu)
            .is_some_and(|keys| keys.contains(&key))
        {
            return PleHydraTier::OwnerCpuHot { owner_cpu };
        }
        if self.shared_cpu_warm_keys.contains(&key) {
            return PleHydraTier::SharedCpuWarm;
        }
        if self.page_cache_warm_keys.contains(&key) {
            return PleHydraTier::PageCacheWarm;
        }
        PleHydraTier::NvmeCold
    }

    /// Resolve a trace without performing real I/O, returning per-access tiers and aggregate counts.
    pub fn simulate_trace(&self, keys: &[u64]) -> PleHydraSimulation {
        let mut tier_counts = PleHydraTierCounts::default();
        let accesses = keys
            .iter()
            .copied()
            .map(|key| {
                let tier = self.tier_for_key(key);
                tier_counts.record(&tier);
                PleHydraAccess { key, tier }
            })
            .collect();

        PleHydraSimulation {
            accesses,
            tier_counts,
        }
    }
}
