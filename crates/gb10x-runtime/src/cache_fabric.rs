//! Deterministic CPU hot-cache ownership derived from Linux cache topology.

use gb10x_core::{CacheType, CpuCache};
use std::collections::BTreeSet;
use thiserror::Error;

/// Failure while deriving a CPU hot-cache ownership policy.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CacheFabricError {
    /// No CPU was selected for the compute/cache-owner role.
    #[error("CPU cache fabric requires at least one compute CPU")]
    NoComputeCpus,
    /// A compute CPU was selected more than once.
    #[error("compute CPU {0} appears more than once")]
    DuplicateComputeCpu(u32),
    /// A selected compute CPU lacks a private data-capable L2 cache.
    #[error("compute CPU {0} has no private data-capable L2 cache")]
    MissingPrivateL2(u32),
    /// The requested hot-state budget would exceed a private L2 cache.
    #[error(
        "CPU {cpu_id} hot-cache budget {requested_bytes} exceeds private L2 capacity {private_l2_bytes}"
    )]
    BudgetExceedsPrivateL2 {
        /// CPU whose private L2 capacity would be exceeded.
        cpu_id: u32,
        /// Requested per-owner hot-cache budget in bytes.
        requested_bytes: u64,
        /// Discovered private L2 capacity in bytes.
        private_l2_bytes: u64,
    },
    /// A candidate cannot be ranked by saved latency per byte when it has no byte cost.
    #[error("hot-cache candidate {key} has zero bytes")]
    ZeroCandidateBytes {
        /// Stable key of the invalid candidate.
        key: u64,
    },
    /// A cache state key was supplied more than once with ambiguous placement metadata.
    #[error("hot-cache candidate key {key} appears more than once")]
    DuplicateCandidateKey {
        /// Repeated stable key.
        key: u64,
    },
}

/// One CPU cache owner selected from the discovered topology.
#[derive(Clone, Debug, Eq, PartialEq)]
struct CpuCacheOwner {
    cpu_id: u32,
    hot_budget_bytes: u64,
}

/// A PLE/cache entry that may be placed in its canonical owner's hot cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HotCacheCandidate {
    /// Stable PLE/cache state key.
    pub key: u64,
    /// Bytes required by this entry in the owner's hot cache.
    pub bytes: u64,
    /// Expected latency saved by keeping the entry hot, in nanoseconds.
    pub expected_savings_ns: u64,
}

/// One entry accepted into a CPU owner's hot-cache plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HotCachePlacement {
    /// Stable PLE/cache state key.
    pub key: u64,
    /// Canonical Linux CPU responsible for the entry.
    pub owner_cpu: u32,
    /// Bytes reserved by the entry.
    pub bytes: u64,
    /// Expected latency saved by the entry, in nanoseconds.
    pub expected_savings_ns: u64,
}

/// Deterministic hot-cache placements and unused budget for each CPU owner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuHotCachePlan {
    owner_cpus: Vec<u32>,
    hot_budget_bytes_per_owner: u64,
    placements: Vec<HotCachePlacement>,
    remaining_bytes_by_cpu: Vec<(u32, u64)>,
}

impl CpuHotCachePlan {
    /// Return the canonical owner CPUs of the fabric that produced this plan.
    pub fn owner_cpus(&self) -> &[u32] {
        &self.owner_cpus
    }

    /// Return the uniform hot-cache budget recorded when this plan was created.
    pub(crate) fn hot_budget_bytes_per_owner(&self) -> u64 {
        self.hot_budget_bytes_per_owner
    }

    /// Return accepted entries in canonical owner and priority order.
    pub fn placements(&self) -> &[HotCachePlacement] {
        &self.placements
    }

    /// Return the unused hot-cache budget for one owner CPU.
    pub fn remaining_bytes_for_cpu(&self, cpu_id: u32) -> Option<u64> {
        self.remaining_bytes_by_cpu
            .iter()
            .find_map(|(owner_cpu, remaining_bytes)| {
                (*owner_cpu == cpu_id).then_some(*remaining_bytes)
            })
    }
}

/// Stable mapping from PLE/cache state keys to canonical CPU owners.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuCacheFabric {
    owners: Vec<CpuCacheOwner>,
}

impl CpuCacheFabric {
    /// Derive canonical owners and validate their hot-cache budgets against private L2 capacity.
    pub fn from_topology(
        caches: &[CpuCache],
        compute_cpus: &[u32],
        hot_budget_bytes_per_owner: u64,
    ) -> Result<Self, CacheFabricError> {
        if compute_cpus.is_empty() {
            return Err(CacheFabricError::NoComputeCpus);
        }
        let mut owner_cpus = compute_cpus.to_vec();
        owner_cpus.sort_unstable();
        if let Some(duplicate) = owner_cpus
            .windows(2)
            .find_map(|pair| (pair[0] == pair[1]).then_some(pair[0]))
        {
            return Err(CacheFabricError::DuplicateComputeCpu(duplicate));
        }
        let mut owners = Vec::with_capacity(owner_cpus.len());
        for cpu_id in owner_cpus {
            let private_l2_bytes = caches
                .iter()
                .filter(|cache| {
                    cache.level == 2
                        && matches!(cache.cache_type, CacheType::Data | CacheType::Unified)
                        && cache.shared_cpu_ids.as_slice() == [cpu_id]
                })
                .map(|cache| cache.size_bytes)
                .max()
                .ok_or(CacheFabricError::MissingPrivateL2(cpu_id))?;
            if hot_budget_bytes_per_owner > private_l2_bytes {
                return Err(CacheFabricError::BudgetExceedsPrivateL2 {
                    cpu_id,
                    requested_bytes: hot_budget_bytes_per_owner,
                    private_l2_bytes,
                });
            }
            owners.push(CpuCacheOwner {
                cpu_id,
                hot_budget_bytes: hot_budget_bytes_per_owner,
            });
        }
        Ok(Self { owners })
    }

    /// Return canonical Linux CPU IDs selected as cache owners.
    pub fn owner_cpus(&self) -> Vec<u32> {
        self.owners.iter().map(|owner| owner.cpu_id).collect()
    }

    /// Select the one owner responsible for a cache/PLE state key.
    pub fn owner_cpu_for_key(&self, key: u64) -> u32 {
        let owner_index = (key % self.owners.len() as u64) as usize;
        self.owners[owner_index].cpu_id
    }

    /// Return the uniform hot-cache budget selected for each canonical owner.
    pub(crate) fn hot_budget_bytes_per_owner(&self) -> u64 {
        self.owners[0].hot_budget_bytes
    }

    /// Greedily reserve each owner's budget for positive-savings entries with the greatest latency saved per byte.
    pub fn plan_hot_candidates(
        &self,
        candidates: &[HotCacheCandidate],
    ) -> Result<CpuHotCachePlan, CacheFabricError> {
        if self.owners.is_empty() {
            return Err(CacheFabricError::NoComputeCpus);
        }
        if let Some(candidate) = candidates.iter().find(|candidate| candidate.bytes == 0) {
            return Err(CacheFabricError::ZeroCandidateBytes { key: candidate.key });
        }
        let mut seen_keys = BTreeSet::new();
        if let Some(candidate) = candidates
            .iter()
            .find(|candidate| !seen_keys.insert(candidate.key))
        {
            return Err(CacheFabricError::DuplicateCandidateKey { key: candidate.key });
        }

        let mut placements = Vec::new();
        let mut remaining_bytes_by_cpu = self
            .owners
            .iter()
            .map(|owner| (owner.cpu_id, owner.hot_budget_bytes))
            .collect::<Vec<_>>();

        for (owner_index, owner) in self.owners.iter().enumerate() {
            let mut owner_candidates = candidates
                .iter()
                .filter(|candidate| self.owner_cpu_for_key(candidate.key) == owner.cpu_id)
                .collect::<Vec<_>>();
            owner_candidates.sort_unstable_by(|left, right| {
                let left_density = u128::from(left.expected_savings_ns) * u128::from(right.bytes);
                let right_density = u128::from(right.expected_savings_ns) * u128::from(left.bytes);
                right_density
                    .cmp(&left_density)
                    .then_with(|| left.key.cmp(&right.key))
                    .then_with(|| left.bytes.cmp(&right.bytes))
                    .then_with(|| left.expected_savings_ns.cmp(&right.expected_savings_ns))
            });

            let remaining_bytes = &mut remaining_bytes_by_cpu[owner_index].1;
            for candidate in owner_candidates {
                if candidate.expected_savings_ns == 0 {
                    continue;
                }
                if candidate.bytes <= *remaining_bytes {
                    *remaining_bytes -= candidate.bytes;
                    placements.push(HotCachePlacement {
                        key: candidate.key,
                        owner_cpu: owner.cpu_id,
                        bytes: candidate.bytes,
                        expected_savings_ns: candidate.expected_savings_ns,
                    });
                }
            }
        }

        Ok(CpuHotCachePlan {
            owner_cpus: self.owner_cpus(),
            hot_budget_bytes_per_owner: self.hot_budget_bytes_per_owner(),
            placements,
            remaining_bytes_by_cpu,
        })
    }
}
