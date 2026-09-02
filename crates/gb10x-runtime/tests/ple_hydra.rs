use gb10x_core::{CacheType, CpuCache};
use gb10x_runtime::cache_fabric::{CpuCacheFabric, HotCacheCandidate};
use gb10x_runtime::ple_hydra::{
    PleHydra, PleHydraError, PleHydraMode, PleHydraTier, PleHydraTierCounts,
};

fn private_l2(cpu_id: u32) -> CpuCache {
    CpuCache {
        level: 2,
        cache_type: CacheType::Unified,
        size_bytes: 2 * 1024 * 1024,
        line_bytes: 64,
        shared_cpu_ids: vec![cpu_id],
    }
}

fn hot_candidate(key: u64) -> HotCacheCandidate {
    HotCacheCandidate {
        key,
        bytes: 320,
        expected_savings_ns: 1_000,
    }
}

#[test]
fn resolves_each_ple_key_to_the_highest_available_logical_tier() {
    let fabric = CpuCacheFabric::from_topology(&[private_l2(2), private_l2(7)], &[2, 7], 320)
        .expect("private L2 caches must form owner shards");
    let hot_plan = fabric
        .plan_hot_candidates(&[hot_candidate(0), hot_candidate(1)])
        .expect("hot candidates must plan");
    let hydra = PleHydra::from_hot_cache_plan(fabric, &hot_plan, &[0, 3], &[0, 4])
        .expect("a cache-fabric plan must seed PLE-Hydra");

    assert_eq!(
        hydra.tier_for_key(0),
        PleHydraTier::OwnerCpuHot { owner_cpu: 2 }
    );
    assert_eq!(
        hydra.tier_for_key(1),
        PleHydraTier::OwnerCpuHot { owner_cpu: 7 }
    );
    assert_eq!(hydra.tier_for_key(3), PleHydraTier::SharedCpuWarm);
    assert_eq!(hydra.tier_for_key(4), PleHydraTier::PageCacheWarm);
    assert_eq!(hydra.tier_for_key(5), PleHydraTier::NvmeCold);
}

#[test]
fn simulates_trace_accesses_and_counts_each_logical_tier() {
    let fabric = CpuCacheFabric::from_topology(&[private_l2(2), private_l2(7)], &[2, 7], 320)
        .expect("private L2 caches must form owner shards");
    let hot_plan = fabric
        .plan_hot_candidates(&[hot_candidate(0), hot_candidate(1)])
        .expect("hot candidates must plan");
    let hydra = PleHydra::from_hot_cache_plan(fabric, &hot_plan, &[3], &[4])
        .expect("a cache-fabric plan must seed PLE-Hydra");

    let simulation = hydra.simulate_trace(&[0, 1, 3, 4, 5, 0]);

    assert_eq!(
        simulation.tier_counts(),
        PleHydraTierCounts {
            owner_cpu_hot_hits: 3,
            shared_cpu_warm_hits: 1,
            page_cache_warm_hits: 1,
            nvme_cold_reads: 1,
        }
    );
    let tiers = simulation
        .accesses()
        .iter()
        .map(|access| access.tier.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        tiers,
        vec![
            PleHydraTier::OwnerCpuHot { owner_cpu: 2 },
            PleHydraTier::OwnerCpuHot { owner_cpu: 7 },
            PleHydraTier::SharedCpuWarm,
            PleHydraTier::PageCacheWarm,
            PleHydraTier::NvmeCold,
            PleHydraTier::OwnerCpuHot { owner_cpu: 2 },
        ]
    );
}

#[test]
fn keeps_exact_and_experimental_ple_hydra_state_explicitly_distinct() {
    let fabric = CpuCacheFabric::from_topology(&[private_l2(2)], &[2], 320)
        .expect("one private L2 cache must form an owner shard");
    let hot_plan = fabric
        .plan_hot_candidates(&[hot_candidate(0)])
        .expect("hot candidates must plan");

    let exact = PleHydra::from_hot_cache_plan(fabric.clone(), &hot_plan, &[], &[])
        .expect("the default PLE-Hydra state must be exact");
    let experimental_mode = PleHydraMode::ExperimentalApproximate {
        label: "ple-nvfp4".to_owned(),
    };
    let experimental = PleHydra::from_hot_cache_plan_with_mode(
        experimental_mode.clone(),
        fabric,
        &hot_plan,
        &[],
        &[],
    )
    .expect("experimental PLE-Hydra state must require an explicit mode");

    assert_eq!(exact.mode(), &PleHydraMode::Exact);
    assert_eq!(experimental.mode(), &experimental_mode);
    assert_ne!(exact.mode(), experimental.mode());
}

#[test]
fn rejects_an_experimental_ple_hydra_state_without_an_experiment_label() {
    let fabric = CpuCacheFabric::from_topology(&[private_l2(2)], &[2], 320)
        .expect("one private L2 cache must form an owner shard");
    let hot_plan = fabric
        .plan_hot_candidates(&[hot_candidate(0)])
        .expect("hot candidates must plan");

    let error = PleHydra::from_hot_cache_plan_with_mode(
        PleHydraMode::ExperimentalApproximate {
            label: "  ".to_owned(),
        },
        fabric,
        &hot_plan,
        &[],
        &[],
    )
    .expect_err("experimental cache state must identify its approximation");

    assert_eq!(error, PleHydraError::EmptyExperimentalModeLabel);
}

#[test]
fn rejects_a_hot_plan_from_a_different_owner_fabric_even_when_one_key_still_maps_to_the_same_cpu() {
    let plan_fabric = CpuCacheFabric::from_topology(&[private_l2(2), private_l2(7)], &[2, 7], 320)
        .expect("two private L2 caches must form owner shards");
    let hot_plan = plan_fabric
        .plan_hot_candidates(&[hot_candidate(0)])
        .expect("hot candidates must plan");
    let hydra_fabric = CpuCacheFabric::from_topology(&[private_l2(2)], &[2], 320)
        .expect("one private L2 cache must form a different owner fabric");

    let error = PleHydra::from_hot_cache_plan(hydra_fabric, &hot_plan, &[], &[])
        .expect_err("a plan must remain bound to the complete fabric that created it");

    assert_eq!(
        error,
        PleHydraError::HotPlanFabricMismatch {
            plan_owner_cpus: vec![2, 7],
            plan_hot_budget_bytes_per_owner: 320,
            hydra_owner_cpus: vec![2],
            hydra_hot_budget_bytes_per_owner: 320,
        }
    );
}

#[test]
fn rejects_a_hot_plan_from_the_same_owner_cpu_with_a_different_hot_budget() {
    let plan_fabric = CpuCacheFabric::from_topology(&[private_l2(2)], &[2], 640)
        .expect("one private L2 cache must form an owner shard");
    let hot_plan = plan_fabric
        .plan_hot_candidates(&[hot_candidate(0)])
        .expect("hot candidates must plan");
    let hydra_fabric = CpuCacheFabric::from_topology(&[private_l2(2)], &[2], 320)
        .expect("the same CPU can form a fabric with a smaller hot budget");

    let error = PleHydra::from_hot_cache_plan(hydra_fabric, &hot_plan, &[], &[])
        .expect_err("a plan must remain bound to the hot budget that created it");

    assert_eq!(
        error,
        PleHydraError::HotPlanFabricMismatch {
            plan_owner_cpus: vec![2],
            plan_hot_budget_bytes_per_owner: 640,
            hydra_owner_cpus: vec![2],
            hydra_hot_budget_bytes_per_owner: 320,
        }
    );
}
