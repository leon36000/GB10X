use gb10x_core::{CacheType, CpuCache};
use gb10x_runtime::cache_fabric::{CacheFabricError, CpuCacheFabric, HotCacheCandidate};

fn private_l2(cpu_id: u32) -> CpuCache {
    CpuCache {
        level: 2,
        cache_type: CacheType::Unified,
        size_bytes: 2 * 1024 * 1024,
        line_bytes: 64,
        shared_cpu_ids: vec![cpu_id],
    }
}

#[test]
fn canonicalizes_compute_owners_and_routes_ple_state_keys_to_one_owner() {
    let fabric =
        CpuCacheFabric::from_topology(&[private_l2(7), private_l2(2)], &[7, 2], 512 * 1024)
            .expect("private L2 caches must form owner shards");

    assert_eq!(fabric.owner_cpus(), &[2, 7]);
    assert_eq!(fabric.owner_cpu_for_key(0), 2);
    assert_eq!(fabric.owner_cpu_for_key(1), 7);
    assert_eq!(fabric.owner_cpu_for_key(2), 2);
}

#[test]
fn rejects_compute_owners_without_private_data_capable_l2() {
    let shared_l2 = CpuCache {
        level: 2,
        cache_type: CacheType::Unified,
        size_bytes: 2 * 1024 * 1024,
        line_bytes: 64,
        shared_cpu_ids: vec![2, 7],
    };

    assert!(CpuCacheFabric::from_topology(&[shared_l2], &[2], 512 * 1024).is_err());
}

#[test]
fn rejects_hot_budget_larger_than_the_owner_private_l2() {
    assert!(CpuCacheFabric::from_topology(&[private_l2(2)], &[2], 2 * 1024 * 1024 + 1).is_err());
}

#[test]
fn rejects_an_empty_compute_owner_set() {
    assert!(CpuCacheFabric::from_topology(&[], &[], 512 * 1024).is_err());
}

#[test]
fn rejects_duplicate_compute_owners() {
    assert!(CpuCacheFabric::from_topology(&[private_l2(2)], &[2, 2], 512 * 1024).is_err());
}

fn hot_candidate(key: u64, bytes: u64, expected_savings_ns: u64) -> HotCacheCandidate {
    HotCacheCandidate {
        key,
        bytes,
        expected_savings_ns,
    }
}

#[test]
fn plans_hot_ple_entries_by_saved_latency_per_byte_per_owner() {
    let fabric = CpuCacheFabric::from_topology(&[private_l2(2), private_l2(7)], &[2, 7], 320)
        .expect("two private L2 caches must form owner shards");

    let plan = fabric
        .plan_hot_candidates(&[
            hot_candidate(0, 320, 1_000),
            hot_candidate(2, 160, 800),
            hot_candidate(1, 320, 500),
        ])
        .expect("valid candidates must form a deterministic plan");

    let placed = plan
        .placements()
        .iter()
        .map(|placement| (placement.key, placement.owner_cpu))
        .collect::<Vec<_>>();
    assert_eq!(placed, vec![(2, 2), (1, 7)]);
    assert_eq!(plan.remaining_bytes_for_cpu(2), Some(160));
    assert_eq!(plan.remaining_bytes_for_cpu(7), Some(0));
}

#[test]
fn rejects_zero_byte_hot_cache_candidates() {
    let fabric = CpuCacheFabric::from_topology(&[private_l2(2)], &[2], 320)
        .expect("one private L2 cache must form an owner shard");

    let error = fabric
        .plan_hot_candidates(&[hot_candidate(0, 0, 1_000)])
        .expect_err("a zero-byte entry has no valid saved-latency-per-byte score");

    assert_eq!(error, CacheFabricError::ZeroCandidateBytes { key: 0 });
}

#[test]
fn rejects_duplicate_hot_cache_candidate_keys() {
    let fabric = CpuCacheFabric::from_topology(&[private_l2(2)], &[2], 640)
        .expect("one private L2 cache must form an owner shard");

    let error = fabric
        .plan_hot_candidates(&[hot_candidate(5, 320, 1_000), hot_candidate(5, 160, 2_000)])
        .expect_err("one PLE/cache state must have one unambiguous candidate");

    assert_eq!(error, CacheFabricError::DuplicateCandidateKey { key: 5 });
}

#[test]
fn does_not_reserve_hot_cache_for_a_candidate_without_predicted_latency_savings() {
    let fabric = CpuCacheFabric::from_topology(&[private_l2(2)], &[2], 320)
        .expect("one private L2 cache must form an owner shard");

    let plan = fabric
        .plan_hot_candidates(&[hot_candidate(0, 320, 0)])
        .expect("a zero-savings candidate is valid but not cache-worthy");

    assert!(plan.placements().is_empty());
    assert_eq!(plan.remaining_bytes_for_cpu(2), Some(320));
}
