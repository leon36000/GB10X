use gb10x_telemetry::{
    CacheCounters, CacheRunState, CorrectnessGate, CpuPlacement, EvidenceError, EvidenceRecord,
    ExecutionConfig, ExecutionMode, PerformanceMetrics, RunIdentity, StageTimings, WorkloadShape,
};

fn full_record() -> EvidenceRecord {
    EvidenceRecord {
        identity: RunIdentity {
            git_commit: "0123456789abcdef0123456789abcdef01234567".into(),
            model_id: "Qwen/Qwen3.8-Flash-Next".into(),
            model_revision: "34567a4".into(),
            model_digest: "sha256:model".into(),
            hardware: "NVIDIA GB10".into(),
            mode: ExecutionMode::Exact,
        },
        execution: ExecutionConfig {
            command: vec!["gb10x-serve".into(), "--profile".into(), "exact".into()],
            runtime_profile: "dedicated-server".into(),
            runtime_config_digest: "sha256:runtime".into(),
            cpu_placement: CpuPlacement::Affinity {
                cpus: vec![0, 1, 2, 3],
            },
            precision_mode: "bf16".into(),
            quantization_mode: "none".into(),
            prefix_cache_state: CacheRunState::Cold,
            ple_cache_state: CacheRunState::Warm,
            kv_cache_state: CacheRunState::Cold,
        },
        workload: WorkloadShape {
            context_tokens: 8192,
            prompt_tokens: 2048,
            output_tokens: 256,
            concurrency: 1,
        },
        stages: StageTimings::default(),
        caches: CacheCounters::default(),
        performance: PerformanceMetrics {
            decode_tokens_per_second: Some(42.0),
            ..PerformanceMetrics::default()
        },
        speculation: None,
        correctness: Some(CorrectnessGate::Passed {
            oracle: "target-greedy-reference".into(),
            checked_tokens: 256,
        }),
    }
}

#[test]
fn complete_execution_configuration_is_required() {
    let mut record = full_record();
    record.execution.command.clear();
    assert!(matches!(
        record.validate(),
        Err(EvidenceError::Missing("execution.command"))
    ));

    let mut record = full_record();
    record.execution.runtime_config_digest.clear();
    assert!(matches!(
        record.validate(),
        Err(EvidenceError::Missing("execution.runtime_config_digest"))
    ));
}

#[test]
fn cpu_affinity_must_be_nonempty_sorted_and_unique() {
    let mut record = full_record();
    record.execution.cpu_placement = CpuPlacement::Affinity { cpus: vec![] };
    assert!(record.validate().is_err());

    let mut record = full_record();
    record.execution.cpu_placement = CpuPlacement::Affinity {
        cpus: vec![2, 1, 1],
    };
    assert!(record.validate().is_err());
}

#[test]
fn exact_and_approximate_cache_states_serialize_without_freeform_json() {
    let record = full_record();
    record.validate().expect("full evidence contract");
    let json = serde_json::to_string(&record).unwrap();
    assert!(json.contains("dedicated-server"));
    assert!(json.contains("\"ple_cache_state\":\"warm\""));
    assert!(json.contains("\"cpu_placement\":{\"kind\":\"affinity\""));
}
