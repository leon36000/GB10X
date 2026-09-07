use gb10x_telemetry::{
    CacheCounters, CacheRunState, CorrectnessGate, CpuPlacement, EvidenceError, EvidenceRecord,
    ExecutionConfig, ExecutionMode, HardwareState, PerformanceMetrics, RunIdentity, StageTimings,
    WorkloadShape,
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
        hardware_state: HardwareState::default(),
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

#[test]
fn exact_evidence_rejects_each_unsupported_precision_label() {
    for precision in ["fp8", "nvfp4", "fp16", "fp32", "unknown", "BF16", "bf16 "] {
        let mut record = full_record();
        record.execution.precision_mode = precision.into();
        assert!(
            matches!(
                record.validate(),
                Err(EvidenceError::Invalid {
                    field: "execution.precision_mode",
                    ..
                })
            ),
            "unsupported exact precision {precision:?} was not rejected",
        );
    }
}

#[test]
fn exact_evidence_rejects_each_quantized_or_unknown_label() {
    for quantization in [
        "fp8",
        "nvfp4",
        "int8",
        "ple-nvfp4",
        "unknown",
        "NONE",
        "none ",
    ] {
        let mut record = full_record();
        record.execution.quantization_mode = quantization.into();
        assert!(
            matches!(
                record.validate(),
                Err(EvidenceError::Invalid {
                    field: "execution.quantization_mode",
                    ..
                })
            ),
            "unsupported exact quantization {quantization:?} was not rejected",
        );
    }
}

#[test]
fn experimental_profiles_remain_explicit_through_json_roundtrip() {
    for (precision, quantization) in [
        ("bf16", "none"),
        ("fp8", "fp8"),
        ("nvfp4", "nvfp4"),
        ("bf16", "ple-nvfp4"),
        ("future-experiment", "custom"),
    ] {
        let mut record = full_record();
        record.identity.mode = ExecutionMode::ExperimentalApproximate {
            label: "independently-gated-experiment".into(),
        };
        record.execution.precision_mode = precision.into();
        record.execution.quantization_mode = quantization.into();
        record.validate().expect("labelled experimental evidence");
        let encoded = serde_json::to_string(&record).unwrap();
        let decoded: EvidenceRecord = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, record);
        decoded.validate().unwrap();
    }
}

#[test]
fn each_present_integer_performance_measurement_must_be_positive() {
    for field in [
        "ttft_micros",
        "p50_latency_micros",
        "p95_latency_micros",
        "unified_memory_bytes",
    ] {
        let mut value = serde_json::to_value(full_record()).unwrap();
        value["performance"][field] = serde_json::json!(0);
        let record: EvidenceRecord = serde_json::from_value(value).unwrap();
        assert!(
            matches!(record.validate(), Err(EvidenceError::Invalid { field: actual, .. }) if actual == field),
            "zero-valued performance field {field} was not rejected",
        );
    }
}

#[test]
fn positive_integer_performance_boundaries_are_accepted() {
    for measured in [1, u64::MAX] {
        let mut record = full_record();
        record.performance.ttft_micros = Some(measured);
        record.performance.p50_latency_micros = Some(measured);
        record.performance.p95_latency_micros = Some(measured);
        record.performance.unified_memory_bytes = Some(measured);
        record.validate().expect("positive integer measurements");
    }
}

#[test]
fn omitted_performance_measurements_and_zero_stage_timers_remain_valid() {
    let mut value = serde_json::to_value(full_record()).unwrap();
    for field in [
        "ttft_micros",
        "p50_latency_micros",
        "p95_latency_micros",
        "unified_memory_bytes",
    ] {
        value["performance"].as_object_mut().unwrap().remove(field);
    }
    for stage in value["stages"].as_object_mut().unwrap().values_mut() {
        *stage = serde_json::json!(0);
    }
    let record: EvidenceRecord = serde_json::from_value(value).unwrap();
    record
        .validate()
        .expect("optional and sub-resolution timings");
}
