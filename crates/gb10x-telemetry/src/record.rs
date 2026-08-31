#[cfg(test)]
mod tests {
    use super::*;

    fn valid_record() -> EvidenceRecord {
        EvidenceRecord {
            identity: RunIdentity {
                git_commit: "0123456789abcdef0123456789abcdef01234567".into(),
                model_id: "Qwen/Qwen3.8-Flash-Next".into(),
                model_revision: "34567a4".into(),
                model_digest: "sha256:fixture".into(),
                hardware: "NVIDIA GB10".into(),
                mode: ExecutionMode::Exact,
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
                prefill_tokens_per_second: Some(1000.0),
                decode_tokens_per_second: Some(42.0),
                ttft_micros: Some(250_000),
                p50_latency_micros: None,
                p95_latency_micros: None,
                unified_memory_bytes: Some(100_000_000_000),
                power_watts: None,
                tokens_per_joule: None,
            },
            speculation: None,
            correctness: Some(CorrectnessGate::Passed {
                oracle: "target-greedy-reference".into(),
                checked_tokens: 256,
            }),
        }
    }

    #[test]
    fn performance_record_without_commit_is_rejected() {
        let mut record = valid_record();
        record.identity.git_commit.clear();
        assert!(matches!(
            record.validate(),
            Err(EvidenceError::Missing("git_commit"))
        ));
    }

    #[test]
    fn record_requires_correctness_result() {
        let mut record = valid_record();
        record.correctness = None;
        assert!(matches!(
            record.validate(),
            Err(EvidenceError::Missing("correctness"))
        ));
    }

    #[test]
    fn exact_and_experimental_modes_are_explicitly_distinct() {
        let mut record = valid_record();
        record.identity.mode = ExecutionMode::ExperimentalApproximate {
            label: "ple-nvfp4".into(),
        };
        assert!(record.validate().is_ok());
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("experimental_approximate"));
        assert!(json.contains("ple-nvfp4"));
    }

    #[test]
    fn speculative_run_requires_acceptance_evidence() {
        let mut record = valid_record();
        record.speculation = Some(SpeculationMetrics {
            strategy: "mtp".into(),
            proposed_tokens: 100,
            accepted_tokens: 0,
            verified_cycles: 0,
        });
        assert!(record.validate().is_err());
    }

    #[test]
    fn impossible_workload_is_rejected() {
        let mut record = valid_record();
        record.workload.output_tokens = 0;
        assert!(record.validate().is_err());
    }
}
