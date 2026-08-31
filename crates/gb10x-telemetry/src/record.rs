//! Strict, serializable evidence records for GB10X correctness and performance runs.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Exactness mode attached to every evidence record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Bit/algorithmically exact execution path.
    Exact,
    /// Explicitly approximate experimental path that must never be mixed with exact evidence.
    ExperimentalApproximate {
        /// Human-readable approximation identifier, for example `ple-nvfp4`.
        label: String,
    },
}

/// Immutable identity of one benchmark or correctness run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunIdentity {
    /// Source commit that produced the executable.
    pub git_commit: String,
    /// Canonical model repository/name.
    pub model_id: String,
    /// Pinned model revision.
    pub model_revision: String,
    /// Content digest or manifest digest for the loaded model artifacts.
    pub model_digest: String,
    /// Hardware identity captured for this run.
    pub hardware: String,
    /// Exact versus explicitly approximate execution mode.
    pub mode: ExecutionMode,
}

/// Token/concurrency geometry of one measured workload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkloadShape {
    /// Effective context/window size used by the request.
    pub context_tokens: u64,
    /// Number of prompt/prefill tokens.
    pub prompt_tokens: u64,
    /// Number of generated tokens included in the measurement.
    pub output_tokens: u64,
    /// Simultaneous request count.
    pub concurrency: u32,
}

/// Optional stage timings in microseconds.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StageTimings {
    /// Prompt tokenization time.
    pub tokenize_micros: Option<u64>,
    /// Scheduler/queue time before GPU execution.
    pub queue_micros: Option<u64>,
    /// Prefill execution time.
    pub prefill_micros: Option<u64>,
    /// Decode execution time.
    pub decode_micros: Option<u64>,
    /// PLE hash computation time.
    pub ple_hash_micros: Option<u64>,
    /// PLE lookup/cache service time.
    pub ple_lookup_micros: Option<u64>,
    /// Attention time when independently measured.
    pub attention_micros: Option<u64>,
    /// MoE routing/expert execution time when independently measured.
    pub moe_micros: Option<u64>,
    /// Speculative/MTP verification time.
    pub speculation_verify_micros: Option<u64>,
    /// Sampling time.
    pub sampling_micros: Option<u64>,
}

/// Integer cache/service counters for one run.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CacheCounters {
    /// PLE requests served from the hottest software/cache tier.
    pub ple_hot_hits: u64,
    /// PLE requests served from the exact cold source.
    pub ple_cold_hits: u64,
    /// PLE rows prefetched speculatively or predictively.
    pub ple_prefetched_rows: u64,
    /// Prefetched PLE rows that were subsequently consumed.
    pub ple_prefetch_useful_rows: u64,
    /// Prefix-cache hits at request granularity.
    pub prefix_cache_hits: u64,
    /// Prefix-cache misses at request granularity.
    pub prefix_cache_misses: u64,
    /// Bytes read from the exact cold PLE source.
    pub cold_source_bytes_read: u64,
    /// Bytes read from the PLEPack hot overlay.
    pub hot_overlay_bytes_read: u64,
}

/// End-to-end performance metrics. Floating-point values are validated as finite and positive.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Prefill throughput.
    pub prefill_tokens_per_second: Option<f64>,
    /// Decode throughput.
    pub decode_tokens_per_second: Option<f64>,
    /// Time to first token in microseconds.
    pub ttft_micros: Option<u64>,
    /// Median end-to-end latency in microseconds.
    pub p50_latency_micros: Option<u64>,
    /// 95th percentile end-to-end latency in microseconds.
    pub p95_latency_micros: Option<u64>,
    /// Unified/system memory footprint attributed to the run.
    pub unified_memory_bytes: Option<u64>,
    /// Average measured power draw in watts.
    pub power_watts: Option<f64>,
    /// Generated tokens per joule.
    pub tokens_per_joule: Option<f64>,
}

/// Evidence captured when speculative decoding/MTP is active.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpeculationMetrics {
    /// Strategy identifier such as `mtp` or a future exact draft strategy.
    pub strategy: String,
    /// Number of draft tokens proposed.
    pub proposed_tokens: u64,
    /// Number of draft tokens accepted by target verification.
    pub accepted_tokens: u64,
    /// Number of target-verification cycles actually executed.
    pub verified_cycles: u64,
}

/// Correctness outcome attached to every run that may be used as evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CorrectnessGate {
    /// Candidate matched its declared correctness oracle for the checked scope.
    Passed {
        /// Oracle/reference path used for comparison.
        oracle: String,
        /// Number of output tokens covered by the comparison.
        checked_tokens: u64,
    },
    /// Candidate failed its declared correctness oracle.
    Failed {
        /// Oracle/reference path used for comparison.
        oracle: String,
        /// Number of output tokens checked before/while detecting the failure.
        checked_tokens: u64,
        /// Concise machine/log-friendly failure description.
        reason: String,
    },
}

/// Complete structured evidence payload for one GB10X run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    /// Commit/model/hardware identity.
    pub identity: RunIdentity,
    /// Workload geometry.
    pub workload: WorkloadShape,
    /// Optional detailed stage timings.
    pub stages: StageTimings,
    /// Cache/service counters.
    pub caches: CacheCounters,
    /// End-to-end performance metrics.
    pub performance: PerformanceMetrics,
    /// Speculation evidence when speculative execution was enabled.
    pub speculation: Option<SpeculationMetrics>,
    /// Mandatory correctness result.
    pub correctness: Option<CorrectnessGate>,
}

/// Validation failure that makes an evidence payload unusable for GB10X decisions.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum EvidenceError {
    /// Required field is absent or empty.
    #[error("missing required evidence field: {0}")]
    Missing(&'static str),
    /// Field value violates a semantic invariant.
    #[error("invalid evidence field {field}: {reason}")]
    Invalid {
        /// Field or logical group that failed validation.
        field: &'static str,
        /// Stable concise reason.
        reason: &'static str,
    },
}

impl EvidenceRecord {
    /// Validate that this record is complete and internally consistent enough to act as evidence.
    pub fn validate(&self) -> Result<(), EvidenceError> {
        require_nonempty(&self.identity.git_commit, "git_commit")?;
        require_nonempty(&self.identity.model_id, "model_id")?;
        require_nonempty(&self.identity.model_revision, "model_revision")?;
        require_nonempty(&self.identity.model_digest, "model_digest")?;
        require_nonempty(&self.identity.hardware, "hardware")?;

        if let ExecutionMode::ExperimentalApproximate { label } = &self.identity.mode {
            require_nonempty(label, "experimental_mode_label")?;
        }

        if self.workload.context_tokens == 0 {
            return invalid("context_tokens", "must be nonzero");
        }
        if self.workload.prompt_tokens == 0 {
            return invalid("prompt_tokens", "must be nonzero");
        }
        if self.workload.output_tokens == 0 {
            return invalid("output_tokens", "must be nonzero");
        }
        if self.workload.concurrency == 0 {
            return invalid("concurrency", "must be nonzero");
        }
        if self.workload.prompt_tokens > self.workload.context_tokens {
            return invalid("prompt_tokens", "cannot exceed context_tokens");
        }

        validate_positive_f64(
            self.performance.prefill_tokens_per_second,
            "prefill_tokens_per_second",
        )?;
        validate_positive_f64(
            self.performance.decode_tokens_per_second,
            "decode_tokens_per_second",
        )?;
        validate_positive_f64(self.performance.power_watts, "power_watts")?;
        validate_positive_f64(self.performance.tokens_per_joule, "tokens_per_joule")?;

        if !self.performance.has_any_measurement() {
            return Err(EvidenceError::Missing("performance"));
        }

        if let (Some(p50), Some(p95)) = (
            self.performance.p50_latency_micros,
            self.performance.p95_latency_micros,
        ) && p95 < p50
        {
            return invalid("p95_latency_micros", "cannot be lower than p50 latency");
        }

        if self.caches.ple_prefetch_useful_rows > self.caches.ple_prefetched_rows {
            return invalid(
                "ple_prefetch_useful_rows",
                "cannot exceed prefetched row count",
            );
        }

        if let Some(speculation) = &self.speculation {
            require_nonempty(&speculation.strategy, "speculation.strategy")?;
            if speculation.proposed_tokens == 0 {
                return invalid("speculation.proposed_tokens", "must be nonzero");
            }
            if speculation.accepted_tokens > speculation.proposed_tokens {
                return invalid(
                    "speculation.accepted_tokens",
                    "cannot exceed proposed tokens",
                );
            }
            if speculation.verified_cycles == 0 {
                return invalid("speculation.verified_cycles", "must be nonzero");
            }
        }

        match self
            .correctness
            .as_ref()
            .ok_or(EvidenceError::Missing("correctness"))?
        {
            CorrectnessGate::Passed {
                oracle,
                checked_tokens,
            } => {
                require_nonempty(oracle, "correctness.oracle")?;
                if *checked_tokens == 0 {
                    return invalid("correctness.checked_tokens", "must be nonzero");
                }
            }
            CorrectnessGate::Failed {
                oracle,
                checked_tokens,
                reason,
            } => {
                require_nonempty(oracle, "correctness.oracle")?;
                require_nonempty(reason, "correctness.reason")?;
                if *checked_tokens == 0 {
                    return invalid("correctness.checked_tokens", "must be nonzero");
                }
            }
        }

        Ok(())
    }
}

impl PerformanceMetrics {
    fn has_any_measurement(&self) -> bool {
        self.prefill_tokens_per_second.is_some()
            || self.decode_tokens_per_second.is_some()
            || self.ttft_micros.is_some()
            || self.p50_latency_micros.is_some()
            || self.p95_latency_micros.is_some()
            || self.unified_memory_bytes.is_some()
            || self.power_watts.is_some()
            || self.tokens_per_joule.is_some()
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), EvidenceError> {
    if value.trim().is_empty() {
        Err(EvidenceError::Missing(field))
    } else {
        Ok(())
    }
}

fn validate_positive_f64(value: Option<f64>, field: &'static str) -> Result<(), EvidenceError> {
    if let Some(value) = value
        && (!value.is_finite() || value <= 0.0)
    {
        return invalid(field, "must be finite and greater than zero");
    }
    Ok(())
}

fn invalid<T>(field: &'static str, reason: &'static str) -> Result<T, EvidenceError> {
    Err(EvidenceError::Invalid { field, reason })
}

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

    #[test]
    fn non_finite_throughput_is_rejected() {
        let mut record = valid_record();
        record.performance.decode_tokens_per_second = Some(f64::NAN);
        assert!(record.validate().is_err());
    }

    #[test]
    fn useful_prefetches_cannot_exceed_prefetches() {
        let mut record = valid_record();
        record.caches.ple_prefetched_rows = 5;
        record.caches.ple_prefetch_useful_rows = 6;
        assert!(record.validate().is_err());
    }

    #[test]
    fn failed_correctness_is_valid_evidence_when_explained() {
        let mut record = valid_record();
        record.correctness = Some(CorrectnessGate::Failed {
            oracle: "target-greedy-reference".into(),
            checked_tokens: 12,
            reason: "token mismatch at position 12".into(),
        });
        assert!(record.validate().is_ok());
    }
}
