//! Thin PLEPack CLI helpers over the exact storage library.

use gb10x_ple::{LayoutPlan, PlePackError, plan_exact_layout};
use thiserror::Error;

/// Failure while converting a JSON workload trace into an exact PLEPack layout plan.
#[derive(Debug, Error)]
pub enum PlanFromTraceError {
    /// Workload trace JSON was malformed.
    #[error("invalid PLEPack trace JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The decoded trace or requested geometry violated exact PLEPack constraints.
    #[error(transparent)]
    Layout(#[from] PlePackError),
}

/// Parse a JSON array-of-arrays trace and build the deterministic exact PLEPack layout.
pub fn plan_from_trace_json(
    row_count: u64,
    row_bytes: u32,
    block_bytes: u32,
    trace_json: &str,
) -> Result<LayoutPlan, PlanFromTraceError> {
    let trace = serde_json::from_str::<Vec<Vec<u32>>>(trace_json)?;
    Ok(plan_exact_layout(
        row_count,
        row_bytes,
        block_bytes,
        &trace,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_json_builds_deterministic_hot_overlay_plan() {
        let trace = r#"[[9,3,7,3],[9,7,11],[2,1],[7,9]]"#;
        let plan = plan_from_trace_json(40, 320, 4096, trace).expect("plan");
        assert_eq!(plan.hot_physical_order(), &[3, 7, 9, 11, 1, 2]);
        assert_eq!(plan.hot_overlay_placements().len(), 6);
    }

    #[test]
    fn invalid_trace_json_is_rejected() {
        assert!(plan_from_trace_json(40, 320, 4096, "not-json").is_err());
    }
}
