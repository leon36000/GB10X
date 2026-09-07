//! Thin PLEPack CLI helpers over the exact storage library.

use gb10x_ple::{LayoutPlan, OverlayAdmissionBudget, PlePackError, plan_exact_layout_with_budget};
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
    plan_from_trace_json_with_budget(
        row_count,
        row_bytes,
        block_bytes,
        trace_json,
        OverlayAdmissionBudget::unbounded(),
    )
}

/// Parse a JSON array-of-arrays trace and build a deterministic exact PLEPack layout under a
/// caller-supplied overlay admission budget.
pub fn plan_from_trace_json_with_budget(
    row_count: u64,
    row_bytes: u32,
    block_bytes: u32,
    trace_json: &str,
    budget: OverlayAdmissionBudget,
) -> Result<LayoutPlan, PlanFromTraceError> {
    let trace = serde_json::from_str::<Vec<Vec<u32>>>(trace_json)?;
    Ok(plan_exact_layout_with_budget(
        row_count,
        row_bytes,
        block_bytes,
        &trace,
        budget,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gb10x_ple::OverlayAdmissionBudget;

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

    #[test]
    fn trace_json_admission_respects_the_requested_overlay_budget() {
        let trace = r#"[[9,3,7,3],[9,7,11],[2,1],[7,9]]"#;
        let budget = OverlayAdmissionBudget::with_hot_row_cap(2).with_overlay_byte_cap(4_096);

        let plan =
            plan_from_trace_json_with_budget(40, 320, 4_096, trace, budget).expect("bounded plan");

        assert_eq!(plan.hot_physical_order(), &[3, 7]);
    }
}
