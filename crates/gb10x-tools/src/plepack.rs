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
