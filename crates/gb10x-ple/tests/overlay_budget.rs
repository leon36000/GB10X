use gb10x_ple::{OverlayAdmissionBudget, PlePackError, plan_exact_layout_with_budget};

#[test]
fn admission_stops_at_the_hot_row_cap_in_deterministic_trace_order() {
    let trace = vec![vec![9, 3, 7, 3], vec![9, 7, 11], vec![2, 1], vec![7, 9]];
    let budget = OverlayAdmissionBudget::with_hot_row_cap(3);

    let plan =
        plan_exact_layout_with_budget(40, 320, 4_096, &trace, budget).expect("bounded layout");

    assert_eq!(plan.hot_physical_order(), &[3, 7, 9]);
    assert_eq!(plan.hot_overlay_placements().len(), 3);
}

#[test]
fn admission_counts_block_padding_against_the_overlay_byte_cap() {
    let trace = vec![vec![6, 1, 8, 3, 5, 2, 4]];
    let budget = OverlayAdmissionBudget::with_hot_row_cap(40).with_overlay_byte_cap(64);

    let plan = plan_exact_layout_with_budget(40, 16, 64, &trace, budget).expect("bounded layout");

    assert_eq!(plan.hot_physical_order(), &[1, 2, 3, 4]);
    assert_eq!(plan.hot_overlay_storage_bytes().unwrap(), 64);
}

#[test]
fn budget_does_not_hide_a_later_out_of_range_trace_row() {
    let budget = OverlayAdmissionBudget::with_hot_row_cap(1);
    let error = plan_exact_layout_with_budget(4, 16, 64, &[vec![1, 2], vec![4]], budget)
        .expect_err("all trace rows remain validated even after admission is full");

    assert_eq!(
        error,
        PlePackError::TraceRowOutOfRange {
            row: 4,
            row_count: 4,
        }
    );
}
