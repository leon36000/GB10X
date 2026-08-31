#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn trace() -> Vec<Vec<u32>> {
        vec![vec![9, 3, 7, 3], vec![9, 7, 11], vec![2, 1], vec![7, 9]]
    }

    #[test]
    fn same_trace_produces_byte_identical_plan() {
        let first = plan_exact_layout(40, 320, 4096, &trace()).expect("layout");
        let second = plan_exact_layout(40, 320, 4096, &trace()).expect("layout");
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
    }

    #[test]
    fn every_logical_row_is_mapped_once() {
        let plan = plan_exact_layout(40, 320, 4096, &trace()).expect("layout");
        assert_eq!(plan.placements().len(), 40);
        let rows = plan
            .placements()
            .iter()
            .map(|placement| placement.logical_row)
            .collect::<BTreeSet<_>>();
        assert_eq!(rows.len(), 40);
        assert_eq!(rows.first().copied(), Some(0));
        assert_eq!(rows.last().copied(), Some(39));
    }

    #[test]
    fn no_row_crosses_block_boundary() {
        let plan = plan_exact_layout(40, 320, 4096, &trace()).expect("layout");
        for placement in plan.placements() {
            assert!(placement.offset_in_block as u64 + 320 <= 4096);
        }
    }

    #[test]
    fn first_observed_coaccess_group_is_physically_adjacent() {
        let plan = plan_exact_layout(40, 320, 4096, &trace()).expect("layout");
        assert_eq!(&plan.physical_order()[..3], &[3, 7, 9]);
    }

    #[test]
    fn rejects_impossible_block_geometry_and_out_of_range_trace_rows() {
        assert!(plan_exact_layout(40, 320, 256, &trace()).is_err());
        assert!(plan_exact_layout(40, 320, 4096, &[vec![40]]).is_err());
    }
}
