#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const EOS: u32 = 248_044;

    fn plan() -> PleHashPlan {
        let vocab_sizes = (0_u64..16).map(|index| 97 + index).collect::<Vec<_>>();
        let mut offsets = Vec::with_capacity(16);
        let mut next = 0_u64;
        for size in &vocab_sizes {
            offsets.push(next);
            next += *size;
        }
        PleHashPlan::new(vec![3, 5, 7], vocab_sizes, offsets, 8, 3, EOS)
            .expect("valid Qwen3.8-shaped PLE plan")
    }

    #[test]
    fn emits_exactly_sixteen_rows_per_token() {
        let plan = plan();
        let mut window = PleTokenWindow::new(3, EOS).expect("window");
        let rows = plan
            .rows_for_token(&mut window, 42)
            .expect("hashing must succeed");
        assert_eq!(rows.len(), 16);
    }

    #[test]
    fn abort_restores_previous_ngram_window() {
        let plan = plan();
        let mut window = PleTokenWindow::new(3, EOS).expect("window");
        plan.rows_for_token(&mut window, 10).unwrap();
        let before = window.snapshot().expect("clean snapshot");

        window.begin_append().expect("begin speculative append");
        plan.rows_for_token(&mut window, 20).unwrap();
        plan.rows_for_token(&mut window, 30).unwrap();
        window.abort_append().expect("abort speculative append");

        assert_eq!(window, before);
    }

    #[test]
    fn commit_prefix_matches_replaying_only_accepted_tokens() {
        let plan = plan();
        let mut speculative = PleTokenWindow::new(3, EOS).expect("window");
        let mut reference = PleTokenWindow::new(3, EOS).expect("window");

        for token in [11, 12] {
            plan.rows_for_token(&mut speculative, token).unwrap();
            plan.rows_for_token(&mut reference, token).unwrap();
        }

        speculative.begin_append().unwrap();
        for token in [21, 22, 23] {
            plan.rows_for_token(&mut speculative, token).unwrap();
        }
        speculative.commit_append_prefix(&[21, 22]).unwrap();

        for token in [21, 22] {
            plan.rows_for_token(&mut reference, token).unwrap();
        }
        assert_eq!(speculative, reference);
    }

    #[test]
    fn eos_resets_left_context() {
        let plan = plan();
        let mut after_history = PleTokenWindow::new(3, EOS).unwrap();
        for token in [1, 2, 3, EOS] {
            plan.rows_for_token(&mut after_history, token).unwrap();
        }

        let fresh = PleTokenWindow::new(3, EOS).unwrap();
        assert_eq!(after_history, fresh);
    }

    proptest! {
        #[test]
        fn every_emitted_row_stays_inside_its_head_table(tokens in prop::collection::vec(any::<u32>(), 1..64)) {
            let plan = plan();
            let mut window = PleTokenWindow::new(3, EOS).unwrap();
            for token in tokens {
                let rows = plan.rows_for_token(&mut window, token).unwrap();
                for (head, &row) in rows.iter().enumerate() {
                    let start = plan.offsets()[head];
                    let end = start + plan.vocab_sizes()[head];
                    prop_assert!((row as u64) >= start);
                    prop_assert!((row as u64) < end);
                }
            }
        }
    }
}
