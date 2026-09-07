//! Numeric expectations come from the pinned upstream PyTorch methods, not this Rust hash.

use gb10x_core::{PleHashPlan, PleTokenWindow};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    schema_version: u32,
    plans: Vec<Plan>,
}

#[derive(Deserialize)]
struct Plan {
    name: String,
    ngram_size: usize,
    heads_per_ngram: usize,
    eos_token_id: u32,
    multipliers: Vec<u64>,
    vocab_sizes: Vec<u64>,
    offsets: Vec<u64>,
    traces: Vec<Trace>,
}

#[derive(Deserialize)]
struct Trace {
    name: String,
    tokens: Vec<u32>,
    rows: Vec<[u32; 16]>,
}

impl Plan {
    fn hash(&self) -> PleHashPlan {
        PleHashPlan::new(
            self.multipliers.clone(),
            self.vocab_sizes.clone(),
            self.offsets.clone(),
            self.heads_per_ngram,
            self.ngram_size,
            self.eos_token_id,
        )
        .expect("reference metadata must be accepted")
    }

    fn window(&self) -> PleTokenWindow {
        PleTokenWindow::new(self.ngram_size, self.eos_token_id).unwrap()
    }

    fn trace(&self, name: &str) -> &Trace {
        self.traces.iter().find(|trace| trace.name == name).unwrap()
    }
}

fn reference() -> Reference {
    let reference: Reference =
        serde_json::from_str(include_str!("../../../tests/fixtures/ple-reference.json")).unwrap();
    assert_eq!(reference.schema_version, 1);
    assert_eq!(reference.plans.len(), 2);
    for plan in &reference.plans {
        assert!(!plan.traces.is_empty());
        for trace in &plan.traces {
            assert!(!trace.tokens.is_empty());
            assert_eq!(trace.tokens.len(), trace.rows.len());
        }
    }
    reference
}

fn assert_trace(plan: &Plan, trace: &Trace) {
    let hash = plan.hash();
    let mut window = plan.window();
    for (index, &token) in trace.tokens.iter().enumerate() {
        assert_eq!(
            hash.rows_for_token(&mut window, token).unwrap(),
            trace.rows[index],
            "plan {}, trace {}, token {index} ({token})",
            plan.name,
            trace.name,
        );
    }
}

#[test]
fn matches_upstream_default_rows_and_eos_boundaries() {
    let reference = reference();
    let plan = &reference.plans[0];
    assert_eq!(plan.name, "upstream_defaults");
    for trace in &plan.traces {
        if trace.name != "u32_boundaries" {
            assert_trace(plan, trace);
        }
    }
}

#[test]
fn matches_upstream_signed_i64_wraparound() {
    let reference = reference();
    let plan = &reference.plans[1];
    assert_eq!(plan.name, "synthetic_signed_overflow");
    assert_trace(plan, plan.trace("overflow"));
}

#[test]
fn matches_upstream_over_full_u32_token_domain() {
    let reference = reference();
    let plan = &reference.plans[0];
    assert_trace(plan, plan.trace("u32_boundaries"));
}

#[test]
fn restored_context_matches_upstream_at_every_sequence_split() {
    let reference = reference();
    let plan = &reference.plans[0];
    let trace = plan.trace("eos_boundaries");
    let hash = plan.hash();
    for split in 0..=trace.tokens.len() {
        let mut prefill = plan.window();
        for &token in &trace.tokens[..split] {
            hash.rows_for_token(&mut prefill, token).unwrap();
        }
        let mut restored = prefill.snapshot().unwrap();
        for index in split..trace.tokens.len() {
            assert_eq!(
                hash.rows_for_token(&mut restored, trace.tokens[index])
                    .unwrap(),
                trace.rows[index],
                "split {split}, token {index}",
            );
        }
    }
}

fn stage_draft(plan: &Plan) -> PleTokenWindow {
    let hash = plan.hash();
    let trace = plan.trace("draft");
    let mut window = plan.window();
    for index in 0..2 {
        assert_eq!(
            hash.rows_for_token(&mut window, trace.tokens[index])
                .unwrap(),
            trace.rows[index],
        );
    }
    window.begin_append().unwrap();
    for index in 2..trace.tokens.len() {
        assert_eq!(
            hash.rows_for_token(&mut window, trace.tokens[index])
                .unwrap(),
            trace.rows[index],
        );
    }
    window
}

#[test]
fn aborted_draft_matches_upstream_recomputed_history() {
    let reference = reference();
    let plan = &reference.plans[0];
    let hash = plan.hash();
    let mut window = stage_draft(plan);
    window.abort_append().unwrap();
    let trace = plan.trace("aborted_prefix");
    for index in 2..trace.tokens.len() {
        assert_eq!(
            hash.rows_for_token(&mut window, trace.tokens[index])
                .unwrap(),
            trace.rows[index],
        );
    }
}

#[test]
fn partial_commit_with_eos_matches_upstream_recomputed_history() {
    let reference = reference();
    let plan = &reference.plans[0];
    let hash = plan.hash();
    let mut window = stage_draft(plan);
    let trace = plan.trace("accepted_prefix");
    window.commit_append_prefix(&trace.tokens[2..4]).unwrap();
    for index in 4..trace.tokens.len() {
        assert_eq!(
            hash.rows_for_token(&mut window, trace.tokens[index])
                .unwrap(),
            trace.rows[index],
        );
    }
}
