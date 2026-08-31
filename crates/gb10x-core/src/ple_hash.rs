//! Exact Qwen3.8-Flash-Next PLE hashing and speculative token-window transactions.

use thiserror::Error;

const QWEN38_NGRAM_SIZE: usize = 3;
const QWEN38_HEADS_PER_ORDER: usize = 8;
const QWEN38_PLE_HEADS: usize = 16;

/// Failure while constructing or advancing the exact Qwen3.8 PLE hash state.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum PleHashError {
    /// A constructor argument does not match the fixed Qwen3.8 PLE shape.
    #[error("unsupported Qwen3.8 PLE shape: {0}")]
    UnsupportedShape(&'static str),
    /// Hash-table metadata is invalid or internally inconsistent.
    #[error("invalid Qwen3.8 PLE hash plan: {0}")]
    InvalidPlan(&'static str),
    /// Hash output does not fit the compact logical row identifier.
    #[error("Qwen3.8 PLE row ID {0} does not fit u32")]
    RowOverflow(u64),
    /// Token-window transaction state is invalid for the requested operation.
    #[error("invalid Qwen3.8 PLE transaction: {0}")]
    Transaction(&'static str),
    /// Plan and token-window immutable parameters do not match.
    #[error("Qwen3.8 PLE plan/window mismatch")]
    IncompatibleWindow,
}

/// Immutable exact hash metadata for Qwen3.8-Flash-Next PLE lookup.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PleHashPlan {
    multipliers: [u64; QWEN38_NGRAM_SIZE],
    vocab_sizes: [u64; QWEN38_PLE_HEADS],
    offsets: [u64; QWEN38_PLE_HEADS],
    eos_token_id: u32,
}

impl PleHashPlan {
    /// Construct and validate the released Qwen3.8 PLE hash-table shape.
    ///
    /// The public constructor accepts vectors because checkpoint metadata is naturally decoded
    /// that way, then converts them to fixed arrays so the token hot path has no dynamic shape.
    pub fn new(
        multipliers: Vec<u64>,
        vocab_sizes: Vec<u64>,
        offsets: Vec<u64>,
        heads_per_ngram: usize,
        ngram_size: usize,
        eos_token_id: u32,
    ) -> Result<Self, PleHashError> {
        if ngram_size != QWEN38_NGRAM_SIZE {
            return Err(PleHashError::UnsupportedShape("ngram_size must equal 3"));
        }
        if heads_per_ngram != QWEN38_HEADS_PER_ORDER {
            return Err(PleHashError::UnsupportedShape(
                "heads_per_ngram must equal 8",
            ));
        }

        let multipliers: [u64; QWEN38_NGRAM_SIZE] = multipliers
            .try_into()
            .map_err(|_| PleHashError::UnsupportedShape("exactly 3 multipliers are required"))?;
        let vocab_sizes: [u64; QWEN38_PLE_HEADS] = vocab_sizes
            .try_into()
            .map_err(|_| PleHashError::UnsupportedShape("exactly 16 vocab sizes are required"))?;
        let offsets: [u64; QWEN38_PLE_HEADS] = offsets
            .try_into()
            .map_err(|_| PleHashError::UnsupportedShape("exactly 16 offsets are required"))?;

        if multipliers.contains(&0) {
            return Err(PleHashError::InvalidPlan("multipliers must be nonzero"));
        }
        if vocab_sizes.contains(&0) {
            return Err(PleHashError::InvalidPlan("vocab sizes must be nonzero"));
        }

        for head in 1..QWEN38_PLE_HEADS {
            let expected = offsets[head - 1]
                .checked_add(vocab_sizes[head - 1])
                .ok_or(PleHashError::InvalidPlan("head table offset overflow"))?;
            if offsets[head] != expected {
                return Err(PleHashError::InvalidPlan(
                    "head tables must be contiguous",
                ));
            }
        }
        let final_end = offsets[QWEN38_PLE_HEADS - 1]
            .checked_add(vocab_sizes[QWEN38_PLE_HEADS - 1])
            .ok_or(PleHashError::InvalidPlan("final head table overflow"))?;
        if final_end > u32::MAX as u64 + 1 {
            return Err(PleHashError::InvalidPlan(
                "logical PLE row space exceeds u32",
            ));
        }

        Ok(Self {
            multipliers,
            vocab_sizes,
            offsets,
            eos_token_id,
        })
    }

    /// Return the per-head vocabulary-table sizes.
    pub fn vocab_sizes(&self) -> &[u64; QWEN38_PLE_HEADS] {
        &self.vocab_sizes
    }

    /// Return the per-head logical row offsets.
    pub fn offsets(&self) -> &[u64; QWEN38_PLE_HEADS] {
        &self.offsets
    }

    /// Hash one token using the current left context, then append that token to the window.
    ///
    /// Multiplication intentionally uses wrapping arithmetic to match the checkpoint/reference
    /// implementation exactly. The returned rows are ordered by the 16 PLE hash heads.
    pub fn rows_for_token(
        &self,
        window: &mut PleTokenWindow,
        token: u32,
    ) -> Result<[u32; QWEN38_PLE_HEADS], PleHashError> {
        if window.eos_token_id != self.eos_token_id {
            return Err(PleHashError::IncompatibleWindow);
        }

        let current = token as u64;
        let mut rows = [0_u32; QWEN38_PLE_HEADS];

        for order in 2..=QWEN38_NGRAM_SIZE {
            let mut mixed = current.wrapping_mul(self.multipliers[0]);
            for previous in 1..order {
                mixed ^= (window.previous[previous - 1] as u64)
                    .wrapping_mul(self.multipliers[previous]);
            }

            let first_head = (order - 2) * QWEN38_HEADS_PER_ORDER;
            for (head, row_slot) in rows
                .iter_mut()
                .enumerate()
                .skip(first_head)
                .take(QWEN38_HEADS_PER_ORDER)
            {
                let row = self.offsets[head] + mixed % self.vocab_sizes[head];
                *row_slot = u32::try_from(row).map_err(|_| PleHashError::RowOverflow(row))?;
            }
        }

        window.push(token);
        Ok(rows)
    }
}

/// Exact two-token left-context window used by the Qwen3.8 PLE hash.
///
/// The window also owns one rollback snapshot so MTP/speculative verification can stage multiple
/// future tokens and later commit only the accepted prefix without recomputing unrelated state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PleTokenWindow {
    previous: [u32; QWEN38_NGRAM_SIZE - 1],
    rollback: Option<[u32; QWEN38_NGRAM_SIZE - 1]>,
    eos_token_id: u32,
}

impl PleTokenWindow {
    /// Create an empty Qwen3.8 PLE window whose missing left context is padded with EOS.
    pub fn new(ngram_size: usize, eos_token_id: u32) -> Result<Self, PleHashError> {
        if ngram_size != QWEN38_NGRAM_SIZE {
            return Err(PleHashError::UnsupportedShape("ngram_size must equal 3"));
        }
        Ok(Self {
            previous: [eos_token_id; QWEN38_NGRAM_SIZE - 1],
            rollback: None,
            eos_token_id,
        })
    }

    /// Start one speculative append transaction.
    pub fn begin_append(&mut self) -> Result<(), PleHashError> {
        if self.rollback.is_some() {
            return Err(PleHashError::Transaction(
                "an append transaction is already active",
            ));
        }
        self.rollback = Some(self.previous);
        Ok(())
    }

    /// Commit all tokens staged since the current transaction began.
    pub fn commit_append(&mut self) -> Result<(), PleHashError> {
        if self.rollback.take().is_none() {
            return Err(PleHashError::Transaction(
                "no append transaction is active",
            ));
        }
        Ok(())
    }

    /// Commit only `accepted_tokens` from the beginning of the current speculative transaction.
    ///
    /// The method restores the pre-transaction state and replays only the accepted target prefix,
    /// guaranteeing that rejected draft suffixes leave no PLE history behind.
    pub fn commit_append_prefix(&mut self, accepted_tokens: &[u32]) -> Result<(), PleHashError> {
        let original = self.rollback.take().ok_or(PleHashError::Transaction(
            "no append transaction is active",
        ))?;
        self.previous = original;
        for &token in accepted_tokens {
            self.push(token);
        }
        Ok(())
    }

    /// Abort the current speculative transaction and restore its exact starting state.
    pub fn abort_append(&mut self) -> Result<(), PleHashError> {
        let original = self.rollback.take().ok_or(PleHashError::Transaction(
            "no append transaction is active",
        ))?;
        self.previous = original;
        Ok(())
    }

    /// Clone committed state for prefix caching or equality/correctness checks.
    pub fn snapshot(&self) -> Result<Self, PleHashError> {
        if self.rollback.is_some() {
            return Err(PleHashError::Transaction(
                "cannot snapshot an active append transaction",
            ));
        }
        Ok(self.clone())
    }

    fn push(&mut self, token: u32) {
        if token == self.eos_token_id {
            self.previous.fill(self.eos_token_id);
            return;
        }
        self.previous[1] = self.previous[0];
        self.previous[0] = token;
    }
}

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
