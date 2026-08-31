//! Pinned Qwen3.8-Flash-Next architecture contract used by GB10X.

use serde_json::Value;
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Per-layer mixer kind in the released Qwen3.8-Flash-Next text stack.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38LayerType {
    /// Gated DeltaNet / linear-attention layer.
    LinearAttention,
    /// Qwen Sparse Attention / full-attention layer.
    FullAttention,
}

/// Error while parsing or validating the one Qwen architecture supported by GB10X.
#[derive(Debug, Error)]
pub enum Qwen38ConfigError {
    /// Configuration file could not be read.
    #[error("failed to read {path}: {source}")]
    Io {
        /// Path that could not be read.
        path: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// JSON syntax was invalid.
    #[error("invalid Qwen3.8 config JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Required field was absent or had the wrong JSON type.
    #[error("missing or invalid field {0}")]
    Field(&'static str),
    /// Parsed architecture differs from GB10X's pinned model contract.
    #[error("Qwen3.8-Flash-Next contract mismatch: {0}")]
    Contract(String),
}

/// Parsed architecture fields that materially determine the GB10X execution plan.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen38Config {
    /// Outer Transformers model type.
    pub model_type: String,
    /// Text-stack model type.
    pub text_model_type: String,
    /// Hidden-state width.
    pub hidden_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Number of text layers.
    pub num_hidden_layers: usize,
    /// Fixed per-layer mixer schedule.
    pub layer_types: Vec<Qwen38LayerType>,
    /// Interval between QSA/full-attention layers.
    pub full_attention_interval: usize,
    /// Query-head count.
    pub num_attention_heads: usize,
    /// KV-head count.
    pub num_key_value_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Partial RoPE factor.
    pub partial_rotary_factor: f64,
    /// GDN key-head count.
    pub linear_num_key_heads: usize,
    /// GDN value-head count.
    pub linear_num_value_heads: usize,
    /// GDN key-head dimension.
    pub linear_key_head_dim: usize,
    /// GDN value-head dimension.
    pub linear_value_head_dim: usize,
    /// GDN convolution width.
    pub linear_conv_kernel_dim: usize,
    /// Routed expert count.
    pub num_experts: usize,
    /// Routed experts selected per token.
    pub num_experts_per_tok: usize,
    /// Routed-expert hidden width.
    pub moe_intermediate_size: usize,
    /// Shared-expert hidden width.
    pub shared_expert_intermediate_size: usize,
    /// Hyperconnection stream count.
    pub hc_count: usize,
    /// Hyperconnection low-rank width.
    pub hc_lowrank: usize,
    /// Zero-based layer carrying PLE.
    pub ple_layer: usize,
    /// Total PLE embedding width.
    pub ple_embed_dim: usize,
    /// PLE convolution width.
    pub ple_conv_kernel_size: usize,
    /// Maximum n-gram order.
    pub ngram_size: usize,
    /// Hash heads per n-gram order.
    pub heads_per_ngram: usize,
    /// Number of released PLE tensor shards.
    pub split_ngram_parts: usize,
    /// Base size of each n-gram vocabulary table.
    pub ngram_vocab_size_base: usize,
    /// Required PLE vocabulary-row alignment.
    pub make_ngram_vocab_size_divisible_by: usize,
    /// QSA indexer query-head count.
    pub indexer_n_heads: usize,
    /// QSA indexer KV-head count.
    pub indexer_kv_heads: usize,
    /// QSA indexer head dimension.
    pub indexer_head_dim: usize,
    /// QSA indexer compression ratio.
    pub indexer_compress_ratio: usize,
    /// Number of selected tokens retained by QSA.
    pub indexer_budget: usize,
    /// Maximum trained position envelope.
    pub max_position_embeddings: usize,
    /// End-of-sequence token ID used by the text config.
    pub eos_token_id: u32,
    /// Output gating function name.
    pub output_gate_type: String,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE theta.
    pub rope_theta: f64,
    /// Number of MTP hidden layers.
    pub mtp_num_hidden_layers: usize,
}

impl Qwen38Config {
    /// Load and parse `config.json` from a model directory.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, Qwen38ConfigError> {
        let path = model_dir.as_ref().join("config.json");
        let bytes = fs::read(&path).map_err(|source| Qwen38ConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let value: Value = serde_json::from_slice(&bytes)?;
        Self::from_value(&value)
    }

    /// Parse a complete Qwen config JSON string.
    pub fn from_json_str(json: &str) -> Result<Self, Qwen38ConfigError> {
        let value: Value = serde_json::from_str(json)?;
        Self::from_value(&value)
    }

    fn from_value(root: &Value) -> Result<Self, Qwen38ConfigError> {
        let model_type = required_str(root, "model_type")?.to_owned();
        let text = root
            .get("text_config")
            .ok_or(Qwen38ConfigError::Field("text_config"))?;
        let text_model_type = required_str(text, "model_type")?.to_owned();
        let layer_types = required_array(text, "layer_types")?
            .iter()
            .map(|value| match value.as_str() {
                Some("linear_attention") => Ok(Qwen38LayerType::LinearAttention),
                Some("full_attention") => Ok(Qwen38LayerType::FullAttention),
                _ => Err(Qwen38ConfigError::Field("text_config.layer_types")),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let ple_layers = required_array(text, "ple_layer_ids")?;
        if ple_layers.len() != 1 {
            return Err(Qwen38ConfigError::Field("text_config.ple_layer_ids"));
        }
        let one_based_ple = value_usize(&ple_layers[0], "text_config.ple_layer_ids[0]")?;
        let ple_layer = one_based_ple.checked_sub(1).ok_or_else(|| {
            Qwen38ConfigError::Contract("PLE layer ID must be one-based and positive".to_owned())
        })?;

        let rope = text
            .get("rope_parameters")
            .ok_or(Qwen38ConfigError::Field("text_config.rope_parameters"))?;
        let eos = required_usize(text, "eos_token_id")?;
        let eos_token_id = u32::try_from(eos).map_err(|_| {
            Qwen38ConfigError::Contract(format!("eos_token_id {eos} does not fit u32"))
        })?;

        Ok(Self {
            model_type,
            text_model_type,
            hidden_size: required_usize(text, "hidden_size")?,
            vocab_size: required_usize(text, "vocab_size")?,
            num_hidden_layers: required_usize(text, "num_hidden_layers")?,
            layer_types,
            full_attention_interval: required_usize(text, "full_attention_interval")?,
            num_attention_heads: required_usize(text, "num_attention_heads")?,
            num_key_value_heads: required_usize(text, "num_key_value_heads")?,
            head_dim: required_usize(text, "head_dim")?,
            partial_rotary_factor: required_f64(text, "partial_rotary_factor")?,
            linear_num_key_heads: required_usize(text, "linear_num_key_heads")?,
            linear_num_value_heads: required_usize(text, "linear_num_value_heads")?,
            linear_key_head_dim: required_usize(text, "linear_key_head_dim")?,
            linear_value_head_dim: required_usize(text, "linear_value_head_dim")?,
            linear_conv_kernel_dim: required_usize(text, "linear_conv_kernel_dim")?,
            num_experts: required_usize(text, "num_experts")?,
            num_experts_per_tok: required_usize(text, "num_experts_per_tok")?,
            moe_intermediate_size: required_usize(text, "moe_intermediate_size")?,
            shared_expert_intermediate_size: required_usize(
                text,
                "shared_expert_intermediate_size",
            )?,
            hc_count: required_usize(text, "hc_count")?,
            hc_lowrank: required_usize(text, "hc_lowrank")?,
            ple_layer,
            ple_embed_dim: required_usize(text, "ple_embed_dim")?,
            ple_conv_kernel_size: required_usize(text, "ple_conv_kernel_size")?,
            ngram_size: required_usize(text, "ngram_size")?,
            heads_per_ngram: required_usize(text, "heads_per_ngram")?,
            split_ngram_parts: required_usize(text, "split_ngram_parts")?,
            ngram_vocab_size_base: required_usize(text, "ngram_vocab_size_base")?,
            make_ngram_vocab_size_divisible_by: required_usize(
                text,
                "make_ngram_vocab_size_divisible_by",
            )?,
            indexer_n_heads: required_usize(text, "indexer_n_heads")?,
            indexer_kv_heads: required_usize(text, "indexer_kv_heads")?,
            indexer_head_dim: required_usize(text, "indexer_head_dim")?,
            indexer_compress_ratio: required_usize(text, "indexer_compress_ratio")?,
            indexer_budget: required_usize(text, "indexer_budget")?,
            max_position_embeddings: required_usize(text, "max_position_embeddings")?,
            eos_token_id,
            output_gate_type: required_str(text, "output_gate_type")?.to_owned(),
            rms_norm_eps: required_f64(text, "rms_norm_eps")?,
            rope_theta: required_f64(rope, "rope_theta")?,
            mtp_num_hidden_layers: required_usize(text, "mtp_num_hidden_layers")?,
        })
    }

    /// Validate every immutable field GB10X bakes into its Qwen3.8 execution plan.
    pub fn validate_exact_contract(&self) -> Result<(), Qwen38ConfigError> {
        expect_str("model_type", &self.model_type, "qwen4_exp")?;
        expect_str("text model_type", &self.text_model_type, "qwen4_exp_text")?;
        expect_usize("hidden_size", self.hidden_size, 2560)?;
        expect_usize("vocab_size", self.vocab_size, 248_320)?;
        expect_usize("num_hidden_layers", self.num_hidden_layers, 48)?;
        expect_usize("full_attention_interval", self.full_attention_interval, 4)?;
        expect_usize("num_attention_heads", self.num_attention_heads, 24)?;
        expect_usize("num_key_value_heads", self.num_key_value_heads, 2)?;
        expect_usize("head_dim", self.head_dim, 256)?;
        expect_f64("partial_rotary_factor", self.partial_rotary_factor, 0.25)?;
        expect_usize("linear_num_key_heads", self.linear_num_key_heads, 16)?;
        expect_usize("linear_num_value_heads", self.linear_num_value_heads, 48)?;
        expect_usize("linear_key_head_dim", self.linear_key_head_dim, 128)?;
        expect_usize("linear_value_head_dim", self.linear_value_head_dim, 128)?;
        expect_usize("linear_conv_kernel_dim", self.linear_conv_kernel_dim, 4)?;
        expect_usize("num_experts", self.num_experts, 512)?;
        expect_usize("num_experts_per_tok", self.num_experts_per_tok, 10)?;
        expect_usize("moe_intermediate_size", self.moe_intermediate_size, 640)?;
        expect_usize(
            "shared_expert_intermediate_size",
            self.shared_expert_intermediate_size,
            640,
        )?;
        expect_usize("hc_count", self.hc_count, 4)?;
        expect_usize("hc_lowrank", self.hc_lowrank, 320)?;
        expect_usize("ple_layer", self.ple_layer, 1)?;
        expect_usize("ple_embed_dim", self.ple_embed_dim, 2560)?;
        expect_usize("ple_conv_kernel_size", self.ple_conv_kernel_size, 4)?;
        expect_usize("ngram_size", self.ngram_size, 3)?;
        expect_usize("heads_per_ngram", self.heads_per_ngram, 8)?;
        expect_usize("split_ngram_parts", self.split_ngram_parts, 128)?;
        expect_usize(
            "ngram_vocab_size_base",
            self.ngram_vocab_size_base,
            20_000_000,
        )?;
        expect_usize(
            "make_ngram_vocab_size_divisible_by",
            self.make_ngram_vocab_size_divisible_by,
            128,
        )?;
        expect_usize("indexer_n_heads", self.indexer_n_heads, 4)?;
        expect_usize("indexer_kv_heads", self.indexer_kv_heads, 1)?;
        expect_usize("indexer_head_dim", self.indexer_head_dim, 128)?;
        expect_usize("indexer_compress_ratio", self.indexer_compress_ratio, 4)?;
        expect_usize("indexer_budget", self.indexer_budget, 2048)?;
        expect_usize(
            "max_position_embeddings",
            self.max_position_embeddings,
            262_144,
        )?;
        if self.eos_token_id != 248_044 {
            return Err(contract_mismatch(
                "eos_token_id",
                self.eos_token_id,
                248_044_u32,
            ));
        }
        expect_str("output_gate_type", &self.output_gate_type, "sigmoid")?;
        expect_f64("rms_norm_eps", self.rms_norm_eps, 0.000001)?;
        expect_f64("rope_theta", self.rope_theta, 10_000_000.0)?;
        expect_usize("mtp_num_hidden_layers", self.mtp_num_hidden_layers, 1)?;

        if self.layer_types.len() != 48 {
            return Err(Qwen38ConfigError::Contract(format!(
                "layer_types has {} entries, expected 48",
                self.layer_types.len()
            )));
        }
        for (index, actual) in self.layer_types.iter().enumerate() {
            let expected = if index % 4 == 3 {
                Qwen38LayerType::FullAttention
            } else {
                Qwen38LayerType::LinearAttention
            };
            if *actual != expected {
                return Err(Qwen38ConfigError::Contract(format!(
                    "layer {index} is {actual:?}, expected {expected:?}"
                )));
            }
        }

        let ple_rows = self.ple_rows_per_token();
        if ple_rows == 0 || !self.ple_embed_dim.is_multiple_of(ple_rows) {
            return Err(Qwen38ConfigError::Contract(
                "PLE width is not divisible by selected row count".to_owned(),
            ));
        }
        Ok(())
    }

    /// Rotary dimensions operated on by the partial-RoPE path.
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor).round() as usize
    }

    /// Number of independent PLE rows selected for one token.
    pub fn ple_rows_per_token(&self) -> usize {
        self.ngram_size
            .saturating_sub(1)
            .saturating_mul(self.heads_per_ngram)
    }

    /// BF16 vector width of one selected PLE row.
    pub fn ple_row_width(&self) -> usize {
        let rows = self.ple_rows_per_token();
        if rows == 0 {
            return 0;
        }
        self.ple_embed_dim / rows
    }

    /// Useful BF16 PLE payload selected per token, excluding storage-alignment amplification.
    pub fn ple_bf16_bytes_per_token(&self) -> usize {
        self.ple_rows_per_token()
            .saturating_mul(self.ple_row_width())
            .saturating_mul(2)
    }

    /// Selected BF16 K+V bytes consumed by one QSA layer for its exact sparse budget.
    pub fn qsa_selected_kv_bf16_bytes(&self) -> usize {
        self.indexer_budget
            .saturating_mul(self.num_key_value_heads)
            .saturating_mul(self.head_dim)
            .saturating_mul(2)
            .saturating_mul(2)
    }
}

fn required_usize(value: &Value, name: &'static str) -> Result<usize, Qwen38ConfigError> {
    value
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|raw| usize::try_from(raw).ok())
        .ok_or(Qwen38ConfigError::Field(name))
}

fn required_f64(value: &Value, name: &'static str) -> Result<f64, Qwen38ConfigError> {
    value
        .get(name)
        .and_then(Value::as_f64)
        .filter(|raw| raw.is_finite())
        .ok_or(Qwen38ConfigError::Field(name))
}

fn required_str<'a>(value: &'a Value, name: &'static str) -> Result<&'a str, Qwen38ConfigError> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or(Qwen38ConfigError::Field(name))
}

fn required_array<'a>(
    value: &'a Value,
    name: &'static str,
) -> Result<&'a [Value], Qwen38ConfigError> {
    value
        .get(name)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or(Qwen38ConfigError::Field(name))
}

fn value_usize(value: &Value, name: &'static str) -> Result<usize, Qwen38ConfigError> {
    value
        .as_u64()
        .and_then(|raw| usize::try_from(raw).ok())
        .ok_or(Qwen38ConfigError::Field(name))
}

fn expect_usize(name: &str, actual: usize, expected: usize) -> Result<(), Qwen38ConfigError> {
    if actual == expected {
        Ok(())
    } else {
        Err(contract_mismatch(name, actual, expected))
    }
}

fn expect_f64(name: &str, actual: f64, expected: f64) -> Result<(), Qwen38ConfigError> {
    if actual.to_bits() == expected.to_bits() {
        Ok(())
    } else {
        Err(contract_mismatch(name, actual, expected))
    }
}

fn expect_str(name: &str, actual: &str, expected: &str) -> Result<(), Qwen38ConfigError> {
    if actual == expected {
        Ok(())
    } else {
        Err(contract_mismatch(name, actual, expected))
    }
}

fn contract_mismatch(
    name: &str,
    actual: impl std::fmt::Display,
    expected: impl std::fmt::Display,
) -> Qwen38ConfigError {
    Qwen38ConfigError::Contract(format!("{name}={actual}, expected {expected}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_config() -> Qwen38Config {
        Qwen38Config::from_json_str(include_str!(
            "../../../tests/fixtures/qwen38-flash-next-config.json"
        ))
        .expect("official-shape fixture must parse")
    }

    #[test]
    fn released_fixture_matches_gb10x_contract() {
        let config = fixture_config();
        config
            .validate_exact_contract()
            .expect("released architecture must validate");
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_hidden_layers, 48);
        assert_eq!(config.num_experts, 512);
        assert_eq!(config.num_experts_per_tok, 10);
        assert_eq!(config.max_position_embeddings, 262_144);
    }

    #[test]
    fn rejects_changed_expert_count() {
        let mut config = fixture_config();
        config.num_experts = 256;
        assert!(config.validate_exact_contract().is_err());
    }

    #[test]
    fn rejects_layer_schedule_drift() {
        let mut config = fixture_config();
        config.layer_types[0] = Qwen38LayerType::FullAttention;
        assert!(config.validate_exact_contract().is_err());
    }

    #[test]
    fn derives_cache_critical_shapes() {
        let config = fixture_config();
        assert_eq!(config.rotary_dim(), 64);
        assert_eq!(config.ple_rows_per_token(), 16);
        assert_eq!(config.ple_row_width(), 160);
        assert_eq!(config.ple_bf16_bytes_per_token(), 5_120);
        assert_eq!(config.qsa_selected_kv_bf16_bytes(), 4 * 1024 * 1024);
    }
}
