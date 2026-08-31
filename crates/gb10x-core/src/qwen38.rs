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
