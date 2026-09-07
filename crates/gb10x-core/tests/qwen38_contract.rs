use gb10x_core::{Qwen38Config, Qwen38ConfigError};
use serde_json::{Value, json};

const QWEN38_FIXTURE: &str = include_str!("../../../tests/fixtures/qwen38-flash-next-config.json");

fn fixture_value() -> Value {
    serde_json::from_str(QWEN38_FIXTURE).expect("Qwen3.8 fixture must be valid JSON")
}

fn parse_and_validate(value: &Value) -> Result<(), Qwen38ConfigError> {
    let json = serde_json::to_string(value).expect("mutated fixture must serialize");
    Qwen38Config::from_json_str(&json)?.validate_exact_contract()
}

fn value_at_mut<'a>(root: &'a mut Value, path: &[&str]) -> &'a mut Value {
    let mut value = root;
    for component in path {
        value = value
            .get_mut(*component)
            .unwrap_or_else(|| panic!("fixture must contain {component}"));
    }
    value
}

fn remove_at(root: &mut Value, parent_path: &[&str], field: &str) {
    value_at_mut(root, parent_path)
        .as_object_mut()
        .expect("fixture parent must be an object")
        .remove(field)
        .unwrap_or_else(|| panic!("fixture must contain {field}"));
}

#[test]
fn fixture_matches_exact_qwen38_contract() {
    Qwen38Config::from_json_str(QWEN38_FIXTURE)
        .expect("Qwen3.8 fixture must parse")
        .validate_exact_contract()
        .expect("Qwen3.8 fixture must match the exact contract");
}

#[test]
fn rejects_each_changed_execution_field() {
    let cases = [
        (
            "text_config.hidden_act",
            &["text_config", "hidden_act"][..],
            json!("relu"),
        ),
        (
            "text_config.attention_bias",
            &["text_config", "attention_bias"][..],
            json!(true),
        ),
        (
            "text_config.dtype",
            &["text_config", "dtype"][..],
            json!("float32"),
        ),
        (
            "text_config.rope_parameters.rope_type",
            &["text_config", "rope_parameters", "rope_type"][..],
            json!("linear"),
        ),
        (
            "text_config.rope_parameters.partial_rotary_factor",
            &["text_config", "rope_parameters", "partial_rotary_factor"][..],
            json!(0.5),
        ),
    ];

    let mut failures = Vec::new();
    for (field, path, replacement) in cases {
        let mut fixture = fixture_value();
        *value_at_mut(&mut fixture, path) = replacement;

        match parse_and_validate(&fixture) {
            Ok(()) => failures.push(format!("{field} mutation was accepted")),
            Err(error) if !error.to_string().contains(field) => failures.push(format!(
                "{field} mutation returned an error that did not name the field: {error}"
            )),
            Err(_) => {}
        }
    }

    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn rejects_missing_execution_fields() {
    let cases = [
        ("text_config.hidden_act", &["text_config"][..], "hidden_act"),
        (
            "text_config.attention_bias",
            &["text_config"][..],
            "attention_bias",
        ),
        ("text_config.dtype", &["text_config"][..], "dtype"),
        (
            "text_config.rope_parameters.rope_type",
            &["text_config", "rope_parameters"][..],
            "rope_type",
        ),
        (
            "text_config.rope_parameters.partial_rotary_factor",
            &["text_config", "rope_parameters"][..],
            "partial_rotary_factor",
        ),
    ];

    let mut failures = Vec::new();
    for (field, parent_path, key) in cases {
        let mut fixture = fixture_value();
        remove_at(&mut fixture, parent_path, key);

        match parse_and_validate(&fixture) {
            Ok(()) => failures.push(format!("missing {field} was accepted")),
            Err(error) if !error.to_string().contains(field) => failures.push(format!(
                "missing {field} returned an error that did not name the field: {error}"
            )),
            Err(_) => {}
        }
    }

    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn rejects_wrong_json_types_for_execution_fields() {
    let cases = [
        (
            "text_config.hidden_act",
            &["text_config", "hidden_act"][..],
            json!(false),
        ),
        (
            "text_config.attention_bias",
            &["text_config", "attention_bias"][..],
            json!("false"),
        ),
        (
            "text_config.dtype",
            &["text_config", "dtype"][..],
            json!(16),
        ),
        (
            "text_config.rope_parameters.rope_type",
            &["text_config", "rope_parameters", "rope_type"][..],
            json!(false),
        ),
        (
            "text_config.rope_parameters.partial_rotary_factor",
            &["text_config", "rope_parameters", "partial_rotary_factor"][..],
            json!("0.25"),
        ),
    ];

    let mut failures = Vec::new();
    for (field, path, replacement) in cases {
        let mut fixture = fixture_value();
        *value_at_mut(&mut fixture, path) = replacement;

        match parse_and_validate(&fixture) {
            Ok(()) => failures.push(format!("wrong JSON type for {field} was accepted")),
            Err(error) if !error.to_string().contains(field) => failures.push(format!(
                "wrong JSON type for {field} returned an error that did not name the field: {error}"
            )),
            Err(_) => {}
        }
    }

    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
