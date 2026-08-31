use gb10x_ple::{
    QWEN38_FLASH_NEXT_MODEL_ID, QWEN38_FLASH_NEXT_REVISION, QWEN38_PLE_PARTS,
    QWEN38_PLE_ROWS, QWEN38_PLE_ROWS_PER_PART, QWEN38_PLE_ROW_ELEMENTS,
    qwen38_ple_manifest_from_index,
};
use serde_json::{Map, Value, json};
use std::fs;
use std::path::Path;
use tempfile::tempdir;

const TENSOR_PREFIX: &str =
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";

fn tensor_name(index: usize) -> String {
    format!("{TENSOR_PREFIX}{index}.weight")
}

fn valid_weight_map() -> Map<String, Value> {
    (0..128)
        .map(|index| {
            (
                tensor_name(index),
                Value::String(format!(
                    "model-{:05}-of-00131.safetensors",
                    5 + index / 4
                )),
            )
        })
        .collect()
}

fn write_index(root: &Path, weight_map: Map<String, Value>) {
    let index = json!({
        "metadata": {"total_size": 1},
        "weight_map": weight_map,
    });
    fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index).unwrap(),
    )
    .unwrap();
}

#[test]
fn official_index_shape_builds_one_pinned_contiguous_manifest() {
    let dir = tempdir().unwrap();
    write_index(dir.path(), valid_weight_map());

    let manifest = qwen38_ple_manifest_from_index(dir.path(), QWEN38_FLASH_NEXT_REVISION)
        .expect("exact pinned index should produce a manifest");

    assert_eq!(manifest.model_id, QWEN38_FLASH_NEXT_MODEL_ID);
    assert_eq!(manifest.model_revision, QWEN38_FLASH_NEXT_REVISION);
    assert_eq!(manifest.row_elements, QWEN38_PLE_ROW_ELEMENTS);
    assert_eq!(manifest.parts.len(), QWEN38_PLE_PARTS);
    assert_eq!(QWEN38_PLE_ROWS_PER_PART * QWEN38_PLE_PARTS as u64, QWEN38_PLE_ROWS);

    for (index, part) in manifest.parts.iter().enumerate() {
        assert_eq!(part.tensor_name, tensor_name(index));
        assert_eq!(part.logical_row_start, index as u64 * QWEN38_PLE_ROWS_PER_PART);
        assert_eq!(part.row_count, QWEN38_PLE_ROWS_PER_PART);
    }
    let last = manifest.parts.last().unwrap();
    assert_eq!(last.logical_row_start + last.row_count, QWEN38_PLE_ROWS);
}

#[test]
fn any_revision_other_than_the_pinned_source_commit_is_rejected() {
    let dir = tempdir().unwrap();
    write_index(dir.path(), valid_weight_map());
    assert!(qwen38_ple_manifest_from_index(dir.path(), "deadbeef").is_err());
}

#[test]
fn missing_shard_index_is_rejected() {
    let dir = tempdir().unwrap();
    let mut map = valid_weight_map();
    map.remove(&tensor_name(73));
    write_index(dir.path(), map);
    assert!(qwen38_ple_manifest_from_index(dir.path(), QWEN38_FLASH_NEXT_REVISION).is_err());
}

#[test]
fn one_based_config_id_is_not_accepted_as_state_dict_layer_two() {
    let dir = tempdir().unwrap();
    let mut map = valid_weight_map();
    map.remove(&tensor_name(64));
    map.insert(
        "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_64.weight".into(),
        Value::String("model-00021-of-00131.safetensors".into()),
    );
    write_index(dir.path(), map);
    assert!(qwen38_ple_manifest_from_index(dir.path(), QWEN38_FLASH_NEXT_REVISION).is_err());
}

#[test]
fn out_of_range_or_noncanonical_shard_names_are_rejected() {
    let dir = tempdir().unwrap();
    let mut map = valid_weight_map();
    map.insert(
        format!("{TENSOR_PREFIX}128.weight"),
        Value::String("model-00038-of-00131.safetensors".into()),
    );
    write_index(dir.path(), map);
    assert!(qwen38_ple_manifest_from_index(dir.path(), QWEN38_FLASH_NEXT_REVISION).is_err());

    let mut map = valid_weight_map();
    map.remove(&tensor_name(1));
    map.insert(
        format!("{TENSOR_PREFIX}01.weight"),
        Value::String("model-00005-of-00131.safetensors".into()),
    );
    write_index(dir.path(), map);
    assert!(qwen38_ple_manifest_from_index(dir.path(), QWEN38_FLASH_NEXT_REVISION).is_err());
}

#[test]
fn mapped_file_path_must_not_escape_model_directory() {
    let dir = tempdir().unwrap();
    let mut map = valid_weight_map();
    map.insert(tensor_name(9), Value::String("../escape.safetensors".into()));
    write_index(dir.path(), map);
    assert!(qwen38_ple_manifest_from_index(dir.path(), QWEN38_FLASH_NEXT_REVISION).is_err());
}

#[test]
fn missing_or_non_string_weight_map_entries_are_rejected() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("model.safetensors.index.json"),
        br#"{"metadata": {"total_size": 1}}"#,
    )
    .unwrap();
    assert!(qwen38_ple_manifest_from_index(dir.path(), QWEN38_FLASH_NEXT_REVISION).is_err());

    let mut map = valid_weight_map();
    map.insert(tensor_name(5), Value::Bool(true));
    write_index(dir.path(), map);
    assert!(qwen38_ple_manifest_from_index(dir.path(), QWEN38_FLASH_NEXT_REVISION).is_err());
}
