use gb10x_ple::{QWEN38_FLASH_NEXT_REVISION, verify_qwen38_ple_hash_buffers};
use serde_json::{Map, Value, json};
use std::fs;
use std::path::Path;
use tempfile::tempdir;

const MULTIPLIERS: &str = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers";
const VOCAB_SIZES: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes";
const OFFSETS: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets";

const EXPECTED_MULTIPLIERS: [i64; 3] = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];
const EXPECTED_VOCAB_SIZES: [i64; 16] = [
    20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069, 20_000_077,
    20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159, 20_000_161, 20_000_171,
];
const EXPECTED_OFFSETS: [i64; 16] = [
    0,
    20_000_003,
    40_000_026,
    60_000_059,
    80_000_106,
    100_000_165,
    120_000_228,
    140_000_297,
    160_000_374,
    180_000_455,
    200_000_548,
    220_000_655,
    240_000_802,
    260_000_955,
    280_001_114,
    300_001_275,
];

fn encode_i64s(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn write_safetensors(path: &Path, mut tensors: Map<String, Value>, payload: &[u8]) {
    tensors.insert("__metadata__".into(), json!({"format": "pt"}));
    let mut header = serde_json::to_vec(&Value::Object(tensors)).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }

    let mut file = Vec::with_capacity(8 + header.len() + payload.len());
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    file.extend_from_slice(payload);
    fs::write(path, file).unwrap();
}

fn write_index(root: &Path, file_name: &str) {
    fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&json!({
            "metadata": {"total_size": 280},
            "weight_map": {
                MULTIPLIERS: file_name,
                VOCAB_SIZES: file_name,
                OFFSETS: file_name,
            }
        }))
        .unwrap(),
    )
    .unwrap();
}

fn write_matching_checkpoint(root: &Path) {
    let multiplier_bytes = encode_i64s(&EXPECTED_MULTIPLIERS);
    let vocab_size_bytes = encode_i64s(&EXPECTED_VOCAB_SIZES);
    let offset_bytes = encode_i64s(&EXPECTED_OFFSETS);
    let mut payload = multiplier_bytes.clone();
    payload.extend_from_slice(&vocab_size_bytes);
    payload.extend_from_slice(&offset_bytes);

    let mut tensors = Map::new();
    tensors.insert(
        MULTIPLIERS.into(),
        json!({"dtype": "I64", "shape": [3], "data_offsets": [0, multiplier_bytes.len()]}),
    );
    tensors.insert(
        VOCAB_SIZES.into(),
        json!({
            "dtype": "I64",
            "shape": [16],
            "data_offsets": [multiplier_bytes.len(), multiplier_bytes.len() + vocab_size_bytes.len()]
        }),
    );
    tensors.insert(
        OFFSETS.into(),
        json!({
            "dtype": "I64",
            "shape": [16],
            "data_offsets": [multiplier_bytes.len() + vocab_size_bytes.len(), payload.len()]
        }),
    );

    let file_name = "model-00001-of-00131.safetensors";
    write_safetensors(&root.join(file_name), tensors, &payload);
    write_index(root, file_name);
}

#[test]
fn physical_i64_hash_buffers_match_pinned_upstream_defaults() {
    let root = tempdir().unwrap();
    write_matching_checkpoint(root.path());

    let verification = verify_qwen38_ple_hash_buffers(root.path(), QWEN38_FLASH_NEXT_REVISION)
        .expect("matching physical hash buffers");

    assert_eq!(verification.buffers_verified, 3);
    assert_eq!(verification.files_read, 1);
    assert_eq!(verification.payload_bytes_read, 280);
    assert!(verification.header_bytes_read > 0);
    assert!(verification.header_bytes_read <= 16 * 1024 * 1024);
}

#[test]
fn changed_physical_multiplier_is_rejected() {
    let root = tempdir().unwrap();
    write_matching_checkpoint(root.path());

    let path = root.path().join("model-00001-of-00131.safetensors");
    let mut bytes = fs::read(&path).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    bytes[8 + header_len] ^= 1;
    fs::write(path, bytes).unwrap();

    assert!(verify_qwen38_ple_hash_buffers(root.path(), QWEN38_FLASH_NEXT_REVISION).is_err());
}

#[test]
fn duplicate_index_member_and_escaping_file_target_are_rejected() {
    let root = tempdir().unwrap();
    fs::write(
        root.path().join("model.safetensors.index.json"),
        format!(
            r#"{{"weight_map":{{"{MULTIPLIERS}":"hashes.safetensors","{VOCAB_SIZES}":"hashes.safetensors","{OFFSETS}":"hashes.safetensors"}},"weight_map":{{"{MULTIPLIERS}":"../escape.safetensors"}}}}"#
        ),
    )
    .unwrap();
    assert!(verify_qwen38_ple_hash_buffers(root.path(), QWEN38_FLASH_NEXT_REVISION).is_err());

    fs::write(
        root.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "weight_map": {
                MULTIPLIERS: "../escape.safetensors",
                VOCAB_SIZES: "hashes.safetensors",
                OFFSETS: "hashes.safetensors",
            }
        }))
        .unwrap(),
    )
    .unwrap();
    assert!(verify_qwen38_ple_hash_buffers(root.path(), QWEN38_FLASH_NEXT_REVISION).is_err());
}

#[test]
fn oversized_safetensors_header_is_rejected_before_payload_read() {
    let root = tempdir().unwrap();
    let file_name = "hashes.safetensors";
    fs::write(
        root.path().join(file_name),
        (16_u64 * 1024 * 1024 + 1).to_le_bytes(),
    )
    .unwrap();
    write_index(root.path(), file_name);

    assert!(verify_qwen38_ple_hash_buffers(root.path(), QWEN38_FLASH_NEXT_REVISION).is_err());
}

#[test]
fn hash_buffers_can_be_read_from_multiple_safe_safetensors_files() {
    let root = tempdir().unwrap();
    let multiplier_bytes = encode_i64s(&EXPECTED_MULTIPLIERS);
    let vocab_size_bytes = encode_i64s(&EXPECTED_VOCAB_SIZES);
    let offset_bytes = encode_i64s(&EXPECTED_OFFSETS);

    let mut multipliers = Map::new();
    multipliers.insert(
        MULTIPLIERS.into(),
        json!({"dtype": "I64", "shape": [3], "data_offsets": [0, multiplier_bytes.len()]}),
    );
    write_safetensors(
        &root.path().join("hashes-a.safetensors"),
        multipliers,
        &multiplier_bytes,
    );

    let mut remaining = vocab_size_bytes.clone();
    remaining.extend_from_slice(&offset_bytes);
    let mut ngram = Map::new();
    ngram.insert(
        VOCAB_SIZES.into(),
        json!({
            "dtype": "I64",
            "shape": [16],
            "data_offsets": [0, vocab_size_bytes.len()]
        }),
    );
    ngram.insert(
        OFFSETS.into(),
        json!({
            "dtype": "I64",
            "shape": [16],
            "data_offsets": [vocab_size_bytes.len(), remaining.len()]
        }),
    );
    write_safetensors(&root.path().join("hashes-b.safetensors"), ngram, &remaining);

    fs::write(
        root.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "weight_map": {
                MULTIPLIERS: "hashes-a.safetensors",
                VOCAB_SIZES: "hashes-b.safetensors",
                OFFSETS: "hashes-b.safetensors",
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let verification = verify_qwen38_ple_hash_buffers(root.path(), QWEN38_FLASH_NEXT_REVISION)
        .expect("safe split hash buffers");
    assert_eq!(verification.files_read, 2);
    assert_eq!(verification.payload_bytes_read, 280);
}

#[test]
fn hash_buffer_dtype_must_be_i64() {
    let root = tempdir().unwrap();
    let multiplier_bytes = encode_i64s(&EXPECTED_MULTIPLIERS);
    let vocab_size_bytes = encode_i64s(&EXPECTED_VOCAB_SIZES);
    let offset_bytes = encode_i64s(&EXPECTED_OFFSETS);
    let mut payload = multiplier_bytes.clone();
    payload.extend_from_slice(&vocab_size_bytes);
    payload.extend_from_slice(&offset_bytes);

    let mut tensors = Map::new();
    tensors.insert(
        MULTIPLIERS.into(),
        json!({"dtype": "F32", "shape": [3], "data_offsets": [0, multiplier_bytes.len()]}),
    );
    tensors.insert(
        VOCAB_SIZES.into(),
        json!({
            "dtype": "I64",
            "shape": [16],
            "data_offsets": [multiplier_bytes.len(), multiplier_bytes.len() + vocab_size_bytes.len()]
        }),
    );
    tensors.insert(
        OFFSETS.into(),
        json!({
            "dtype": "I64",
            "shape": [16],
            "data_offsets": [multiplier_bytes.len() + vocab_size_bytes.len(), payload.len()]
        }),
    );
    let file_name = "wrong-type.safetensors";
    write_safetensors(&root.path().join(file_name), tensors, &payload);
    write_index(root.path(), file_name);

    assert!(verify_qwen38_ple_hash_buffers(root.path(), QWEN38_FLASH_NEXT_REVISION).is_err());
}

#[test]
fn hash_buffer_verification_rejects_any_other_observed_revision_before_reading_files() {
    let root = tempdir().unwrap();
    assert!(verify_qwen38_ple_hash_buffers(root.path(), "not-the-pinned-revision").is_err());
}
