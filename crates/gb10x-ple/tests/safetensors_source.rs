use gb10x_ple::{
    ExactPleRowSource, SafetensorsPleManifest, SafetensorsPlePart, SafetensorsPleSource,
};
use serde_json::{Value, json};
use std::fs;
use std::path::Path;
use tempfile::tempdir;

const ROW_ELEMENTS: u32 = 160;
const ROW_BYTES: usize = ROW_ELEMENTS as usize * 2;

fn row(seed: u8) -> Vec<u8> {
    (0..ROW_BYTES)
        .map(|index| seed.wrapping_add(index as u8))
        .collect()
}

fn write_safetensors(path: &Path, mut header: Value, data: &[u8]) {
    if let Some(object) = header.as_object_mut() {
        object.insert("__metadata__".into(), json!({"format": "pt"}));
    }
    let mut header_bytes = serde_json::to_vec(&header).unwrap();
    while !header_bytes.len().is_multiple_of(8) {
        header_bytes.push(b' ');
    }

    let mut file = Vec::with_capacity(8 + header_bytes.len() + data.len());
    file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(&header_bytes);
    file.extend_from_slice(data);
    fs::write(path, file).unwrap();
}

fn one_tensor_header(name: &str, dtype: &str, rows: u64, data_len: usize) -> Value {
    json!({
        name: {
            "dtype": dtype,
            "shape": [rows, ROW_ELEMENTS],
            "data_offsets": [0, data_len]
        }
    })
}

fn manifest() -> SafetensorsPleManifest {
    SafetensorsPleManifest {
        model_id: "Qwen/Qwen3.8-Flash-Next".into(),
        model_revision: "test-revision".into(),
        row_elements: ROW_ELEMENTS,
        parts: vec![
            SafetensorsPlePart {
                file: "part-000.safetensors".into(),
                tensor_name: "ple.part0".into(),
                logical_row_start: 0,
                row_count: 2,
            },
            SafetensorsPlePart {
                file: "part-001.safetensors".into(),
                tensor_name: "ple.part1".into(),
                logical_row_start: 2,
                row_count: 2,
            },
        ],
    }
}

fn write_valid_fixture(root: &Path) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let r0 = row(0x10);
    let r1 = row(0x20);
    let r2 = row(0x30);
    let r3 = row(0x40);

    let mut first = r0.clone();
    first.extend_from_slice(&r1);
    write_safetensors(
        &root.join("part-000.safetensors"),
        one_tensor_header("ple.part0", "BF16", 2, first.len()),
        &first,
    );

    let mut second = r2.clone();
    second.extend_from_slice(&r3);
    write_safetensors(
        &root.join("part-001.safetensors"),
        one_tensor_header("ple.part1", "BF16", 2, second.len()),
        &second,
    );

    (r0, r1, r2, r3)
}

#[test]
fn mmap_source_reads_exact_rows_across_physical_shard_boundary() {
    let dir = tempdir().unwrap();
    let (_, r1, r2, _) = write_valid_fixture(dir.path());
    let source = SafetensorsPleSource::open(dir.path(), &manifest()).expect("valid source");

    assert_eq!(source.row_count(), 4);
    assert_eq!(source.row_bytes(), ROW_BYTES as u32);
    assert_ne!(source.source_digest(), [0_u8; 32]);

    let mut dst = vec![0_u8; ROW_BYTES];
    source.read_exact_row(1, &mut dst).unwrap();
    assert_eq!(dst, r1);
    source.read_exact_row(2, &mut dst).unwrap();
    assert_eq!(dst, r2);
}

#[test]
fn non_bf16_tensor_is_rejected() {
    let dir = tempdir().unwrap();
    let data = vec![0_u8; ROW_BYTES * 2];
    write_safetensors(
        &dir.path().join("part-000.safetensors"),
        one_tensor_header("ple.part0", "F16", 2, data.len()),
        &data,
    );
    let mut manifest = manifest();
    manifest.parts.truncate(1);
    assert!(SafetensorsPleSource::open(dir.path(), &manifest).is_err());
}

#[test]
fn wrong_row_width_is_rejected() {
    let dir = tempdir().unwrap();
    let data = vec![0_u8; 2 * 159 * 2];
    let header = json!({
        "ple.part0": {
            "dtype": "BF16",
            "shape": [2, 159],
            "data_offsets": [0, data.len()]
        }
    });
    write_safetensors(&dir.path().join("part-000.safetensors"), header, &data);
    let mut manifest = manifest();
    manifest.parts.truncate(1);
    assert!(SafetensorsPleSource::open(dir.path(), &manifest).is_err());
}

#[test]
fn truncated_header_or_payload_is_rejected() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("part-000.safetensors"),
        1024_u64.to_le_bytes(),
    )
    .unwrap();
    let mut manifest = manifest();
    manifest.parts.truncate(1);
    assert!(SafetensorsPleSource::open(dir.path(), &manifest).is_err());

    let data = vec![0_u8; ROW_BYTES * 2 - 1];
    write_safetensors(
        &dir.path().join("part-000.safetensors"),
        one_tensor_header("ple.part0", "BF16", 2, ROW_BYTES * 2),
        &data,
    );
    assert!(SafetensorsPleSource::open(dir.path(), &manifest).is_err());
}

#[test]
fn overlapping_tensor_ranges_are_rejected() {
    let dir = tempdir().unwrap();
    let data = vec![0_u8; ROW_BYTES * 2];
    let header = json!({
        "ple.part0": {
            "dtype": "BF16",
            "shape": [1, ROW_ELEMENTS],
            "data_offsets": [0, ROW_BYTES]
        },
        "other": {
            "dtype": "BF16",
            "shape": [1, ROW_ELEMENTS],
            "data_offsets": [ROW_BYTES / 2, ROW_BYTES + ROW_BYTES / 2]
        }
    });
    write_safetensors(&dir.path().join("part-000.safetensors"), header, &data);
    let mut manifest = manifest();
    manifest.parts.truncate(1);
    manifest.parts[0].row_count = 1;
    assert!(SafetensorsPleSource::open(dir.path(), &manifest).is_err());
}

#[test]
fn source_digest_changes_when_referenced_tensor_bytes_change() {
    let dir = tempdir().unwrap();
    write_valid_fixture(dir.path());
    let manifest = manifest();
    let first = SafetensorsPleSource::open(dir.path(), &manifest)
        .unwrap()
        .source_digest();

    let path = dir.path().join("part-001.safetensors");
    let mut bytes = fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x80;
    fs::write(&path, bytes).unwrap();

    let second = SafetensorsPleSource::open(dir.path(), &manifest)
        .unwrap()
        .source_digest();
    assert_ne!(first, second);
}

#[test]
fn manifest_parts_must_cover_one_contiguous_logical_row_space() {
    let dir = tempdir().unwrap();
    write_valid_fixture(dir.path());
    let mut manifest = manifest();
    manifest.parts[1].logical_row_start = 3;
    assert!(SafetensorsPleSource::open(dir.path(), &manifest).is_err());
}
