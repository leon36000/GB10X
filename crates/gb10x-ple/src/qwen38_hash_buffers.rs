//! Bounded physical verification of persistent Qwen3.8 PLE hash buffers.

use crate::PlePackIoError;
use crate::qwen38_manifest::QWEN38_FLASH_NEXT_REVISION;
use crate::safetensors::{parse_json_without_duplicate_members, read_safetensors_header_bounded};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

const SAFETENSORS_INDEX_FILE: &str = "model.safetensors.index.json";
const MAX_INDEX_BYTES: u64 = 16 * 1024 * 1024;
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const I64_BYTES: u64 = 8;

const LAYER_MULTIPLIERS_NAME: &str =
    "model.language_model.layers.1.ple.ple_embedding.layer_multipliers";
const NGRAM_HEADS_VOCAB_SIZES_NAME: &str =
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes";
const NGRAM_HEADS_OFFSETS_NAME: &str =
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets";

const LAYER_MULTIPLIERS: [i64; 3] = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];
const NGRAM_HEADS_VOCAB_SIZES: [i64; 16] = [
    20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069, 20_000_077,
    20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159, 20_000_161, 20_000_171,
];
const NGRAM_HEADS_OFFSETS: [i64; 16] = [
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

/// Number of persistent Qwen3.8 PLE hash buffers checked by the bounded verifier.
pub const QWEN38_PLE_HASH_BUFFER_COUNT: usize = 3;

/// Exact payload bytes read by a successful bounded Qwen3.8 PLE hash-buffer verification.
pub const QWEN38_PLE_HASH_BUFFER_PAYLOAD_BYTES: u64 = 280;

#[derive(Clone, Copy)]
struct HashBufferSpec {
    tensor_name: &'static str,
    expected: &'static [i64],
}

const HASH_BUFFER_SPECS: [HashBufferSpec; QWEN38_PLE_HASH_BUFFER_COUNT] = [
    HashBufferSpec {
        tensor_name: LAYER_MULTIPLIERS_NAME,
        expected: &LAYER_MULTIPLIERS,
    },
    HashBufferSpec {
        tensor_name: NGRAM_HEADS_VOCAB_SIZES_NAME,
        expected: &NGRAM_HEADS_VOCAB_SIZES,
    },
    HashBufferSpec {
        tensor_name: NGRAM_HEADS_OFFSETS_NAME,
        expected: &NGRAM_HEADS_OFFSETS,
    },
];

/// Local byte-read evidence returned by [`verify_qwen38_ple_hash_buffers`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen38PleHashBufferVerification {
    /// Number of required persistent hash buffers that matched their pinned values.
    pub buffers_verified: usize,
    /// Number of distinct safetensors files opened for the three buffers.
    pub files_read: usize,
    /// Exact bytes read from `model.safetensors.index.json`.
    pub index_bytes_read: u64,
    /// Exact prefix-plus-header bytes read from referenced safetensors files.
    pub header_bytes_read: u64,
    /// Exact selected tensor payload bytes read from referenced safetensors files.
    pub payload_bytes_read: u64,
}

/// Verify physical Qwen3.8 PLE hash buffers without mapping or reading full model shards.
///
/// The caller supplies the checkpoint revision it observed. The function accepts only the pinned
/// GB10X revision and then compares the three `I64` persistent buffers that determine the PLE hash
/// against the pinned upstream-constructor defaults. It validates local bytes only; it does not
/// establish remote checkpoint provenance, verify unrelated tensors or execute model inference.
pub fn verify_qwen38_ple_hash_buffers(
    model_dir: impl AsRef<Path>,
    observed_revision: &str,
) -> Result<Qwen38PleHashBufferVerification, PlePackIoError> {
    if observed_revision != QWEN38_FLASH_NEXT_REVISION {
        return Err(PlePackIoError::Format(
            "Qwen3.8 checkpoint revision does not match pinned GB10X source",
        ));
    }

    let model_dir = model_dir.as_ref();
    let index_path = model_dir.join(SAFETENSORS_INDEX_FILE);
    let index_bytes = read_file_bounded(
        &index_path,
        MAX_INDEX_BYTES,
        "model.safetensors.index.json exceeds configured read bound",
    )?;
    let index_bytes_read = u64::try_from(index_bytes.len())
        .map_err(|_| PlePackIoError::Format("safetensors index length exceeds u64"))?;
    let index =
        parse_json_without_duplicate_members(&index_bytes, "invalid model.safetensors.index.json")?;
    let weight_map =
        index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or(PlePackIoError::Format(
                "safetensors index is missing weight_map object",
            ))?;

    let mut by_file: BTreeMap<PathBuf, Vec<HashBufferSpec>> = BTreeMap::new();
    for spec in HASH_BUFFER_SPECS {
        let mapped_file = weight_map
            .get(spec.tensor_name)
            .and_then(Value::as_str)
            .ok_or(PlePackIoError::Format(
                "safetensors index is missing a required Qwen3.8 PLE hash buffer",
            ))?;
        let relative = safe_safetensors_relative_path(mapped_file)?;
        by_file.entry(relative).or_default().push(spec);
    }

    let mut header_bytes_read = 0_u64;
    let mut payload_bytes_read = 0_u64;
    let mut buffers_verified = 0_usize;
    let files_read = by_file.len();

    for (relative, specs) in by_file {
        let mut file = File::open(model_dir.join(relative))?;
        let header = read_safetensors_header_bounded(&mut file, MAX_SAFETENSORS_HEADER_BYTES)?;
        header_bytes_read = header_bytes_read
            .checked_add(header.header_bytes_read)
            .ok_or(PlePackIoError::Format(
                "safetensors header-byte count overflow",
            ))?;

        for spec in specs {
            let tensor = header
                .tensors
                .iter()
                .find(|tensor| tensor.name == spec.tensor_name)
                .ok_or(PlePackIoError::Format(
                    "safetensors hash-buffer tensor is missing from header",
                ))?;
            if tensor.dtype != "I64" {
                return Err(PlePackIoError::Format(
                    "Qwen3.8 PLE hash-buffer tensor dtype must be I64",
                ));
            }
            let expected_elements = u64::try_from(spec.expected.len()).map_err(|_| {
                PlePackIoError::Format("Qwen3.8 PLE hash-buffer element count exceeds u64")
            })?;
            if tensor.shape.as_slice() != [expected_elements] {
                return Err(PlePackIoError::Format(
                    "Qwen3.8 PLE hash-buffer tensor shape does not match pinned geometry",
                ));
            }
            let expected_bytes =
                expected_elements
                    .checked_mul(I64_BYTES)
                    .ok_or(PlePackIoError::Format(
                        "Qwen3.8 PLE hash-buffer byte count overflow",
                    ))?;
            let actual_bytes =
                tensor
                    .data_end
                    .checked_sub(tensor.data_start)
                    .ok_or(PlePackIoError::Format(
                        "safetensors hash-buffer offsets are descending",
                    ))?;
            if actual_bytes != expected_bytes {
                return Err(PlePackIoError::Format(
                    "Qwen3.8 PLE hash-buffer payload length does not match I64 shape",
                ));
            }
            let absolute_start = header
                .data_section_start
                .checked_add(tensor.data_start)
                .ok_or(PlePackIoError::Format(
                    "safetensors hash-buffer absolute offset overflow",
                ))?;
            let payload_len = usize::try_from(expected_bytes).map_err(|_| {
                PlePackIoError::Format("Qwen3.8 PLE hash-buffer payload length does not fit usize")
            })?;
            let mut payload = vec![0_u8; payload_len];
            file.seek(SeekFrom::Start(absolute_start))?;
            file.read_exact(&mut payload)?;
            verify_i64_payload(&payload, spec.expected)?;

            payload_bytes_read =
                payload_bytes_read
                    .checked_add(expected_bytes)
                    .ok_or(PlePackIoError::Format(
                        "safetensors hash-buffer payload-byte count overflow",
                    ))?;
            buffers_verified = buffers_verified
                .checked_add(1)
                .ok_or(PlePackIoError::Format(
                    "Qwen3.8 PLE hash-buffer count overflow",
                ))?;
        }
    }

    if payload_bytes_read != QWEN38_PLE_HASH_BUFFER_PAYLOAD_BYTES {
        return Err(PlePackIoError::Format(
            "Qwen3.8 PLE hash-buffer payload contract is internally inconsistent",
        ));
    }
    if buffers_verified != QWEN38_PLE_HASH_BUFFER_COUNT {
        return Err(PlePackIoError::Format(
            "Qwen3.8 PLE hash-buffer count is internally inconsistent",
        ));
    }

    Ok(Qwen38PleHashBufferVerification {
        buffers_verified,
        files_read,
        index_bytes_read,
        header_bytes_read,
        payload_bytes_read,
    })
}

fn read_file_bounded(
    path: &Path,
    max_bytes: u64,
    too_large_message: &'static str,
) -> Result<Vec<u8>, PlePackIoError> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    if length > max_bytes {
        return Err(PlePackIoError::Format(too_large_message));
    }
    let length = usize::try_from(length)
        .map_err(|_| PlePackIoError::Format("bounded file length does not fit usize"))?;
    let mut bytes = vec![0_u8; length];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn safe_safetensors_relative_path(value: &str) -> Result<PathBuf, PlePackIoError> {
    if value.trim().is_empty() {
        return Err(PlePackIoError::Format(
            "Qwen3.8 hash-buffer mapped file path is empty",
        ));
    }
    if !value.ends_with(".safetensors") {
        return Err(PlePackIoError::Format(
            "Qwen3.8 hash-buffer mapped file path must end in .safetensors",
        ));
    }

    let path = Path::new(value);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(PlePackIoError::Format(
            "Qwen3.8 hash-buffer mapped file path escapes model directory",
        ));
    }
    Ok(path.to_owned())
}

fn verify_i64_payload(payload: &[u8], expected: &[i64]) -> Result<(), PlePackIoError> {
    let expected_bytes =
        expected
            .len()
            .checked_mul(I64_BYTES as usize)
            .ok_or(PlePackIoError::Format(
                "Qwen3.8 PLE hash-buffer expected byte count overflow",
            ))?;
    if payload.len() != expected_bytes {
        return Err(PlePackIoError::Format(
            "Qwen3.8 PLE hash-buffer payload width is invalid",
        ));
    }

    for (bytes, expected) in payload.as_chunks::<8>().0.iter().zip(expected) {
        let actual = i64::from_le_bytes(*bytes);
        if actual != *expected {
            return Err(PlePackIoError::Format(
                "physical Qwen3.8 PLE hash-buffer value differs from pinned upstream default",
            ));
        }
    }
    Ok(())
}
