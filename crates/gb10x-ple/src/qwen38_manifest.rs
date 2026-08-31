//! Pinned Qwen3.8-Flash-Next PLE tensor manifest derived from the official safetensors index.

use crate::{PlePackIoError, SafetensorsPleManifest, SafetensorsPlePart};
use serde_json::Value;
use std::fs;
use std::path::{Component, Path};

/// Canonical Hugging Face model ID supported by GB10X.
pub const QWEN38_FLASH_NEXT_MODEL_ID: &str = "Qwen/Qwen3.8-Flash-Next";

/// Immutable source revision pinned by GB10X for the initial Qwen3.8-Flash-Next checkpoint.
pub const QWEN38_FLASH_NEXT_REVISION: &str = "34567a4712bc9766c4449e2e98e4468bfa24d915";

/// Number of physical PLE tensor parts in the pinned checkpoint.
pub const QWEN38_PLE_PARTS: usize = 128;

/// BF16 elements in one logical PLE row.
pub const QWEN38_PLE_ROW_ELEMENTS: u32 = 160;

/// Logical rows stored in each physical PLE tensor part.
pub const QWEN38_PLE_ROWS_PER_PART: u64 = 2_500_012;

/// Total logical PLE rows in the pinned checkpoint.
pub const QWEN38_PLE_ROWS: u64 = 320_001_536;

const QWEN38_PLE_TENSOR_PREFIX: &str =
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";
const QWEN38_PLE_TENSOR_SUFFIX: &str = ".weight";
const SAFETENSORS_INDEX_FILE: &str = "model.safetensors.index.json";

/// Build the exact Qwen3.8 PLE manifest from the checkpoint's safetensors weight map.
///
/// The caller must provide independently observed checkpoint provenance. The function rejects any
/// revision other than [`QWEN38_FLASH_NEXT_REVISION`] and accepts only canonical shard names
/// `shard_0.weight` through `shard_127.weight` under the pinned zero-based PLE layer path.
pub fn qwen38_ple_manifest_from_index(
    model_dir: impl AsRef<Path>,
    observed_revision: &str,
) -> Result<SafetensorsPleManifest, PlePackIoError> {
    if observed_revision != QWEN38_FLASH_NEXT_REVISION {
        return Err(PlePackIoError::Format(
            "Qwen3.8 checkpoint revision does not match pinned GB10X source",
        ));
    }

    let index_path = model_dir.as_ref().join(SAFETENSORS_INDEX_FILE);
    let bytes = fs::read(index_path)?;
    let root: Value = serde_json::from_slice(&bytes)
        .map_err(|_| PlePackIoError::Format("invalid model.safetensors.index.json"))?;
    let weight_map =
        root.get("weight_map")
            .and_then(Value::as_object)
            .ok_or(PlePackIoError::Format(
                "safetensors index is missing weight_map object",
            ))?;

    let mut parts: Vec<Option<SafetensorsPlePart>> = vec![None; QWEN38_PLE_PARTS];
    for (tensor_name, mapped_file) in weight_map {
        let Some(shard_text) = tensor_name
            .strip_prefix(QWEN38_PLE_TENSOR_PREFIX)
            .and_then(|rest| rest.strip_suffix(QWEN38_PLE_TENSOR_SUFFIX))
        else {
            continue;
        };

        if shard_text.is_empty() || !shard_text.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(PlePackIoError::Format(
                "Qwen3.8 PLE shard index must be canonical decimal digits",
            ));
        }
        let shard_index = shard_text
            .parse::<usize>()
            .map_err(|_| PlePackIoError::Format("Qwen3.8 PLE shard index does not fit usize"))?;
        if shard_index >= QWEN38_PLE_PARTS || shard_text != shard_index.to_string() {
            return Err(PlePackIoError::Format(
                "Qwen3.8 PLE shard index is out of range or noncanonical",
            ));
        }
        if parts[shard_index].is_some() {
            return Err(PlePackIoError::Format(
                "Qwen3.8 PLE shard index appears more than once",
            ));
        }

        let file = mapped_file.as_str().ok_or(PlePackIoError::Format(
            "Qwen3.8 PLE weight_map value must be a safetensors file string",
        ))?;
        validate_relative_model_path(file)?;
        if !file.ends_with(".safetensors") {
            return Err(PlePackIoError::Format(
                "Qwen3.8 PLE weight_map target must be a .safetensors file",
            ));
        }

        let logical_row_start = (shard_index as u64)
            .checked_mul(QWEN38_PLE_ROWS_PER_PART)
            .ok_or(PlePackIoError::Format(
                "Qwen3.8 PLE logical row start overflow",
            ))?;
        parts[shard_index] = Some(SafetensorsPlePart {
            file: file.to_owned(),
            tensor_name: tensor_name.clone(),
            logical_row_start,
            row_count: QWEN38_PLE_ROWS_PER_PART,
        });
    }

    let parts = parts
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(PlePackIoError::Format(
            "safetensors index does not contain all 128 pinned Qwen3.8 PLE shards",
        ))?;
    let total_rows = QWEN38_PLE_ROWS_PER_PART
        .checked_mul(QWEN38_PLE_PARTS as u64)
        .ok_or(PlePackIoError::Format(
            "Qwen3.8 PLE total row count overflow",
        ))?;
    if total_rows != QWEN38_PLE_ROWS {
        return Err(PlePackIoError::Format(
            "Qwen3.8 PLE constants are internally inconsistent",
        ));
    }

    Ok(SafetensorsPleManifest {
        model_id: QWEN38_FLASH_NEXT_MODEL_ID.to_owned(),
        model_revision: QWEN38_FLASH_NEXT_REVISION.to_owned(),
        row_elements: QWEN38_PLE_ROW_ELEMENTS,
        parts,
    })
}

fn validate_relative_model_path(value: &str) -> Result<(), PlePackIoError> {
    if value.trim().is_empty() {
        return Err(PlePackIoError::Format(
            "Qwen3.8 PLE mapped file path is empty",
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
            "Qwen3.8 PLE mapped file path escapes model directory",
        ));
    }
    Ok(())
}
