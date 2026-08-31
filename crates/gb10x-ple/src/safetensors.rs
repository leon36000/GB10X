//! Exact mmap-backed PLE rows sourced directly from safetensors shards.

use crate::{ExactPleRowSource, PlePackError, PlePackIoError};
use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::{Component, Path, PathBuf};

const SAFETENSORS_PREFIX_BYTES: usize = 8;
const BF16_BYTES: u64 = 2;
const DIGEST_DOMAIN: &[u8] = b"GB10X-SAFETENSORS-PLE-V1\0";

/// One logical PLE tensor range inside a safetensors file.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SafetensorsPlePart {
    /// File path relative to the pinned model directory.
    pub file: String,
    /// Exact safetensors tensor name containing this PLE part.
    pub tensor_name: String,
    /// First logical PLE row exposed by this tensor.
    pub logical_row_start: u64,
    /// Number of exact PLE rows in this tensor.
    pub row_count: u64,
}

/// Immutable manifest binding logical PLE rows to pinned safetensors tensors.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SafetensorsPleManifest {
    /// Canonical model repository identifier.
    pub model_id: String,
    /// Exact immutable model revision/commit.
    pub model_revision: String,
    /// Number of BF16 elements in one logical PLE row.
    pub row_elements: u32,
    /// Ordered physical PLE tensor parts covering one contiguous logical row space.
    pub parts: Vec<SafetensorsPlePart>,
}

/// Exact safetensors tensor location retained for diagnostics/evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsTensorRef {
    /// Relative safetensors file path.
    pub file: String,
    /// Exact tensor name.
    pub tensor_name: String,
    /// Tensor dtype string from the safetensors header.
    pub dtype: String,
    /// Tensor shape from the safetensors header.
    pub shape: Vec<u64>,
    /// Absolute file offset of the first tensor byte.
    pub data_offset: u64,
}

struct MappedPart {
    mmap: Mmap,
    logical_row_start: u64,
    row_count: u64,
    data_start: usize,
    data_end: usize,
}

/// Read-only exact PLE source that maps the original safetensors shards without materializing
/// the complete PLE tensor stream in RAM.
pub struct SafetensorsPleSource {
    parts: Vec<MappedPart>,
    row_count: u64,
    row_bytes: u32,
    digest: [u8; 32],
}

impl SafetensorsPleSource {
    /// Open and validate every manifest-referenced PLE tensor under `model_dir`.
    pub fn open(
        model_dir: impl AsRef<Path>,
        manifest: &SafetensorsPleManifest,
    ) -> Result<Self, PlePackIoError> {
        validate_manifest(manifest)?;
        let row_bytes_u64 = u64::from(manifest.row_elements)
            .checked_mul(BF16_BYTES)
            .ok_or(PlePackIoError::Format("safetensors PLE row width overflow"))?;
        let row_bytes = u32::try_from(row_bytes_u64)
            .map_err(|_| PlePackIoError::Format("safetensors PLE row width exceeds u32"))?;

        let mut mapped = Vec::with_capacity(manifest.parts.len());
        let mut hasher = Sha256::new();
        hasher.update(DIGEST_DOMAIN);
        digest_string(&mut hasher, &manifest.model_id)?;
        digest_string(&mut hasher, &manifest.model_revision)?;
        hasher.update(manifest.row_elements.to_le_bytes());
        hasher.update((manifest.parts.len() as u64).to_le_bytes());

        for part in &manifest.parts {
            let relative = safe_relative_path(&part.file)?;
            let full_path = model_dir.as_ref().join(relative);
            let file = File::open(&full_path)?;
            let file_len = file.metadata()?.len();
            if file_len < SAFETENSORS_PREFIX_BYTES as u64 {
                return Err(PlePackIoError::Format(
                    "safetensors file is shorter than header-length prefix",
                ));
            }

            // SAFETY: each mapping is read-only, owned by this source, and GB10X requires the
            // pinned model files to remain immutable for the lifetime of the source.
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let parsed = parse_header(&mmap)?;
            let target = parsed
                .tensors
                .iter()
                .find(|tensor| tensor.name == part.tensor_name)
                .ok_or(PlePackIoError::Format(
                    "manifest tensor is missing from safetensors header",
                ))?;

            if target.dtype != "BF16" {
                return Err(PlePackIoError::Format(
                    "PLE safetensors tensor dtype must be BF16",
                ));
            }
            if target.shape.len() != 2
                || target.shape[0] != part.row_count
                || target.shape[1] != u64::from(manifest.row_elements)
            {
                return Err(PlePackIoError::Format(
                    "PLE safetensors tensor shape does not match manifest geometry",
                ));
            }

            let expected_tensor_bytes =
                part.row_count
                    .checked_mul(row_bytes_u64)
                    .ok_or(PlePackIoError::Format(
                        "PLE safetensors tensor byte length overflow",
                    ))?;
            let actual_tensor_bytes =
                target
                    .data_end
                    .checked_sub(target.data_start)
                    .ok_or(PlePackIoError::Format(
                        "safetensors tensor offsets are descending",
                    ))?;
            if actual_tensor_bytes != expected_tensor_bytes {
                return Err(PlePackIoError::Format(
                    "PLE safetensors tensor byte length does not match shape",
                ));
            }

            let absolute_start_u64 = parsed
                .data_section_start
                .checked_add(target.data_start)
                .ok_or(PlePackIoError::Format(
                    "safetensors tensor absolute offset overflow",
                ))?;
            let absolute_end_u64 = parsed
                .data_section_start
                .checked_add(target.data_end)
                .ok_or(PlePackIoError::Format(
                    "safetensors tensor absolute end overflow",
                ))?;
            if absolute_end_u64 > file_len {
                return Err(PlePackIoError::Format(
                    "safetensors tensor payload lies outside file",
                ));
            }
            let data_start = usize::try_from(absolute_start_u64).map_err(|_| {
                PlePackIoError::Format("safetensors tensor offset does not fit usize")
            })?;
            let data_end = usize::try_from(absolute_end_u64)
                .map_err(|_| PlePackIoError::Format("safetensors tensor end does not fit usize"))?;
            let tensor_bytes = mmap
                .get(data_start..data_end)
                .ok_or(PlePackIoError::Format(
                    "safetensors tensor payload lies outside mapping",
                ))?;

            digest_string(&mut hasher, &part.file)?;
            digest_string(&mut hasher, &part.tensor_name)?;
            hasher.update(part.logical_row_start.to_le_bytes());
            hasher.update(part.row_count.to_le_bytes());
            hasher.update(tensor_bytes);

            mapped.push(MappedPart {
                mmap,
                logical_row_start: part.logical_row_start,
                row_count: part.row_count,
                data_start,
                data_end,
            });
        }

        let row_count = manifest.parts.iter().try_fold(0_u64, |total, part| {
            total
                .checked_add(part.row_count)
                .ok_or(PlePackIoError::Format("safetensors PLE row count overflow"))
        })?;
        if row_count > u32::MAX as u64 + 1 {
            return Err(PlePackError::InvalidGeometry(
                "logical row space must fit u32 row identifiers",
            )
            .into());
        }

        Ok(Self {
            parts: mapped,
            row_count,
            row_bytes,
            digest: hasher.finalize().into(),
        })
    }
}

impl ExactPleRowSource for SafetensorsPleSource {
    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn row_bytes(&self) -> u32 {
        self.row_bytes
    }

    fn source_digest(&self) -> [u8; 32] {
        self.digest
    }

    fn read_exact_row(&self, logical_row: u32, dst: &mut [u8]) -> Result<(), PlePackIoError> {
        if logical_row as u64 >= self.row_count {
            return Err(PlePackError::LogicalRowOutOfRange {
                row: logical_row,
                row_count: self.row_count,
            }
            .into());
        }
        let expected = self.row_bytes as usize;
        if dst.len() != expected {
            return Err(PlePackIoError::RowBufferWidth {
                expected,
                actual: dst.len(),
            });
        }

        let logical = u64::from(logical_row);
        let part = self
            .parts
            .iter()
            .find(|part| {
                logical >= part.logical_row_start
                    && logical < part.logical_row_start + part.row_count
            })
            .ok_or(PlePackIoError::Format(
                "logical PLE row is not covered by mapped safetensors parts",
            ))?;
        let relative = logical - part.logical_row_start;
        let byte_offset_u64 =
            relative
                .checked_mul(u64::from(self.row_bytes))
                .ok_or(PlePackIoError::Format(
                    "safetensors PLE row offset overflow",
                ))?;
        let byte_offset = usize::try_from(byte_offset_u64)
            .map_err(|_| PlePackIoError::Format("safetensors PLE row offset does not fit usize"))?;
        let start = part
            .data_start
            .checked_add(byte_offset)
            .ok_or(PlePackIoError::Format(
                "safetensors PLE absolute row offset overflow",
            ))?;
        let end = start
            .checked_add(expected)
            .ok_or(PlePackIoError::Format("safetensors PLE row end overflow"))?;
        if end > part.data_end {
            return Err(PlePackIoError::Format(
                "safetensors PLE row exceeds tensor payload",
            ));
        }
        let row = part.mmap.get(start..end).ok_or(PlePackIoError::Format(
            "safetensors PLE row lies outside mapping",
        ))?;
        dst.copy_from_slice(row);
        Ok(())
    }
}

struct ParsedHeader {
    data_section_start: u64,
    tensors: Vec<ParsedTensor>,
}

struct ParsedTensor {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    data_start: u64,
    data_end: u64,
}

fn parse_header(bytes: &[u8]) -> Result<ParsedHeader, PlePackIoError> {
    let prefix: [u8; SAFETENSORS_PREFIX_BYTES] = bytes
        .get(..SAFETENSORS_PREFIX_BYTES)
        .ok_or(PlePackIoError::Format("missing safetensors header length"))?
        .try_into()
        .map_err(|_| PlePackIoError::Format("invalid safetensors header prefix"))?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len == 0 {
        return Err(PlePackIoError::Format(
            "safetensors header length must be nonzero",
        ));
    }
    let data_section_start = (SAFETENSORS_PREFIX_BYTES as u64)
        .checked_add(header_len)
        .ok_or(PlePackIoError::Format(
            "safetensors header boundary overflow",
        ))?;
    let header_end = usize::try_from(data_section_start)
        .map_err(|_| PlePackIoError::Format("safetensors header end does not fit usize"))?;
    if header_end > bytes.len() {
        return Err(PlePackIoError::Format(
            "safetensors header extends beyond file",
        ));
    }

    let header: Value = serde_json::from_slice(
        bytes
            .get(SAFETENSORS_PREFIX_BYTES..header_end)
            .ok_or(PlePackIoError::Format("invalid safetensors header range"))?,
    )
    .map_err(|_| PlePackIoError::Format("invalid safetensors header JSON"))?;
    let object = header.as_object().ok_or(PlePackIoError::Format(
        "safetensors header must be a JSON object",
    ))?;

    let data_len =
        (bytes.len() as u64)
            .checked_sub(data_section_start)
            .ok_or(PlePackIoError::Format(
                "safetensors data-section boundary is invalid",
            ))?;
    let mut tensors = Vec::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let tensor = value.as_object().ok_or(PlePackIoError::Format(
            "safetensors tensor entry must be an object",
        ))?;
        let dtype = tensor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or(PlePackIoError::Format(
                "safetensors tensor dtype is missing",
            ))?
            .to_owned();
        let shape = tensor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or(PlePackIoError::Format(
                "safetensors tensor shape is missing",
            ))?
            .iter()
            .map(|dimension| {
                dimension.as_u64().ok_or(PlePackIoError::Format(
                    "safetensors tensor shape contains a non-u64 dimension",
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offsets =
            tensor
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or(PlePackIoError::Format(
                    "safetensors tensor data_offsets is missing",
                ))?;
        if offsets.len() != 2 {
            return Err(PlePackIoError::Format(
                "safetensors tensor data_offsets must contain exactly two values",
            ));
        }
        let data_start = offsets[0].as_u64().ok_or(PlePackIoError::Format(
            "invalid safetensors tensor start offset",
        ))?;
        let data_end = offsets[1].as_u64().ok_or(PlePackIoError::Format(
            "invalid safetensors tensor end offset",
        ))?;
        if data_end < data_start {
            return Err(PlePackIoError::Format(
                "safetensors tensor offsets are descending",
            ));
        }
        if data_end > data_len {
            return Err(PlePackIoError::Format(
                "safetensors tensor payload lies outside data section",
            ));
        }
        tensors.push(ParsedTensor {
            name: name.clone(),
            dtype,
            shape,
            data_start,
            data_end,
        });
    }
    if tensors.is_empty() {
        return Err(PlePackIoError::Format(
            "safetensors file contains no tensors",
        ));
    }

    let mut ranges = tensors
        .iter()
        .map(|tensor| (tensor.data_start, tensor.data_end))
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|range| (range.0, range.1));
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(PlePackIoError::Format(
                "safetensors tensor payload ranges overlap",
            ));
        }
    }

    Ok(ParsedHeader {
        data_section_start,
        tensors,
    })
}

fn validate_manifest(manifest: &SafetensorsPleManifest) -> Result<(), PlePackIoError> {
    if manifest.model_id.trim().is_empty() {
        return Err(PlePackIoError::Format("PLE manifest model_id is empty"));
    }
    if manifest.model_revision.trim().is_empty() {
        return Err(PlePackIoError::Format(
            "PLE manifest model_revision is empty",
        ));
    }
    if manifest.row_elements == 0 {
        return Err(PlePackIoError::Format(
            "PLE manifest row_elements must be nonzero",
        ));
    }
    if manifest.parts.is_empty() {
        return Err(PlePackIoError::Format("PLE manifest contains no parts"));
    }

    let mut next_row = 0_u64;
    for part in &manifest.parts {
        if part.file.trim().is_empty() {
            return Err(PlePackIoError::Format("PLE manifest part file is empty"));
        }
        if part.tensor_name.trim().is_empty() {
            return Err(PlePackIoError::Format("PLE manifest tensor name is empty"));
        }
        safe_relative_path(&part.file)?;
        if part.row_count == 0 {
            return Err(PlePackIoError::Format(
                "PLE manifest part row_count must be nonzero",
            ));
        }
        if part.logical_row_start != next_row {
            return Err(PlePackIoError::Format(
                "PLE manifest parts must cover a contiguous logical row space",
            ));
        }
        next_row = next_row
            .checked_add(part.row_count)
            .ok_or(PlePackIoError::Format("PLE manifest row count overflow"))?;
    }
    Ok(())
}

fn safe_relative_path(value: &str) -> Result<PathBuf, PlePackIoError> {
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
            "PLE manifest file path must remain inside model directory",
        ));
    }
    Ok(path.to_owned())
}

fn digest_string(hasher: &mut Sha256, value: &str) -> Result<(), PlePackIoError> {
    let len = u64::try_from(value.len())
        .map_err(|_| PlePackIoError::Format("digest string length exceeds u64"))?;
    hasher.update(len.to_le_bytes());
    hasher.update(value.as_bytes());
    Ok(())
}
