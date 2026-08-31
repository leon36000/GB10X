//! Exact cold PLE row-source abstractions used beneath hot PLEPack overlays.

use crate::{PlePackError, PlePackIoError};
use memmap2::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::Path;

/// Immutable exact logical-row provider for the original Qwen PLE tensors.
///
/// Production implementations may read directly from pinned safetensors shards. PLEPack never
/// changes a cold row: an overlay miss delegates to this source byte-for-byte.
pub trait ExactPleRowSource {
    /// Number of logical PLE rows exposed by the source.
    fn row_count(&self) -> u64;

    /// Exact byte width of one logical row.
    fn row_bytes(&self) -> u32;

    /// SHA-256 digest identifying the immutable logical source row stream.
    fn source_digest(&self) -> [u8; 32];

    /// Read one logical row byte-for-byte into `dst`.
    fn read_exact_row(&self, logical_row: u32, dst: &mut [u8]) -> Result<(), PlePackIoError>;
}

/// Read-only mmap source for a prepared flat file of fixed-width exact PLE rows.
///
/// This adapter is intentionally format-minimal: the file contains only consecutive row bytes.
/// It is useful for the M1 CLI and for prepared data exported from the model's immutable source.
pub struct RawFileRowSource {
    mmap: Mmap,
    row_count: u64,
    row_bytes: u32,
    digest: [u8; 32],
}

impl RawFileRowSource {
    /// Open and validate a flat exact-row source.
    pub fn open(path: impl AsRef<Path>, row_bytes: u32) -> Result<Self, PlePackIoError> {
        if row_bytes == 0 {
            return Err(PlePackIoError::Format(
                "raw source row width must be nonzero",
            ));
        }

        let file = File::open(path)?;
        let file_bytes = file.metadata()?.len();
        if file_bytes == 0 {
            return Err(PlePackIoError::Format(
                "raw source must contain at least one row",
            ));
        }
        if !file_bytes.is_multiple_of(row_bytes as u64) {
            return Err(PlePackIoError::Format(
                "raw source byte length is not divisible by row width",
            ));
        }

        let row_count = file_bytes / row_bytes as u64;
        if row_count > u32::MAX as u64 + 1 {
            return Err(PlePackIoError::Format(
                "raw source row count exceeds u32 logical row space",
            ));
        }

        // SAFETY: mapping is read-only and RawFileRowSource owns the resulting mapping. GB10X
        // treats prepared row sources as immutable for the lifetime of a pack build/verification.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let digest = Sha256::digest(&mmap).into();

        Ok(Self {
            mmap,
            row_count,
            row_bytes,
            digest,
        })
    }
}

impl ExactPleRowSource for RawFileRowSource {
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

        let start_u64 = (logical_row as u64)
            .checked_mul(self.row_bytes as u64)
            .ok_or(PlePackIoError::Format("raw source row offset overflow"))?;
        let start = usize::try_from(start_u64)
            .map_err(|_| PlePackIoError::Format("raw source row offset does not fit usize"))?;
        let end = start
            .checked_add(expected)
            .ok_or(PlePackIoError::Format("raw source row end overflow"))?;
        let row = self.mmap.get(start..end).ok_or(PlePackIoError::Format(
            "raw source row lies outside mapping",
        ))?;
        dst.copy_from_slice(row);
        Ok(())
    }
}
