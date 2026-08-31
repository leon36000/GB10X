//! Exact PLEPack metadata shared by the layout, writer and reader.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Eight-byte file-format identity for exact GB10X PLEPack artifacts.
pub const PLEPACK_MAGIC: [u8; 8] = *b"GB10XPLE";
/// Initial exact PLEPack format version.
pub const PLEPACK_VERSION: u32 = 1;

/// PLEPack construction, layout or validation failure.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum PlePackError {
    /// Row/block geometry cannot represent an exact pack.
    #[error("invalid PLEPack geometry: {0}")]
    InvalidGeometry(&'static str),
    /// A workload trace referenced a logical row outside the immutable base region.
    #[error("trace row {row} is outside PLEPack base row count {row_count}")]
    TraceRowOutOfRange {
        /// Invalid logical row ID.
        row: u32,
        /// Number of rows in the exact cold base region.
        row_count: u64,
    },
    /// A logical row lookup addressed outside the immutable base region.
    #[error("logical row {row} is outside PLEPack base row count {row_count}")]
    LogicalRowOutOfRange {
        /// Invalid logical row ID.
        row: u32,
        /// Number of rows in the exact cold base region.
        row_count: u64,
    },
    /// Checked integer arithmetic failed while computing a file layout.
    #[error("PLEPack arithmetic overflow: {0}")]
    Overflow(&'static str),
}

/// I/O or integrity failure while building or consuming an exact PLEPack sidecar.
#[derive(Debug, Error)]
pub enum PlePackIoError {
    /// Layout/provenance geometry is invalid.
    #[error(transparent)]
    Layout(#[from] PlePackError),
    /// Filesystem operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// On-disk structure is malformed or internally inconsistent.
    #[error("invalid PLEPack sidecar: {0}")]
    Format(&'static str),
    /// Exact source provider rejected a request.
    #[error("exact PLE source error: {0}")]
    Source(&'static str),
    /// Caller provided a row buffer of the wrong width.
    #[error("row buffer width mismatch: expected {expected}, got {actual}")]
    RowBufferWidth {
        /// Exact source row width.
        expected: usize,
        /// Caller-provided buffer width.
        actual: usize,
    },
    /// Sidecar was created against a different immutable source row stream.
    #[error("PLEPack source digest does not match the exact source")]
    SourceDigestMismatch,
    /// Serialized hot-overlay index was modified or corrupted.
    #[error("PLEPack hot-overlay index digest mismatch")]
    IndexDigestMismatch,
}

/// Immutable provenance and geometry for one exact PLEPack dataset.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PlePackHeader {
    /// File-format identity.
    pub magic: [u8; 8],
    /// File-format version.
    pub version: u32,
    /// SHA-256 digest of the exact source PLE row stream.
    pub source_digest: [u8; 32],
    /// Number of logical BF16 rows in the cold identity-mapped base region.
    pub row_count: u64,
    /// Exact byte width of one logical row.
    pub row_bytes: u32,
    /// Physical block size used by locality-optimized overlay storage.
    pub block_bytes: u32,
    /// SHA-256 digest of the serialized hot-overlay index payload.
    pub index_digest: [u8; 32],
}

impl PlePackHeader {
    /// Construct validated exact PLEPack metadata.
    pub fn new(
        source_digest: [u8; 32],
        row_count: u64,
        row_bytes: u32,
        block_bytes: u32,
        index_digest: [u8; 32],
    ) -> Result<Self, PlePackError> {
        if row_count == 0 {
            return Err(PlePackError::InvalidGeometry(
                "row_count must be nonzero",
            ));
        }
        if row_bytes == 0 {
            return Err(PlePackError::InvalidGeometry("row_bytes must be nonzero"));
        }
        if block_bytes == 0 {
            return Err(PlePackError::InvalidGeometry(
                "block_bytes must be nonzero",
            ));
        }
        if block_bytes < row_bytes {
            return Err(PlePackError::InvalidGeometry(
                "block_bytes must fit at least one complete row",
            ));
        }
        if row_count > u32::MAX as u64 + 1 {
            return Err(PlePackError::InvalidGeometry(
                "logical row space must fit u32 row identifiers",
            ));
        }

        Ok(Self {
            magic: PLEPACK_MAGIC,
            version: PLEPACK_VERSION,
            source_digest,
            row_count,
            row_bytes,
            block_bytes,
            index_digest,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_rejects_zero_dimensions() {
        assert!(PlePackHeader::new([1; 32], 0, 320, 4096, [2; 32]).is_err());
        assert!(PlePackHeader::new([1; 32], 10, 0, 4096, [2; 32]).is_err());
        assert!(PlePackHeader::new([1; 32], 10, 320, 0, [2; 32]).is_err());
    }

    #[test]
    fn header_keeps_source_and_index_provenance() {
        let header = PlePackHeader::new([0xAA; 32], 100, 320, 4096, [0xBB; 32])
            .expect("valid exact PLEPack header");
        assert_eq!(header.source_digest, [0xAA; 32]);
        assert_eq!(header.index_digest, [0xBB; 32]);
        assert_eq!(header.row_count, 100);
        assert_eq!(header.row_bytes, 320);
        assert_eq!(header.block_bytes, 4096);
    }
}
