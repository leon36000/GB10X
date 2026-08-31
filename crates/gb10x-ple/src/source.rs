//! Exact cold PLE row-source abstraction used beneath hot PLEPack overlays.

use crate::PlePackIoError;

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
