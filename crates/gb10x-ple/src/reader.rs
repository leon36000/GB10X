//! mmap-backed exact PLEPack hot-overlay reader with cold-source fallback.

use crate::disk::{DISK_HEADER_BYTES, DiskHeader, INDEX_ENTRY_BYTES, decode_index_entry};
use crate::{ExactPleRowSource, PlePackHeader, PlePackIoError};
use memmap2::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::Path;

/// Exact logical PLE reader backed by a verified hot sidecar and immutable cold source.
pub struct PlePackReader<S: ExactPleRowSource> {
    mmap: Mmap,
    source: S,
    header: PlePackHeader,
    hot_row_count: usize,
    index_offset: usize,
    data_offset: usize,
    overlay_bytes: u64,
}

impl<S: ExactPleRowSource> PlePackReader<S> {
    /// Open, hash-verify and structurally validate an exact PLEPack hot-overlay sidecar.
    pub fn open(path: impl AsRef<Path>, source: S) -> Result<Self, PlePackIoError> {
        let file = File::open(path)?;
        // SAFETY: the mapping is read-only, the File remains valid for map creation, and Mmap owns
        // the OS mapping after construction. GB10X never mutates a published sidecar in place.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let disk = DiskHeader::decode(&mmap)?;

        if disk.logical.source_digest != source.source_digest() {
            return Err(PlePackIoError::SourceDigestMismatch);
        }
        if disk.logical.row_count != source.row_count() {
            return Err(PlePackIoError::Format(
                "sidecar/source logical row counts differ",
            ));
        }
        if disk.logical.row_bytes != source.row_bytes() {
            return Err(PlePackIoError::Format(
                "sidecar/source logical row widths differ",
            ));
        }

        let expected_file_bytes = disk
            .data_offset
            .checked_add(disk.overlay_bytes)
            .ok_or(PlePackIoError::Format("sidecar file length overflow"))?;
        if u64::try_from(mmap.len())
            .map_err(|_| PlePackIoError::Format("mapped sidecar length does not fit u64"))?
            != expected_file_bytes
        {
            return Err(PlePackIoError::Format(
                "sidecar file length does not match header",
            ));
        }

        let index_offset = usize::try_from(DISK_HEADER_BYTES)
            .map_err(|_| PlePackIoError::Format("index offset does not fit usize"))?;
        let index_bytes = usize::try_from(disk.index_bytes)
            .map_err(|_| PlePackIoError::Format("index byte length does not fit usize"))?;
        let index_end = index_offset
            .checked_add(index_bytes)
            .ok_or(PlePackIoError::Format("index end does not fit usize"))?;
        let index = mmap
            .get(index_offset..index_end)
            .ok_or(PlePackIoError::Format("hot index lies outside sidecar"))?;
        let actual_index_digest: [u8; 32] = Sha256::digest(index).into();
        if actual_index_digest != disk.logical.index_digest {
            return Err(PlePackIoError::IndexDigestMismatch);
        }

        let hot_row_count = usize::try_from(disk.hot_row_count)
            .map_err(|_| PlePackIoError::Format("hot row count does not fit usize"))?;
        validate_index(index, hot_row_count, &disk)?;

        let data_offset = usize::try_from(disk.data_offset)
            .map_err(|_| PlePackIoError::Format("overlay data offset does not fit usize"))?;

        Ok(Self {
            mmap,
            source,
            header: disk.logical,
            hot_row_count,
            index_offset,
            data_offset,
            overlay_bytes: disk.overlay_bytes,
        })
    }

    /// Immutable exact sidecar provenance/geometry.
    pub fn header(&self) -> &PlePackHeader {
        &self.header
    }

    /// Return whether `logical_row` has a verified duplicate in the hot overlay.
    pub fn has_hot_overlay(&self, logical_row: u32) -> bool {
        self.find_overlay_offset(logical_row)
            .ok()
            .flatten()
            .is_some()
    }

    /// Verify every stored hot-overlay row byte-for-byte against the immutable exact source.
    pub fn verify_hot_overlay(&self) -> Result<u64, PlePackIoError> {
        let mut expected = vec![0_u8; self.header.row_bytes as usize];

        for ordinal in 0..self.hot_row_count {
            let (logical_row, overlay_offset) = self.index_entry(ordinal)?;
            self.source.read_exact_row(logical_row, &mut expected)?;
            let actual = self.overlay_row(overlay_offset, "overlay verification")?;
            if actual != expected.as_slice() {
                return Err(PlePackIoError::OverlayDataMismatch { logical_row });
            }
        }

        u64::try_from(self.hot_row_count)
            .map_err(|_| PlePackIoError::Format("hot row count does not fit u64"))
    }

    /// Read one logical row exactly, preferring the mmap hot overlay and falling back to source.
    pub fn read_exact_row(&self, logical_row: u32, dst: &mut [u8]) -> Result<(), PlePackIoError> {
        if logical_row as u64 >= self.header.row_count {
            return Err(crate::PlePackError::LogicalRowOutOfRange {
                row: logical_row,
                row_count: self.header.row_count,
            }
            .into());
        }
        let expected = self.header.row_bytes as usize;
        if dst.len() != expected {
            return Err(PlePackIoError::RowBufferWidth {
                expected,
                actual: dst.len(),
            });
        }

        if let Some(overlay_offset) = self.find_overlay_offset(logical_row)? {
            dst.copy_from_slice(self.overlay_row(overlay_offset, "overlay")?);
            return Ok(());
        }

        self.source.read_exact_row(logical_row, dst)
    }

    fn overlay_row(
        &self,
        overlay_offset: u64,
        context: &'static str,
    ) -> Result<&[u8], PlePackIoError> {
        let row_bytes = self.header.row_bytes as u64;
        let relative_end = overlay_offset
            .checked_add(row_bytes)
            .ok_or(PlePackIoError::Format(match context {
                "overlay verification" => "overlay verification row end overflow",
                _ => "overlay row end overflow",
            }))?;
        if relative_end > self.overlay_bytes {
            return Err(PlePackIoError::Format(match context {
                "overlay verification" => "overlay verification row exceeds data region",
                _ => "overlay row exceeds verified data region",
            }));
        }
        let absolute = (self.data_offset as u64)
            .checked_add(overlay_offset)
            .ok_or(PlePackIoError::Format(match context {
                "overlay verification" => "overlay verification absolute offset overflow",
                _ => "overlay absolute row offset overflow",
            }))?;
        let start = usize::try_from(absolute).map_err(|_| {
            PlePackIoError::Format(match context {
                "overlay verification" => "overlay verification offset does not fit usize",
                _ => "overlay row offset does not fit usize",
            })
        })?;
        let row_len = self.header.row_bytes as usize;
        let end = start.checked_add(row_len).ok_or(PlePackIoError::Format(
            match context {
                "overlay verification" => "overlay verification slice end overflow",
                _ => "overlay row slice end overflow",
            },
        ))?;
        self.mmap.get(start..end).ok_or(PlePackIoError::Format(
            match context {
                "overlay verification" => "overlay verification row lies outside sidecar",
                _ => "overlay row lies outside sidecar",
            },
        ))
    }

    fn find_overlay_offset(&self, logical_row: u32) -> Result<Option<u64>, PlePackIoError> {
        let mut low = 0_usize;
        let mut high = self.hot_row_count;
        while low < high {
            let middle = low + (high - low) / 2;
            let (row, offset) = self.index_entry(middle)?;
            match row.cmp(&logical_row) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Greater => high = middle,
                std::cmp::Ordering::Equal => return Ok(Some(offset)),
            }
        }
        Ok(None)
    }

    fn index_entry(&self, ordinal: usize) -> Result<(u32, u64), PlePackIoError> {
        if ordinal >= self.hot_row_count {
            return Err(PlePackIoError::Format("hot index ordinal out of range"));
        }
        let byte_offset = ordinal
            .checked_mul(INDEX_ENTRY_BYTES as usize)
            .and_then(|offset| self.index_offset.checked_add(offset))
            .ok_or(PlePackIoError::Format("hot index entry offset overflow"))?;
        let end = byte_offset
            .checked_add(INDEX_ENTRY_BYTES as usize)
            .ok_or(PlePackIoError::Format("hot index entry end overflow"))?;
        let bytes = self
            .mmap
            .get(byte_offset..end)
            .ok_or(PlePackIoError::Format("hot index entry outside sidecar"))?;
        decode_index_entry(bytes)
    }
}

fn validate_index(
    index: &[u8],
    hot_row_count: usize,
    disk: &DiskHeader,
) -> Result<(), PlePackIoError> {
    let expected_bytes = hot_row_count
        .checked_mul(INDEX_ENTRY_BYTES as usize)
        .ok_or(PlePackIoError::Format("hot index validation size overflow"))?;
    if index.len() != expected_bytes {
        return Err(PlePackIoError::Format(
            "hot index length does not match row count",
        ));
    }

    let row_bytes = disk.logical.row_bytes as u64;
    let block_bytes = disk.logical.block_bytes as u64;
    let mut previous_row = None;
    for ordinal in 0..hot_row_count {
        let start = ordinal
            .checked_mul(INDEX_ENTRY_BYTES as usize)
            .ok_or(PlePackIoError::Format("hot index scan offset overflow"))?;
        let end = start
            .checked_add(INDEX_ENTRY_BYTES as usize)
            .ok_or(PlePackIoError::Format("hot index scan end overflow"))?;
        let (logical_row, overlay_offset) = decode_index_entry(&index[start..end])?;
        if logical_row as u64 >= disk.logical.row_count {
            return Err(PlePackIoError::Format(
                "hot index contains logical row outside source",
            ));
        }
        if previous_row.is_some_and(|previous| previous >= logical_row) {
            return Err(PlePackIoError::Format(
                "hot index logical rows are not strictly sorted",
            ));
        }
        previous_row = Some(logical_row);

        let row_end = overlay_offset
            .checked_add(row_bytes)
            .ok_or(PlePackIoError::Format("hot index row end overflow"))?;
        if row_end > disk.overlay_bytes {
            return Err(PlePackIoError::Format(
                "hot index row exceeds overlay data region",
            ));
        }
        let offset_in_block = overlay_offset % block_bytes;
        if offset_in_block
            .checked_add(row_bytes)
            .ok_or(PlePackIoError::Format("hot index block boundary overflow"))?
            > block_bytes
        {
            return Err(PlePackIoError::Format(
                "hot index row crosses overlay block boundary",
            ));
        }
    }
    Ok(())
}
