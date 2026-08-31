//! Atomic exact hot-overlay PLEPack sidecar writer.

use crate::disk::{DISK_HEADER_BYTES, INDEX_ENTRY_BYTES, DiskHeader, encode_index_entry};
use crate::{ExactPleRowSource, LayoutPlan, PlePackHeader, PlePackIoError};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Measured byte geometry of one successfully published hot-overlay sidecar.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlePackWriteReport {
    /// Number of unique exact logical rows duplicated into the hot overlay.
    pub hot_rows: u64,
    /// Byte offset of the sorted hot index in the sidecar.
    pub index_offset: u64,
    /// Serialized hot-index byte count.
    pub index_bytes: u64,
    /// Block-padded hot-overlay data byte count.
    pub overlay_bytes: u64,
    /// Final complete sidecar file size.
    pub file_bytes: u64,
}

/// Stateless exact PLEPack hot-overlay builder.
pub struct PlePackWriter;

impl PlePackWriter {
    /// Build and atomically publish an exact hot-overlay sidecar.
    ///
    /// The immutable cold base remains in `source`; only traced hot rows are duplicated. The final
    /// path is replaced only after the temporary sidecar has been completely written and synced.
    pub fn write_overlay<S: ExactPleRowSource>(
        path: impl AsRef<Path>,
        source: &S,
        plan: &LayoutPlan,
    ) -> Result<PlePackWriteReport, PlePackIoError> {
        let path = path.as_ref();
        validate_source_plan(source, plan)?;

        let mut index_entries = Vec::with_capacity(plan.hot_overlay_placements().len());
        for placement in plan.hot_overlay_placements() {
            let overlay_offset = placement
                .block_id
                .checked_mul(plan.block_bytes() as u64)
                .and_then(|base| base.checked_add(placement.offset_in_block as u64))
                .ok_or(PlePackIoError::Format("hot overlay offset overflow"))?;
            index_entries.push((placement.logical_row, overlay_offset));
        }
        index_entries.sort_unstable_by_key(|entry| entry.0);
        for pair in index_entries.windows(2) {
            if pair[0].0 >= pair[1].0 {
                return Err(PlePackIoError::Format(
                    "hot overlay logical rows are not unique",
                ));
            }
        }

        let mut index_payload = Vec::with_capacity(
            index_entries
                .len()
                .checked_mul(INDEX_ENTRY_BYTES as usize)
                .ok_or(PlePackIoError::Format("hot index allocation overflow"))?,
        );
        for &(logical_row, overlay_offset) in &index_entries {
            index_payload.extend_from_slice(&encode_index_entry(logical_row, overlay_offset));
        }
        let index_digest: [u8; 32] = Sha256::digest(&index_payload).into();
        let index_bytes = u64::try_from(index_payload.len())
            .map_err(|_| PlePackIoError::Format("hot index size does not fit u64"))?;
        let index_offset = DISK_HEADER_BYTES;
        let index_end = index_offset
            .checked_add(index_bytes)
            .ok_or(PlePackIoError::Format("hot index end overflow"))?;
        let data_offset = align_up(index_end, plan.block_bytes() as u64)?;
        let overlay_bytes = overlay_storage_bytes(plan)?;

        let logical_header = PlePackHeader::new(
            source.source_digest(),
            source.row_count(),
            source.row_bytes(),
            plan.block_bytes(),
            index_digest,
        )?;
        let disk_header = DiskHeader::new(
            logical_header,
            u64::try_from(index_entries.len())
                .map_err(|_| PlePackIoError::Format("hot row count does not fit u64"))?,
            index_bytes,
            data_offset,
            overlay_bytes,
        )?;
        let file_bytes = data_offset
            .checked_add(overlay_bytes)
            .ok_or(PlePackIoError::Format("sidecar file size overflow"))?;

        let temp_path = temporary_path(path);
        let result = (|| {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&temp_path)?;
            file.set_len(file_bytes)?;
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&disk_header.encode())?;
            file.seek(SeekFrom::Start(index_offset))?;
            file.write_all(&index_payload)?;

            let mut row = vec![0_u8; source.row_bytes() as usize];
            for placement in plan.hot_overlay_placements() {
                source.read_exact_row(placement.logical_row, &mut row)?;
                let overlay_offset = placement
                    .block_id
                    .checked_mul(plan.block_bytes() as u64)
                    .and_then(|base| base.checked_add(placement.offset_in_block as u64))
                    .ok_or(PlePackIoError::Format("hot overlay write offset overflow"))?;
                let absolute = data_offset
                    .checked_add(overlay_offset)
                    .ok_or(PlePackIoError::Format("hot overlay absolute offset overflow"))?;
                file.seek(SeekFrom::Start(absolute))?;
                file.write_all(&row)?;
            }

            file.sync_all()?;
            drop(file);
            std::fs::rename(&temp_path, path)?;
            sync_parent_directory(path)?;

            Ok(PlePackWriteReport {
                hot_rows: u64::try_from(index_entries.len())
                    .map_err(|_| PlePackIoError::Format("hot row count does not fit u64"))?,
                index_offset,
                index_bytes,
                overlay_bytes,
                file_bytes,
            })
        })();

        if result.is_err() {
            let _ = std::fs::remove_file(&temp_path);
        }
        result
    }
}

fn validate_source_plan<S: ExactPleRowSource>(
    source: &S,
    plan: &LayoutPlan,
) -> Result<(), PlePackIoError> {
    if source.row_count() != plan.base_row_count() {
        return Err(PlePackIoError::Format(
            "source and layout row counts differ",
        ));
    }
    if source.row_bytes() != plan.row_bytes() {
        return Err(PlePackIoError::Format(
            "source and layout row widths differ",
        ));
    }
    if source.row_count() == 0 || source.row_bytes() == 0 {
        return Err(PlePackIoError::Format("source geometry is empty"));
    }
    Ok(())
}

fn overlay_storage_bytes(plan: &LayoutPlan) -> Result<u64, PlePackIoError> {
    let Some(last) = plan.hot_overlay_placements().last() else {
        return Ok(0);
    };
    last.block_id
        .checked_add(1)
        .and_then(|blocks| blocks.checked_mul(plan.block_bytes() as u64))
        .ok_or(PlePackIoError::Format("hot overlay storage size overflow"))
}

fn align_up(value: u64, alignment: u64) -> Result<u64, PlePackIoError> {
    if alignment == 0 {
        return Err(PlePackIoError::Format("zero sidecar alignment"));
    }
    let remainder = value % alignment;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - remainder)
        .ok_or(PlePackIoError::Format("sidecar alignment overflow"))
}

fn temporary_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("plepack");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    path.with_file_name(format!(
        ".{file_name}.tmp-{}-{nonce}",
        std::process::id()
    ))
}

fn sync_parent_directory(path: &Path) -> Result<(), PlePackIoError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)?.sync_all()?;
    Ok(())
}
