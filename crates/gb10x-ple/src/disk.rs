//! Private fixed-width PLEPack sidecar header and index encoding.

use crate::{PLEPACK_MAGIC, PLEPACK_VERSION, PlePackHeader, PlePackIoError};

pub(crate) const DISK_HEADER_BYTES: u64 = 128;
pub(crate) const INDEX_ENTRY_BYTES: u64 = 12;
const EXACT_OVERLAY_FLAGS: u32 = 1;

#[derive(Clone, Debug)]
pub(crate) struct DiskHeader {
    pub(crate) logical: PlePackHeader,
    pub(crate) hot_row_count: u64,
    pub(crate) index_bytes: u64,
    pub(crate) data_offset: u64,
    pub(crate) overlay_bytes: u64,
}

impl DiskHeader {
    pub(crate) fn new(
        logical: PlePackHeader,
        hot_row_count: u64,
        index_bytes: u64,
        data_offset: u64,
        overlay_bytes: u64,
    ) -> Result<Self, PlePackIoError> {
        let expected_index_bytes = hot_row_count
            .checked_mul(INDEX_ENTRY_BYTES)
            .ok_or(PlePackIoError::Format("hot index byte count overflow"))?;
        if index_bytes != expected_index_bytes {
            return Err(PlePackIoError::Format(
                "hot index byte count does not match hot row count",
            ));
        }
        let minimum_data_offset = DISK_HEADER_BYTES
            .checked_add(index_bytes)
            .ok_or(PlePackIoError::Format("index end overflow"))?;
        if data_offset < minimum_data_offset {
            return Err(PlePackIoError::Format("overlay data overlaps index"));
        }
        if !data_offset.is_multiple_of(logical.block_bytes as u64) {
            return Err(PlePackIoError::Format(
                "overlay data offset is not block aligned",
            ));
        }
        if !overlay_bytes.is_multiple_of(logical.block_bytes as u64) {
            return Err(PlePackIoError::Format(
                "overlay byte length is not block aligned",
            ));
        }
        Ok(Self {
            logical,
            hot_row_count,
            index_bytes,
            data_offset,
            overlay_bytes,
        })
    }

    pub(crate) fn encode(&self) -> [u8; DISK_HEADER_BYTES as usize] {
        let mut bytes = [0_u8; DISK_HEADER_BYTES as usize];
        bytes[0..8].copy_from_slice(&PLEPACK_MAGIC);
        bytes[8..12].copy_from_slice(&PLEPACK_VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&EXACT_OVERLAY_FLAGS.to_le_bytes());
        bytes[16..48].copy_from_slice(&self.logical.source_digest);
        bytes[48..56].copy_from_slice(&self.logical.row_count.to_le_bytes());
        bytes[56..60].copy_from_slice(&self.logical.row_bytes.to_le_bytes());
        bytes[60..64].copy_from_slice(&self.logical.block_bytes.to_le_bytes());
        bytes[64..72].copy_from_slice(&self.hot_row_count.to_le_bytes());
        bytes[72..80].copy_from_slice(&self.index_bytes.to_le_bytes());
        bytes[80..88].copy_from_slice(&self.data_offset.to_le_bytes());
        bytes[88..96].copy_from_slice(&self.overlay_bytes.to_le_bytes());
        bytes[96..128].copy_from_slice(&self.logical.index_digest);
        bytes
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, PlePackIoError> {
        if bytes.len() < DISK_HEADER_BYTES as usize {
            return Err(PlePackIoError::Format("sidecar is shorter than header"));
        }
        if bytes.get(0..8) != Some(PLEPACK_MAGIC.as_slice()) {
            return Err(PlePackIoError::Format("PLEPack magic mismatch"));
        }
        if read_u32(bytes, 8)? != PLEPACK_VERSION {
            return Err(PlePackIoError::Format("PLEPack version mismatch"));
        }
        if read_u32(bytes, 12)? != EXACT_OVERLAY_FLAGS {
            return Err(PlePackIoError::Format("unsupported PLEPack flags"));
        }

        let mut source_digest = [0_u8; 32];
        source_digest.copy_from_slice(&bytes[16..48]);
        let row_count = read_u64(bytes, 48)?;
        let row_bytes = read_u32(bytes, 56)?;
        let block_bytes = read_u32(bytes, 60)?;
        let hot_row_count = read_u64(bytes, 64)?;
        let index_bytes = read_u64(bytes, 72)?;
        let data_offset = read_u64(bytes, 80)?;
        let overlay_bytes = read_u64(bytes, 88)?;
        let mut index_digest = [0_u8; 32];
        index_digest.copy_from_slice(&bytes[96..128]);

        let logical = PlePackHeader::new(
            source_digest,
            row_count,
            row_bytes,
            block_bytes,
            index_digest,
        )?;
        Self::new(
            logical,
            hot_row_count,
            index_bytes,
            data_offset,
            overlay_bytes,
        )
    }
}

pub(crate) fn encode_index_entry(logical_row: u32, overlay_offset: u64) -> [u8; 12] {
    let mut bytes = [0_u8; 12];
    bytes[0..4].copy_from_slice(&logical_row.to_le_bytes());
    bytes[4..12].copy_from_slice(&overlay_offset.to_le_bytes());
    bytes
}

pub(crate) fn decode_index_entry(bytes: &[u8]) -> Result<(u32, u64), PlePackIoError> {
    if bytes.len() != INDEX_ENTRY_BYTES as usize {
        return Err(PlePackIoError::Format("invalid hot index entry width"));
    }
    Ok((read_u32(bytes, 0)?, read_u64(bytes, 4)?))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, PlePackIoError> {
    let end = offset
        .checked_add(4)
        .ok_or(PlePackIoError::Format("u32 header offset overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or(PlePackIoError::Format("truncated u32 header field"))?;
    Ok(u32::from_le_bytes(
        slice
            .try_into()
            .map_err(|_| PlePackIoError::Format("invalid u32 header field"))?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, PlePackIoError> {
    let end = offset
        .checked_add(8)
        .ok_or(PlePackIoError::Format("u64 header offset overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or(PlePackIoError::Format("truncated u64 header field"))?;
    Ok(u64::from_le_bytes(
        slice
            .try_into()
            .map_err(|_| PlePackIoError::Format("invalid u64 header field"))?,
    ))
}
