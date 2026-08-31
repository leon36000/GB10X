//! Deterministic exact PLEPack base/overlay layout planning.

use crate::PlePackError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Physical location of one duplicated hot logical row inside the overlay region.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HotOverlayPlacement {
    /// Logical PLE row ID in the immutable exact base region.
    pub logical_row: u32,
    /// Zero-based block ID relative to the beginning of the hot overlay region.
    pub block_id: u64,
    /// Byte offset of the complete row inside its overlay block.
    pub offset_in_block: u32,
}

/// Exact PLEPack layout: identity-mapped cold base plus bounded locality-optimized hot overlay.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LayoutPlan {
    row_count: u64,
    row_bytes: u32,
    block_bytes: u32,
    hot_physical_order: Vec<u32>,
    hot_overlay_placements: Vec<HotOverlayPlacement>,
}

impl LayoutPlan {
    /// Number of exact rows in the immutable identity-mapped cold base region.
    pub fn base_row_count(&self) -> u64 {
        self.row_count
    }

    /// Byte offset of `logical_row` inside the exact cold base region.
    ///
    /// The base is deliberately index-free: every row retains logical order.
    pub fn base_offset(&self, logical_row: u32) -> Result<u64, PlePackError> {
        if logical_row as u64 >= self.row_count {
            return Err(PlePackError::LogicalRowOutOfRange {
                row: logical_row,
                row_count: self.row_count,
            });
        }
        (logical_row as u64)
            .checked_mul(self.row_bytes as u64)
            .ok_or(PlePackError::Overflow("cold base row offset"))
    }

    /// Hot rows in their deterministic physical overlay order.
    pub fn hot_physical_order(&self) -> &[u32] {
        &self.hot_physical_order
    }

    /// Physical overlay placements corresponding to [`Self::hot_physical_order`].
    pub fn hot_overlay_placements(&self) -> &[HotOverlayPlacement] {
        &self.hot_overlay_placements
    }

    /// Exact logical row width in bytes.
    pub fn row_bytes(&self) -> u32 {
        self.row_bytes
    }

    /// Overlay block size in bytes.
    pub fn block_bytes(&self) -> u32 {
        self.block_bytes
    }
}

/// Build a deterministic exact PLEPack layout from observed row co-access traces.
///
/// All logical rows remain present exactly once in the index-free cold base. Only rows observed in
/// `trace` are duplicated into the hot overlay. Within each first-seen trace group, logical row IDs
/// are sorted before placement so the same trace produces byte-identical output independent of
/// duplicate order. Memory usage is proportional to the number of unique traced rows, not to the
/// complete PLE row space.
pub fn plan_exact_layout(
    row_count: u64,
    row_bytes: u32,
    block_bytes: u32,
    trace: &[Vec<u32>],
) -> Result<LayoutPlan, PlePackError> {
    if row_count == 0 {
        return Err(PlePackError::InvalidGeometry("row_count must be nonzero"));
    }
    if row_count > u32::MAX as u64 + 1 {
        return Err(PlePackError::InvalidGeometry(
            "logical row space must fit u32 row identifiers",
        ));
    }
    if row_bytes == 0 {
        return Err(PlePackError::InvalidGeometry("row_bytes must be nonzero"));
    }
    if block_bytes < row_bytes {
        return Err(PlePackError::InvalidGeometry(
            "block_bytes must fit at least one complete row",
        ));
    }

    let rows_per_block = block_bytes / row_bytes;
    if rows_per_block == 0 {
        return Err(PlePackError::InvalidGeometry(
            "block geometry yields zero complete rows",
        ));
    }

    let mut seen = BTreeSet::new();
    let mut hot_physical_order = Vec::new();

    for batch in trace {
        let mut group = batch.clone();
        group.sort_unstable();
        group.dedup();
        for row in group {
            if row as u64 >= row_count {
                return Err(PlePackError::TraceRowOutOfRange { row, row_count });
            }
            if seen.insert(row) {
                hot_physical_order.push(row);
            }
        }
    }

    let mut hot_overlay_placements = Vec::with_capacity(hot_physical_order.len());
    for (ordinal, &logical_row) in hot_physical_order.iter().enumerate() {
        let ordinal = u64::try_from(ordinal)
            .map_err(|_| PlePackError::Overflow("hot overlay physical ordinal"))?;
        let rows_per_block = rows_per_block as u64;
        let block_id = ordinal / rows_per_block;
        let slot = ordinal % rows_per_block;
        let offset = slot
            .checked_mul(row_bytes as u64)
            .ok_or(PlePackError::Overflow("hot overlay block offset"))?;
        let offset_in_block = u32::try_from(offset)
            .map_err(|_| PlePackError::Overflow("hot overlay block offset u32"))?;
        hot_overlay_placements.push(HotOverlayPlacement {
            logical_row,
            block_id,
            offset_in_block,
        });
    }

    Ok(LayoutPlan {
        row_count,
        row_bytes,
        block_bytes,
        hot_physical_order,
        hot_overlay_placements,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn trace() -> Vec<Vec<u32>> {
        vec![vec![9, 3, 7, 3], vec![9, 7, 11], vec![2, 1], vec![7, 9]]
    }

    #[test]
    fn same_trace_produces_byte_identical_plan() {
        let first = plan_exact_layout(40, 320, 4096, &trace()).expect("layout");
        let second = plan_exact_layout(40, 320, 4096, &trace()).expect("layout");
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
    }

    #[test]
    fn cold_base_region_maps_every_logical_row_without_an_index() {
        let plan = plan_exact_layout(40, 320, 4096, &trace()).expect("layout");
        assert_eq!(plan.base_row_count(), 40);
        assert_eq!(plan.base_offset(0).unwrap(), 0);
        assert_eq!(plan.base_offset(1).unwrap(), 320);
        assert_eq!(plan.base_offset(39).unwrap(), 39 * 320);
        assert!(plan.base_offset(40).is_err());
    }

    #[test]
    fn hot_overlay_contains_each_observed_row_at_most_once() {
        let plan = plan_exact_layout(40, 320, 4096, &trace()).expect("layout");
        let rows = plan
            .hot_overlay_placements()
            .iter()
            .map(|placement| placement.logical_row)
            .collect::<BTreeSet<_>>();
        assert_eq!(rows, BTreeSet::from([1, 2, 3, 7, 9, 11]));
        assert_eq!(rows.len(), plan.hot_overlay_placements().len());
    }

    #[test]
    fn no_hot_overlay_row_crosses_block_boundary() {
        let plan = plan_exact_layout(40, 320, 4096, &trace()).expect("layout");
        for placement in plan.hot_overlay_placements() {
            assert!(placement.offset_in_block as u64 + 320 <= 4096);
        }
    }

    #[test]
    fn first_observed_coaccess_group_is_physically_adjacent() {
        let plan = plan_exact_layout(40, 320, 4096, &trace()).expect("layout");
        assert_eq!(&plan.hot_physical_order()[..3], &[3, 7, 9]);
    }

    #[test]
    fn rejects_impossible_block_geometry_and_out_of_range_trace_rows() {
        assert!(plan_exact_layout(40, 320, 256, &trace()).is_err());
        assert!(plan_exact_layout(40, 320, 4096, &[vec![40]]).is_err());
    }
}
