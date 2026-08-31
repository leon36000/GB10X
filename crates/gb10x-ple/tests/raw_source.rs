use gb10x_ple::{ExactPleRowSource, PlePackIoError, RawFileRowSource};
use std::fs;
use tempfile::tempdir;

#[test]
fn raw_source_maps_exact_fixed_width_rows() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ple.raw");
    let bytes = (0_u8..24).collect::<Vec<_>>();
    fs::write(&path, &bytes).unwrap();

    let source = RawFileRowSource::open(&path, 8).expect("raw source");
    assert_eq!(source.row_count(), 3);
    assert_eq!(source.row_bytes(), 8);

    let mut row = [0_u8; 8];
    source.read_exact_row(1, &mut row).unwrap();
    assert_eq!(row, [8, 9, 10, 11, 12, 13, 14, 15]);
}

#[test]
fn raw_source_rejects_partial_rows_and_bad_buffers() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("partial.raw");
    fs::write(&path, [1_u8; 17]).unwrap();
    assert!(RawFileRowSource::open(&path, 8).is_err());

    fs::write(&path, [1_u8; 16]).unwrap();
    let source = RawFileRowSource::open(&path, 8).unwrap();
    let mut wrong = [0_u8; 7];
    assert!(matches!(
        source.read_exact_row(0, &mut wrong),
        Err(PlePackIoError::RowBufferWidth { .. })
    ));
}

#[test]
fn raw_source_digest_changes_when_source_bytes_change() {
    let dir = tempdir().unwrap();
    let first = dir.path().join("first.raw");
    let second = dir.path().join("second.raw");
    fs::write(&first, [3_u8; 16]).unwrap();
    fs::write(&second, [4_u8; 16]).unwrap();

    let a = RawFileRowSource::open(&first, 8).unwrap();
    let b = RawFileRowSource::open(&second, 8).unwrap();
    assert_ne!(a.source_digest(), b.source_digest());
}
