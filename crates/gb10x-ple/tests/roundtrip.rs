use gb10x_ple::{
    ExactPleRowSource, PlePackIoError, PlePackReader, PlePackWriter, plan_exact_layout,
};
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use tempfile::tempdir;

#[derive(Clone)]
struct MemoryRows {
    rows: Vec<Vec<u8>>,
    digest: [u8; 32],
}

impl MemoryRows {
    fn new(row_count: u32, row_bytes: usize) -> Self {
        let rows = (0..row_count)
            .map(|row| {
                (0..row_bytes)
                    .map(|column| ((row as usize * 131 + column * 17) % 251) as u8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let digest = digest_rows(&rows);
        Self { rows, digest }
    }
}

fn digest_rows(rows: &[Vec<u8>]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for row in rows {
        hasher.update(row);
    }
    hasher.finalize().into()
}

impl ExactPleRowSource for MemoryRows {
    fn row_count(&self) -> u64 {
        self.rows.len() as u64
    }

    fn row_bytes(&self) -> u32 {
        self.rows[0].len() as u32
    }

    fn source_digest(&self) -> [u8; 32] {
        self.digest
    }

    fn read_exact_row(&self, logical_row: u32, dst: &mut [u8]) -> Result<(), PlePackIoError> {
        let row = self
            .rows
            .get(logical_row as usize)
            .ok_or(PlePackIoError::Source(
                "logical row outside synthetic source",
            ))?;
        if dst.len() != row.len() {
            return Err(PlePackIoError::Source(
                "synthetic row buffer width mismatch",
            ));
        }
        dst.copy_from_slice(row);
        Ok(())
    }
}

fn trace() -> Vec<Vec<u32>> {
    vec![vec![9, 3, 7, 3], vec![9, 7, 11], vec![2, 1], vec![7, 9]]
}

#[test]
fn exact_overlay_roundtrips_every_logical_row() {
    let source = MemoryRows::new(40, 320);
    let plan = plan_exact_layout(40, 320, 4096, &trace()).expect("layout");
    let dir = tempdir().unwrap();
    let path = dir.path().join("qwen38.plepack");

    let report = PlePackWriter::write_overlay(&path, &source, &plan).expect("write overlay");
    assert_eq!(report.hot_rows, 6);
    assert!(report.overlay_bytes > 0);

    let reader = PlePackReader::open(&path, source.clone()).expect("open exact overlay");
    assert!(reader.has_hot_overlay(3));
    assert!(reader.has_hot_overlay(11));
    assert!(!reader.has_hot_overlay(12));

    let mut output = vec![0_u8; 320];
    for logical_row in 0_u32..40 {
        reader
            .read_exact_row(logical_row, &mut output)
            .expect("read exact logical row");
        assert_eq!(output, source.rows[logical_row as usize]);
    }
}

#[test]
fn corrupted_overlay_index_digest_is_rejected() {
    let source = MemoryRows::new(40, 320);
    let plan = plan_exact_layout(40, 320, 4096, &trace()).unwrap();
    let dir = tempdir().unwrap();
    let path = dir.path().join("corrupt.plepack");
    let report = PlePackWriter::write_overlay(&path, &source, &plan).unwrap();
    assert!(report.index_bytes > 0);

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    file.seek(SeekFrom::Start(report.index_offset)).unwrap();
    let mut byte = [0_u8; 1];
    file.read_exact(&mut byte).unwrap();
    byte[0] ^= 0x80;
    file.seek(SeekFrom::Start(report.index_offset)).unwrap();
    file.write_all(&byte).unwrap();
    file.sync_all().unwrap();

    assert!(matches!(
        PlePackReader::open(&path, source),
        Err(PlePackIoError::IndexDigestMismatch)
    ));
}

#[test]
fn source_digest_mismatch_is_rejected() {
    let source = MemoryRows::new(40, 320);
    let plan = plan_exact_layout(40, 320, 4096, &trace()).unwrap();
    let dir = tempdir().unwrap();
    let path = dir.path().join("wrong-source.plepack");
    PlePackWriter::write_overlay(&path, &source, &plan).unwrap();

    let mut wrong_source = source;
    wrong_source.digest[0] ^= 1;
    assert!(matches!(
        PlePackReader::open(&path, wrong_source),
        Err(PlePackIoError::SourceDigestMismatch)
    ));
}
