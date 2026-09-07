use gb10x_ple::{
    ExactPleRowSource, PlePackIoError, PlePackReader, PlePackWriter, RawFileRowSource,
    plan_exact_layout,
};
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
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

struct DestinationCreatingRows {
    rows: MemoryRows,
    destination: PathBuf,
    sentinel: Vec<u8>,
    created: Cell<bool>,
}

impl ExactPleRowSource for DestinationCreatingRows {
    fn row_count(&self) -> u64 {
        self.rows.row_count()
    }

    fn row_bytes(&self) -> u32 {
        self.rows.row_bytes()
    }

    fn source_digest(&self) -> [u8; 32] {
        self.rows.source_digest()
    }

    fn read_exact_row(&self, logical_row: u32, dst: &mut [u8]) -> Result<(), PlePackIoError> {
        if !self.created.replace(true) {
            fs::write(&self.destination, &self.sentinel)?;
        }
        self.rows.read_exact_row(logical_row, dst)
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
    assert_eq!(reader.verify_hot_overlay().unwrap(), 6);

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
fn corrupted_hot_overlay_data_is_rejected_on_open() {
    let source = MemoryRows::new(40, 320);
    let plan = plan_exact_layout(40, 320, 4096, &trace()).unwrap();
    let dir = tempdir().unwrap();
    let path = dir.path().join("corrupt-data.plepack");
    let report = PlePackWriter::write_overlay(&path, &source, &plan).unwrap();
    let data_offset = report.file_bytes - report.overlay_bytes;

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    file.seek(SeekFrom::Start(data_offset)).unwrap();
    let mut byte = [0_u8; 1];
    file.read_exact(&mut byte).unwrap();
    assert_eq!(byte[0], source.rows[3][0], "fixture must target hot row 3");
    byte[0] ^= 0x01;
    file.seek(SeekFrom::Start(data_offset)).unwrap();
    file.write_all(&byte).unwrap();
    file.sync_all().unwrap();
    let mut persisted = [0_u8; 1];
    file.seek(SeekFrom::Start(data_offset)).unwrap();
    file.read_exact(&mut persisted).unwrap();
    assert_eq!(persisted, byte);
    drop(file);

    let error = PlePackReader::open(&path, source)
        .err()
        .expect("corrupted hot data must fail during open");
    assert!(
        matches!(error, PlePackIoError::OverlayDataMismatch { .. }),
        "unexpected open error: {error:?}"
    );
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

#[test]
fn writer_preserves_an_existing_destination() {
    let source = MemoryRows::new(4, 8);
    let plan = plan_exact_layout(4, 8, 64, &[vec![1, 2]]).unwrap();
    let dir = tempdir().unwrap();
    let path = dir.path().join("existing.plepack");
    let sentinel = b"existing destination".to_vec();
    fs::write(&path, &sentinel).unwrap();

    assert!(PlePackWriter::write_overlay(&path, &source, &plan).is_err());
    assert_eq!(fs::read(&path).unwrap(), sentinel);
}

#[test]
fn writer_rejects_the_exact_source_path_as_destination() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("source.raw");
    let original = (0_u8..32).collect::<Vec<_>>();
    fs::write(&path, &original).unwrap();
    let source = RawFileRowSource::open(&path, 8).unwrap();
    let plan = plan_exact_layout(4, 8, 64, &[vec![1, 2]]).unwrap();

    assert!(PlePackWriter::write_overlay(&path, &source, &plan).is_err());
    assert_eq!(fs::read(&path).unwrap(), original);
}

#[cfg(unix)]
#[test]
fn writer_preserves_a_symlink_destination_and_its_target() {
    use std::os::unix::fs::symlink;

    let source = MemoryRows::new(4, 8);
    let plan = plan_exact_layout(4, 8, 64, &[vec![1, 2]]).unwrap();
    let dir = tempdir().unwrap();
    let target = dir.path().join("target");
    let path = dir.path().join("output");
    let sentinel = b"symlink target".to_vec();
    fs::write(&target, &sentinel).unwrap();
    symlink(&target, &path).unwrap();

    assert!(PlePackWriter::write_overlay(&path, &source, &plan).is_err());
    assert!(
        fs::symlink_metadata(&path)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(fs::read(&target).unwrap(), sentinel);
}

#[test]
fn writer_preserves_a_hard_link_destination_and_its_peer() {
    let source = MemoryRows::new(4, 8);
    let plan = plan_exact_layout(4, 8, 64, &[vec![1, 2]]).unwrap();
    let dir = tempdir().unwrap();
    let peer = dir.path().join("peer");
    let path = dir.path().join("output");
    let sentinel = b"hard link contents".to_vec();
    fs::write(&peer, &sentinel).unwrap();
    fs::hard_link(&peer, &path).unwrap();

    assert!(PlePackWriter::write_overlay(&path, &source, &plan).is_err());
    assert_eq!(fs::read(&path).unwrap(), sentinel);
    assert_eq!(fs::read(&peer).unwrap(), sentinel);
}

#[test]
fn writer_preserves_an_existing_directory_destination() {
    let source = MemoryRows::new(4, 8);
    let plan = plan_exact_layout(4, 8, 64, &[vec![1, 2]]).unwrap();
    let dir = tempdir().unwrap();
    let path = dir.path().join("output");
    fs::create_dir(&path).unwrap();

    assert!(PlePackWriter::write_overlay(&path, &source, &plan).is_err());
    assert!(path.is_dir());
}

#[test]
fn writer_loses_a_publication_race_without_replacing_the_winner() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("raced.plepack");
    let sentinel = b"race winner".to_vec();
    let source = DestinationCreatingRows {
        rows: MemoryRows::new(4, 8),
        destination: path.clone(),
        sentinel: sentinel.clone(),
        created: Cell::new(false),
    };
    let plan = plan_exact_layout(4, 8, 64, &[vec![1, 2]]).unwrap();

    assert!(PlePackWriter::write_overlay(&path, &source, &plan).is_err());
    assert_eq!(fs::read(&path).unwrap(), sentinel);
}
