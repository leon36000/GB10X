use clap::{Args, Parser, Subcommand};
use gb10x_ple::{
    ExactPleRowSource, OverlayAdmissionBudget, PlePackReader, PlePackWriter,
    QWEN38_FLASH_NEXT_MODEL_ID, QWEN38_FLASH_NEXT_REVISION, QWEN38_PLE_HASH_BUFFER_PAYLOAD_BYTES,
    RawFileRowSource, SafetensorsPleSource, qwen38_ple_manifest_from_index,
    verify_qwen38_ple_hash_buffers,
};
use gb10x_tools::plepack::plan_from_trace_json_with_budget;
use serde_json::json;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "gb10x-plepack",
    about = "Plan, build and verify exact GB10X PLE hot-overlay sidecars"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Build a deterministic locality plan from a measured logical-row trace.
    Plan(PlanArgs),
    /// Build an exact hot-overlay sidecar from an immutable flat row source.
    Build(BuildArgs),
    /// Verify every stored hot row byte-for-byte against its immutable source.
    Verify(VerifyArgs),
    /// Validate and hash the pinned Qwen PLE tensors directly from safetensors.
    SourceVerify(SourceVerifyArgs),
    /// Compare the persistent Qwen PLE hash buffers using bounded local reads.
    HashVerify(HashVerifyArgs),
}

#[derive(Debug, Args)]
struct PlanArgs {
    /// JSON array-of-arrays containing co-accessed logical row IDs.
    #[arg(long)]
    trace: PathBuf,
    /// Output path for the deterministic JSON layout plan.
    #[arg(long)]
    out: PathBuf,
    /// Number of logical rows in the immutable exact source.
    #[arg(long)]
    row_count: u64,
    /// Exact byte width of one logical row.
    #[arg(long)]
    row_bytes: u32,
    /// Physical hot-overlay block size.
    #[arg(long, default_value_t = 4096)]
    block_bytes: u32,
    /// Maximum number of trace-distinct rows duplicated into the hot overlay.
    #[arg(long)]
    max_hot_rows: Option<u64>,
    /// Maximum block-padded hot-overlay data bytes; header and index bytes are reported separately.
    #[arg(long)]
    max_overlay_bytes: Option<u64>,
}

#[derive(Debug, Args)]
struct BuildArgs {
    /// Prepared immutable flat file containing consecutive exact row bytes.
    #[arg(long)]
    source: PathBuf,
    /// JSON array-of-arrays containing co-accessed logical row IDs.
    #[arg(long)]
    trace: PathBuf,
    /// Output path for the exact PLEPack hot-overlay sidecar.
    #[arg(long)]
    out: PathBuf,
    /// Exact byte width of one logical row in the source.
    #[arg(long)]
    row_bytes: u32,
    /// Physical hot-overlay block size.
    #[arg(long, default_value_t = 4096)]
    block_bytes: u32,
    /// Maximum number of trace-distinct rows duplicated into the hot overlay.
    #[arg(long)]
    max_hot_rows: Option<u64>,
    /// Maximum block-padded hot-overlay data bytes; header and index bytes are reported separately.
    #[arg(long)]
    max_overlay_bytes: Option<u64>,
}

#[derive(Debug, Args)]
struct VerifyArgs {
    /// Prepared immutable flat file used to build the sidecar.
    #[arg(long)]
    source: PathBuf,
    /// Exact PLEPack hot-overlay sidecar to verify.
    #[arg(long)]
    pack: PathBuf,
    /// Exact byte width of one logical row in the source.
    #[arg(long)]
    row_bytes: u32,
}

#[derive(Debug, Args)]
struct SourceVerifyArgs {
    /// Directory containing the pinned Qwen checkpoint and model.safetensors.index.json.
    #[arg(long)]
    model_dir: PathBuf,
}

#[derive(Debug, Args)]
struct HashVerifyArgs {
    /// Directory containing the pinned Qwen checkpoint and model.safetensors.index.json.
    #[arg(long)]
    model_dir: PathBuf,
    /// Checkpoint revision observed by the caller; it must equal the GB10X pinned revision.
    #[arg(long)]
    observed_revision: String,
}

fn main() {
    let cli = Cli::parse();
    if let Err(error) = run(cli) {
        eprintln!("gb10x-plepack: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Plan(args) => plan(args),
        Command::Build(args) => build(args),
        Command::Verify(args) => verify(args),
        Command::SourceVerify(args) => source_verify(args),
        Command::HashVerify(args) => hash_verify(args),
    }
}

fn plan(args: PlanArgs) -> Result<(), Box<dyn std::error::Error>> {
    let trace = fs::read_to_string(args.trace)?;
    let plan = plan_from_trace_json_with_budget(
        args.row_count,
        args.row_bytes,
        args.block_bytes,
        &trace,
        overlay_budget(args.max_hot_rows, args.max_overlay_bytes),
    )?;
    fs::write(args.out, serde_json::to_vec_pretty(&plan)?)?;
    Ok(())
}

fn build(args: BuildArgs) -> Result<(), Box<dyn std::error::Error>> {
    let source = RawFileRowSource::open(&args.source, args.row_bytes)?;
    let trace = fs::read_to_string(args.trace)?;
    let plan = plan_from_trace_json_with_budget(
        source.row_count(),
        source.row_bytes(),
        args.block_bytes,
        &trace,
        overlay_budget(args.max_hot_rows, args.max_overlay_bytes),
    )?;
    let report = PlePackWriter::write_overlay(&args.out, &source, &plan)?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "hot_rows": report.hot_rows,
            "index_bytes": report.index_bytes,
            "overlay_bytes": report.overlay_bytes,
            "file_bytes": report.file_bytes,
            "row_count": source.row_count(),
            "row_bytes": source.row_bytes(),
        }))?
    );
    Ok(())
}

fn overlay_budget(
    max_hot_rows: Option<u64>,
    max_overlay_bytes: Option<u64>,
) -> OverlayAdmissionBudget {
    let mut budget = OverlayAdmissionBudget::unbounded();
    if let Some(max_hot_rows) = max_hot_rows {
        budget = OverlayAdmissionBudget::with_hot_row_cap(max_hot_rows);
    }
    if let Some(max_overlay_bytes) = max_overlay_bytes {
        budget = budget.with_overlay_byte_cap(max_overlay_bytes);
    }
    budget
}

fn verify(args: VerifyArgs) -> Result<(), Box<dyn std::error::Error>> {
    let source = RawFileRowSource::open(&args.source, args.row_bytes)?;
    let reader = PlePackReader::open(&args.pack, source)?;
    let hot_rows_verified = reader.verify_hot_overlay()?;
    let header = reader.header();

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "hot_rows_verified": hot_rows_verified,
            "row_count": header.row_count,
            "row_bytes": header.row_bytes,
            "block_bytes": header.block_bytes,
            "source_digest": header.source_digest,
            "index_digest": header.index_digest,
        }))?
    );
    Ok(())
}

fn source_verify(args: SourceVerifyArgs) -> Result<(), Box<dyn std::error::Error>> {
    let index_path = args.model_dir.join("model.safetensors.index.json");
    if !index_path.is_file() {
        return Err(format!(
            "pinned Qwen source is missing model.safetensors.index.json at {}",
            index_path.display()
        )
        .into());
    }

    let manifest = qwen38_ple_manifest_from_index(&args.model_dir, QWEN38_FLASH_NEXT_REVISION)?;
    let source = SafetensorsPleSource::open(&args.model_dir, &manifest)?;
    let digest = digest_hex(source.source_digest());

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "state": "verified-local-bytes",
            "model_id": manifest.model_id,
            "revision_contract": manifest.model_revision,
            "parts": manifest.parts.len(),
            "row_count": source.row_count(),
            "row_bytes": source.row_bytes(),
            "source_digest_sha256": digest,
            "digest_scope": "GB10X-SAFETENSORS-PLE-V1 manifest+referenced-PLE-bytes",
            "remote_digest_match": null,
        }))?
    );
    Ok(())
}

fn hash_verify(args: HashVerifyArgs) -> Result<(), Box<dyn std::error::Error>> {
    let verification = verify_qwen38_ple_hash_buffers(&args.model_dir, &args.observed_revision)?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "state": "verified-local-hash-buffers",
            "model_id": QWEN38_FLASH_NEXT_MODEL_ID,
            "revision_contract": QWEN38_FLASH_NEXT_REVISION,
            "observed_revision": args.observed_revision,
            "buffers_verified": verification.buffers_verified,
            "files_read": verification.files_read,
            "index_bytes_read": verification.index_bytes_read,
            "header_bytes_read": verification.header_bytes_read,
            "payload_bytes_read": verification.payload_bytes_read,
            "payload_bytes_expected": QWEN38_PLE_HASH_BUFFER_PAYLOAD_BYTES,
            "validation_scope": "local physical PLE hash buffers only; no full checkpoint, remote provenance, inference, GPU, or logits claim",
        }))?
    );
    Ok(())
}

fn digest_hex(digest: [u8; 32]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}
