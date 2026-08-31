use gb10x_tools::plepack::plan_from_trace_json;
use std::fs;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("gb10x-plepack: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os();
    let _program = args.next();
    let command = args.next().and_then(|value| value.into_string().ok());
    if command.as_deref() != Some("plan") {
        return Err(usage().into());
    }

    let trace_path = PathBuf::from(args.next().ok_or_else(usage)?);
    let output_path = PathBuf::from(args.next().ok_or_else(usage)?);
    let row_count = parse_arg::<u64>(args.next(), "row_count")?;
    let row_bytes = parse_arg::<u32>(args.next(), "row_bytes")?;
    let block_bytes = match args.next() {
        Some(value) => value
            .into_string()
            .map_err(|_| "block_bytes must be valid UTF-8")?
            .parse::<u32>()?,
        None => 4096,
    };
    if args.next().is_some() {
        return Err(usage().into());
    }

    let trace = fs::read_to_string(trace_path)?;
    let plan = plan_from_trace_json(row_count, row_bytes, block_bytes, &trace)?;
    let bytes = serde_json::to_vec_pretty(&plan)?;
    fs::write(output_path, bytes)?;
    Ok(())
}

fn parse_arg<T>(value: Option<std::ffi::OsString>, name: &str) -> Result<T, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + 'static,
{
    let value = value.ok_or_else(usage)?;
    let value = value
        .into_string()
        .map_err(|_| format!("{name} must be valid UTF-8"))?;
    Ok(value.parse::<T>()?)
}

fn usage() -> String {
    "usage: gb10x-plepack plan <trace.json> <plan.json> <row_count> <row_bytes> [block_bytes]"
        .to_owned()
}
