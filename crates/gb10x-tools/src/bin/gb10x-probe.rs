use clap::Parser;
use gb10x_runtime::linux_probe::probe_host;

#[cfg(feature = "native-cuda")]
use gb10x_cuda::probe_device;
#[cfg(feature = "native-cuda")]
use gb10x_runtime::validate_gb10;
#[cfg(not(feature = "native-cuda"))]
use gb10x_tools::probe::render_host_probe_json;
#[cfg(feature = "native-cuda")]
use gb10x_tools::probe::{gpu_snapshot_from_cuda, render_native_probe_json};

#[derive(Debug, Parser)]
#[command(
    name = "gb10x-probe",
    version,
    about = "Probe exact host and optional native CUDA facts for GB10X"
)]
struct Cli {
    /// Emit the probe as JSON.
    #[arg(long)]
    json: bool,
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    if !cli.json {
        return Err("gb10x-probe currently requires --json".into());
    }

    let host = probe_host()?;

    #[cfg(feature = "native-cuda")]
    {
        let device = probe_device(0)?;
        let gpu = gpu_snapshot_from_cuda(&device);
        let snapshot = host.clone().into_platform_snapshot(gpu);
        let validation = validate_gb10(&snapshot)?;
        println!("{}", render_native_probe_json(&host, &device, &validation)?);
    }

    #[cfg(not(feature = "native-cuda"))]
    {
        println!("{}", render_host_probe_json(&host)?);
    }

    Ok(())
}

fn main() {
    if let Err(error) = run(Cli::parse()) {
        eprintln!("gb10x-probe: {error}");
        std::process::exit(1);
    }
}
