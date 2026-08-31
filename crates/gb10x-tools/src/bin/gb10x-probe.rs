use clap::Parser;
use gb10x_runtime::linux_probe::probe_host;
use gb10x_tools::probe::render_host_probe_json;

#[derive(Debug, Parser)]
#[command(name = "gb10x-probe", about = "Probe GB10X Linux host facts without assumptions")]
struct Cli {
    /// Emit the complete host probe as JSON.
    #[arg(long, required = true)]
    json: bool,
}

fn main() {
    let cli = Cli::parse();
    if let Err(error) = run(cli) {
        eprintln!("gb10x-probe: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    debug_assert!(cli.json);
    let probe = probe_host()?;
    println!("{}", render_host_probe_json(&probe)?);
    Ok(())
}
