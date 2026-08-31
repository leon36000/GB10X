use gb10x_runtime::linux_probe::probe_host;
use gb10x_tools::probe::render_host_probe_json;

fn main() {
    if let Err(error) = run() {
        eprintln!("gb10x-probe: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let probe = probe_host()?;
    println!("{}", render_host_probe_json(&probe)?);
    Ok(())
}
