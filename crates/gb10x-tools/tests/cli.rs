use serde_json::Value;
use std::fs;
use std::process::{Command, Output};
use tempfile::tempdir;

fn probe() -> Command {
    Command::new(env!("CARGO_BIN_EXE_gb10x-probe"))
}

fn plepack() -> Command {
    Command::new(env!("CARGO_BIN_EXE_gb10x-plepack"))
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("UTF-8 stdout")
}

#[test]
fn probe_cli_has_help_json_and_rejects_unknown_flags() {
    let help = probe().arg("--help").output().expect("probe help");
    assert!(help.status.success());
    assert!(stdout(&help).contains("Usage:"));

    let json = probe().arg("--json").output().expect("probe JSON");
    assert!(json.status.success());
    let value: Value = serde_json::from_slice(&json.stdout).expect("valid probe JSON");
    assert!(value.get("arch").and_then(Value::as_str).is_some());
    assert!(value.get("caches").and_then(Value::as_array).is_some());
    assert_eq!(value["cuda_native"]["state"], "unavailable");
    assert_eq!(
        value["cuda_native"]["reason"],
        "binary built without native-cuda feature"
    );
    assert!(value["cuda_native"].get("device").is_none());

    let bad = probe().arg("--definitely-unknown").output().unwrap();
    assert!(!bad.status.success());
}

#[test]
fn plepack_cli_has_help() {
    let help = plepack().arg("--help").output().expect("PLEPack help");
    assert!(help.status.success());
    let text = stdout(&help);
    assert!(text.contains("Usage:"));
    assert!(text.contains("build"));
    assert!(text.contains("verify"));
    assert!(text.contains("source-verify"));
}

#[test]
fn plepack_source_verify_fails_closed_without_pinned_index() {
    let dir = tempdir().unwrap();
    let output = plepack()
        .args(["source-verify", "--model-dir"])
        .arg(dir.path())
        .output()
        .expect("source-verify invocation");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("model.safetensors.index.json"));
}

#[test]
fn plepack_build_and_verify_exact_hot_overlay() {
    let dir = tempdir().unwrap();
    let source = dir.path().join("ple.raw");
    let trace = dir.path().join("trace.json");
    let pack = dir.path().join("hot.plepack");

    let source_bytes = (0_u32..40)
        .flat_map(|row| (0_u32..8).map(move |column| ((row * 31 + column * 7) % 251) as u8))
        .collect::<Vec<_>>();
    fs::write(&source, source_bytes).unwrap();
    fs::write(&trace, r#"[[9,3,7,3],[9,7,11],[2,1],[7,9]]"#).unwrap();

    let build = plepack()
        .args(["build", "--source"])
        .arg(&source)
        .args(["--trace"])
        .arg(&trace)
        .args(["--out"])
        .arg(&pack)
        .args(["--row-bytes", "8", "--block-bytes", "64"])
        .output()
        .expect("PLEPack build");
    assert!(
        build.status.success(),
        "build stderr: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let build_json: Value = serde_json::from_slice(&build.stdout).expect("build JSON");
    assert_eq!(build_json["hot_rows"], 6);
    assert!(pack.exists());

    let verify = plepack()
        .args(["verify", "--source"])
        .arg(&source)
        .args(["--pack"])
        .arg(&pack)
        .args(["--row-bytes", "8"])
        .output()
        .expect("PLEPack verify");
    assert!(
        verify.status.success(),
        "verify stderr: {}",
        String::from_utf8_lossy(&verify.stderr)
    );
    let verify_json: Value = serde_json::from_slice(&verify.stdout).expect("verify JSON");
    assert_eq!(verify_json["hot_rows_verified"], 6);
}

#[test]
fn plepack_verify_rejects_changed_exact_source() {
    let dir = tempdir().unwrap();
    let source = dir.path().join("ple.raw");
    let trace = dir.path().join("trace.json");
    let pack = dir.path().join("hot.plepack");

    fs::write(&source, [5_u8; 64]).unwrap();
    fs::write(&trace, r#"[[1,2],[2,3]]"#).unwrap();
    let build = plepack()
        .args(["build", "--source"])
        .arg(&source)
        .args(["--trace"])
        .arg(&trace)
        .args(["--out"])
        .arg(&pack)
        .args(["--row-bytes", "8", "--block-bytes", "64"])
        .output()
        .unwrap();
    assert!(build.status.success());

    let mut changed = fs::read(&source).unwrap();
    changed[0] ^= 1;
    fs::write(&source, changed).unwrap();

    let verify = plepack()
        .args(["verify", "--source"])
        .arg(&source)
        .args(["--pack"])
        .arg(&pack)
        .args(["--row-bytes", "8"])
        .output()
        .unwrap();
    assert!(!verify.status.success());
}
