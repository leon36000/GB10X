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
fn plepack_build_accepts_a_basename_output_in_its_current_directory() {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("ple.raw"), [5_u8; 64]).unwrap();
    fs::write(dir.path().join("trace.json"), r#"[[1,2],[2,3]]"#).unwrap();

    let build = plepack()
        .current_dir(dir.path())
        .args([
            "build",
            "--source",
            "ple.raw",
            "--trace",
            "trace.json",
            "--out",
            "hot.plepack",
            "--row-bytes",
            "8",
            "--block-bytes",
            "64",
        ])
        .output()
        .expect("PLEPack basename-relative build");

    assert!(
        build.status.success(),
        "build stderr: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(dir.path().join("hot.plepack").is_file());
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
fn plepack_build_honors_hot_row_and_overlay_byte_caps() {
    let dir = tempdir().unwrap();
    let source = dir.path().join("ple.raw");
    let trace = dir.path().join("trace.json");
    let pack = dir.path().join("bounded.plepack");

    fs::write(&source, [7_u8; 320]).unwrap();
    fs::write(&trace, r#"[[9,3,7,3],[9,7,11],[2,1],[7,9]]"#).unwrap();

    let build = plepack()
        .args(["build", "--source"])
        .arg(&source)
        .args(["--trace"])
        .arg(&trace)
        .args(["--out"])
        .arg(&pack)
        .args([
            "--row-bytes",
            "8",
            "--block-bytes",
            "64",
            "--max-hot-rows",
            "2",
            "--max-overlay-bytes",
            "64",
        ])
        .output()
        .expect("bounded PLEPack build");

    assert!(
        build.status.success(),
        "build stderr: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let build_json: Value = serde_json::from_slice(&build.stdout).expect("build JSON");
    assert_eq!(build_json["hot_rows"], 2);
    assert_eq!(build_json["overlay_bytes"], 64);
}

#[test]
fn plepack_plan_honors_a_byte_cap_without_a_row_cap() {
    let dir = tempdir().unwrap();
    let trace = dir.path().join("trace.json");
    let plan_path = dir.path().join("bounded-plan.json");
    fs::write(&trace, r#"[[6,1,8,3,5,2,4]]"#).unwrap();

    let plan = plepack()
        .args(["plan", "--trace"])
        .arg(&trace)
        .args(["--out"])
        .arg(&plan_path)
        .args([
            "--row-count",
            "40",
            "--row-bytes",
            "16",
            "--block-bytes",
            "64",
            "--max-overlay-bytes",
            "64",
        ])
        .output()
        .expect("bounded PLEPack plan");

    assert!(
        plan.status.success(),
        "plan stderr: {}",
        String::from_utf8_lossy(&plan.stderr)
    );
    let value: Value = serde_json::from_slice(&fs::read(&plan_path).unwrap()).expect("plan JSON");
    assert_eq!(value["hot_physical_order"], serde_json::json!([1, 2, 3, 4]));
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

#[test]
fn plepack_cli_lists_hash_verify_and_requires_an_observed_revision() {
    let help = plepack().arg("--help").output().expect("PLEPack help");
    assert!(help.status.success());
    assert!(stdout(&help).contains("hash-verify"));

    let dir = tempdir().unwrap();
    let missing_revision = plepack()
        .args(["hash-verify", "--model-dir"])
        .arg(dir.path())
        .output()
        .expect("hash-verify invocation");
    assert!(!missing_revision.status.success());
    assert!(String::from_utf8_lossy(&missing_revision.stderr).contains("--observed-revision"));
}
