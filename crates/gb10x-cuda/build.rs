use std::env;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{Command, Output};

fn main() {
    println!("cargo:rerun-if-changed=native/gb10x_cuda.h");
    println!("cargo:rerun-if-changed=native/probe.cu");
    println!("cargo:rerun-if-changed=native/smoke.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=AR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if env::var_os("CARGO_FEATURE_NATIVE_CUDA").is_none() {
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("Cargo must provide target OS");
    if target_os != "linux" {
        panic!("GB10X native-cuda currently supports Linux only, found {target_os}");
    }

    let nvcc = env::var_os("NVCC").unwrap_or_else(|| OsString::from("nvcc"));
    let version_output = Command::new(&nvcc)
        .arg("--version")
        .output()
        .unwrap_or_else(|error| panic!("failed to execute nvcc for GB10X native-cuda: {error}"));
    require_success(&version_output, "nvcc --version");
    validate_nvcc_version(&version_output);

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    let archive = out_dir.join("libgb10x_cuda_native.a");
    let mut objects = Vec::new();

    for (source, object_name) in [("native/probe.cu", "probe.o"), ("native/smoke.cu", "smoke.o")] {
        let object = out_dir.join(object_name);
        let mut compile = Command::new(&nvcc);
        compile
            .arg("-std=c++17")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg("--generate-code=arch=compute_121a,code=sm_121a")
            .arg("-I")
            .arg("native")
            .arg("-c")
            .arg(source)
            .arg("-o")
            .arg(&object);
        run_checked(
            &mut compile,
            &format!("nvcc failed to compile GB10X sm_121a source {source}"),
        );
        objects.push(object);
    }

    let ar = env::var_os("AR").unwrap_or_else(|| OsString::from("ar"));
    let mut archive_command = Command::new(&ar);
    archive_command.arg("crs").arg(&archive);
    for object in &objects {
        archive_command.arg(object);
    }
    run_checked(
        &mut archive_command,
        "failed to archive the GB10X native CUDA objects",
    );

    let cuda_home = env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let cuda_lib64 = cuda_home.join("lib64");
    if !cuda_lib64.is_dir() {
        panic!(
            "GB10X native-cuda requires CUDA lib64 at {}, set CUDA_HOME/CUDA_PATH if needed",
            cuda_lib64.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=gb10x_cuda_native");
    println!("cargo:rustc-link-search=native={}", cuda_lib64.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn validate_nvcc_version(output: &Output) {
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    let Some((major, minor)) = parse_nvcc_release(&text) else {
        panic!("could not parse nvcc release from output:\n{text}");
    };
    if (major, minor) < (12, 9) {
        panic!("GB10X sm_121a requires CUDA 12.9 or newer, found {major}.{minor}");
    }
}

fn parse_nvcc_release(text: &str) -> Option<(u32, u32)> {
    text.lines().find_map(|line| {
        let release = line.split_once("release ")?.1.split(',').next()?.trim();
        let mut parts = release.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        Some((major, minor))
    })
}

fn require_success(output: &Output, label: &str) {
    if !output.status.success() {
        panic!(
            "{label} failed with {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn run_checked(command: &mut Command, label: &str) {
    let rendered = format!("{command:?}");
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("{label}: could not execute {rendered}: {error}"));
    if !output.status.success() {
        panic!(
            "{label}: {rendered}\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
