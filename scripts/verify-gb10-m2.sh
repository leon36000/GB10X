#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'GB10X M2 verification: %s\n' "$*" >&2
  exit 64
}

usage() {
  cat <<'EOF'
Usage: verify-gb10-m2.sh --model-dir PATH

Run the fail-closed M2 correctness/evidence gate on the real DGX Spark.
The Qwen model directory may alternatively be supplied through GB10X_QWEN_MODEL_DIR.
EOF
}

kernel="$(uname -s)"
arch="$(uname -m)"
if [[ "$kernel" != "Linux" || "$arch" != "aarch64" ]]; then
  fail "requires Linux aarch64 on the real DGX Spark; found ${kernel} ${arch}"
fi

model_dir="${GB10X_QWEN_MODEL_DIR:-}"
while (( $# > 0 )); do
  case "$1" in
    --model-dir)
      shift
      (( $# > 0 )) || fail "--model-dir requires a path"
      model_dir="$1"
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
  shift
done

[[ -n "$model_dir" ]] || fail \
  "requires --model-dir PATH or GB10X_QWEN_MODEL_DIR for the pinned Qwen checkpoint"
[[ -d "$model_dir" ]] || fail "Qwen model directory does not exist: ${model_dir}"
model_dir="$(cd -- "$model_dir" && pwd -P)" || fail "cannot resolve Qwen model directory"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "$repo_root"

for command_name in git cargo rustc nvcc nvidia-smi cuobjdump python3 tee find mktemp rm; do
  command -v "$command_name" >/dev/null 2>&1 || fail "required command is missing: ${command_name}"
done

# Reject ambient Cargo/Rust/CUDA build overrides before GB10X creates its own isolated target tree.
build_env_guard_output="$(bash scripts/check-gb10-build-env.sh "$repo_root")"

# Evidence must bind to an exact tracked source state. Generated/untracked evidence directories are
# intentionally allowed, but tracked source/index modifications are not.
git diff --quiet || fail "tracked worktree changes are present"
git diff --cached --quiet || fail "staged source changes are present"
git ls-files --error-unmatch Cargo.lock >/dev/null 2>&1 || fail "tracked Cargo.lock is missing"
cargo metadata --locked --format-version 1 >/dev/null || fail "Cargo.lock does not match the workspace"

git_sha="$(git rev-parse HEAD)"
short_sha="$(git rev-parse --short=12 HEAD)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
evidence_dir="${GB10X_M2_EVIDENCE_DIR:-${repo_root}/docs/evidence/native-runs/${timestamp}-${short_sha}}"
mkdir -p "$evidence_dir"

# Build in a fresh target tree so artifact inspection cannot accidentally select an object produced
# by an older commit or an earlier local experiment. The target tree is disposable; raw evidence is
# written separately under evidence_dir and survives cleanup.
native_target_dir="$(mktemp -d "${TMPDIR:-/tmp}/gb10x-m2-target.XXXXXX")"
cleanup_native_target() {
  rm -rf -- "$native_target_dir"
}
trap cleanup_native_target EXIT
export CARGO_TARGET_DIR="$native_target_dir"

exec > >(tee -a "${evidence_dir}/verification.log") 2>&1

printf 'GB10X M2 native verification\n'
printf 'git_sha=%s\n' "$git_sha"
printf 'evidence_dir=%s\n' "$evidence_dir"
printf 'isolated_cargo_target=%s\n' "$CARGO_TARGET_DIR"
printf 'qwen_model_dir=%s\n' "$model_dir"
printf '%s\n' "$build_env_guard_output" | tee "${evidence_dir}/build-env.txt"

uname -a | tee "${evidence_dir}/uname.txt"
rustc --version | tee "${evidence_dir}/rustc.txt"
cargo --version | tee "${evidence_dir}/cargo.txt"

nvcc_text="$(nvcc --version)"
printf '%s\n' "$nvcc_text" | tee "${evidence_dir}/nvcc.txt"
if [[ "$nvcc_text" =~ release[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
  nvcc_major="${BASH_REMATCH[1]}"
  nvcc_minor="${BASH_REMATCH[2]}"
else
  fail "could not parse nvcc release"
fi
if (( nvcc_major < 12 || (nvcc_major == 12 && nvcc_minor < 9) )); then
  fail "CUDA 12.9 or newer is required for sm_121a; found ${nvcc_major}.${nvcc_minor}"
fi

nvidia-smi | tee "${evidence_dir}/nvidia-smi.txt"

model_source_json="$(cargo run --locked -p gb10x-tools --bin gb10x-plepack -- source-verify --model-dir "$model_dir")"
printf '%s\n' "$model_source_json" | tee "${evidence_dir}/model-source.json"

python3 - "${evidence_dir}/model-source.json" <<'PY'
import json
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("state") != "verified-local-bytes":
    raise SystemExit(f"Qwen local-byte verification did not pass: {data!r}")
if data.get("model_id") != "Qwen/Qwen3.8-Flash-Next":
    raise SystemExit(f"unexpected Qwen model id: {data.get('model_id')!r}")
if data.get("revision_contract") != "34567a4712bc9766c4449e2e98e4468bfa24d915":
    raise SystemExit(f"unexpected Qwen revision contract: {data.get('revision_contract')!r}")
if data.get("parts") != 128:
    raise SystemExit(f"unexpected PLE part count: {data.get('parts')!r}")
if data.get("row_count") != 320_001_536:
    raise SystemExit(f"unexpected PLE row count: {data.get('row_count')!r}")
if data.get("row_bytes") != 320:
    raise SystemExit(f"unexpected PLE row width: {data.get('row_bytes')!r}")
digest = data.get("source_digest_sha256")
if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
    raise SystemExit(f"invalid local PLE source digest: {digest!r}")
if data.get("remote_digest_match") is not None:
    raise SystemExit("source verifier must not fabricate a remote digest match")
PY

probe_json="$(cargo run --locked -p gb10x-tools --features native-cuda --bin gb10x-probe -- --json)"
printf '%s\n' "$probe_json" | tee "${evidence_dir}/probe.json"

python3 - "${evidence_dir}/probe.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("arch") != "aarch64":
    raise SystemExit(f"probe arch is not aarch64: {data.get('arch')!r}")

cuda = data.get("cuda_native")
if not isinstance(cuda, dict) or cuda.get("state") != "verified":
    raise SystemExit(f"native CUDA evidence is not verified: {cuda!r}")

device = cuda.get("device")
if not isinstance(device, dict):
    raise SystemExit("native CUDA device facts are missing")
if (device.get("compute_major"), device.get("compute_minor")) != (12, 1):
    raise SystemExit(
        "expected GB10 compute capability 12.1, found "
        f"{device.get('compute_major')}.{device.get('compute_minor')}"
    )
if "GB10" not in str(device.get("name", "")).upper():
    raise SystemExit(f"CUDA device identity is not GB10: {device.get('name')!r}")
for field in ("total_memory_bytes", "l2_bytes", "sm_count", "warp_size"):
    value = device.get(field)
    if not isinstance(value, int) or value <= 0:
        raise SystemExit(f"native CUDA device field {field} is invalid: {value!r}")
persisting = device.get("persisting_l2_max_bytes")
if not isinstance(persisting, int) or persisting < 0 or persisting > device["l2_bytes"]:
    raise SystemExit(f"persisting-L2 value is invalid: {persisting!r}")

validation = cuda.get("validation")
if not isinstance(validation, dict) or validation.get("state") != "passed":
    raise SystemExit(f"GB10 validation did not pass: {validation!r}")
PY

cargo test --locked -p gb10x-cuda --features native-cuda --test native_probe -- --nocapture \
  2>&1 | tee "${evidence_dir}/native-probe-test.txt"
cargo test --locked -p gb10x-cuda --features native-cuda --test native_smoke -- --nocapture \
  2>&1 | tee "${evidence_dir}/native-smoke-test.txt"
cargo test --locked -p gb10x-cuda --features native-cuda --test rmsnorm -- --nocapture \
  2>&1 | tee "${evidence_dir}/rmsnorm-test.txt"

artifact_log="${evidence_dir}/cuobjdump.txt"
: > "$artifact_log"
for object_name in probe.o smoke.o rmsnorm.o; do
  mapfile -t candidates < <(
    find "$CARGO_TARGET_DIR/debug/build" -type f -path "*/gb10x-cuda-*/out/${object_name}" -print
  )
  (( ${#candidates[@]} == 1 )) || fail \
    "expected exactly one isolated CUDA object ${object_name}, found ${#candidates[@]}"
  object_path="${candidates[0]}"
  printf '\n## %s\n' "$object_path" | tee -a "$artifact_log"
  cuobjdump --list-elf "$object_path" 2>&1 | tee -a "$artifact_log"
done

python3 - "${evidence_dir}/probe.json" "${evidence_dir}/model-source.json" \
  "${evidence_dir}/summary.json" "$git_sha" "$timestamp" "${nvcc_major}.${nvcc_minor}" <<'PY'
import json
import pathlib
import sys

probe_path = pathlib.Path(sys.argv[1])
model_source_path = pathlib.Path(sys.argv[2])
summary_path = pathlib.Path(sys.argv[3])
summary = {
    "schema_version": 1,
    "result": "pass",
    "proof_class": "gb10-device-execution",
    "git_sha": sys.argv[4],
    "timestamp_utc": sys.argv[5],
    "nvcc_release": sys.argv[6],
    "model_source": json.loads(model_source_path.read_text(encoding="utf-8")),
    "probe": json.loads(probe_path.read_text(encoding="utf-8")),
    "native_tests": {
        "device_probe": "pass",
        "smoke_checksum": "pass",
        "bf16_rmsnorm_correctness": "pass",
    },
    "artifact_inspection": "pass",
    "performance_claim": None,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

cat > "${evidence_dir}/summary.md" <<EOF
# GB10X M2 device-execution evidence

- Result: **PASS**
- Proof class: **GB10 device execution**
- Git SHA: \`${git_sha}\`
- UTC timestamp: \`${timestamp}\`
- Host: Linux \`aarch64\`
- CUDA compiler: \`${nvcc_major}.${nvcc_minor}\` or newer patch level as recorded in \`nvcc.txt\`
- Pinned Qwen local PLE bytes: PASS (observed local digest recorded; remote digest match is not claimed)
- Native probe / GB10 validation: PASS
- CUDA smoke checksum: PASS
- BF16 RMSNorm correctness gate: PASS
- CUDA artifact inspection: PASS
- Performance claim: **none**

Raw command output is retained in this evidence directory.
EOF

printf 'GB10X M2 native verification PASS: %s\n' "$evidence_dir"
