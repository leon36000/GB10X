#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'GB10X M2 verification: %s\n' "$*" >&2
  exit 64
}

kernel="$(uname -s)"
arch="$(uname -m)"
if [[ "$kernel" != "Linux" || "$arch" != "aarch64" ]]; then
  fail "requires Linux aarch64 on the real DGX Spark; found ${kernel} ${arch}"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "$repo_root"

for command_name in git cargo rustc nvcc nvidia-smi cuobjdump python3 tee find mktemp rm; do
  command -v "$command_name" >/dev/null 2>&1 || fail "required command is missing: ${command_name}"
done

# Evidence must bind to an exact tracked source state. Generated/untracked evidence directories are
# intentionally allowed, but tracked source/index modifications are not.
git diff --quiet || fail "tracked worktree changes are present"
git diff --cached --quiet || fail "staged source changes are present"

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

probe_json="$(cargo run -p gb10x-tools --features native-cuda --bin gb10x-probe -- --json)"
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

cargo test -p gb10x-cuda --features native-cuda --test native_probe -- --nocapture \
  2>&1 | tee "${evidence_dir}/native-probe-test.txt"
cargo test -p gb10x-cuda --features native-cuda --test native_smoke -- --nocapture \
  2>&1 | tee "${evidence_dir}/native-smoke-test.txt"
cargo test -p gb10x-cuda --features native-cuda --test rmsnorm -- --nocapture \
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

python3 - "${evidence_dir}/probe.json" "${evidence_dir}/summary.json" \
  "$git_sha" "$timestamp" "${nvcc_major}.${nvcc_minor}" <<'PY'
import json
import pathlib
import sys

probe_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
summary = {
    "schema_version": 1,
    "result": "pass",
    "proof_class": "gb10-device-execution",
    "git_sha": sys.argv[3],
    "timestamp_utc": sys.argv[4],
    "nvcc_release": sys.argv[5],
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
- Native probe / GB10 validation: PASS
- CUDA smoke checksum: PASS
- BF16 RMSNorm correctness gate: PASS
- CUDA artifact inspection: PASS
- Performance claim: **none**

Raw command output is retained in this evidence directory.
EOF

printf 'GB10X M2 native verification PASS: %s\n' "$evidence_dir"
