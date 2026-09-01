#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'GB10X build environment: %s\n' "$*" >&2
  exit 64
}

[[ $# -eq 1 ]] || fail 'usage: check-gb10-build-env.sh REPO_ROOT'
[[ -d "$1" ]] || fail "repository root does not exist: $1"
repo_root="$(cd -- "$1" && pwd -P)"
cd "$repo_root"

git rev-parse --show-toplevel >/dev/null 2>&1 || fail 'repository root is not inside a Git worktree'
actual_root="$(git rev-parse --show-toplevel)"
actual_root="$(cd -- "$actual_root" && pwd -P)"
[[ "$actual_root" == "$repo_root" ]] || fail "repository root mismatch: expected ${repo_root}, Git reports ${actual_root}"

reject_env_var() {
  local name="$1"
  if [[ -n "${!name-}" ]]; then
    fail "ambient build override ${name} is set"
  fi
}

for name in \
  RUSTFLAGS \
  CARGO_ENCODED_RUSTFLAGS \
  RUSTC \
  RUSTC_WRAPPER \
  RUSTC_WORKSPACE_WRAPPER \
  RUSTUP_TOOLCHAIN \
  CARGO_BUILD_TARGET \
  CARGO_BUILD_RUSTFLAGS \
  CARGO_TARGET_DIR \
  NVCC \
  AR \
  NVCC_PREPEND_FLAGS \
  NVCC_APPEND_FLAGS \
  NVCC_CCBIN; do
  reject_env_var "$name"
done

while IFS='=' read -r name _; do
  case "$name" in
    CARGO_TARGET_*_RUSTFLAGS|CARGO_PROFILE_*)
      fail "ambient Cargo codegen override ${name} is set"
      ;;
  esac
done < <(env)

for relative in .cargo/config .cargo/config.toml; do
  candidate="${repo_root}/${relative}"
  if [[ -e "$candidate" || -L "$candidate" ]]; then
    [[ ! -L "$candidate" ]] || fail "tracked Cargo config must not be a symlink: ${relative}"
    git ls-files --error-unmatch -- "$relative" >/dev/null 2>&1 || \
      fail "untracked Cargo config can alter the build: ${relative}"
  fi
done

parent="$(dirname -- "$repo_root")"
while true; do
  for suffix in .cargo/config .cargo/config.toml; do
    candidate="${parent}/${suffix}"
    if [[ -e "$candidate" || -L "$candidate" ]]; then
      fail "inherited Cargo config can alter the build: ${candidate}"
    fi
  done
  [[ "$parent" == / ]] && break
  parent="$(dirname -- "$parent")"
done

cargo_home="${CARGO_HOME:-${HOME:-}/.cargo}"
if [[ -n "$cargo_home" ]]; then
  if [[ "$cargo_home" != /* ]]; then
    cargo_home="${repo_root}/${cargo_home}"
  fi
  for suffix in config config.toml; do
    candidate="${cargo_home}/${suffix}"
    if [[ -e "$candidate" || -L "$candidate" ]]; then
      fail "CARGO_HOME config can alter the build: ${candidate}"
    fi
  done
fi

printf 'GB10X build environment: PASS\n'
