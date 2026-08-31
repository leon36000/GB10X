//! Linux sysfs CPU/cache topology parsing for the GB10 Cache Fabric.

use gb10x_core::{CacheType, CpuCache, GpuSnapshot, PlatformSnapshot};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Failure while reading or validating Linux hardware topology.
#[derive(Debug, Error)]
pub enum ProbeError {
    /// A required sysfs or procfs entry could not be read.
    #[error("failed to read {path}: {source}")]
    Io {
        /// Path that could not be read.
        path: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// A Linux topology value was syntactically or semantically invalid.
    #[error("invalid topology value for {field}: {value}")]
    InvalidValue {
        /// Logical field being parsed.
        field: &'static str,
        /// Invalid textual value.
        value: String,
    },
    /// No usable CPU cache objects were discovered.
    #[error("no CPU cache objects discovered under {0}")]
    NoCaches(String),
}

/// CPU/OS/memory facts collected independently of the CUDA device probe.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HostProbe {
    /// Rust/Linux architecture string.
    pub arch: String,
    /// Linux kernel release string.
    pub kernel_release: String,
    /// CPU model description reported by `/proc/cpuinfo`.
    pub cpu_model: String,
    /// Online Linux CPU IDs.
    pub online_cpus: Vec<u32>,
    /// De-duplicated physical CPU cache objects.
    pub caches: Vec<CpuCache>,
    /// Physical memory reported by `/proc/meminfo` in bytes.
    pub mem_total_bytes: u64,
    /// Base Linux page size in bytes.
    pub page_size_bytes: u64,
}

impl HostProbe {
    /// Combine already-probed host facts with an independently probed CUDA device snapshot.
    pub fn into_platform_snapshot(self, gpu: GpuSnapshot) -> PlatformSnapshot {
        PlatformSnapshot {
            arch: self.arch,
            kernel_release: self.kernel_release,
            cpu_model: self.cpu_model,
            online_cpus: self.online_cpus,
            caches: self.caches,
            gpu,
            mem_total_bytes: self.mem_total_bytes,
            page_size_bytes: self.page_size_bytes,
        }
    }
}

fn read_trimmed(path: &Path) -> Result<String, ProbeError> {
    fs::read_to_string(path)
        .map(|value| value.trim().to_owned())
        .map_err(|source| ProbeError::Io {
            path: path.display().to_string(),
            source,
        })
}

fn parse_u32(field: &'static str, value: &str) -> Result<u32, ProbeError> {
    value
        .trim()
        .parse::<u32>()
        .map_err(|_| ProbeError::InvalidValue {
            field,
            value: value.to_owned(),
        })
}

fn parse_cpu_list_for(field: &'static str, value: &str) -> Result<Vec<u32>, ProbeError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ProbeError::InvalidValue {
            field,
            value: value.to_owned(),
        });
    }

    let mut cpus = Vec::new();
    for segment in value.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            return Err(ProbeError::InvalidValue {
                field,
                value: value.to_owned(),
            });
        }

        if let Some((start, end)) = segment.split_once('-') {
            let start = parse_u32(field, start)?;
            let end = parse_u32(field, end)?;
            if start > end {
                return Err(ProbeError::InvalidValue {
                    field,
                    value: value.to_owned(),
                });
            }
            cpus.extend(start..=end);
        } else {
            cpus.push(parse_u32(field, segment)?);
        }
    }

    cpus.sort_unstable();
    cpus.dedup();
    Ok(cpus)
}

fn parse_cpu_list(value: &str) -> Result<Vec<u32>, ProbeError> {
    parse_cpu_list_for("shared_cpu_list", value)
}

fn parse_cache_size(value: &str) -> Result<u64, ProbeError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ProbeError::InvalidValue {
            field: "cache size",
            value: value.to_owned(),
        });
    }

    let (number, multiplier) = match value.as_bytes().last().copied() {
        Some(b'K') | Some(b'k') => (&value[..value.len() - 1], 1024_u64),
        Some(b'M') | Some(b'm') => (&value[..value.len() - 1], 1024_u64 * 1024),
        Some(b'G') | Some(b'g') => (&value[..value.len() - 1], 1024_u64 * 1024 * 1024),
        _ => (value, 1_u64),
    };

    let base = number
        .parse::<u64>()
        .map_err(|_| ProbeError::InvalidValue {
            field: "cache size",
            value: value.to_owned(),
        })?;
    base.checked_mul(multiplier)
        .ok_or_else(|| ProbeError::InvalidValue {
            field: "cache size",
            value: value.to_owned(),
        })
}

fn parse_cache_type(value: &str) -> Result<CacheType, ProbeError> {
    match value.trim() {
        "Data" => Ok(CacheType::Data),
        "Instruction" => Ok(CacheType::Instruction),
        "Unified" => Ok(CacheType::Unified),
        other => Err(ProbeError::InvalidValue {
            field: "cache type",
            value: other.to_owned(),
        }),
    }
}

fn cache_type_rank(cache_type: CacheType) -> u8 {
    match cache_type {
        CacheType::Instruction => 0,
        CacheType::Data => 1,
        CacheType::Unified => 2,
    }
}

fn is_numbered_dir(path: &Path, prefix: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix(prefix))
        .is_some_and(|suffix| {
            !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
        })
}

fn parse_cpu_model(cpuinfo: &str) -> Result<String, ProbeError> {
    for candidate in ["model name", "Model", "Processor", "Hardware"] {
        for line in cpuinfo.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            if key.trim() == candidate && !value.trim().is_empty() {
                return Ok(value.trim().to_owned());
            }
        }
    }

    Err(ProbeError::InvalidValue {
        field: "CPU model",
        value: "missing from /proc/cpuinfo".to_owned(),
    })
}

fn parse_mem_total_bytes(meminfo: &str) -> Result<u64, ProbeError> {
    for line in meminfo.lines() {
        let Some(rest) = line.strip_prefix("MemTotal:") else {
            continue;
        };
        let mut fields = rest.split_whitespace();
        let value = fields
            .next()
            .ok_or_else(|| ProbeError::InvalidValue {
                field: "MemTotal",
                value: line.to_owned(),
            })?
            .parse::<u64>()
            .map_err(|_| ProbeError::InvalidValue {
                field: "MemTotal",
                value: line.to_owned(),
            })?;
        if fields.next() != Some("kB") || fields.next().is_some() {
            return Err(ProbeError::InvalidValue {
                field: "MemTotal",
                value: line.to_owned(),
            });
        }
        return value
            .checked_mul(1024)
            .ok_or_else(|| ProbeError::InvalidValue {
                field: "MemTotal",
                value: line.to_owned(),
            });
    }

    Err(ProbeError::InvalidValue {
        field: "MemTotal",
        value: "missing from /proc/meminfo".to_owned(),
    })
}

/// Read Linux sysfs CPU cache entries and return each physical cache object exactly once.
///
/// `root` must point at a directory containing `cpuN/cache/indexM` entries, such as
/// `/sys/devices/system/cpu`. The same shared L3 is normally reported below every CPU that
/// participates in it; GB10X de-duplicates those copies using the complete cache descriptor.
pub fn read_cpu_cache_tree(root: &Path) -> Result<Vec<CpuCache>, ProbeError> {
    let cpu_entries = fs::read_dir(root).map_err(|source| ProbeError::Io {
        path: root.display().to_string(),
        source,
    })?;

    let mut caches = Vec::new();
    for cpu_entry in cpu_entries {
        let cpu_entry = cpu_entry.map_err(|source| ProbeError::Io {
            path: root.display().to_string(),
            source,
        })?;
        let cpu_path = cpu_entry.path();
        if !is_numbered_dir(&cpu_path, "cpu") {
            continue;
        }

        let cache_root = cpu_path.join("cache");
        if !cache_root.is_dir() {
            continue;
        }
        let index_entries = fs::read_dir(&cache_root).map_err(|source| ProbeError::Io {
            path: cache_root.display().to_string(),
            source,
        })?;

        for index_entry in index_entries {
            let index_entry = index_entry.map_err(|source| ProbeError::Io {
                path: cache_root.display().to_string(),
                source,
            })?;
            let index_path = index_entry.path();
            if !is_numbered_dir(&index_path, "index") {
                continue;
            }

            let level = parse_u32("cache level", &read_trimmed(&index_path.join("level"))?)?;
            let level = u8::try_from(level).map_err(|_| ProbeError::InvalidValue {
                field: "cache level",
                value: level.to_string(),
            })?;
            let cache_type = parse_cache_type(&read_trimmed(&index_path.join("type"))?)?;
            let size_bytes = parse_cache_size(&read_trimmed(&index_path.join("size"))?)?;
            let line_bytes = parse_u32(
                "coherency_line_size",
                &read_trimmed(&index_path.join("coherency_line_size"))?,
            )?;
            let shared_cpu_ids =
                parse_cpu_list(&read_trimmed(&index_path.join("shared_cpu_list"))?)?;

            let cache = CpuCache {
                level,
                cache_type,
                size_bytes,
                line_bytes,
                shared_cpu_ids,
            };
            if !caches.contains(&cache) {
                caches.push(cache);
            }
        }
    }

    if caches.is_empty() {
        return Err(ProbeError::NoCaches(root.display().to_string()));
    }

    caches.sort_by(|left, right| {
        left.level
            .cmp(&right.level)
            .then_with(|| cache_type_rank(left.cache_type).cmp(&cache_type_rank(right.cache_type)))
            .then_with(|| left.shared_cpu_ids.cmp(&right.shared_cpu_ids))
            .then_with(|| left.size_bytes.cmp(&right.size_bytes))
    });
    Ok(caches)
}

/// Build a host probe from explicit Linux paths and externally supplied kernel/page facts.
///
/// The indirection keeps all parsing deterministic in host-independent tests while production
/// code uses the same parser against the real `/proc` and `/sys` filesystems.
pub fn probe_host_from_paths(
    cpu_root: &Path,
    online_path: &Path,
    cpuinfo_path: &Path,
    meminfo_path: &Path,
    arch: &str,
    kernel_release: &str,
    page_size_bytes: u64,
) -> Result<HostProbe, ProbeError> {
    if arch.trim().is_empty() {
        return Err(ProbeError::InvalidValue {
            field: "architecture",
            value: arch.to_owned(),
        });
    }
    if kernel_release.trim().is_empty() {
        return Err(ProbeError::InvalidValue {
            field: "kernel release",
            value: kernel_release.to_owned(),
        });
    }
    if page_size_bytes == 0 {
        return Err(ProbeError::InvalidValue {
            field: "page size",
            value: page_size_bytes.to_string(),
        });
    }

    let online_cpus = parse_cpu_list_for("online CPUs", &read_trimmed(online_path)?)?;
    let cpu_model = parse_cpu_model(&read_trimmed(cpuinfo_path)?)?;
    let mem_total_bytes = parse_mem_total_bytes(&read_trimmed(meminfo_path)?)?;
    let caches = read_cpu_cache_tree(cpu_root)?;

    Ok(HostProbe {
        arch: arch.to_owned(),
        kernel_release: kernel_release.to_owned(),
        cpu_model,
        online_cpus,
        caches,
        mem_total_bytes,
        page_size_bytes,
    })
}

/// Probe the real Linux host facts needed before the independent CUDA GB10 probe runs.
pub fn probe_host() -> Result<HostProbe, ProbeError> {
    let raw_page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if raw_page_size <= 0 {
        return Err(ProbeError::InvalidValue {
            field: "page size",
            value: raw_page_size.to_string(),
        });
    }
    let kernel_release = read_trimmed(Path::new("/proc/sys/kernel/osrelease"))?;
    probe_host_from_paths(
        Path::new("/sys/devices/system/cpu"),
        Path::new("/sys/devices/system/cpu/online"),
        Path::new("/proc/cpuinfo"),
        Path::new("/proc/meminfo"),
        std::env::consts::ARCH,
        &kernel_release,
        raw_page_size as u64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/cache-topology")
    }

    fn host_fixture_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/host-probe")
    }

    #[test]
    fn parses_shared_cpu_list_ranges() {
        assert_eq!(
            parse_cpu_list("0-3,8,10-11").expect("valid CPU list"),
            vec![0, 1, 2, 3, 8, 10, 11]
        );
    }

    #[test]
    fn rejects_descending_cpu_ranges() {
        assert!(parse_cpu_list("4-2").is_err());
    }

    #[test]
    fn cache_fixture_preserves_private_and_shared_groups_without_duplicates() {
        let caches = read_cpu_cache_tree(&fixture_root()).expect("cache fixture");
        assert!(
            caches
                .iter()
                .any(|cache| cache.level == 2 && cache.shared_cpu_ids == vec![0])
        );
        assert!(
            caches
                .iter()
                .any(|cache| cache.level == 3 && cache.shared_cpu_ids == vec![0, 1, 2, 3])
        );
        assert_eq!(
            caches
                .iter()
                .filter(|cache| cache.level == 3 && cache.shared_cpu_ids == vec![0, 1, 2, 3])
                .count(),
            1,
            "the same physical L3 reported by multiple CPUs must be de-duplicated"
        );
    }

    #[test]
    fn parses_binary_and_decimal_cache_sizes() {
        assert_eq!(parse_cache_size("64K").unwrap(), 64 * 1024);
        assert_eq!(parse_cache_size("2M").unwrap(), 2 * 1024 * 1024);
        assert_eq!(parse_cache_size("4096").unwrap(), 4096);
    }

    #[test]
    fn host_probe_fixture_preserves_discovered_facts() {
        let root = host_fixture_root();
        let probe = probe_host_from_paths(
            &fixture_root(),
            &root.join("online"),
            &root.join("cpuinfo"),
            &root.join("meminfo"),
            "aarch64",
            "6.11.0-gb10x-test",
            4096,
        )
        .expect("host probe fixture");

        assert_eq!(probe.arch, "aarch64");
        assert_eq!(probe.kernel_release, "6.11.0-gb10x-test");
        assert_eq!(probe.cpu_model, "NVIDIA Grace Blackwell GB10");
        assert_eq!(probe.online_cpus, vec![0, 1, 2, 3]);
        assert_eq!(probe.mem_total_bytes, 131_072_000 * 1024);
        assert_eq!(probe.page_size_bytes, 4096);
        assert!(!probe.caches.is_empty());
    }

    #[test]
    fn meminfo_requires_memtotal() {
        assert!(parse_mem_total_bytes("MemFree: 42 kB\n").is_err());
    }
}
