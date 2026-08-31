//! Linux sysfs CPU/cache topology parsing for the GB10 Cache Fabric.

use gb10x_core::{CacheType, CpuCache};
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

fn parse_cpu_list(value: &str) -> Result<Vec<u32>, ProbeError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ProbeError::InvalidValue {
            field: "shared_cpu_list",
            value: value.to_owned(),
        });
    }

    let mut cpus = Vec::new();
    for segment in value.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            return Err(ProbeError::InvalidValue {
                field: "shared_cpu_list",
                value: value.to_owned(),
            });
        }

        if let Some((start, end)) = segment.split_once('-') {
            let start = parse_u32("shared_cpu_list", start)?;
            let end = parse_u32("shared_cpu_list", end)?;
            if start > end {
                return Err(ProbeError::InvalidValue {
                    field: "shared_cpu_list",
                    value: value.to_owned(),
                });
            }
            cpus.extend(start..=end);
        } else {
            cpus.push(parse_u32("shared_cpu_list", segment)?);
        }
    }

    cpus.sort_unstable();
    cpus.dedup();
    Ok(cpus)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/cache-topology")
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
}
