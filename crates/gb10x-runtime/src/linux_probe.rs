//! Linux sysfs CPU/cache topology parsing for the GB10 Cache Fabric.

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/cache-topology")
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
