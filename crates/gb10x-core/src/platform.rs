#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gb10_fixture_exposes_nonzero_cache_and_memory_facts() {
        let fixture = PlatformSnapshot::gb10_test_fixture();
        assert_eq!(fixture.arch, "aarch64");
        assert!(!fixture.caches.is_empty());
        assert!(fixture.mem_total_bytes > 0);
        assert!(fixture.gpu.l2_bytes > 0);
    }
}
