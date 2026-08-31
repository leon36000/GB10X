#[cfg(test)]
mod tests {
    use super::*;
    use gb10x_core::PlatformSnapshot;

    #[test]
    fn rejects_x86_even_if_gpu_claims_sm121() {
        let mut p = PlatformSnapshot::gb10_test_fixture();
        p.arch = "x86_64".into();
        assert!(matches!(
            validate_gb10(&p),
            Err(PlatformError::Architecture(_))
        ));
    }

    #[test]
    fn rejects_non_121_compute_capability() {
        let mut p = PlatformSnapshot::gb10_test_fixture();
        p.gpu.compute_major = 12;
        p.gpu.compute_minor = 0;
        assert!(matches!(
            validate_gb10(&p),
            Err(PlatformError::ComputeCapability { .. })
        ));
    }

    #[test]
    fn accepts_exact_gb10_fixture() {
        let p = PlatformSnapshot::gb10_test_fixture();
        let result = validate_gb10(&p).expect("GB10 fixture must validate");
        assert_eq!(result.compute_capability, (12, 1));
        assert!(result.discovered_l2_bytes > 0);
    }
}
