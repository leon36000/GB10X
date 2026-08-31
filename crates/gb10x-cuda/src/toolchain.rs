#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gb10_contract_is_architecture_specific() {
        let contract = CudaBuildContract::gb10();
        assert_eq!(contract.virtual_arch(), "compute_121a");
        assert_eq!(contract.real_arch(), "sm_121a");
        assert!(contract.is_arch_specific());
    }

    #[test]
    fn nvcc_release_line_is_parsed_exactly() {
        let version = parse_nvcc_version("Cuda compilation tools, release 13.2, V13.2.51")
            .expect("valid nvcc release line");
        assert_eq!(
            (version.major, version.minor, version.patch),
            (13, 2, Some(51))
        );
    }

    #[test]
    fn cuda_12_8_is_rejected_for_sm121a_m2() {
        let version = NvccVersion {
            major: 12,
            minor: 8,
            patch: None,
        };
        assert!(validate_nvcc_for_gb10(&version).is_err());
    }

    #[test]
    fn cuda_12_9_is_accepted_for_sm121a_m2() {
        let version = NvccVersion {
            major: 12,
            minor: 9,
            patch: None,
        };
        validate_nvcc_for_gb10(&version).expect("CUDA 12.9 supports sm_121a");
    }
}
