use thiserror::Error;

const MIN_CUDA_MAJOR: u32 = 12;
const MIN_CUDA_MINOR: u32 = 9;

/// CUDA architecture target supported by GB10X.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaArchTarget {
    /// NVIDIA GB10 architecture-specific target.
    Gb10Sm121a,
}

/// Immutable CUDA code-generation contract for GB10X.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaBuildContract {
    target: CudaArchTarget,
}

impl CudaBuildContract {
    /// Return the only CUDA build contract supported by GB10X.
    pub const fn gb10() -> Self {
        Self {
            target: CudaArchTarget::Gb10Sm121a,
        }
    }

    /// Return the virtual PTX architecture required by the build contract.
    pub const fn virtual_arch(&self) -> &'static str {
        match self.target {
            CudaArchTarget::Gb10Sm121a => "compute_121a",
        }
    }

    /// Return the real cubin architecture required by the build contract.
    pub const fn real_arch(&self) -> &'static str {
        match self.target {
            CudaArchTarget::Gb10Sm121a => "sm_121a",
        }
    }

    /// Whether the contract uses an architecture-specific `a` target.
    pub const fn is_arch_specific(&self) -> bool {
        matches!(self.target, CudaArchTarget::Gb10Sm121a)
    }
}

/// Parsed `nvcc` toolkit version.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NvccVersion {
    /// CUDA Toolkit major release.
    pub major: u32,
    /// CUDA Toolkit minor release.
    pub minor: u32,
    /// Optional build/patch component from the `Vx.y.z` token.
    pub patch: Option<u32>,
}

/// Failure while parsing or validating the CUDA toolchain used by GB10X.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CudaToolchainError {
    /// The `nvcc --version` output did not contain a release token.
    #[error("nvcc output is missing a CUDA release version")]
    MissingRelease,
    /// The release token was present but malformed.
    #[error("invalid nvcc CUDA release version: {0}")]
    InvalidRelease(String),
    /// The toolkit predates architecture-specific SM 12.1 compiler support required by M2.
    #[error(
        "GB10X requires CUDA Toolkit {required_major}.{required_minor}+ for sm_121a, found {found_major}.{found_minor}"
    )]
    UnsupportedVersion {
        /// Discovered toolkit major version.
        found_major: u32,
        /// Discovered toolkit minor version.
        found_minor: u32,
        /// Required toolkit major version.
        required_major: u32,
        /// Required toolkit minor version.
        required_minor: u32,
    },
}

/// Parse the canonical CUDA release/build tokens emitted by `nvcc --version`.
pub fn parse_nvcc_version(output: &str) -> Result<NvccVersion, CudaToolchainError> {
    let release = output
        .split("release ")
        .nth(1)
        .ok_or(CudaToolchainError::MissingRelease)?
        .split([',', ' ', '\n', '\r'])
        .next()
        .ok_or(CudaToolchainError::MissingRelease)?;

    let mut release_parts = release.split('.');
    let major = parse_component(release_parts.next(), release)?;
    let minor = parse_component(release_parts.next(), release)?;
    if release_parts.next().is_some() {
        return Err(CudaToolchainError::InvalidRelease(release.to_owned()));
    }

    let patch = output
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .find_map(|token| token.strip_prefix('V'))
        .and_then(|build| {
            let mut parts = build.split('.');
            let build_major = parts.next()?.parse::<u32>().ok()?;
            let build_minor = parts.next()?.parse::<u32>().ok()?;
            let build_patch = parts.next()?.parse::<u32>().ok()?;
            if parts.next().is_none() && build_major == major && build_minor == minor {
                Some(build_patch)
            } else {
                None
            }
        });

    Ok(NvccVersion {
        major,
        minor,
        patch,
    })
}

/// Validate that an `nvcc` version can target GB10X's `sm_121a` contract.
pub fn validate_nvcc_for_gb10(version: &NvccVersion) -> Result<(), CudaToolchainError> {
    if (version.major, version.minor) < (MIN_CUDA_MAJOR, MIN_CUDA_MINOR) {
        return Err(CudaToolchainError::UnsupportedVersion {
            found_major: version.major,
            found_minor: version.minor,
            required_major: MIN_CUDA_MAJOR,
            required_minor: MIN_CUDA_MINOR,
        });
    }
    Ok(())
}

fn parse_component(value: Option<&str>, whole: &str) -> Result<u32, CudaToolchainError> {
    value
        .and_then(|component| component.parse::<u32>().ok())
        .ok_or_else(|| CudaToolchainError::InvalidRelease(whole.to_owned()))
}

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
