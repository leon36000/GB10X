#![cfg(feature = "native-cuda")]

use gb10x_cuda::run_smoke;

const MULTIPLIER: u64 = 0x9E37_79B9_7F4A_7C15;
const OFFSET: u64 = 0xD1B5_4A32_D192_ED03;
const XOR_MASK: u64 = 0x94D0_49BB_1331_11EB;

fn expected_value(index: u64) -> u64 {
    index
        .wrapping_mul(MULTIPLIER)
        .wrapping_add(OFFSET)
        .rotate_left(17)
        ^ XOR_MASK
}

fn expected_checksum(elements: u64) -> u64 {
    (0..elements).fold(0_u64, |sum, index| sum.wrapping_add(expected_value(index)))
}

#[test]
fn native_smoke_matches_cpu_checkable_checksum() {
    let elements = 1_u64 << 20;
    let expected = expected_checksum(elements);
    let actual = run_smoke(elements).expect("sm_121a CUDA smoke kernel must execute");
    assert_eq!(actual, expected);
}
