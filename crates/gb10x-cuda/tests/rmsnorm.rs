#![cfg(feature = "native-cuda")]

use gb10x_cuda::rmsnorm_bf16_host_for_test;

const WIDTH: usize = 2560;
const EPSILON: f32 = 1.0e-6;

fn bf16_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7fff + lsb) >> 16) as u16
}

fn f32_from_bf16(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn oracle(input: &[u16], weight: &[u16]) -> Vec<u16> {
    assert_eq!(input.len(), WIDTH);
    assert_eq!(weight.len(), WIDTH);

    let sum_squares = input.iter().fold(0.0_f32, |sum, &bits| {
        let value = f32_from_bf16(bits);
        sum + value * value
    });
    let inverse_rms = 1.0_f32 / (sum_squares / WIDTH as f32 + EPSILON).sqrt();

    input
        .iter()
        .zip(weight)
        .map(|(&input_bits, &weight_bits)| {
            bf16_from_f32(
                f32_from_bf16(input_bits) * inverse_rms * f32_from_bf16(weight_bits),
            )
        })
        .collect()
}

fn ordered_bf16(bits: u16) -> u32 {
    if bits & 0x8000 != 0 {
        u32::from(!bits)
    } else {
        u32::from(bits | 0x8000)
    }
}

fn assert_within_one_bf16_ulp(actual: &[u16], expected: &[u16]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual_bits, &expected_bits)) in actual.iter().zip(expected).enumerate() {
        let actual_ordered = ordered_bf16(actual_bits);
        let expected_ordered = ordered_bf16(expected_bits);
        let distance = actual_ordered.abs_diff(expected_ordered);
        assert!(
            distance <= 1,
            "BF16 mismatch at {index}: actual=0x{actual_bits:04x} expected=0x{expected_bits:04x} ulp={distance}"
        );
    }
}

fn constant_case(value: f32, weight: f32) -> (Vec<u16>, Vec<u16>) {
    (
        vec![bf16_from_f32(value); WIDTH],
        vec![bf16_from_f32(weight); WIDTH],
    )
}

fn alternating_case() -> (Vec<u16>, Vec<u16>) {
    let input = (0..WIDTH)
        .map(|index| bf16_from_f32(if index % 2 == 0 { 0.5 } else { -0.5 }))
        .collect();
    let weight = (0..WIDTH)
        .map(|index| bf16_from_f32(0.75 + (index % 7) as f32 * 0.03125))
        .collect();
    (input, weight)
}

fn pseudorandom_case() -> (Vec<u16>, Vec<u16>) {
    let mut state = 0xD1B5_4A32_D192_ED03_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state
    };

    let input = (0..WIDTH)
        .map(|_| {
            let sample = ((next() >> 40) as i32 - (1 << 23)) as f32 / (1 << 22) as f32;
            bf16_from_f32(sample)
        })
        .collect();
    let weight = (0..WIDTH)
        .map(|_| {
            let sample = 0.5 + ((next() >> 48) as u16 as f32 / u16::MAX as f32);
            bf16_from_f32(sample)
        })
        .collect();
    (input, weight)
}

#[test]
fn bf16_rmsnorm_matches_fp32_oracle_with_narrow_rounding_gate() {
    let cases = [
        constant_case(0.0, 1.0),
        constant_case(1.0, 1.0),
        alternating_case(),
        pseudorandom_case(),
    ];

    for (input, weight) in cases {
        let expected = oracle(&input, &weight);
        let actual = rmsnorm_bf16_host_for_test(&input, &weight)
            .expect("GB10 BF16 RMSNorm CUDA path must execute");
        assert_within_one_bf16_ulp(&actual, &expected);
    }
}
