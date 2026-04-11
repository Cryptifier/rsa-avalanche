#[cfg(all(feature = "aarch64-hamming-accel", target_arch = "aarch64"))]
use core::arch::aarch64::{vaddvq_u8, vcntq_u8, veorq_u8, vld1q_u8};

/// Computes the Hamming distance between two bit slices.
///
/// # Parameters
/// - `left`: First bit slice.
/// - `right`: Second bit slice.
///
/// # Returns
/// - `usize`: Number of differing bit positions.
///
/// # Expected Output
/// - Returns the distance; no side effects.
pub fn hamming_distance_bits(left: &[bool], right: &[bool]) -> usize {
    let shared_len = left.len().min(right.len());
    if shared_len == 0 {
        return 0;
    }

    #[cfg(all(feature = "aarch64-hamming-accel", target_arch = "aarch64"))]
    {
        let packed_left = pack_bits_to_bytes(&left[..shared_len]);
        let packed_right = pack_bits_to_bytes(&right[..shared_len]);
        return hamming_distance_packed_bytes(&packed_left, &packed_right);
    }

    left[..shared_len]
        .iter()
        .zip(right[..shared_len].iter())
        .filter(|(a, b)| a != b)
        .count()
}

/// Packs a bit slice into bytes so eight booleans share one storage byte.
///
/// # Parameters
/// - `bits`: Bit slice to pack in little-endian order within each byte.
///
/// # Returns
/// - `Vec<u8>`: Packed bytes with one bit per source boolean.
///
/// # Expected Output
/// - Returns packed storage; no stdout/stderr output.
#[cfg(all(feature = "aarch64-hamming-accel", target_arch = "aarch64"))]
pub(crate) fn pack_bits_to_bytes(bits: &[bool]) -> Vec<u8> {
    let mut packed = vec![0u8; bits.len().div_ceil(8)];
    for (index, bit) in bits.iter().enumerate() {
        if *bit {
            packed[index / 8] |= 1u8 << (index % 8);
        }
    }
    packed
}

/// Computes the Hamming distance between packed byte slices.
///
/// # Parameters
/// - `left`: First packed byte slice.
/// - `right`: Second packed byte slice.
///
/// # Returns
/// - `usize`: Number of differing bits across the packed bytes.
///
/// # Expected Output
/// - Returns the distance; no stdout/stderr output.
#[cfg(all(feature = "aarch64-hamming-accel", target_arch = "aarch64"))]
pub(crate) fn hamming_distance_packed_bytes(left: &[u8], right: &[u8]) -> usize {
    if std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { hamming_distance_packed_bytes_neon(left, right) }
    } else {
        hamming_distance_packed_bytes_scalar(left, right)
    }
}

/// Computes the Hamming distance between packed bytes with scalar popcount.
///
/// # Parameters
/// - `left`: First packed byte slice.
/// - `right`: Second packed byte slice.
///
/// # Returns
/// - `usize`: Number of differing bits across the packed bytes.
///
/// # Expected Output
/// - Returns the distance; no stdout/stderr output.
#[cfg(all(feature = "aarch64-hamming-accel", target_arch = "aarch64"))]
fn hamming_distance_packed_bytes_scalar(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a ^ b).count_ones() as usize)
        .sum()
}

/// Computes the Hamming distance between packed bytes with NEON popcount.
///
/// # Parameters
/// - `left`: First packed byte slice.
/// - `right`: Second packed byte slice.
///
/// # Returns
/// - `usize`: Number of differing bits across the packed bytes.
///
/// # Expected Output
/// - Returns the distance; no stdout/stderr output.
#[cfg(all(feature = "aarch64-hamming-accel", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn hamming_distance_packed_bytes_neon(left: &[u8], right: &[u8]) -> usize {
    let mut total = 0usize;
    let mut index = 0usize;

    while index + 16 <= left.len().min(right.len()) {
        let lhs = unsafe { vld1q_u8(left.as_ptr().add(index)) };
        let rhs = unsafe { vld1q_u8(right.as_ptr().add(index)) };
        let diff = veorq_u8(lhs, rhs);
        let counts = vcntq_u8(diff);
        total += vaddvq_u8(counts) as usize;
        index += 16;
    }

    total + hamming_distance_packed_bytes_scalar(&left[index..], &right[index..])
}

/// Normalizes avalanche biases into the [0.0, 1.0] range using max-abs scaling.
///
/// # Parameters
/// - `biases`: Raw avalanche bias values.
///
/// # Returns
/// - `Vec<f64>`: Normalized bias values.
///
/// # Expected Output
/// - Returns normalized biases; no stdout/stderr output.
pub fn normalize_avalanche_biases(biases: &[f64]) -> Vec<f64> {
    let max_abs = biases
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    if max_abs == 0.0 {
        return vec![0.0; biases.len()];
    }
    biases
        .iter()
        .map(|bias| (bias.abs() / max_abs).clamp(0.0, 1.0))
        .collect()
}

/// Spreads normalized avalanche probabilities around the `0.5` decision boundary.
///
/// # Parameters
/// - `normalized_biases`: Bias values already normalized into `[0.0, 1.0]`.
/// - `spread_exponent`: Power exponent applied to each bit's confidence away from
///   `0.5`; values below `1.0` sharpen confidence and values above `1.0` soften it.
///
/// # Returns
/// - `Vec<f64>`: Spread probabilities clamped to `[0.0, 1.0]`.
///
/// # Expected Output
/// - Returns transformed probabilities while preserving which side of `0.5` each
///   input probability was on; no stdout/stderr output.
pub fn spread_normalized_avalanche_biases(
    normalized_biases: &[f64],
    spread_exponent: f64,
) -> Vec<f64> {
    let exponent = if spread_exponent.is_finite() && spread_exponent > 0.0 {
        spread_exponent
    } else {
        1.0
    };
    normalized_biases
        .iter()
        .map(|bias| {
            let probability = bias.clamp(0.0, 1.0);
            let centered = (probability - 0.5) * 2.0;
            if centered == 0.0 {
                return 0.5;
            }
            let spread_confidence = centered.abs().powf(exponent);
            (0.5 + centered.signum() * spread_confidence * 0.5).clamp(0.0, 1.0)
        })
        .collect()
}

/// Interprets a stored beam value as a boolean bit using a configurable cutoff.
///
/// # Parameters
/// - `value`: Stored floating-point value for the bit.
/// - `one_threshold`: Minimum value treated as bit `1`.
///
/// # Returns
/// - `bool`: `true` when `value` is at least `one_threshold`.
///
/// # Expected Output
/// - Returns the interpreted bit; no stdout/stderr output.
pub fn stored_beam_value_is_one(value: f64, one_threshold: f64) -> bool {
    value >= one_threshold
}

/// Formats a floating-point value with a fixed precision.
///
/// # Parameters
/// - `value`: Value to format.
/// - `precision`: Number of decimal places to include.
///
/// # Returns
/// - `String`: Formatted string with the requested precision.
///
/// # Expected Output
/// - Returns a formatted string; no stdout/stderr output.
pub fn format_beam_float(value: f64, precision: usize) -> String {
    format!("{:.precision$}", value, precision = precision)
}

#[cfg(test)]
mod tests {
    use super::{
        hamming_distance_bits, spread_normalized_avalanche_biases, stored_beam_value_is_one,
    };

    #[test]
    fn hamming_distance_bits_counts_differences() {
        assert_eq!(
            hamming_distance_bits(&[true, false, true, false], &[true, true, false, false]),
            2
        );
    }

    #[test]
    fn spread_normalized_biases_preserves_probability_side_of_half() {
        let spread = spread_normalized_avalanche_biases(&[0.25, 0.5, 0.75], 0.5);
        assert!(spread[0] < 0.5);
        assert!((spread[1] - 0.5).abs() < 1e-12);
        assert!(spread[2] > 0.5);
        assert!((spread[0] - 0.1464466094067262).abs() < 1e-12);
        assert!((spread[2] - 0.8535533905932737).abs() < 1e-12);
    }

    #[test]
    fn spread_normalized_biases_softens_toward_half_for_superunit_exponents() {
        let spread = spread_normalized_avalanche_biases(&[0.25, 0.75], 2.0);
        assert!((spread[0] - 0.375).abs() < 1e-12);
        assert!((spread[1] - 0.625).abs() < 1e-12);
    }

    #[test]
    fn spread_normalized_biases_falls_back_to_identity_for_invalid_exponents() {
        let spread = spread_normalized_avalanche_biases(&[0.2, 0.8], 0.0);
        assert!((spread[0] - 0.2).abs() < 1e-12);
        assert!((spread[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn stored_beam_value_uses_configured_threshold() {
        assert!(!stored_beam_value_is_one(0.39, 0.4));
        assert!(stored_beam_value_is_one(0.4, 0.4));
        assert!(stored_beam_value_is_one(1.0, 0.4));
    }

    #[cfg(all(feature = "aarch64-hamming-accel", target_arch = "aarch64"))]
    #[test]
    fn packed_hamming_distance_matches_scalar_path() {
        let left = [
            true, false, true, true, false, false, true, false, true, true, false, true, false,
            true, false, false, true, false, false, true,
        ];
        let right = [
            false, false, true, false, false, true, true, false, true, false, true, true, false,
            false, false, true, false, false, true, true,
        ];

        let packed_left = super::pack_bits_to_bytes(&left);
        let packed_right = super::pack_bits_to_bytes(&right);

        assert_eq!(
            super::hamming_distance_packed_bytes(&packed_left, &packed_right),
            left.iter()
                .zip(right.iter())
                .filter(|(a, b)| a != b)
                .count()
        );
    }
}
