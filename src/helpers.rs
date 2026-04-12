#[cfg(all(feature = "aarch64-hamming-accel", target_arch = "aarch64"))]
use core::arch::aarch64::{vaddvq_u8, vcntq_u8, veorq_u8, vld1q_u8};
#[cfg(all(feature = "x86-hamming-accel", target_arch = "x86_64"))]
use core::arch::x86_64::{
    __m256i, __m512i, _mm256_add_epi8, _mm256_and_si256, _mm256_loadu_si256, _mm256_sad_epu8,
    _mm256_set1_epi8, _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
    _mm256_srli_epi16, _mm256_storeu_si256, _mm256_xor_si256, _mm512_loadu_si512,
    _mm512_popcnt_epi8, _mm512_popcnt_epi64, _mm512_storeu_si512, _mm512_xor_si512,
};

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

    if let Some(distance) =
        hamming_distance_bits_accelerated(&left[..shared_len], &right[..shared_len])
    {
        return distance;
    }

    left[..shared_len]
        .iter()
        .zip(right[..shared_len].iter())
        .filter(|(a, b)| a != b)
        .count()
}

/// Counts matching bits across packed little-endian byte slices.
///
/// # Parameters
/// - `left`: First packed byte slice with least-significant byte first.
/// - `right`: Second packed byte slice with least-significant byte first.
/// - `bit_len`: Number of low-order bits to compare.
///
/// # Returns
/// - `(usize, usize)`: `(matching_lsb_run, matching_total)` counts.
///
/// # Expected Output
/// - Returns packed-byte match counts; no stdout/stderr output.
pub(crate) fn matching_bit_counts_bytes_le(
    left: &[u8],
    right: &[u8],
    bit_len: usize,
) -> (usize, usize) {
    if bit_len == 0 {
        return (0, 0);
    }

    let byte_len = bit_len.div_ceil(8);
    let tail_bits = bit_len % 8;
    let tail_mask = if tail_bits == 0 {
        u8::MAX
    } else {
        ((1u16 << tail_bits) - 1) as u8
    };
    let mut xor_bytes = vec![0u8; byte_len];

    for byte_idx in 0..byte_len {
        let left_byte = left.get(byte_idx).copied().unwrap_or(0);
        let right_byte = right.get(byte_idx).copied().unwrap_or(0);
        let mask = if byte_idx + 1 == byte_len {
            tail_mask
        } else {
            u8::MAX
        };
        xor_bytes[byte_idx] = (left_byte ^ right_byte) & mask;
    }

    let differing = xor_bytes
        .iter()
        .map(|byte| byte.count_ones() as usize)
        .sum::<usize>();
    let matching_total = bit_len.saturating_sub(differing);

    let mut matching_lsb = 0usize;
    for (byte_idx, diff) in xor_bytes.iter().enumerate() {
        if *diff == 0 {
            let full_bits = if byte_idx + 1 == byte_len {
                if tail_bits == 0 { 8 } else { tail_bits }
            } else {
                8
            };
            matching_lsb += full_bits;
            continue;
        }

        let first_diff = diff.trailing_zeros() as usize;
        let byte_limit = if byte_idx + 1 == byte_len {
            if tail_bits == 0 { 8 } else { tail_bits }
        } else {
            8
        };
        matching_lsb += first_diff.min(byte_limit);
        break;
    }

    (matching_lsb.min(bit_len), matching_total)
}

/// Computes the Hamming distance through an accelerated packed path when enabled.
///
/// # Parameters
/// - `left`: First bit slice with matching length already enforced by the caller.
/// - `right`: Second bit slice with matching length already enforced by the caller.
///
/// # Returns
/// - `Option<usize>`: Accelerated distance when a packed backend is available.
///
/// # Expected Output
/// - Returns an optional distance; no stdout/stderr output.
fn hamming_distance_bits_accelerated(left: &[bool], right: &[bool]) -> Option<usize> {
    let packed_left = pack_bits_to_bytes(left);
    let packed_right = pack_bits_to_bytes(right);
    Some(hamming_distance_packed_bytes(&packed_left, &packed_right))
}

/// Computes the Hamming distance through an accelerated packed path when enabled.
///
/// # Parameters
/// - `left`: First bit slice with matching length already enforced by the caller.
/// - `right`: Second bit slice with matching length already enforced by the caller.
///
/// # Returns
/// - `Option<usize>`: `None` when no accelerated backend is compiled in.
///
/// # Expected Output
/// - Returns `None`; no stdout/stderr output.
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
fn hamming_distance_packed_bytes_scalar(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a ^ b).count_ones() as usize)
        .sum()
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
pub(crate) fn hamming_distance_packed_bytes(left: &[u8], right: &[u8]) -> usize {
    #[cfg(all(feature = "aarch64-hamming-accel", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { hamming_distance_packed_bytes_neon(left, right) };
        }
    }

    #[cfg(all(feature = "x86-hamming-accel", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx512bitalg") {
            return unsafe { hamming_distance_packed_bytes_avx512bitalg(left, right) };
        }
        if std::arch::is_x86_feature_detected!("avx512vpopcntdq") {
            return unsafe { hamming_distance_packed_bytes_avx512vpopcntdq(left, right) };
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { hamming_distance_packed_bytes_avx2(left, right) };
        }
    }

    hamming_distance_packed_bytes_scalar(left, right)
}

/// Computes the Hamming distance between packed bytes with AVX-512 byte popcount.
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
#[cfg(all(feature = "x86-hamming-accel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bitalg")]
unsafe fn hamming_distance_packed_bytes_avx512bitalg(left: &[u8], right: &[u8]) -> usize {
    let len = left.len().min(right.len());
    let mut total = 0usize;
    let mut index = 0usize;

    while index + 64 <= len {
        let lhs = unsafe { _mm512_loadu_si512(left.as_ptr().add(index).cast::<__m512i>()) };
        let rhs = unsafe { _mm512_loadu_si512(right.as_ptr().add(index).cast::<__m512i>()) };
        let diff = _mm512_xor_si512(lhs, rhs);
        let counts = _mm512_popcnt_epi8(diff);
        let mut lanes = [0u8; 64];
        unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast::<__m512i>(), counts) };
        total += lanes.iter().map(|value| *value as usize).sum::<usize>();
        index += 64;
    }

    total + hamming_distance_packed_bytes_scalar(&left[index..len], &right[index..len])
}

/// Computes the Hamming distance between packed bytes with AVX-512 quadword popcount.
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
#[cfg(all(feature = "x86-hamming-accel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn hamming_distance_packed_bytes_avx512vpopcntdq(left: &[u8], right: &[u8]) -> usize {
    let len = left.len().min(right.len());
    let mut total = 0usize;
    let mut index = 0usize;

    while index + 64 <= len {
        let lhs = unsafe { _mm512_loadu_si512(left.as_ptr().add(index).cast::<__m512i>()) };
        let rhs = unsafe { _mm512_loadu_si512(right.as_ptr().add(index).cast::<__m512i>()) };
        let diff = _mm512_xor_si512(lhs, rhs);
        let counts = _mm512_popcnt_epi64(diff);
        let mut lanes = [0u64; 8];
        unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast::<__m512i>(), counts) };
        total += lanes.iter().map(|value| *value as usize).sum::<usize>();
        index += 64;
    }

    total + hamming_distance_packed_bytes_scalar(&left[index..len], &right[index..len])
}

/// Computes the Hamming distance between packed bytes with AVX2 nibble lookup popcount.
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
#[cfg(all(feature = "x86-hamming-accel", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hamming_distance_packed_bytes_avx2(left: &[u8], right: &[u8]) -> usize {
    let len = left.len().min(right.len());
    let nibble_popcounts = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f_u8 as i8);
    let zero = _mm256_setzero_si256();
    let mut total = 0usize;
    let mut index = 0usize;

    while index + 32 <= len {
        let lhs = unsafe { _mm256_loadu_si256(left.as_ptr().add(index).cast::<__m256i>()) };
        let rhs = unsafe { _mm256_loadu_si256(right.as_ptr().add(index).cast::<__m256i>()) };
        let diff = _mm256_xor_si256(lhs, rhs);
        let low_nibbles = _mm256_and_si256(diff, low_mask);
        let high_nibbles = _mm256_and_si256(_mm256_srli_epi16::<4>(diff), low_mask);
        let low_counts = _mm256_shuffle_epi8(nibble_popcounts, low_nibbles);
        let high_counts = _mm256_shuffle_epi8(nibble_popcounts, high_nibbles);
        let counts = _mm256_add_epi8(low_counts, high_counts);
        let sums = _mm256_sad_epu8(counts, zero);
        let mut lanes = [0u64; 4];
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), sums) };
        total += lanes.iter().map(|value| *value as usize).sum::<usize>();
        index += 32;
    }

    total + hamming_distance_packed_bytes_scalar(&left[index..len], &right[index..len])
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
        hamming_distance_bits, matching_bit_counts_bytes_le, spread_normalized_avalanche_biases,
        stored_beam_value_is_one,
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

    #[test]
    fn matching_bit_counts_bytes_handles_partial_tail_bits() {
        let left = [0b1011_0101u8, 0b0000_0011u8];
        let right = [0b1011_0001u8, 0b0000_0010u8];
        assert_eq!(matching_bit_counts_bytes_le(&left, &right, 10), (2, 8));
    }

    #[test]
    fn matching_bit_counts_bytes_treats_missing_high_bytes_as_zero() {
        let left = [0u8];
        let right = [0b0000_1000u8];
        assert_eq!(matching_bit_counts_bytes_le(&left, &right, 4), (3, 3));
    }

    #[cfg(any(
        all(feature = "aarch64-hamming-accel", target_arch = "aarch64"),
        all(feature = "x86-hamming-accel", target_arch = "x86_64"),
    ))]
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
