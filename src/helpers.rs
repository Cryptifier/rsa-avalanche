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
    left.iter()
        .zip(right.iter())
        .filter(|(a, b)| a != b)
        .count()
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
    use super::{spread_normalized_avalanche_biases, stored_beam_value_is_one};

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
}
