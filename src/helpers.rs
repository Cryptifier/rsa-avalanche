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
    use super::stored_beam_value_is_one;

    #[test]
    fn stored_beam_value_uses_configured_threshold() {
        assert!(!stored_beam_value_is_one(0.39, 0.4));
        assert!(stored_beam_value_is_one(0.4, 0.4));
        assert!(stored_beam_value_is_one(1.0, 0.4));
    }
}
