use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};

use crate::config::{PolynomialFieldConfig, PolynomialFieldsConfig};
use crate::math::is_probable_prime_big;

const MIN_POLY_BITS: u64 = 8;
const MAX_POLY_BITS: u64 = 64;
const POLY_DEGREE: usize = 3;
const MAX_FIELDS: usize = 16;

/// Polynomial field definition used to score ciphertext residues.
#[derive(Debug, Clone)]
pub struct PolynomialField {
    prime: BigUint,
    seed: u64,
    coefficients: Vec<BigUint>,
}

/// Coordinate produced by evaluating a polynomial field.
#[derive(Debug, Clone)]
pub struct PolynomialCoordinate {
    /// Label for the coordinate (X, Y, Z, ...).
    pub label: String,
    /// Normalized value in the range [0.0, 1.0].
    pub value: f64,
}

/// Builds polynomial fields from configuration.
///
/// # Parameters
/// - `config`: Polynomial field configuration loaded from JSON.
///
/// # Returns
/// - `Result<Vec<PolynomialField>, String>`: Parsed polynomial fields or an error message.
///
/// # Expected Output
/// - Returns validated fields without side effects.
pub fn build_polynomial_fields(
    config: &PolynomialFieldsConfig,
) -> Result<Vec<PolynomialField>, String> {
    if config.fields.len() > MAX_FIELDS {
        return Err(format!(
            "polynomial field count {} exceeds max {}",
            config.fields.len(),
            MAX_FIELDS
        ));
    }

    config
        .fields
        .iter()
        .map(PolynomialField::from_config)
        .collect()
}

/// Generates normalized coordinates for a ciphertext using the given fields.
///
/// # Parameters
/// - `ciphertext`: Ciphertext value to score.
/// - `fields`: Polynomial fields used to compute coordinates.
///
/// # Returns
/// - `Vec<PolynomialCoordinate>`: Coordinate list in field order.
///
/// # Expected Output
/// - Returns normalized coordinates; no stdout/stderr output.
pub fn coordinates_for_ciphertext(
    ciphertext: &BigUint,
    fields: &[PolynomialField],
) -> Vec<PolynomialCoordinate> {
    fields
        .iter()
        .enumerate()
        .map(|(idx, field)| PolynomialCoordinate {
            label: coordinate_label(idx),
            value: field.score_normalized(ciphertext),
        })
        .collect()
}

impl PolynomialField {
    /// Builds a polynomial field from config, validating prime size and coefficients.
    ///
    /// # Parameters
    /// - `config`: Field config containing prime modulus and seed.
    ///
    /// # Returns
    /// - `Result<PolynomialField, String>`: Parsed field or an error message.
    ///
    /// # Expected Output
    /// - Returns a validated field; no stdout/stderr output.
    pub fn from_config(config: &PolynomialFieldConfig) -> Result<Self, String> {
        let bits = config.prime.bits();
        if bits < MIN_POLY_BITS || bits > MAX_POLY_BITS {
            return Err(format!(
                "prime modulus bit length {} outside {}..={}",
                bits, MIN_POLY_BITS, MAX_POLY_BITS
            ));
        }
        if config.prime.is_zero() || config.prime.is_one() {
            return Err("prime modulus must be > 1".to_string());
        }
        if !is_probable_prime_big(&config.prime) {
            return Err(format!("prime modulus {} is not prime", config.prime));
        }

        let coefficients = generate_coefficients(config.seed, &config.prime, POLY_DEGREE);
        Ok(Self {
            prime: config.prime.clone(),
            seed: config.seed,
            coefficients,
        })
    }

    fn score_normalized(&self, ciphertext: &BigUint) -> f64 {
        let value = self.score(ciphertext);
        let max_value = self.prime.clone().saturating_sub(BigUint::one());
        if max_value.is_zero() {
            return 0.0;
        }

        let value_u128 = value.to_u128().unwrap_or(u128::MAX);
        let max_u128 = max_value.to_u128().unwrap_or(u128::MAX);
        if max_u128 == 0 {
            return 0.0;
        }

        let ratio = (value_u128.min(max_u128)) as f64 / max_u128 as f64;
        ratio.clamp(0.0, 1.0)
    }

    fn score(&self, ciphertext: &BigUint) -> BigUint {
        let x = ciphertext % &self.prime;
        evaluate_polynomial(&self.coefficients, &x, &self.prime)
    }
}

/// Generates polynomial coefficients from a seed within a modulus.
///
/// # Parameters
/// - `seed`: Seed value used to derive coefficients.
/// - `modulus`: Prime modulus defining coefficient bounds.
/// - `degree`: Polynomial degree to generate.
///
/// # Returns
/// - `Vec<BigUint>`: Coefficients in ascending degree order.
///
/// # Expected Output
/// - Returns a deterministic coefficient list; no stdout/stderr output.
fn generate_coefficients(seed: u64, modulus: &BigUint, degree: usize) -> Vec<BigUint> {
    let mut state = seed;
    let mut coeffs = Vec::with_capacity(degree + 1);
    for _ in 0..=degree {
        state = lcg_next(state);
        let coeff = BigUint::from(state) % modulus;
        coeffs.push(coeff);
    }
    coeffs
}

/// Advances the local LCG used for coefficient generation.
///
/// # Parameters
/// - `state`: Current LCG state.
///
/// # Returns
/// - `u64`: Next LCG state.
///
/// # Expected Output
/// - Returns the next state; no stdout/stderr output.
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1)
}

/// Evaluates a polynomial at `x` within the provided modulus.
///
/// # Parameters
/// - `coeffs`: Coefficients in ascending degree order.
/// - `x`: Input value to evaluate.
/// - `modulus`: Prime modulus for field arithmetic.
///
/// # Returns
/// - `BigUint`: Polynomial value reduced modulo `modulus`.
///
/// # Expected Output
/// - Returns the reduced polynomial value; no stdout/stderr output.
fn evaluate_polynomial(coeffs: &[BigUint], x: &BigUint, modulus: &BigUint) -> BigUint {
    let mut result = BigUint::zero();
    let mut power = BigUint::one();
    for coeff in coeffs {
        let term = (coeff * &power) % modulus;
        result = (result + term) % modulus;
        power = (&power * x) % modulus;
    }
    result
}

/// Returns a coordinate label for the given index.
///
/// # Parameters
/// - `index`: Coordinate index.
///
/// # Returns
/// - `String`: Coordinate label string.
///
/// # Expected Output
/// - Returns a label with no side effects.
fn coordinate_label(index: usize) -> String {
    const LABELS: [&str; 16] = [
        "X", "Y", "Z", "W", "V", "U", "T", "S", "R", "Q", "P", "O", "N", "M", "L", "K",
    ];
    if index < LABELS.len() {
        LABELS[index].to_string()
    } else {
        format!("C{}", index + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_yaml_snapshot;

    fn make_config(primes: &[u64], seeds: &[u64]) -> PolynomialFieldsConfig {
        let fields = primes
            .iter()
            .zip(seeds.iter())
            .map(|(prime, seed)| PolynomialFieldConfig {
                prime: BigUint::from(*prime),
                seed: *seed,
            })
            .collect();
        PolynomialFieldsConfig { fields }
    }

    fn snapshot_coords(ciphertext: u64, primes: &[u64], seeds: &[u64]) -> Vec<String> {
        let config = make_config(primes, seeds);
        let fields = build_polynomial_fields(&config).expect("fields");
        let coords = coordinates_for_ciphertext(&BigUint::from(ciphertext), &fields);
        coords
            .iter()
            .map(|coord| format!("{}:{:.8}", coord.label, coord.value))
            .collect()
    }

    #[test]
    fn test_coordinates_vector_1() {
        let out = snapshot_coords(123, &[251], &[1]);
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_2() {
        let out = snapshot_coords(456, &[251, 257], &[1, 2]);
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_3() {
        let out = snapshot_coords(789, &[251, 257, 263], &[1, 2, 3]);
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_4() {
        let out = snapshot_coords(1024, &[251, 257, 263, 269], &[1, 2, 3, 4]);
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_5() {
        let out = snapshot_coords(2048, &[251, 257, 263, 269, 271], &[1, 2, 3, 4, 5]);
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_6() {
        let out = snapshot_coords(
            4096,
            &[251, 257, 263, 269, 271, 277],
            &[1, 2, 3, 4, 5, 6],
        );
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_7() {
        let out = snapshot_coords(
            8192,
            &[251, 257, 263, 269, 271, 277, 281],
            &[1, 2, 3, 4, 5, 6, 7],
        );
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_8() {
        let out = snapshot_coords(
            16384,
            &[251, 257, 263, 269, 271, 277, 281, 283],
            &[1, 2, 3, 4, 5, 6, 7, 8],
        );
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_9() {
        let out = snapshot_coords(
            32768,
            &[251, 257, 263, 269, 271, 277, 281, 283, 293],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9],
        );
        assert_yaml_snapshot!(out);
    }

    #[test]
    fn test_coordinates_vector_10() {
        let out = snapshot_coords(
            65535,
            &[251, 257, 263, 269, 271, 277, 281, 283, 293, 307],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        );
        assert_yaml_snapshot!(out);
    }
}
