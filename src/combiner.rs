/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use rand::{Rng, RngCore};
use rayon::prelude::*;

use crate::helpers::PackedBits;
use crate::rng::RngChoice;

#[derive(Debug, Clone)]
pub struct CombinerConfig {
    pub k_oracles: usize,
    pub match_probability: f64,
    pub tie_breaker: bool,
}

#[derive(Debug, Clone)]
pub struct CombinerResult {
    pub majority_bits: Vec<bool>,
    pub correct_bits: usize,
    pub total_bits: usize,
    pub accuracy: f64,
}

#[derive(Debug, Clone)]
pub struct MajorityDistribution {
    pub majority_bits: Vec<bool>,
    pub ones_count: Vec<usize>,
    pub zeros_count: Vec<usize>,
    pub probability_one: Vec<f64>,
    pub total_oracles: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CombinerError {
    EmptyOracles,
    InconsistentLengths,
    EmptyMajorityBits,
    InvalidProbability,
    InvalidOracleCount,
}

impl std::fmt::Display for CombinerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CombinerError::EmptyOracles => write!(f, "no oracles provided"),
            CombinerError::InconsistentLengths => write!(f, "oracle bit lengths differ"),
            CombinerError::EmptyMajorityBits => write!(f, "no majority bits provided"),
            CombinerError::InvalidProbability => {
                write!(f, "match probability must be within [0,1]")
            }
            CombinerError::InvalidOracleCount => write!(f, "oracle count must be >= 1"),
        }
    }
}

impl std::error::Error for CombinerError {}

/// Validates oracle vectors for non-emptiness and consistent bit lengths.
///
/// # Parameters
/// - `oracles`: Slice of oracle bit vectors to validate.
///
/// # Returns
/// - `Result<usize, CombinerError>`: On success, the common bit length for all oracles.
///
/// # Expected Output
/// - Returns the bit length when valid; otherwise a `CombinerError`.
fn validate_oracles(oracles: &[Vec<bool>]) -> Result<usize, CombinerError> {
    if oracles.is_empty() {
        return Err(CombinerError::EmptyOracles);
    }
    let bit_len = oracles[0].len();
    if bit_len == 0 {
        return Err(CombinerError::EmptyMajorityBits);
    }
    if oracles.iter().any(|o| o.len() != bit_len) {
        return Err(CombinerError::InconsistentLengths);
    }
    Ok(bit_len)
}

/// Validates packed oracle vectors for non-emptiness and consistent bit lengths.
///
/// # Parameters
/// - `oracles`: Slice of packed oracle bit vectors to validate.
///
/// # Returns
/// - `Result<usize, CombinerError>`: On success, the common bit length for all packed oracles.
///
/// # Expected Output
/// - Returns the bit length when valid; otherwise a `CombinerError`.
fn validate_packed_oracles(oracles: &[PackedBits]) -> Result<usize, CombinerError> {
    if oracles.is_empty() {
        return Err(CombinerError::EmptyOracles);
    }
    let bit_len = oracles[0].len();
    if bit_len == 0 {
        return Err(CombinerError::EmptyMajorityBits);
    }
    if oracles.iter().any(|oracle| oracle.len() != bit_len) {
        return Err(CombinerError::InconsistentLengths);
    }
    Ok(bit_len)
}

/// Generates a noisy oracle sample by flipping bits with a given mismatch rate.
///
/// # Parameters
/// - `majority_bits`: Reference bit string used as the oracle target.
/// - `match_probability`: Probability that each oracle bit matches the reference.
/// - `rng`: Random number generator used to sample matches vs. flips.
///
/// # Returns
/// - `Vec<bool>`: A bit vector with each position matching or flipped from `majority_bits`.
///
/// # Expected Output
/// - Returns a vector of equal length to `majority_bits`; no stdout/stderr output.
pub fn generate_oracle_samples(
    majority_bits: &[bool],
    match_probability: f64,
    rng: &mut impl Rng,
) -> Vec<bool> {
    majority_bits
        .iter()
        .map(|&bit| {
            let roll: f64 = rng.r#gen();
            if roll <= match_probability { bit } else { !bit }
        })
        .collect()
}

/// Computes the majority bit per position across multiple oracles.
///
/// # Parameters
/// - `oracles`: Collection of oracle bit vectors (must be non-empty and equal length).
/// - `tie_breaker`: Bit value to use when a position has an equal number of 0s and 1s.
///
/// # Returns
/// - `Result<Vec<bool>, CombinerError>`: The majority bit vector, or a validation error.
///
/// # Expected Output
/// - Returns a bit vector with the same length as each oracle; no side effects.
pub fn majority_vote_per_bit(
    oracles: &[Vec<bool>],
    tie_breaker: bool,
) -> Result<Vec<bool>, CombinerError> {
    let bit_len = validate_oracles(oracles)?;

    let majority = (0..bit_len)
        .into_par_iter()
        .map(|idx| {
            let mut ones = 0usize;
            let mut zeros = 0usize;
            for oracle in oracles {
                if oracle[idx] {
                    ones += 1;
                } else {
                    zeros += 1;
                }
            }
            if ones == zeros {
                tie_breaker
            } else {
                ones > zeros
            }
        })
        .collect::<Vec<_>>();

    Ok(majority)
}

/// Computes the majority bit per position and returns full per-bit distribution stats.
///
/// # Parameters
/// - `oracles`: Collection of oracle bit vectors (must be non-empty and equal length).
/// - `tie_breaker`: Bit value to use when a position has an equal number of 0s and 1s.
///
/// # Returns
/// - `Result<MajorityDistribution, CombinerError>`: Majority bits plus counts/probabilities.
///
/// # Expected Output
/// - Returns distribution information for each bit position; no stdout/stderr output.
pub fn majority_vote_with_distribution(
    oracles: &[Vec<bool>],
    tie_breaker: bool,
) -> Result<MajorityDistribution, CombinerError> {
    let bit_len = validate_oracles(oracles)?;
    let total_oracles = oracles.len();
    let mut ones_count = vec![0usize; bit_len];

    for oracle in oracles {
        for (idx, &bit) in oracle.iter().enumerate() {
            if bit {
                ones_count[idx] += 1;
            }
        }
    }

    Ok(build_majority_distribution(
        ones_count,
        total_oracles,
        tie_breaker,
    ))
}

/// Computes the majority bit per position and returns distribution stats from packed bit storage.
///
/// # Parameters
/// - `oracles`: Collection of packed oracle bit vectors (must be non-empty and equal length).
/// - `tie_breaker`: Bit value to use when a position has an equal number of 0s and 1s.
///
/// # Returns
/// - `Result<MajorityDistribution, CombinerError>`: Majority bits plus counts/probabilities.
///
/// # Expected Output
/// - Returns distribution information for each bit position; no stdout/stderr output.
pub(crate) fn majority_vote_with_distribution_packed(
    oracles: &[PackedBits],
    tie_breaker: bool,
) -> Result<MajorityDistribution, CombinerError> {
    let bit_len = validate_packed_oracles(oracles)?;
    let total_oracles = oracles.len();
    let byte_len = bit_len.div_ceil(8);
    let tail_bits = bit_len % 8;
    let tail_mask = if tail_bits == 0 {
        u8::MAX
    } else {
        ((1u16 << tail_bits) - 1) as u8
    };
    let mut ones_count = vec![0usize; bit_len];

    for oracle in oracles {
        for byte_idx in 0..byte_len {
            let mask = if byte_idx + 1 == byte_len {
                tail_mask
            } else {
                u8::MAX
            };
            let mut byte = oracle.bytes_le().get(byte_idx).copied().unwrap_or(0) & mask;
            while byte != 0 {
                let bit_offset = byte.trailing_zeros() as usize;
                let bit_idx = byte_idx * 8 + bit_offset;
                if bit_idx < bit_len {
                    ones_count[bit_idx] += 1;
                }
                byte &= byte - 1;
            }
        }
    }

    Ok(build_majority_distribution(
        ones_count,
        total_oracles,
        tie_breaker,
    ))
}

/// Builds majority-vote distribution fields from per-bit one counts.
///
/// # Parameters
/// - `ones_count`: Count of `1` bits observed at each position.
/// - `total_oracles`: Number of oracle inputs contributing to `ones_count`.
/// - `tie_breaker`: Bit value to use when ones and zeros tie.
///
/// # Returns
/// - `MajorityDistribution`: Majority bits plus counts and probabilities.
///
/// # Expected Output
/// - Returns a fully populated distribution record; no stdout/stderr output.
fn build_majority_distribution(
    ones_count: Vec<usize>,
    total_oracles: usize,
    tie_breaker: bool,
) -> MajorityDistribution {
    let zeros_count: Vec<usize> = ones_count
        .iter()
        .map(|&ones| total_oracles - ones)
        .collect();
    let probability_one: Vec<f64> = ones_count
        .iter()
        .map(|&ones| ones as f64 / total_oracles as f64)
        .collect();
    let majority_bits = ones_count
        .iter()
        .zip(zeros_count.iter())
        .map(|(&ones, &zeros)| {
            if ones == zeros {
                tie_breaker
            } else {
                ones > zeros
            }
        })
        .collect();

    MajorityDistribution {
        majority_bits,
        ones_count,
        zeros_count,
        probability_one,
        total_oracles,
    }
}

/// Runs a full combiner experiment by sampling oracles and voting against a target.
///
/// # Parameters
/// - `majority_bits`: Ground-truth bit vector to compare against.
/// - `config`: Combiner configuration controlling oracle count and sampling.
/// - `rng`: Random number generator used to seed oracle sampling.
///
/// # Returns
/// - `Result<CombinerResult, CombinerError>`: Majority vote results and accuracy metrics.
///
/// # Expected Output
/// - Returns accuracy statistics for the simulated combiner run; no stdout/stderr output.
pub fn optimal_combiner_test(
    majority_bits: &[bool],
    config: &CombinerConfig,
    rng: &mut RngChoice,
) -> Result<CombinerResult, CombinerError> {
    if config.k_oracles == 0 {
        return Err(CombinerError::InvalidOracleCount);
    }
    if majority_bits.is_empty() {
        return Err(CombinerError::EmptyMajorityBits);
    }
    if !(0.0..=1.0).contains(&config.match_probability) {
        return Err(CombinerError::InvalidProbability);
    }

    let mut seeds = Vec::with_capacity(config.k_oracles);
    for _ in 0..config.k_oracles {
        seeds.push(rng.next_u64());
    }

    let oracles = seeds
        .into_par_iter()
        .map(|seed| {
            let mut local_rng = RngChoice::from_seed(rng.mode(), seed);
            generate_oracle_samples(majority_bits, config.match_probability, &mut local_rng)
        })
        .collect::<Vec<_>>();

    let majority = majority_vote_per_bit(&oracles, config.tie_breaker)?;
    let mut correct = 0usize;
    for (a, b) in majority.iter().zip(majority_bits.iter()) {
        if a == b {
            correct += 1;
        }
    }
    let total = majority_bits.len();
    let accuracy = correct as f64 / total as f64;

    Ok(CombinerResult {
        majority_bits: majority,
        correct_bits: correct,
        total_bits: total,
        accuracy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::RngMode;

    #[test]
    fn test_generate_oracle_samples_all_match() {
        let bits = vec![true, false, true, true];
        let mut rng = RngChoice::from_seed(RngMode::Standard, 1);
        let sample = generate_oracle_samples(&bits, 1.0, &mut rng);
        assert_eq!(sample, bits);
    }

    #[test]
    fn test_generate_oracle_samples_all_flip() {
        let bits = vec![true, false, true, true];
        let mut rng = RngChoice::from_seed(RngMode::Standard, 2);
        let sample = generate_oracle_samples(&bits, 0.0, &mut rng);
        assert_eq!(sample, vec![false, true, false, false]);
    }

    #[test]
    fn test_majority_vote_per_bit_basic() {
        let oracles = vec![vec![true, false], vec![true, true], vec![false, true]];
        let majority = majority_vote_per_bit(&oracles, false).expect("majority failed");
        assert_eq!(majority, vec![true, true]);
    }

    #[test]
    fn test_majority_vote_per_bit_tie_breaker() {
        let oracles = vec![vec![true], vec![false]];
        let majority = majority_vote_per_bit(&oracles, true).expect("majority failed");
        assert_eq!(majority, vec![true]);
    }

    #[test]
    fn test_majority_vote_distribution_basic() {
        let oracles = vec![
            vec![true, false, true],
            vec![false, false, true],
            vec![true, true, false],
        ];
        let result = majority_vote_with_distribution(&oracles, false).expect("distribution failed");
        assert_eq!(result.total_oracles, 3);
        assert_eq!(result.ones_count, vec![2, 1, 2]);
        assert_eq!(result.zeros_count, vec![1, 2, 1]);
        assert_eq!(result.majority_bits, vec![true, false, true]);
        assert_eq!(
            result.probability_one,
            vec![2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0]
        );
    }

    #[test]
    fn test_optimal_combiner_high_accuracy() {
        let bits = vec![
            true, false, true, false, true, true, false, false, true, false, true, false, true,
            true, false, true,
        ];
        let mut rng = RngChoice::from_seed(RngMode::Standard, 3);
        let config = CombinerConfig {
            k_oracles: 5,
            match_probability: 0.9,
            tie_breaker: true,
        };
        let result = optimal_combiner_test(&bits, &config, &mut rng).expect("combiner failed");
        assert!(
            result.accuracy >= 0.6,
            "accuracy too low: {}",
            result.accuracy
        );
    }

    #[test]
    fn test_optimal_combiner_invalid_oracles() {
        let bits = vec![true, false];
        let mut rng = RngChoice::from_seed(RngMode::Standard, 4);
        let config = CombinerConfig {
            k_oracles: 0,
            match_probability: 0.9,
            tie_breaker: false,
        };
        let err = optimal_combiner_test(&bits, &config, &mut rng).expect_err("expected error");
        assert_eq!(err, CombinerError::InvalidOracleCount);
    }
}
