use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

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
            CombinerError::InvalidProbability => write!(f, "match probability must be within [0,1]"),
            CombinerError::InvalidOracleCount => write!(f, "oracle count must be >= 1"),
        }
    }
}

impl std::error::Error for CombinerError {}

pub fn generate_oracle_samples(
    majority_bits: &[bool],
    match_probability: f64,
    rng: &mut impl Rng,
) -> Vec<bool> {
    majority_bits
        .iter()
        .map(|&bit| {
            let roll: f64 = rng.r#gen();
            if roll <= match_probability {
                bit
            } else {
                !bit
            }
        })
        .collect()
}

pub fn majority_vote_per_bit(
    oracles: &[Vec<bool>],
    tie_breaker: bool,
) -> Result<Vec<bool>, CombinerError> {
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
            if ones == zeros { tie_breaker } else { ones > zeros }
        })
        .collect::<Vec<_>>();

    Ok(majority)
}

pub fn optimal_combiner_test(
    majority_bits: &[bool],
    config: &CombinerConfig,
    rng: &mut impl Rng,
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
            let mut local_rng = StdRng::seed_from_u64(seed);
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
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_generate_oracle_samples_all_match() {
        let bits = vec![true, false, true, true];
        let mut rng = StdRng::seed_from_u64(1);
        let sample = generate_oracle_samples(&bits, 1.0, &mut rng);
        assert_eq!(sample, bits);
    }

    #[test]
    fn test_generate_oracle_samples_all_flip() {
        let bits = vec![true, false, true, true];
        let mut rng = StdRng::seed_from_u64(2);
        let sample = generate_oracle_samples(&bits, 0.0, &mut rng);
        assert_eq!(sample, vec![false, true, false, false]);
    }

    #[test]
    fn test_majority_vote_per_bit_basic() {
        let oracles = vec![
            vec![true, false],
            vec![true, true],
            vec![false, true],
        ];
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
    fn test_optimal_combiner_high_accuracy() {
        let bits = vec![true, false, true, false, true, true, false, false, true, false, true, false, true, true, false, true];
        let mut rng = StdRng::seed_from_u64(3);
        let config = CombinerConfig {
            k_oracles: 5,
            match_probability: 0.9,
            tie_breaker: true,
        };
        let result = optimal_combiner_test(&bits, &config, &mut rng).expect("combiner failed");
        assert!(result.accuracy >= 0.6, "accuracy too low: {}", result.accuracy);
    }

    #[test]
    fn test_optimal_combiner_invalid_oracles() {
        let bits = vec![true, false];
        let mut rng = StdRng::seed_from_u64(4);
        let config = CombinerConfig {
            k_oracles: 0,
            match_probability: 0.9,
            tie_breaker: false,
        };
        let err = optimal_combiner_test(&bits, &config, &mut rng).expect_err("expected error");
        assert_eq!(err, CombinerError::InvalidOracleCount);
    }
}
