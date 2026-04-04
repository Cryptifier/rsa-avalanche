use crate::math::{
    coalesce_factors, factor_composite_with_timeout, floor_biguint_pow_bigdecimal,
    is_probable_prime_big,
    next_prime_from_biguint_pow_bigdecimal, pollard_rho, random_bigdecimal_partition_with_min,
    random_biguint_below, random_biguint_bits,
};
use bigdecimal::BigDecimal;
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use rand::RngCore;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use crate::rng::RngChoice;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RCandidateMode {
    Factoring,
    SmallPrimes,
}

impl Default for RCandidateMode {
    fn default() -> Self {
        Self::Factoring
    }
}

/// Mutable r-candidate metadata used by speculative-oracle flows.
#[derive(Debug, Clone)]
pub struct RCandidate {
    /// Candidate modulus value.
    pub r: BigUint,
    /// Prime-power factorization for `r`.
    pub factors: Vec<(BigUint, u64)>,
    /// Decimal target exponent used when retargeting this candidate.
    pub target_exponent: BigDecimal,
}

impl RCandidate {
    /// Builds an `RCandidate` with an unset target exponent.
    ///
    /// # Parameters
    /// - `r`: Candidate modulus value.
    /// - `factors`: Prime-power factorization metadata.
    ///
    /// # Returns
    /// - `RCandidate`: Candidate wrapper around the supplied values.
    ///
    /// # Expected Output
    /// - Returns a new candidate with `target_exponent = 0`; no side effects.
    pub fn new(r: BigUint, factors: Vec<(BigUint, u64)>) -> Self {
        Self {
            r,
            factors,
            target_exponent: BigDecimal::zero(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RCandidateSettings {
    pub mode: RCandidateMode,
    pub override_best_r: Option<BigUint>,
    pub process_min_factor: BigUint,
    pub process_count: u64,
    pub process_min_count: u64,
    pub process_scale: u32,
    pub reuse_r_candidates_path: String,
    pub reuse_r_candidates: bool,
    pub reuse_r_candidates_append_only: bool,
    pub small_primes: Vec<BigUint>,
    pub small_prime_factors_per_candidate: usize,
    pub max_factors_per_candidate: usize,
    pub target_bit_length: Option<u64>,
    pub random_power_window: bool,
    pub target_exponent_minimum: BigDecimal,
    pub target_exponent: BigDecimal,
    pub retarget_partition_count: usize,
    pub retarget_minimum_exponent: BigDecimal,
}

/// Generates `r` candidates using the configured strategy.
///
/// # Parameters
/// - `n`: RSA modulus used to bound/scale candidates in factoring mode.
/// - `settings`: Candidate generation configuration.
/// - `rng`: Random number generator for sampling candidates.
///
/// # Returns
/// - `Vec<RCandidate>`: List of mutable candidate records.
///
/// # Expected Output
/// - Returns an empty list when no candidates are found; may print progress logs.
pub fn generate_r_candidates(
    n: &BigUint,
    settings: &RCandidateSettings,
    rng: &mut RngChoice,
) -> Vec<RCandidate> {
    match settings.mode {
        RCandidateMode::Factoring => generate_r_candidates_via_factoring(n, settings, rng),
        RCandidateMode::SmallPrimes => {
            let mut adjusted = settings.clone();
            if adjusted.target_bit_length.is_none() {
                adjusted.target_bit_length = n.bits().checked_add(1);
            }
            generate_r_candidates_from_small_primes(&adjusted, rng)
        }
    }
}

/// Generates a batch of `r` candidates with a fixed batch size.
///
/// # Parameters
/// - `n`: RSA modulus used to bound/scale candidates in factoring mode.
/// - `settings`: Candidate generation configuration (cloned and adjusted for batch size).
/// - `rng`: Random number generator for sampling candidates.
/// - `batch_size`: Target number of candidates to produce.
///
/// # Returns
/// - `Vec<RCandidate>`: List of mutable candidate records.
///
/// # Expected Output
/// - Returns a list with up to `batch_size` entries; may print progress logs.
pub fn generate_r_candidates_batch(
    n: &BigUint,
    settings: &RCandidateSettings,
    rng: &mut RngChoice,
    batch_size: usize,
) -> Vec<RCandidate> {
    let target = batch_size.max(1) as u64;
    let mut batch_settings = settings.clone();
    batch_settings.process_count = target;
    batch_settings.process_min_count = target;
    generate_r_candidates(n, &batch_settings, rng)
}

/// Generates `r` candidates from a ciphertext stream using `r = c^x mod n`.
///
/// # Parameters
/// - `ciphertext`: Ciphertext value reduced modulo `n`.
/// - `n`: RSA modulus used for modular exponentiation.
/// - `count`: Number of `r` candidates to produce.
/// - `start_exponent`: Initial exponent `x` (increments by one per candidate).
///
/// # Returns
/// - `Vec<RCandidate>`: List of candidate records with empty factor lists.
///
/// # Expected Output
/// - Returns a deterministic sequence; no stdout/stderr output.
pub fn generate_r_candidates_from_ciphertext_stream(
    ciphertext: &BigUint,
    n: &BigUint,
    count: usize,
    start_exponent: u64,
) -> Vec<RCandidate> {
    if count == 0 || n.is_zero() {
        return Vec::new();
    }

    let mut exponent = start_exponent;
    let mut results = Vec::with_capacity(count);
    for _ in 0..count {
        let r = ciphertext_stream_next(ciphertext, n, &mut exponent);
        results.push(RCandidate::new(r, Vec::new()));
    }
    results
}

/// Computes the next ciphertext-derived `r` using `r = c^x mod n`.
///
/// # Parameters
/// - `ciphertext`: Ciphertext value reduced modulo `n`.
/// - `n`: RSA modulus for modular exponentiation.
/// - `exponent`: Mutable exponent counter, incremented after each use.
///
/// # Returns
/// - `BigUint`: Next `r` value in the sequence.
///
/// # Expected Output
/// - Increments `exponent` by one; no stdout/stderr output.
fn ciphertext_stream_next(ciphertext: &BigUint, n: &BigUint, exponent: &mut u64) -> BigUint {
    let exp = BigUint::from(*exponent);
    *exponent = exponent.saturating_add(1);
    ciphertext.modpow(&exp, n)
}

/// Builds `r` candidates by combining small primes with generated larger primes.
///
/// # Parameters
/// - `settings`: Candidate generation configuration (uses `small_primes` list).
/// - `rng`: Random number generator for shuffling prime selections.
///
/// # Returns
/// - `Vec<RCandidate>`: List of mutable candidate records.
///
/// # Expected Output
/// - Returns an empty list if not enough primes are available; may read/write reuse files.
pub fn generate_r_candidates_from_small_primes(
    settings: &RCandidateSettings,
    rng: &mut RngChoice,
) -> Vec<RCandidate> {
    let count = settings
        .process_count
        .max(settings.process_min_count)
        .max(1) as usize;
    let target_count = count.max(1);
    let target_bits = settings.target_bit_length;
    let min_small_factors = settings.small_prime_factors_per_candidate.max(1);
    let max_factors = settings
        .max_factors_per_candidate
        .max(min_small_factors + 1);

    let mut collected: Vec<RCandidate> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    let load_reuse = settings.reuse_r_candidates && !settings.reuse_r_candidates_append_only;
    let append_reuse = settings.reuse_r_candidates || settings.reuse_r_candidates_append_only;

    if load_reuse {
        let reuse_path = settings.reuse_r_candidates_path.as_str();
        println!("Reuse enabled; loading r candidates from {}", reuse_path);
        let mut loaded = load_reuse_candidates(reuse_path);
        loaded.shuffle(rng);
        for candidate in loaded {
            if seen.insert(candidate.r.to_string()) {
                collected.push(candidate);
                if collected.len() >= target_count {
                    println!(
                        "Loaded {} r candidates from reuse file {}",
                        collected.len(),
                        reuse_path
                    );
                    return collected.into_iter().take(target_count).collect();
                }
            }
        }
        if !collected.is_empty() {
            println!(
                "Loaded {} r candidates from reuse file {}",
                collected.len(),
                reuse_path
            );
        }
    } else if settings.reuse_r_candidates_append_only {
        println!(
            "Reuse append-only enabled; will append new r candidates to {} but will not load from it",
            settings.reuse_r_candidates_path
        );
    }

    let Some(target_bits) = target_bits else {
        return collected;
    };

    if target_bits == 0 || max_factors <= min_small_factors {
        return collected;
    }

    let min_factor = settings.process_min_factor.clone();
    let mut primes: Vec<BigUint> = settings
        .small_primes
        .iter()
        .filter(|p| *p >= &min_factor)
        .cloned()
        .collect();
    primes.sort();

    if primes.len() < min_small_factors {
        return collected;
    }

    let max_small_prime = primes.last().cloned().unwrap_or_else(|| BigUint::from(2u8));
    let min_large_value = if max_small_prime >= min_factor {
        &max_small_prime + BigUint::one()
    } else {
        min_factor.clone()
    };
    let min_large_bits = min_large_value.bits().max(2);

    let remaining = target_count.saturating_sub(collected.len());
    if remaining == 0 {
        return collected;
    }

    let max_attempts = remaining.saturating_mul(250).max(10);
    let mut seeds = Vec::with_capacity(max_attempts);
    for _ in 0..max_attempts {
        seeds.push(rng.next_u64());
    }

    let found = Arc::new(AtomicUsize::new(0));
    let generated = seeds
        .into_par_iter()
        .filter_map(|seed| {
            if found.load(Ordering::Relaxed) >= remaining {
                return None;
            }
            let mut local_rng = RngChoice::from_seed(rng.mode(), seed);
            let candidate = build_small_primes_candidate(
                target_bits,
                min_large_bits,
                &min_large_value,
                &primes,
                min_small_factors,
                max_factors,
                &mut local_rng,
            )?;
            let prev = found.fetch_add(1, Ordering::Relaxed);
            if prev >= remaining {
                return None;
            }
            Some(candidate)
        })
        .collect::<Vec<_>>();

    let mut new_candidates = Vec::new();
    for entry in generated {
        if collected.len() >= target_count {
            break;
        }
        if seen.insert(entry.r.to_string()) {
            new_candidates.push(entry.clone());
            collected.push(entry);
        }
    }

    if append_reuse && !new_candidates.is_empty() {
        append_reuse_candidates(&settings.reuse_r_candidates_path, &new_candidates);
    }

    collected.truncate(target_count);
    collected
}

const MAX_LARGE_PRIME_ATTEMPTS: usize = 128;
const POLLARD_RHO_PRIMALITY_TIMEOUT_MS: u64 = 25;
const RANDOM_POWER_WINDOW_MIN_UNITS: u64 = 800;
const RANDOM_POWER_WINDOW_MAX_UNITS: u64 = 900;
const RANDOM_POWER_WINDOW_SCALE_UNITS: u64 = 1000;
const RETARGET_EXPONENT_SCALE_UNITS: u64 = 1000;

/// Builds a single r candidate from distinct small primes and generated larger primes.
///
/// # Parameters
/// - `target_bits`: Target bit length derived from the RSA modulus.
/// - `min_large_bits`: Minimum bit length required for large primes.
/// - `min_large_value`: Minimum value for large primes to ensure they exceed small primes.
/// - `small_primes`: Available small prime list (all >= min_factor).
/// - `small_factor_count`: Number of small prime factors to include.
/// - `max_factors`: Maximum total factor count per candidate.
/// - `rng`: Random number generator for selecting primes.
///
/// # Returns
/// - `Option<RCandidate>`: Candidate record or `None` if invalid.
///
/// # Expected Output
/// - Returns `None` when the constraints cannot be met; no side effects.
fn build_small_primes_candidate(
    target_bits: u64,
    min_large_bits: u64,
    min_large_value: &BigUint,
    small_primes: &[BigUint],
    small_factor_count: usize,
    max_factors: usize,
    rng: &mut RngChoice,
) -> Option<RCandidate> {
    if small_factor_count == 0 || max_factors <= small_factor_count {
        return None;
    }

    let mut indices: Vec<usize> = (0..small_primes.len()).collect();
    indices.shuffle(rng);
    let selected = indices
        .into_iter()
        .take(small_factor_count)
        .collect::<Vec<_>>();

    let mut r = BigUint::one();
    let mut factors = Vec::with_capacity(max_factors);
    for idx in selected {
        let p = &small_primes[idx];
        r *= p;
        factors.push((p.clone(), 1));
    }

    let remaining_budget = max_factors - small_factor_count;
    let remaining_bits = target_bits.saturating_sub(r.bits());
    if remaining_bits < min_large_bits {
        return None;
    }

    let max_large_count = remaining_budget
        .min((remaining_bits / min_large_bits) as usize)
        .max(1);
    let large_count = if max_large_count == 1 {
        1
    } else {
        (rng.next_u64() as usize % max_large_count) + 1
    };

    for idx in 0..large_count {
        let remaining_primes = large_count - idx;
        let bits_left = target_bits.saturating_sub(r.bits());
        if bits_left == 0 {
            return None;
        }

        let min_bits_required = min_large_bits * remaining_primes as u64;
        if bits_left < min_bits_required {
            return None;
        }

        let bits_for_prime = if remaining_primes == 1 {
            bits_left
        } else {
            let max_bits_for_prime = bits_left - min_large_bits * (remaining_primes as u64 - 1);
            let span = max_bits_for_prime.saturating_sub(min_large_bits);
            if span == 0 {
                min_large_bits
            } else {
                min_large_bits + (rng.next_u64() % (span + 1))
            }
        };

        let prime =
            sample_large_prime_with_pollard(bits_for_prime, min_large_value, &factors, rng)?;
        r *= &prime;
        if r.bits() > target_bits {
            return None;
        }
        factors.push((prime, 1));
    }

    if factors.len() <= small_factor_count {
        return None;
    }

    factors.sort_by(|a, b| a.0.cmp(&b.0));
    Some(RCandidate::new(r, factors))
}

/// Samples a prime candidate of the requested bit width and validates it with Pollard Rho.
///
/// # Parameters
/// - `bits`: Bit width for the prime candidate.
/// - `min_value`: Minimum acceptable prime value.
/// - `used_factors`: Existing factors to avoid reuse.
/// - `rng`: Random number generator for sampling.
///
/// # Returns
/// - `Option<BigUint>`: A prime of the requested size or `None` on failure.
///
/// # Expected Output
/// - Returns `None` when sampling fails within the attempt budget; no side effects.
fn sample_large_prime_with_pollard(
    bits: u64,
    min_value: &BigUint,
    used_factors: &[(BigUint, u64)],
    rng: &mut RngChoice,
) -> Option<BigUint> {
    if bits < 2 {
        return None;
    }
    let bits_u32 = u32::try_from(bits).ok()?;
    if min_value.bits() > bits {
        return None;
    }

    for _ in 0..MAX_LARGE_PRIME_ATTEMPTS {
        let mut candidate = random_biguint_bits(bits_u32, rng);
        candidate |= BigUint::one();
        if &candidate <= min_value {
            continue;
        }
        if used_factors.iter().any(|(p, _)| p == &candidate) {
            continue;
        }
        if !is_probable_prime_big(&candidate) {
            continue;
        }
        let deadline = Instant::now() + Duration::from_millis(POLLARD_RHO_PRIMALITY_TIMEOUT_MS);
        if pollard_rho(&candidate, rng, deadline).is_none() {
            return Some(candidate);
        }
    }

    None
}

/// Samples a random exponent in the factoring-mode `N^a` window.
///
/// # Parameters
/// - `rng`: Random number generator used for exponent sampling.
///
/// # Returns
/// - `BigDecimal`: Random exponent `a` with `0.8 <= a <= 0.9`.
///
/// # Expected Output
/// - Returns a normalized decimal exponent; no stdout/stderr output.
fn sample_random_power_window_exponent(rng: &mut RngChoice) -> BigDecimal {
    let span = RANDOM_POWER_WINDOW_MAX_UNITS - RANDOM_POWER_WINDOW_MIN_UNITS;
    let units = RANDOM_POWER_WINDOW_MIN_UNITS + (rng.next_u64() % (span + 1));
    (BigDecimal::from(units) / BigDecimal::from(RANDOM_POWER_WINDOW_SCALE_UNITS)).normalized()
}

/// Computes the factoring-mode upper bound used to sample a candidate `r`.
///
/// # Parameters
/// - `n`: RSA modulus used as the size reference.
/// - `scale`: Legacy additive scale used by the default sampler.
/// - `attempt_index`: Zero-based generation attempt index.
/// - `settings`: Candidate generation configuration.
/// - `rng`: Random number generator used for exponent sampling.
///
/// # Returns
/// - `BigUint`: Positive upper bound used for candidate sampling.
///
/// # Expected Output
/// - Returns a positive upper bound; no stdout/stderr output.
fn factoring_candidate_upper_bound(
    n: &BigUint,
    scale: &BigUint,
    attempt_index: usize,
    settings: &RCandidateSettings,
    rng: &mut RngChoice,
) -> BigUint {
    if settings.random_power_window {
        let exponent = sample_random_power_window_exponent(rng);
        let upper = floor_biguint_pow_bigdecimal(n, &exponent);
        if upper.is_zero() {
            BigUint::one()
        } else {
            upper
        }
    } else {
        n + scale + BigUint::from((attempt_index as u64) + 1)
    }
}

/// Samples a retarget total exponent between the configured lower and upper limits.
///
/// # Parameters
/// - `minimum_target_exponent`: Configured lower bound for the sampled total exponent.
/// - `maximum_target_exponent`: Configured upper bound for the sampled total exponent.
/// - `rng`: Random number generator used for exponent sampling.
///
/// # Returns
/// - `BigDecimal`: Sampled total exponent for retargeting.
///
/// # Expected Output
/// - Returns the normalized upper bound when the configured range is invalid; no stdout/stderr output.
fn sample_retarget_total_exponent(
    minimum_target_exponent: &BigDecimal,
    maximum_target_exponent: &BigDecimal,
    rng: &mut RngChoice,
) -> BigDecimal {
    let Some(lower_f64) = minimum_target_exponent.to_f64() else {
        return maximum_target_exponent.normalized();
    };
    let Some(upper_f64) = maximum_target_exponent.to_f64() else {
        return maximum_target_exponent.normalized();
    };
    let lower_units = (lower_f64 * RETARGET_EXPONENT_SCALE_UNITS as f64).floor() as u64;
    let upper_units = (upper_f64 * RETARGET_EXPONENT_SCALE_UNITS as f64).floor() as u64;
    if upper_units <= lower_units {
        return maximum_target_exponent.normalized();
    }

    let units = lower_units + (rng.next_u64() % (upper_units - lower_units + 1));
    (BigDecimal::from(units) / BigDecimal::from(RETARGET_EXPONENT_SCALE_UNITS)).normalized()
}

/// Builds `r` candidates by sampling composites and factoring them.
///
/// # Parameters
/// - `n`: RSA modulus used to scale candidate selection.
/// - `settings`: Candidate generation configuration (including reuse and override options).
/// - `rng`: Random number generator for candidate sampling.
///
/// # Returns
/// - `Vec<RCandidate>`: List of mutable candidate records.
///
/// # Expected Output
/// - Returns a list of candidates meeting factor constraints; may print progress logs.
pub fn generate_r_candidates_via_factoring(
    n: &BigUint,
    settings: &RCandidateSettings,
    rng: &mut RngChoice,
) -> Vec<RCandidate> {
    if let Some(ref override_r) = settings.override_best_r {
        if !override_r.is_zero() {
            if is_probable_prime_big(override_r) {
                return Vec::new();
            }
            let deadline = Instant::now() + Duration::from_secs(10);
            if let Some(factors) = factor_composite_with_timeout(override_r, rng, deadline) {
                if factors.len() >= 3
                    && factors
                        .iter()
                        .all(|(p, _)| p >= &settings.process_min_factor)
                {
                    return vec![RCandidate::new(override_r.clone(), factors)];
                }
            }
        }
    }

    let min_factor = settings.process_min_factor.clone();
    let scale = BigUint::one() << settings.process_scale;
    let count = settings
        .process_count
        .max(settings.process_min_count)
        .max(1);
    let target_count = count as usize;

    let mut collected: Vec<RCandidate> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let reserved = Arc::new(Mutex::new(HashSet::<String>::new()));

    let load_reuse = settings.reuse_r_candidates && !settings.reuse_r_candidates_append_only;
    let append_reuse = settings.reuse_r_candidates || settings.reuse_r_candidates_append_only;

    if load_reuse {
        let reuse_path = settings.reuse_r_candidates_path.as_str();
        println!("Reuse enabled; loading r candidates from {}", reuse_path);
        let mut loaded = load_reuse_candidates(reuse_path);
        loaded.shuffle(rng);
        for candidate in loaded {
            let key = candidate.r.to_string();
            if seen.insert(key.clone()) {
                if let Ok(mut guard) = reserved.lock() {
                    guard.insert(key);
                }
                collected.push(candidate);
                if collected.len() >= target_count {
                    println!(
                        "Loaded {} r candidates from reuse file {}",
                        collected.len(),
                        reuse_path
                    );
                    return collected.into_iter().take(target_count).collect();
                }
            }
        }
        if !collected.is_empty() {
            println!(
                "Loaded {} r candidates from reuse file {}",
                collected.len(),
                reuse_path
            );
        }
    } else if settings.reuse_r_candidates_append_only {
        println!(
            "Reuse append-only enabled; will append new r candidates to {} but will not load from it",
            settings.reuse_r_candidates_path
        );
    }
    if settings.random_power_window {
        println!(
            "Factoring-mode r candidates will sample from a random N^a window with a in [0.8, 0.9]"
        );
    }

    let found = Arc::new(AtomicUsize::new(collected.len()));

    let max_attempts = count.saturating_mul(1000);
    let mut seeds = Vec::with_capacity(max_attempts as usize);
    for _ in 0..max_attempts {
        seeds.push(rng.next_u64());
    }

    println!("Generating r candidates... {} attempts", seeds.len());

    let generated = seeds
        .into_par_iter()
        .enumerate()
        .filter_map(|(idx, seed)| {
            if found.load(Ordering::Relaxed) >= target_count {
                return None;
            }

            let mut local_rng = RngChoice::from_seed(rng.mode(), seed);
            let upper =
                factoring_candidate_upper_bound(n, &scale, idx, settings, &mut local_rng);
            let candidate = random_biguint_below(&upper, &mut local_rng) + BigUint::one();
            let candidate_key = candidate.to_string();
            let Ok(mut reserved_guard) = reserved.lock() else {
                return None;
            };
            if !reserved_guard.insert(candidate_key) {
                return None;
            }
            drop(reserved_guard);
            if is_probable_prime_big(&candidate) {
                println!("Skipping prime r candidate: {}", candidate);
                return None;
            }
            let deadline = Instant::now() + Duration::from_millis(5000);
            let Some(factors) = factor_composite_with_timeout(&candidate, &mut local_rng, deadline)
            else {
                return None;
            };
            if factors.len() < 3 {
                return None;
            }
            if factors.iter().any(|(p, _)| p < &min_factor) {
                return None;
            }

            let prev = found.fetch_add(1, Ordering::Relaxed);
            if prev >= target_count {
                return None;
            }

            println!(
                "Generated r candidate: {}, factors {:?}",
                candidate, factors
            );
            Some(RCandidate::new(candidate, factors))
        })
        .collect::<Vec<_>>();

    let mut new_candidates = Vec::new();
    for candidate in generated {
        if seen.insert(candidate.r.to_string()) {
            new_candidates.push(candidate);
        }
    }

    collected.extend(new_candidates.iter().cloned());
    collected.truncate(target_count);

    if append_reuse && !new_candidates.is_empty() {
        append_reuse_candidates(&settings.reuse_r_candidates_path, &new_candidates);
    }

    collected
}

/// Loads previously generated `r` candidates from a CSV file.
///
/// # Parameters
/// - `path`: Path to the reuse CSV file.
///
/// # Returns
/// - `Vec<RCandidate>`: Parsed candidate records.
///
/// # Expected Output
/// - Returns an empty list on missing/invalid files; may print parsing errors.
fn load_reuse_candidates(path: &str) -> Vec<RCandidate> {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(err) => {
            if err.kind() != std::io::ErrorKind::NotFound {
                println!("Failed to open reuse file {}: {}", path, err);
            }
            return Vec::new();
        }
    };

    let reader = BufReader::new(file);
    let mut entries = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(err) => {
                println!(
                    "Skipping line {} in reuse file due to read error: {}",
                    idx + 1,
                    err
                );
                continue;
            }
        };

        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.splitn(2, ',');
        let r_str = parts.next().unwrap_or("").trim();
        let factors_str = parts.next().unwrap_or("").trim();

        if r_str.is_empty() || factors_str.is_empty() {
            println!(
                "Skipping line {} in reuse file: missing r or factors entry",
                idx + 1
            );
            continue;
        }

        let r = match r_str.parse::<BigUint>() {
            Ok(val) => val,
            Err(err) => {
                println!(
                    "Skipping line {} in reuse file: invalid r '{}': {}",
                    idx + 1,
                    r_str,
                    err
                );
                continue;
            }
        };

        let Some(factors) = parse_factors_csv(factors_str) else {
            println!(
                "Skipping line {} in reuse file: invalid factors '{}': expected p^e;...",
                idx + 1,
                factors_str
            );
            continue;
        };

        entries.push(RCandidate::new(r, factors));
    }

    entries
}

/// Appends newly generated `r` candidates to a reuse CSV file.
///
/// # Parameters
/// - `path`: Path to the reuse CSV file.
/// - `entries`: Candidate entries to append.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Appends lines to the file when possible; may print I/O errors.
fn append_reuse_candidates(path: &str, entries: &[RCandidate]) {
    if entries.is_empty() {
        return;
    }

    let mut file = match OpenOptions::new().create(true).append(true).open(path) {
        Ok(f) => f,
        Err(err) => {
            println!("Failed to append reuse file {}: {}", path, err);
            return;
        }
    };

    for candidate in entries {
        let factors_str = format_factors_csv(&candidate.factors);
        if let Err(err) = writeln!(file, "{},{}", candidate.r, factors_str) {
            println!(
                "Failed to write r candidate {} to {}: {}",
                candidate.r, path, err
            );
            break;
        }
    }
}

/// Retargets candidates using random decimal exponent partitions.
///
/// # Parameters
/// - `n`: Original RSA modulus used as the base for `N^a`, `N^b`, and `N^c`.
/// - `candidates`: Mutable candidate list to rewrite in place.
/// - `minimum_target_exponent`: Configured lower bound for the sampled total exponent budget.
/// - `target_exponent`: Configured upper bound for the sampled total exponent budget.
/// - `partition_count`: Maximum number of exponent partitions to generate per candidate.
/// - `minimum_component_exponent`: Minimum exponent allowed for each retargeted partition.
/// - `rng`: Random number generator used for the exponent partitioning.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Rewrites each candidate's modulus and factors in place; no stdout/stderr output.
pub fn retarget_r_candidates_for_speculative_oracles(
    n: &BigUint,
    candidates: &mut [RCandidate],
    minimum_target_exponent: &BigDecimal,
    target_exponent: &BigDecimal,
    partition_count: usize,
    minimum_component_exponent: &BigDecimal,
    rng: &mut RngChoice,
) {
    if candidates.is_empty() || partition_count == 0 {
        return;
    }

    for candidate in candidates {
        let sampled_target_exponent =
            sample_retarget_total_exponent(minimum_target_exponent, target_exponent, rng);
        let parts = random_bigdecimal_partition_with_min(
            &sampled_target_exponent,
            partition_count,
            minimum_component_exponent,
            rng,
        );
        if parts.is_empty() {
            continue;
        }

        let mut factors = Vec::with_capacity(parts.len());
        for part in parts {
            let prime = next_prime_from_biguint_pow_bigdecimal(n, &part);
            factors.push((prime, 1));
        }
        let factors = coalesce_factors(factors);
        let r = factors
            .iter()
            .fold(BigUint::one(), |acc, (p, e)| acc * p.pow(*e as u32));

        candidate.r = r;
        candidate.factors = factors;
        candidate.target_exponent = sampled_target_exponent;
    }
}

/// Parses a `p^e;...` factor list from CSV form.
///
/// # Parameters
/// - `raw`: Raw CSV factors string (e.g., `"3^1;5^2"`).
///
/// # Returns
/// - `Option<Vec<(BigUint, u64)>>`: Parsed factor list or `None` if invalid.
///
/// # Expected Output
/// - Returns `None` on parse errors or empty input; no side effects.
fn parse_factors_csv(raw: &str) -> Option<Vec<(BigUint, u64)>> {
    let mut factors = Vec::new();

    for entry in raw.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let mut parts = entry.split('^');
        let p_str = parts.next()?;
        let e_str = parts.next().unwrap_or("1");

        let p = p_str.parse::<BigUint>().ok()?;
        let e = e_str.parse::<u64>().ok()?;
        factors.push((p, e));
    }

    if factors.is_empty() {
        None
    } else {
        Some(factors)
    }
}

/// Formats a factor list as `p^e;...` for CSV output.
///
/// # Parameters
/// - `factors`: Factor list to format.
///
/// # Returns
/// - `String`: CSV-friendly factor string (empty for no factors).
///
/// # Expected Output
/// - Returns a formatted string; no side effects.
fn format_factors_csv(factors: &[(BigUint, u64)]) -> String {
    factors
        .iter()
        .map(|(p, e)| format!("{}^{}", p, e))
        .collect::<Vec<_>>()
        .join(";")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{RngChoice, RngMode};
    use rand::RngCore;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let mut rng = RngChoice::from_entropy(RngMode::Crypto).expect("rng entropy");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "rsademo_{}_{}_{}.csv",
            name,
            std::process::id(),
            rng.next_u64()
        ));
        path
    }

    #[test]
    fn test_parse_factors_csv_valid() {
        let raw = "3^1;5^2";
        let factors = parse_factors_csv(raw).expect("missing factors");
        assert_eq!(factors.len(), 2);
        assert_eq!(factors[0].0, BigUint::from(3u8));
        assert_eq!(factors[0].1, 1);
        assert_eq!(factors[1].0, BigUint::from(5u8));
        assert_eq!(factors[1].1, 2);
    }

    #[test]
    fn test_parse_factors_csv_invalid() {
        let raw = "not_a_number";
        assert!(parse_factors_csv(raw).is_none());
    }

    #[test]
    fn test_format_factors_csv_basic() {
        let factors = vec![(BigUint::from(3u8), 1), (BigUint::from(5u8), 2)];
        let formatted = format_factors_csv(&factors);
        assert_eq!(formatted, "3^1;5^2");
    }

    #[test]
    fn test_format_factors_csv_empty() {
        let formatted = format_factors_csv(&[]);
        assert_eq!(formatted, "");
    }

    #[test]
    fn test_load_reuse_candidates_missing_file() {
        let missing = temp_path("missing");
        let entries = load_reuse_candidates(missing.to_str().unwrap());
        assert!(entries.is_empty());
    }

    #[test]
    fn test_load_reuse_candidates_parses() {
        let path = temp_path("load");
        let content = "# header\n105,3^1;5^1;7^1\n";
        fs::write(&path, content).expect("write failed");
        let entries = load_reuse_candidates(path.to_str().unwrap());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].r, BigUint::from(105u8));
        assert_eq!(entries[0].factors.len(), 3);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_append_reuse_candidates_writes() {
        let path = temp_path("append");
        let entries = vec![RCandidate::new(
            BigUint::from(105u8),
            vec![
                (BigUint::from(3u8), 1),
                (BigUint::from(5u8), 1),
                (BigUint::from(7u8), 1),
            ],
        )];
        append_reuse_candidates(path.to_str().unwrap(), &entries);
        let data = fs::read_to_string(&path).expect("read failed");
        assert!(data.contains("105,3^1;5^1;7^1"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_append_reuse_candidates_empty() {
        let path = temp_path("append_empty");
        append_reuse_candidates(path.to_str().unwrap(), &[]);
        assert!(!path.exists());
    }

    #[test]
    fn test_generate_r_candidates_from_small_primes() {
        let settings = RCandidateSettings {
            mode: RCandidateMode::SmallPrimes,
            override_best_r: None,
            process_min_factor: BigUint::from(3u8),
            process_count: 2,
            process_min_count: 1,
            process_scale: 8,
            reuse_r_candidates_path: "".to_string(),
            reuse_r_candidates: false,
            reuse_r_candidates_append_only: false,
            small_primes: vec![3u8, 5u8, 7u8, 11u8]
                .into_iter()
                .map(BigUint::from)
                .collect(),
            small_prime_factors_per_candidate: 3,
            max_factors_per_candidate: 5,
            target_bit_length: Some(16),
            random_power_window: false,
            target_exponent_minimum: BigDecimal::parse_bytes(b"0.8", 10)
                .expect("valid exponent"),
            target_exponent: BigDecimal::parse_bytes(b"2.005", 10).expect("valid exponent"),
            retarget_partition_count: 3,
            retarget_minimum_exponent: BigDecimal::parse_bytes(b"0.45", 10)
                .expect("valid minimum exponent"),
        };
        let mut rng = RngChoice::from_seed(RngMode::Standard, 42);
        let candidates = generate_r_candidates_from_small_primes(&settings, &mut rng);
        assert!(!candidates.is_empty());
        let candidate = &candidates[0];
        let product = candidate
            .factors
            .iter()
            .fold(BigUint::one(), |acc, (p, e)| acc * p.pow(*e as u32));
        assert_eq!(product, candidate.r);
        assert!(candidate.factors.len() >= settings.small_prime_factors_per_candidate + 1);
        let max_small = settings
            .small_primes
            .iter()
            .max()
            .cloned()
            .unwrap_or_else(|| BigUint::from(2u8));
        assert!(candidate.factors.iter().any(|(p, _)| p > &max_small));
    }

    #[test]
    fn test_generate_r_candidates_from_small_primes_empty() {
        let settings = RCandidateSettings {
            mode: RCandidateMode::SmallPrimes,
            override_best_r: None,
            process_min_factor: BigUint::from(3u8),
            process_count: 1,
            process_min_count: 1,
            process_scale: 8,
            reuse_r_candidates_path: "".to_string(),
            reuse_r_candidates: false,
            reuse_r_candidates_append_only: false,
            small_primes: vec![3u8, 5u8].into_iter().map(BigUint::from).collect(),
            small_prime_factors_per_candidate: 3,
            max_factors_per_candidate: 4,
            target_bit_length: Some(12),
            random_power_window: false,
            target_exponent_minimum: BigDecimal::parse_bytes(b"0.8", 10)
                .expect("valid exponent"),
            target_exponent: BigDecimal::parse_bytes(b"2.005", 10).expect("valid exponent"),
            retarget_partition_count: 3,
            retarget_minimum_exponent: BigDecimal::parse_bytes(b"0.45", 10)
                .expect("valid minimum exponent"),
        };
        let mut rng = RngChoice::from_seed(RngMode::Standard, 43);
        let candidates = generate_r_candidates_from_small_primes(&settings, &mut rng);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_generate_r_candidates_small_primes_mode() {
        let settings = RCandidateSettings {
            mode: RCandidateMode::SmallPrimes,
            override_best_r: None,
            process_min_factor: BigUint::from(3u8),
            process_count: 1,
            process_min_count: 1,
            process_scale: 8,
            reuse_r_candidates_path: "".to_string(),
            reuse_r_candidates: false,
            reuse_r_candidates_append_only: false,
            small_primes: vec![3u8, 5u8, 7u8].into_iter().map(BigUint::from).collect(),
            small_prime_factors_per_candidate: 3,
            max_factors_per_candidate: 4,
            target_bit_length: Some(14),
            random_power_window: false,
            target_exponent_minimum: BigDecimal::parse_bytes(b"0.8", 10)
                .expect("valid exponent"),
            target_exponent: BigDecimal::parse_bytes(b"2.005", 10).expect("valid exponent"),
            retarget_partition_count: 3,
            retarget_minimum_exponent: BigDecimal::parse_bytes(b"0.45", 10)
                .expect("valid minimum exponent"),
        };
        let mut rng = RngChoice::from_seed(RngMode::Standard, 44);
        let candidates = generate_r_candidates(&BigUint::from(100u8), &settings, &mut rng);
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_generate_r_candidates_factoring_mode_dispatch() {
        let settings = RCandidateSettings {
            mode: RCandidateMode::Factoring,
            override_best_r: Some(BigUint::from(105u8)),
            process_min_factor: BigUint::from(3u8),
            process_count: 1,
            process_min_count: 1,
            process_scale: 8,
            reuse_r_candidates_path: "".to_string(),
            reuse_r_candidates: false,
            reuse_r_candidates_append_only: false,
            small_primes: Vec::new(),
            small_prime_factors_per_candidate: 3,
            max_factors_per_candidate: 6,
            target_bit_length: None,
            random_power_window: false,
            target_exponent_minimum: BigDecimal::parse_bytes(b"0.8", 10)
                .expect("valid exponent"),
            target_exponent: BigDecimal::parse_bytes(b"2.005", 10).expect("valid exponent"),
            retarget_partition_count: 3,
            retarget_minimum_exponent: BigDecimal::parse_bytes(b"0.45", 10)
                .expect("valid minimum exponent"),
        };
        let mut rng = RngChoice::from_seed(RngMode::Standard, 46);
        let candidates = generate_r_candidates(&BigUint::from(100u8), &settings, &mut rng);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].r, BigUint::from(105u8));
    }

    #[test]
    fn test_generate_r_candidates_override_factoring() {
        let settings = RCandidateSettings {
            mode: RCandidateMode::Factoring,
            override_best_r: Some(BigUint::from(105u8)),
            process_min_factor: BigUint::from(3u8),
            process_count: 1,
            process_min_count: 1,
            process_scale: 8,
            reuse_r_candidates_path: "".to_string(),
            reuse_r_candidates: false,
            reuse_r_candidates_append_only: false,
            small_primes: Vec::new(),
            small_prime_factors_per_candidate: 3,
            max_factors_per_candidate: 6,
            target_bit_length: None,
            random_power_window: false,
            target_exponent_minimum: BigDecimal::parse_bytes(b"0.8", 10)
                .expect("valid exponent"),
            target_exponent: BigDecimal::parse_bytes(b"2.005", 10).expect("valid exponent"),
            retarget_partition_count: 3,
            retarget_minimum_exponent: BigDecimal::parse_bytes(b"0.45", 10)
                .expect("valid minimum exponent"),
        };
        let mut rng = RngChoice::from_seed(RngMode::Standard, 45);
        let candidates =
            generate_r_candidates_via_factoring(&BigUint::from(100u8), &settings, &mut rng);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].r, BigUint::from(105u8));
        assert!(candidates[0].factors.len() >= 3);
    }

    #[test]
    fn test_generate_r_candidates_via_factoring_rejects_prime_override() {
        let settings = RCandidateSettings {
            mode: RCandidateMode::Factoring,
            override_best_r: Some(BigUint::from(101u8)),
            process_min_factor: BigUint::from(3u8),
            process_count: 1,
            process_min_count: 1,
            process_scale: 8,
            reuse_r_candidates_path: "".to_string(),
            reuse_r_candidates: false,
            reuse_r_candidates_append_only: false,
            small_primes: Vec::new(),
            small_prime_factors_per_candidate: 3,
            max_factors_per_candidate: 6,
            target_bit_length: None,
            random_power_window: false,
            target_exponent_minimum: BigDecimal::parse_bytes(b"0.8", 10)
                .expect("valid exponent"),
            target_exponent: BigDecimal::parse_bytes(b"2.005", 10).expect("valid exponent"),
            retarget_partition_count: 3,
            retarget_minimum_exponent: BigDecimal::parse_bytes(b"0.45", 10)
                .expect("valid minimum exponent"),
        };
        let mut rng = RngChoice::from_seed(RngMode::Standard, 47);
        let candidates =
            generate_r_candidates_via_factoring(&BigUint::from(100u8), &settings, &mut rng);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_ciphertext_stream_sequence() {
        let n = BigUint::from(97u32);
        let c = BigUint::from(5u32);
        let candidates = generate_r_candidates_from_ciphertext_stream(&c, &n, 3, 1);
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].r, BigUint::from(5u32));
        assert_eq!(candidates[1].r, BigUint::from(25u32));
        assert_eq!(candidates[2].r, BigUint::from(28u32));
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.factors.is_empty())
        );
    }

    #[test]
    fn test_ciphertext_stream_exponent_increments() {
        let n = BigUint::from(101u32);
        let c = BigUint::from(3u32);
        let mut exponent = 0u64;
        let first = ciphertext_stream_next(&c, &n, &mut exponent);
        let second = ciphertext_stream_next(&c, &n, &mut exponent);
        assert_eq!(exponent, 2);
        assert_eq!(first, BigUint::one());
        assert_eq!(second, BigUint::from(3u32));
    }

    #[test]
    fn test_ciphertext_stream_empty_on_zero_modulus() {
        let n = BigUint::zero();
        let c = BigUint::from(7u32);
        let candidates = generate_r_candidates_from_ciphertext_stream(&c, &n, 2, 0);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_sample_random_power_window_exponent_stays_in_range() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 49);
        let minimum = BigDecimal::parse_bytes(b"0.8", 10).expect("valid minimum");
        let maximum = BigDecimal::parse_bytes(b"0.9", 10).expect("valid maximum");

        for _ in 0..64 {
            let exponent = sample_random_power_window_exponent(&mut rng);
            assert!(exponent >= minimum);
            assert!(exponent <= maximum);
        }
    }

    #[test]
    fn test_factoring_candidate_upper_bound_uses_power_window_when_enabled() {
        let settings = RCandidateSettings {
            mode: RCandidateMode::Factoring,
            override_best_r: None,
            process_min_factor: BigUint::from(3u8),
            process_count: 1,
            process_min_count: 1,
            process_scale: 8,
            reuse_r_candidates_path: "".to_string(),
            reuse_r_candidates: false,
            reuse_r_candidates_append_only: false,
            small_primes: Vec::new(),
            small_prime_factors_per_candidate: 3,
            max_factors_per_candidate: 6,
            target_bit_length: None,
            random_power_window: true,
            target_exponent_minimum: BigDecimal::parse_bytes(b"0.8", 10)
                .expect("valid exponent"),
            target_exponent: BigDecimal::parse_bytes(b"2.005", 10).expect("valid exponent"),
            retarget_partition_count: 3,
            retarget_minimum_exponent: BigDecimal::parse_bytes(b"0.45", 10)
                .expect("valid minimum exponent"),
        };
        let n = BigUint::from(1_000_000u64);
        let minimum = floor_biguint_pow_bigdecimal(
            &n,
            &BigDecimal::parse_bytes(b"0.8", 10).expect("valid minimum"),
        );
        let maximum = floor_biguint_pow_bigdecimal(
            &n,
            &BigDecimal::parse_bytes(b"0.9", 10).expect("valid maximum"),
        );
        let mut rng = RngChoice::from_seed(RngMode::Standard, 50);

        let upper =
            factoring_candidate_upper_bound(&n, &BigUint::from(256u16), 0, &settings, &mut rng);

        assert!(upper >= minimum);
        assert!(upper <= maximum);
    }

    #[test]
    fn test_sample_retarget_total_exponent_stays_in_range() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 51);
        let upper = BigDecimal::parse_bytes(b"0.9", 10).expect("valid upper");
        let lower = BigDecimal::parse_bytes(b"0.8", 10).expect("valid lower");

        for _ in 0..64 {
            let sampled = sample_retarget_total_exponent(&lower, &upper, &mut rng);
            assert!(sampled >= lower);
            assert!(sampled <= upper);
        }
    }

    #[test]
    fn test_retarget_r_candidates_for_speculative_oracles() {
        let mut candidates = vec![RCandidate::new(BigUint::from(21u8), vec![])];
        let mut rng = RngChoice::from_seed(RngMode::Standard, 48);
        retarget_r_candidates_for_speculative_oracles(
            &BigUint::from(1000u16),
            &mut candidates,
            &BigDecimal::parse_bytes(b"0.8", 10).expect("valid lower exponent"),
            &BigDecimal::parse_bytes(b"0.9", 10).expect("valid exponent"),
            3,
            &BigDecimal::parse_bytes(b"0.45", 10).expect("valid minimum exponent"),
            &mut rng,
        );

        let lower = BigDecimal::parse_bytes(b"0.8", 10).expect("valid lower");
        let upper = BigDecimal::parse_bytes(b"0.9", 10).expect("valid upper");
        assert!(candidates[0].target_exponent >= lower);
        assert!(candidates[0].target_exponent <= upper);
        let product = candidates[0]
            .factors
            .iter()
            .fold(BigUint::one(), |acc, (p, e)| acc * p.pow(*e as u32));
        assert_eq!(product, candidates[0].r);
        assert!(!candidates[0].factors.is_empty());
        assert!(candidates[0].factors.len() <= 3);
        assert!(
            candidates[0]
                .factors
                .iter()
                .all(|(p, _)| is_probable_prime_big(p))
        );
    }
}
