use crate::math::{factor_composite_with_timeout, is_probable_prime_big, random_biguint_below};
use num_bigint::BigUint;
use num_traits::{One, Zero};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

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
}

pub fn generate_r_candidates(
    n: &BigUint,
    settings: &RCandidateSettings,
    rng: &mut StdRng,
) -> Vec<(BigUint, Vec<(BigUint, u64)>)> {
    match settings.mode {
        RCandidateMode::Factoring => generate_r_candidates_via_factoring(n, settings, rng),
        RCandidateMode::SmallPrimes => generate_r_candidates_from_small_primes(settings, rng),
    }
}

pub fn generate_r_candidates_batch(
    n: &BigUint,
    settings: &RCandidateSettings,
    rng: &mut StdRng,
    batch_size: usize,
) -> Vec<(BigUint, Vec<(BigUint, u64)>)> {
    let target = batch_size.max(1) as u64;
    let mut batch_settings = settings.clone();
    batch_settings.process_count = target;
    batch_settings.process_min_count = target;
    generate_r_candidates(n, &batch_settings, rng)
}

pub fn generate_r_candidates_from_small_primes(
    settings: &RCandidateSettings,
    rng: &mut StdRng,
) -> Vec<(BigUint, Vec<(BigUint, u64)>)> {
    let count = settings.process_count.max(settings.process_min_count).max(1) as usize;
    let target_count = count.max(1);
    let min_factors = settings.small_prime_factors_per_candidate.max(3);

    if settings.small_primes.len() < min_factors {
        return Vec::new();
    }

    let mut collected: Vec<(BigUint, Vec<(BigUint, u64)>)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    let load_reuse = settings.reuse_r_candidates && !settings.reuse_r_candidates_append_only;
    let append_reuse = settings.reuse_r_candidates || settings.reuse_r_candidates_append_only;

    if load_reuse {
        let reuse_path = settings.reuse_r_candidates_path.as_str();
        println!("Reuse enabled; loading r candidates from {}", reuse_path);
        let mut loaded = load_reuse_candidates(reuse_path);
        loaded.shuffle(rng);
        for (r, factors) in loaded {
            if seen.insert(r.to_string()) {
                collected.push((r, factors));
                if collected.len() >= target_count {
                    println!(
                        "Loaded {} r candidates from reuse file {}",
                        collected.len(), reuse_path
                    );
                    return collected.into_iter().take(target_count).collect();
                }
            }
        }
        if !collected.is_empty() {
            println!(
                "Loaded {} r candidates from reuse file {}",
                collected.len(), reuse_path
            );
        }
    } else if settings.reuse_r_candidates_append_only {
        println!(
            "Reuse append-only enabled; will append new r candidates to {} but will not load from it",
            settings.reuse_r_candidates_path
        );
    }

    let remaining = target_count.saturating_sub(collected.len());
    if remaining == 0 {
        return collected;
    }
    let max_attempts = remaining.saturating_mul(50).max(10);
    let min_factor = settings.process_min_factor.clone();
    let primes = &settings.small_primes;
    let mut seeds = Vec::with_capacity(max_attempts);
    for _ in 0..max_attempts {
        seeds.push(rng.next_u64());
    }

    let generated = seeds
        .into_par_iter()
        .filter_map(|seed| {
            let mut local_rng = StdRng::seed_from_u64(seed);
            let mut indices: Vec<usize> = (0..primes.len()).collect();
            indices.shuffle(&mut local_rng);
            let selected = indices.into_iter().take(min_factors).collect::<Vec<_>>();

            let mut r = BigUint::one();
            let mut factors = Vec::with_capacity(selected.len());
            for idx in selected {
                let p = &primes[idx];
                if p < &min_factor {
                    return None;
                }
                r *= p;
                factors.push((p.clone(), 1));
            }
            factors.sort_by(|a, b| a.0.cmp(&b.0));
            if factors.len() < 3 {
                return None;
            }
            Some((r, factors))
        })
        .collect::<Vec<_>>();

    let mut new_candidates = Vec::new();
    for entry in generated {
        if collected.len() >= target_count {
            break;
        }
        if seen.insert(entry.0.to_string()) {
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

pub fn generate_r_candidates_via_factoring(
    n: &BigUint,
    settings: &RCandidateSettings,
    rng: &mut StdRng,
) -> Vec<(BigUint, Vec<(BigUint, u64)>)> {
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
                    return vec![(override_r.clone(), factors)];
                }
            }
        }
    }

    let min_factor = settings.process_min_factor.clone();
    let scale = BigUint::one() << settings.process_scale;
    let count = settings.process_count.max(settings.process_min_count).max(1);
    let target_count = count as usize;

    let mut collected: Vec<(BigUint, Vec<(BigUint, u64)>)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    let load_reuse = settings.reuse_r_candidates && !settings.reuse_r_candidates_append_only;
    let append_reuse = settings.reuse_r_candidates || settings.reuse_r_candidates_append_only;

    if load_reuse {
        let reuse_path = settings.reuse_r_candidates_path.as_str();
        println!("Reuse enabled; loading r candidates from {}", reuse_path);
        let mut loaded = load_reuse_candidates(reuse_path);
        loaded.shuffle(rng);
        for (r, factors) in loaded {
            if seen.insert(r.to_string()) {
                collected.push((r, factors));
                if collected.len() >= target_count {
                    println!(
                        "Loaded {} r candidates from reuse file {}",
                        collected.len(), reuse_path
                    );
                    return collected.into_iter().take(target_count).collect();
                }
            }
        }
        if !collected.is_empty() {
            println!(
                "Loaded {} r candidates from reuse file {}",
                collected.len(), reuse_path
            );
        }
    } else if settings.reuse_r_candidates_append_only {
        println!(
            "Reuse append-only enabled; will append new r candidates to {} but will not load from it",
            settings.reuse_r_candidates_path
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

            let mut local_rng = StdRng::seed_from_u64(seed);
            let upper = n + &scale + BigUint::from((idx as u64) + 1);
            let candidate = random_biguint_below(&upper, &mut local_rng) + BigUint::one();
            if is_probable_prime_big(&candidate) {
                println!("Skipping prime r candidate: {}", candidate);
                return None;
            }
            let deadline = Instant::now() + Duration::from_millis(5000);
            let Some(factors) = factor_composite_with_timeout(&candidate, &mut local_rng, deadline) else {
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

            println!("Generated r candidate: {}, factors {:?}", candidate, factors);
            Some((candidate, factors))
        })
        .collect::<Vec<_>>();

    let mut new_candidates = Vec::new();
    for (r, factors) in generated {
        if seen.insert(r.to_string()) {
            new_candidates.push((r, factors));
        }
    }

    collected.extend(new_candidates.iter().cloned());
    collected.truncate(target_count);

    if append_reuse && !new_candidates.is_empty() {
        append_reuse_candidates(&settings.reuse_r_candidates_path, &new_candidates);
    }

    collected
}

fn load_reuse_candidates(path: &str) -> Vec<(BigUint, Vec<(BigUint, u64)>)> {
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
                println!("Skipping line {} in reuse file due to read error: {}", idx + 1, err);
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
                println!("Skipping line {} in reuse file: invalid r '{}': {}", idx + 1, r_str, err);
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

        entries.push((r, factors));
    }

    entries
}

fn append_reuse_candidates(path: &str, entries: &[(BigUint, Vec<(BigUint, u64)>)]) {
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

    for (r, factors) in entries {
        let factors_str = format_factors_csv(factors);
        if let Err(err) = writeln!(file, "{},{}", r, factors_str) {
            println!("Failed to write r candidate {} to {}: {}", r, path, err);
            break;
        }
    }
}

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
    use rand::SeedableRng;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("rsademo_{}_{}_{}.csv", name, std::process::id(), rand::random::<u64>()));
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
        assert_eq!(entries[0].0, BigUint::from(105u8));
        assert_eq!(entries[0].1.len(), 3);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_append_reuse_candidates_writes() {
        let path = temp_path("append");
        let entries = vec![(
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
            small_primes: vec![3u8, 5u8, 7u8, 11u8].into_iter().map(BigUint::from).collect(),
            small_prime_factors_per_candidate: 3,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let candidates = generate_r_candidates_from_small_primes(&settings, &mut rng);
        assert!(!candidates.is_empty());
        let (r, factors) = &candidates[0];
        let product = factors
            .iter()
            .fold(BigUint::one(), |acc, (p, e)| acc * p.pow(*e as u32));
        assert_eq!(&product, r);
        assert!(factors.len() >= 3);
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
        };
        let mut rng = StdRng::seed_from_u64(43);
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
        };
        let mut rng = StdRng::seed_from_u64(44);
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
        };
        let mut rng = StdRng::seed_from_u64(46);
        let candidates = generate_r_candidates(&BigUint::from(100u8), &settings, &mut rng);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, BigUint::from(105u8));
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
        };
        let mut rng = StdRng::seed_from_u64(45);
        let candidates = generate_r_candidates_via_factoring(&BigUint::from(100u8), &settings, &mut rng);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, BigUint::from(105u8));
        assert!(candidates[0].1.len() >= 3);
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
        };
        let mut rng = StdRng::seed_from_u64(47);
        let candidates = generate_r_candidates_via_factoring(&BigUint::from(100u8), &settings, &mut rng);
        assert!(candidates.is_empty());
    }
}
