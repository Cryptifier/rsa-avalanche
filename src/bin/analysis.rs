/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>

use std::{
    collections::HashSet,
    error::Error,
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::Path,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use rayon::prelude::*;

use clap::Parser;
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Signed, Zero};
use rand::rngs::StdRng;
use rand::{seq::SliceRandom, Rng, RngCore, SeedableRng};
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(name = "analysis", about = "Lightweight RSA round-trip demo", author, version)]
struct Args {
    /// Bit-length of the primes to generate (kept small for a quick demo)
    #[arg(short, long, default_value_t = 56, value_parser = clap::value_parser!(u32).range(16..=63))]
    bits: u32,

    /// Plaintext message to encrypt and decrypt (overrides config if set)
    #[arg(short, long)]
    message: Option<String>,

    /// Public exponent e (must remain odd)
    #[arg(short = 'e', long, default_value_t = 65_537u64)]
    public_exponent: u64,

    /// Optional deterministic seed for reproducible key generation
    #[arg(long)]
    seed: Option<u64>,

    /// Path to a JSON config matching the original rsa_demo.sage schema
    #[arg(short = 'c', long, default_value = "rsa_config.json")]
    config: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let config = load_config(&args.config)?;
    run_demo(args, config)
}

fn run_demo(args: Args, config: Config) -> Result<(), Box<dyn Error>> {
    let mut rng: StdRng = match args.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_rng(rand::thread_rng())?,
    };

    let (p, q): (BigUint, BigUint) = if config.key.generate {
        let p = random_prime_with_bits(args.bits, &mut rng);
        let mut q = random_prime_with_bits(args.bits, &mut rng);
        while q == p {
            q = random_prime_with_bits(args.bits, &mut rng);
        }
        (BigUint::from(p), BigUint::from(q))
    } else {
        let p = config
            .key
            .p
            .clone()
            .ok_or("config.key.p must be set when generate is false")?;
        let q = config
            .key
            .q
            .clone()
            .ok_or("config.key.q must be set when generate is false")?;
        (p, q)
    };

    let one = BigUint::one();
    let n = &p * &q;
    let phi = (&p - &one) * (&q - &one);

    let start_e = if args.public_exponent != 65_537 {
        args.public_exponent
    } else {
        config.key.e
    };
    let e = choose_exponent(start_e, &phi);
    let d = mod_inverse(&e, &phi)
        .ok_or("public exponent is not invertible; try a different size or exponent")?;

    let message = select_message(args.message.clone(), &config.engine, &mut rng);
    if message.is_zero() {
        return Err("message cannot be empty".into());
    }
    if message >= n {
        return Err("message must be smaller than the modulus n".into());
    }

    let ciphertext = message.modpow(&e, &n);
    //let recovered = ciphertext.modpow(&d, &n);
    //if recovered != message {
    //    return Err("RSA round trip failed".into());
    //}

    println!("Prime p ({} bits): {p}", bit_length(&p));
    println!("Prime q ({} bits): {q}", bit_length(&q));
    println!("Modulus n ({} bits): {n}", n.bits());
    println!("phi(n): {phi}");
    println!("Public exponent e: {e}");
    println!("Private exponent d: {d}");
    println!("Plaintext (hex): {}", to_hex(&message));
    println!("Ciphertext (hex): {}", to_hex(&ciphertext));
    //println!("Recovered (hex): {}", to_hex(&recovered));

    if let Some(seed) = args.seed {
        println!("RNG seed: {seed}");
    }

    let ctx = RSAContext {
        p: p.clone(),
        q: q.clone(),
        n: n.clone(),
        phi: phi.clone(),
        e: e.clone(),
        d: d.clone(),
    };

    if config.engine.test_iterations > 0 {
        let iterations = config.engine.test_iterations;
        let mut reports = Vec::new();
        let mut next_pct = 10u64;
        for i in 0..iterations {
            let msg = if i == 0 && args.message.is_some() {
                message.clone()
            } else if config.engine.message.is_random {
                random_biguint_bits(config.engine.message.bits, &mut rng)
            } else {
                BigUint::from_bytes_be(config.engine.message.fixed_message.as_bytes())
            };
            let report = run_message_trial(
                &ctx,
                &config,
                &config.engine,
                &msg,
                config.engine.min_message_trials,
                &mut rng,
            )?;
            reports.push(report);

            log_progress_every_ten_percent(i + 1, iterations, &mut next_pct, "Test iterations");
        }

        let score = |r: &TestReport| (r.matching_total, r.matching_lsb);

        let best_idx = reports
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| score(r))
            .map(|(idx, _)| idx);

        let mut worst_idx = reports
            .iter()
            .enumerate()
            .min_by_key(|(_, r)| score(r))
            .map(|(idx, _)| idx);

        if let (Some(bi), Some(wi)) = (best_idx, worst_idx) {
            if bi == wi && reports.len() > 1 {
                worst_idx = reports
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx != bi)
                    .min_by_key(|(_, r)| score(r))
                    .map(|(idx, _)| idx)
                    .or(Some(wi));
            }
        }

        let best_match = best_idx.map(|idx| reports[idx].clone());
        let worst_match = worst_idx.map(|idx| reports[idx].clone());

        if let Some(best) = &best_match {
            println!("Best r candidate: {}", best.best_r);
            println!("Factors: {:?}", best.factors);
            println!(
                "Matching bits: LSB run {} / overlap {} of {} bits",
                best.matching_lsb, best.matching_total, best.message_bits
            );
        }
        if let Some(worst) = &worst_match {
            println!("Worst r candidate: {}", worst.best_r);
            println!("Factors: {:?}", worst.factors);
            println!(
                "Matching bits: LSB run {} / overlap {} of {} bits",
                worst.matching_lsb, worst.matching_total, worst.message_bits
            );
        }

        let bits_values: Vec<f64> = reports.iter().map(|r| r.matching_lsb as f64).collect();
        if let Some(bits_stats) = compute_stats(&bits_values) {
            println!(
                "Matching bits stats: mean {:.4}, std dev {:.4}, min {:.4}, max {:.4}, n {}",
                bits_stats.mean,
                bits_stats.stddev,
                bits_stats.min,
                bits_stats.max,
                bits_stats.count
            );
        } else {
            println!("Matching bits stats: no samples");
        }

        let overlaps_pct: Vec<f64> = reports
            .iter()
            .map(|r| (r.matching_total as f64) / (r.message_bits.max(1) as f64) * 100.0)
            .collect();

        if let Some(overlap_stats) = compute_stats(&overlaps_pct) {
            println!(
                "Matching overlap stats (%): mean {:.4}, std dev {:.4}, min {:.4}, max {:.4}, n {}",
                overlap_stats.mean,
                overlap_stats.stddev,
                overlap_stats.min,
                overlap_stats.max,
                overlap_stats.count
            );

            let threshold = config.engine.overlap_report_threshold;
            let over_threshold_count = overlaps_pct.iter().filter(|v| **v >= threshold).count();
            println!(
                "Overlaps >= {:.2}%: count {}",
                threshold,
                over_threshold_count
            );
        } else {
            println!("Matching overlap stats: no samples");
        }

        if iterations > 1 {
            println!(
                "Max matching bits over all test cases: {}",
                reports
                    .iter()
                    .map(|r| r.matching_lsb)
                    .max()
                    .unwrap_or(0)
            );
        }

        if config.engine.alt_iterations > 0 {
            if let Some(best) = &best_match {
                if let Some((avg_bits, avg_overlap, max_bits)) = run_fixed_r_trials(
                    &ctx,
                    &config,
                    best,
                    config.engine.alt_iterations,
                    &mut rng,
                ) {
                    println!("\nAlt iterations on best r ({} runs):", config.engine.alt_iterations);
                    println!("Average matching bits: {:.4}", avg_bits);
                    println!("Average matching overlap: {:.4}%", avg_overlap);
                    println!("Max matching bits: {}", max_bits);
                }
            }

            if let Some(worst) = &worst_match {
                if let Some((avg_bits, avg_overlap, max_bits)) = run_fixed_r_trials(
                    &ctx,
                    &config,
                    worst,
                    config.engine.alt_iterations,
                    &mut rng,
                ) {
                    println!("\nAlt iterations on worst r ({} runs):", config.engine.alt_iterations);
                    println!("Average matching bits: {:.4}", avg_bits);
                    println!("Average matching overlap: {:.4}%", avg_overlap);
                    println!("Max matching bits: {}", max_bits);
                }
            }
        }

        // r_use_list stress testing
        if config.engine.r_use_list_enable && !config.engine.r_use_list.is_empty() {
            println!("\nRunning r_use_list stress tests...");
            for r_str in &config.engine.r_use_list {
                if let Ok(r) = r_str.parse::<BigUint>() {
                    if is_probable_prime_big(&r) {
                        continue;
                    }
                    run_r_stress_entry("r_use_list", &r, &ctx, &config, &config.engine, &mut rng);
                }
            }
        }

        // r_stress range testing
        if config.engine.r_stress_test_enable {
            if let (Some(start), Some(end)) = (&config.engine.r_stress_start, &config.engine.r_stress_end) {
                println!("\nRunning r_stress range tests...");
                let mut current = start.clone();
                while &current <= end {
                    if !is_probable_prime_big(&current) {
                        run_r_stress_entry("r_stress", &current, &ctx, &config, &config.engine, &mut rng);
                    }
                    current += BigUint::one();
                }
            }
        }
    }

    Ok(())
}

#[derive(Debug, Deserialize, Clone)]
struct Config {
    #[serde(default)]
    key: KeyConfig,
    #[serde(default)]
    engine: EngineConfig,
}

#[derive(Debug, Deserialize, Clone)]
struct KeyConfig {
    #[serde(default = "default_generate")]
    generate: bool,
    #[serde(default, deserialize_with = "deserialize_biguint_option")]
    p: Option<BigUint>,
    #[serde(default, deserialize_with = "deserialize_biguint_option")]
    q: Option<BigUint>,
    #[serde(default = "default_e")]
    e: u64,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct MessageConfig {
    #[serde(default = "default_fixed_message")]
    fixed_message: String,
    #[serde(default = "default_message_random")]
    is_random: bool,
    #[serde(default = "default_message_bits")]
    bits: u32,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
struct EngineConfig {
    #[serde(default = "default_base_convert")]
    base_convert: bool,
    #[serde(default = "default_rabin_exponent")]
    rabin_exponent: u64,
    #[serde(default = "default_min_message_trials")]
    min_message_trials: u64,
    #[serde(default = "default_overlap_report_threshold")]
    overlap_report_threshold: f64,
    #[serde(default = "default_process_min_count")]
    process_min_count: u64,
    #[serde(default = "default_process_count")]
    process_count: u64,
    #[serde(default = "default_process_scale")]
    process_scale: u32,
    #[serde(default = "default_process_max_best_attempts")]
    process_max_best_attempts: u64,
    #[serde(default = "default_process_min_factor")]
    process_min_factor: u64,
    #[serde(default = "default_use_rs_decrypt")]
    use_rs_decrypt: bool,
    #[serde(default = "default_test_iterations")]
    test_iterations: u64,
    #[serde(default = "default_alt_iterations")]
    alt_iterations: u64,
    #[serde(default = "default_r_use_list_enable")]
    r_use_list_enable: bool,
    #[serde(default)]
    r_use_list: Vec<String>,
    #[serde(default = "default_r_stress_test_enable")]
    r_stress_test_enable: bool,
    #[serde(default, deserialize_with = "deserialize_biguint_option")]
    r_stress_start: Option<BigUint>,
    #[serde(default, deserialize_with = "deserialize_biguint_option")]
    r_stress_end: Option<BigUint>,
    #[serde(default)]
    override_best_r: Option<String>,
    #[serde(default = "default_reuse_r_candidates_path")]
    reuse_r_candidates_path: String,
    #[serde(default = "default_reuse_r_candidates")]
    reuse_r_candidates: bool,
    #[serde(default = "default_reuse_r_candidates_append_only")]
    reuse_r_candidates_append_only: bool,
    #[serde(default)]
    message: MessageConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            key: KeyConfig::default(),
            engine: EngineConfig::default(),
        }
    }
}

impl Default for KeyConfig {
    fn default() -> Self {
        Self {
            generate: default_generate(),
            p: None,
            q: None,
            e: default_e(),
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            base_convert: default_base_convert(),
            rabin_exponent: default_rabin_exponent(),
            min_message_trials: default_min_message_trials(),
            overlap_report_threshold: default_overlap_report_threshold(),
            process_min_count: default_process_min_count(),
            process_count: default_process_count(),
            process_scale: default_process_scale(),
            process_max_best_attempts: default_process_max_best_attempts(),
            process_min_factor: default_process_min_factor(),
            use_rs_decrypt: default_use_rs_decrypt(),
            test_iterations: default_test_iterations(),
            alt_iterations: default_alt_iterations(),
            r_use_list_enable: default_r_use_list_enable(),
            r_use_list: Vec::new(),
            r_stress_test_enable: default_r_stress_test_enable(),
            r_stress_start: None,
            r_stress_end: None,
            override_best_r: None,
            reuse_r_candidates_path: default_reuse_r_candidates_path(),
            reuse_r_candidates: default_reuse_r_candidates(),
            reuse_r_candidates_append_only: default_reuse_r_candidates_append_only(),
            message: MessageConfig::default(),
        }
    }
}

fn load_config(path: &str) -> Result<Config, Box<dyn Error>> {
    let cfg_path = Path::new(path);
    if !cfg_path.exists() {
        println!("Config file {path} not found; using defaults");
        return Ok(Config::default());
    }

    let raw = fs::read_to_string(cfg_path)?;
    let config = match serde_json::from_str(&raw) {
        Ok(cfg) => cfg,
        Err(json_err) => match json5::from_str(&raw) {
            Ok(cfg) => cfg,
            Err(json5_err) => {
                return Err(format!(
                    "failed to parse config file {path}: json error: {json_err}; json5 fallback error: {json5_err}"
                )
                .into())
            }
        },
    };

    Ok(config)
}

fn log_progress_every_ten_percent(done: u64, total: u64, next_pct: &mut u64, label: &str) {
    if total == 0 {
        return;
    }

    let pct = done.saturating_mul(100) / total;
    if pct >= *next_pct || done == total {
        let display_pct = if done == total { 100 } else { ((pct / 10) * 10).min(100) };
        println!("{label} progress: {}% ({}/{})", display_pct, done, total);

        while *next_pct <= pct && *next_pct < 100 {
            *next_pct += 10;
        }
        if done == total {
            *next_pct = 100;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct StatSummary {
    mean: f64,
    stddev: f64,
    min: f64,
    max: f64,
    count: usize,
}

fn compute_stats(values: &[f64]) -> Option<StatSummary> {
    if values.is_empty() {
        return None;
    }

    let count = values.len();
    let sum: f64 = values.iter().sum();
    let mean = sum / count as f64;
    let variance = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / count as f64;
    let stddev = variance.sqrt();
    let min = values
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max = values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    Some(StatSummary {
        mean,
        stddev,
        min,
        max,
        count,
    })
}

fn select_message(args_message: Option<String>, engine: &EngineConfig, rng: &mut StdRng) -> BigUint {
    if let Some(explicit) = args_message {
        return BigUint::from_bytes_be(explicit.as_bytes());
    }
    if engine.message.is_random {
        return random_message_under_n(engine, &BigUint::zero(), rng);
    }
    BigUint::from_bytes_be(engine.message.fixed_message.as_bytes())
}

fn random_message_under_n(engine: &EngineConfig, n: &BigUint, rng: &mut StdRng) -> BigUint {
    let mut target_bits = engine.message.bits.max(1);
    if !n.is_zero() {
        target_bits = target_bits.min(n.bits().saturating_sub(1) as u32).max(1);
    }

    loop {
        let candidate = random_biguint_bits(target_bits, rng);
        if candidate.is_zero() {
            continue;
        }
        if n.is_zero() || candidate < *n {
            return candidate;
        }
    }
}

fn default_generate() -> bool {
    true
}

fn default_e() -> u64 {
    65_537
}

fn default_fixed_message() -> String {
    "afterstate".to_string()
}

fn default_message_random() -> bool {
    false
}

fn default_message_bits() -> u32 {
    56
}

fn default_base_convert() -> bool {
    true
}

fn default_rabin_exponent() -> u64 {
    2
}

fn default_min_message_trials() -> u64 {
    1
}

fn default_overlap_report_threshold() -> f64 {
    51.0
}

fn default_process_min_count() -> u64 {
    1
}

fn default_process_count() -> u64 {
    8
}

fn default_process_scale() -> u32 {
    8
}

fn default_process_max_best_attempts() -> u64 {
    4
}

fn default_process_min_factor() -> u64 {
    3
}

fn default_use_rs_decrypt() -> bool {
    true
}

fn default_test_iterations() -> u64 {
    1
}

fn default_alt_iterations() -> u64 {
    0
}

fn default_r_use_list_enable() -> bool {
    false
}

fn default_r_stress_test_enable() -> bool {
    false
}

fn default_reuse_r_candidates() -> bool {
    false
}

fn default_reuse_r_candidates_path() -> String {
    "r_candidates.csv".to_string()
}

fn default_reuse_r_candidates_append_only() -> bool {
    false
}

fn deserialize_biguint_option<'de, D>(deserializer: D) -> Result<Option<BigUint>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as DeError;

    let maybe_value = Option::<serde_json::Value>::deserialize(deserializer)?;
    match maybe_value {
        Some(serde_json::Value::String(s)) => s.parse::<BigUint>().map(Some).map_err(DeError::custom),
        Some(serde_json::Value::Number(num)) => num
            .to_string()
            .parse::<BigUint>()
            .map(Some)
            .map_err(DeError::custom),
        Some(other) => Err(DeError::custom(format!(
            "expected string or number for big integer, got {other}"
        ))),
        None => Ok(None),
    }
}

const DETERMINISTIC_BASES: [u64; 7] = [2, 3, 5, 7, 11, 13, 17];

fn random_prime_with_bits(bits: u32, rng: &mut StdRng) -> u64 {
    let lower = 1u64 << (bits - 1);
    let upper = (1u64 << bits) - 1;

    loop {
        let mut candidate = rng.gen_range(lower..=upper);
        candidate |= 1; // force odd
        if is_probable_prime(candidate) {
            return candidate;
        }
    }
}

fn random_biguint_bits(bits: u32, rng: &mut StdRng) -> BigUint {
    if bits == 0 {
        return BigUint::zero();
    }
    let byte_len = ((bits as usize) + 7) / 8;
    let mut bytes = vec![0u8; byte_len];
    rng.fill_bytes(&mut bytes);
    let leading_bits = (bits % 8) as u8;
    if leading_bits != 0 {
        let mask = (1u8 << leading_bits) - 1;
        bytes[0] &= mask;
    }
    // Ensure the top bit is set so the value uses the requested width when possible.
    let top_byte_index = 0;
    let top_bit = if leading_bits == 0 { 0x80 } else { 1u8 << (leading_bits - 1) };
    bytes[top_byte_index] |= top_bit;
    BigUint::from_bytes_be(&bytes)
}

fn is_probable_prime(n: u64) -> bool {
    if n < 4 {
        return n == 2 || n == 3;
    }
    if n % 2 == 0 {
        return false;
    }

    let (d, s) = decompose(n - 1);
    for &a in &DETERMINISTIC_BASES {
        if a % n == 0 {
            continue;
        }
        let mut x = mod_pow_u64(a % n, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        let mut composite = true;
        for _ in 1..s {
            x = mod_pow_u64(x, 2, n);
            if x == n - 1 {
                composite = false;
                break;
            }
        }
        if composite {
            return false;
        }
    }
    true
}

fn decompose(mut value: u64) -> (u64, u32) {
    let mut s = 0;
    while value % 2 == 0 {
        value >>= 1;
        s += 1;
    }
    (value, s)
}

fn mod_pow_u64(base: u64, exponent: u64, modulus: u64) -> u64 {
    let mut result = 1u128;
    let mut base = base as u128 % modulus as u128;
    let mut exp = exponent;
    let m = modulus as u128;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % m;
        }
        base = (base * base) % m;
        exp >>= 1;
    }

    result as u64
}

fn choose_exponent(start: u64, phi: &BigUint) -> BigUint {
    let mut candidate = BigUint::from(if start % 2 == 0 { start + 1 } else { start });
    let step = BigUint::from(2u8);

    while candidate.gcd(phi) != BigUint::one() {
        candidate += &step;
    }

    candidate
}

fn mod_inverse(a: &BigUint, modulus: &BigUint) -> Option<BigUint> {
    let a_int = BigInt::from(a.clone());
    let m_int = BigInt::from(modulus.clone());

    let egcd = a_int.extended_gcd(&m_int);
    if egcd.gcd != BigInt::one() {
        return None;
    }

    let mut x = egcd.x % &m_int;
    if x.is_negative() {
        x += m_int;
    }

    x.to_biguint()
}

fn to_hex(value: &BigUint) -> String {
    let bytes = value.to_bytes_be();
    if bytes.is_empty() {
        return "0".to_string();
    }
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = std::fmt::Write::write_fmt(&mut hex, format_args!("{:02x}", byte));
    }
    hex
}

fn bit_length(value: &BigUint) -> u64 {
    value.bits()
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct RSAContext {
    p: BigUint,
    q: BigUint,
    n: BigUint,
    phi: BigUint,
    e: BigUint,
    d: BigUint,
}

#[derive(Clone, Debug)]
struct TestReport {
    best_r: BigUint,
    factors: Vec<(BigUint, u64)>,
    matching_lsb: usize,
    matching_total: usize,
    message_bits: usize,
}

fn run_message_trial(
    ctx: &RSAContext,
    config: &Config,
    engine: &EngineConfig,
    message: &BigUint,
    min_message_trials: u64,
    rng: &mut StdRng,
) -> Result<TestReport, Box<dyn Error>> {
    let attempts = min_message_trials.max(1);
    let mut best: Option<TestReport> = None;
    let mut worst: Option<TestReport> = None;

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);

    let candidates = generate_r_candidates(&ctx.n, engine, rng);
    if candidates.is_empty() {
        return Err("no r candidates generated".into());
    } else {
        println!("Generated {} r candidates for testing", candidates.len());
    }

    for attempt_idx in 0..attempts {
        let msg = if attempt_idx == 0 {
            message.clone()
        } else {
            random_message_under_n(engine, &ctx.n, rng)
        };
        
        for (r, factors) in &candidates {
            //println!("factors for r {}: {:?}", r, factors);
            let phi_new = compute_totient(factors);
            let k_val = ((((&phi_new / config.key.e.clone()) + BigUint::from(2u32)) * BigUint::from(2u32)) * config.key.e.clone() + BigUint::from(1u32)) % &phi_new;
            
            let ciphertext = msg.modpow(&k_val, &ctx.n);
            let result_default = get_larger_number(&ciphertext, &ctx.n, y, true, false);
            //let recovered = ciphertext.modpow(&ctx.d, &ctx.n);
            //if recovered != msg {
            //    return Err("RSA round trip failed".into());
            //}

            //let d_new2 = mod_inverse(&k_val, &phi_new).unwrap_or(BigUint::zero());
            
            //Reduce k_val mod phi_new before inverse.
            let Some(d_new) = mod_inverse(&(&k_val % &phi_new), &phi_new) else {
                //println!("Skipping r candidate {} due to non-invertible k_val {}, phi_new {}", r, &k_val, &phi_new);
                continue;
            };

            let hbc_result = hbc(&result_default, r, &n_pow_y, engine);
            let recovered_new = if engine.use_rs_decrypt {
                hbc_result.modpow(&d_new, r)
            } else {
                hbc_result
            };

            let r_pow_y = r.pow(y);
            let result2_default = get_larger_number(&recovered_new, r, y, true, false);
            let hbc_default = hbc(&result2_default, &ctx.n, &r_pow_y, engine);
            //take the 1's NOT of dm
            let dm = &hbc_default % &ctx.n;            
            let inverted_dm = (BigUint::from(1u32) << (dm.bits() + 1)) - BigUint::from(1u32) - dm; // Invert all bits

            let (matching_lsb, matching_total) = count_matching_bits(&inverted_dm, &msg);
            //println!("Trial {}, r candidate {}: matching bits LSB run {} / overlap {} of {} bits", attempt_idx + 1, r, matching_lsb, matching_total, msg.bits());
            let report = TestReport {
                best_r: r.clone(),
                factors: factors.clone(),
                matching_lsb,
                matching_total,
                message_bits: msg.bits() as usize,
            };

            if best
                .as_ref()
                .map(|b| (matching_total, matching_lsb) > (b.matching_total, b.matching_lsb))
                .unwrap_or(true)
            {
                //println!("Best candidate updated: r = {}, factors = {:?}, matching bits LSB run {} / overlap {} of {} bits", r, factors, matching_lsb, matching_total, msg.bits());
                best = Some(report.clone());
            }
            if worst
                .as_ref()
                .map(|b| (matching_total, matching_lsb) < (b.matching_total, b.matching_lsb))
                .unwrap_or(true)
            {
                worst = Some(report);
            }
        }
    }

    best.ok_or_else(|| "no valid r candidates after filtering".into())
}

fn run_fixed_r_trials(
    ctx: &RSAContext,
    config: &Config,
    r_report: &TestReport,
    iterations: u64,
    rng: &mut StdRng,
) -> Option<(f64, f64, usize)> {
    if iterations == 0 {
        return None;
    }

    let engine = &config.engine;

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);
    let r = &r_report.best_r;
    let r_pow_y = r.pow(y);
    let phi_new = compute_totient(&r_report.factors);
    let k_val = ((((&phi_new / config.key.e.clone()) + BigUint::from(2u32)) * BigUint::from(2u32)) * config.key.e.clone() + BigUint::from(1u32)) % &phi_new;
    let Some(d_new2) = mod_inverse(&(&k_val % &phi_new), &phi_new) else {
        println!("Skipping r candidate {} due to non-invertible k_val {}, phi_new {}", r, &k_val, &phi_new);
        return None;        
    };
    //let d_new = mod_inverse(&ctx.e, &phi_new)?;

    let iter_count = iterations as usize;
    let mut seeds = Vec::with_capacity(iter_count);
    for _ in 0..iter_count {
        seeds.push(rng.next_u64());
    }

    let done = Arc::new(AtomicU64::new(0));
    let next_pct = Arc::new(AtomicU64::new(10));

    let samples: Vec<(f64, f64, usize)> = seeds
        .into_par_iter()
        .map(|seed| {
            let mut local_rng = StdRng::seed_from_u64(seed);

            //let k_val = ((((&phi_new / config.key.e.clone()) + BigUint::from(2u32)) * BigUint::from(2u32)) * config.key.e.clone() + BigUint::from(1u32)) % &phi_new;
            let msg = random_message_under_n(engine, &ctx.n, &mut local_rng);

            let ciphertext = msg.modpow(&k_val, &ctx.n);
            
            let result_default = get_larger_number(&ciphertext, &ctx.n, y, true, false);

            let hbc_result = hbc(&result_default, r, &n_pow_y, engine);
            let recovered_new = if engine.use_rs_decrypt {
                hbc_result.modpow(&d_new2, r)
            } else {
                hbc_result
            };

            let result2_default = get_larger_number(&recovered_new, r, y, true, false);
            let hbc_default = hbc(&result2_default, &ctx.n, &r_pow_y, engine);
            //take the 1's NOT of dm
            let dm = &hbc_default % &ctx.n;            
            let inverted_dm = (BigUint::from(1u32) << (dm.bits() + 1)) - BigUint::from(1u32) - dm; // Invert all bits

            let (matching_lsb, matching_total) = count_matching_bits(&inverted_dm, &msg);
            let overlap = (matching_total as f64) / (msg.bits().max(1) as f64);
            let lsb_f = matching_lsb as f64;

            let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
            let pct = finished.saturating_mul(100) / iterations;
            let mut current_next = next_pct.load(Ordering::Relaxed);
            while pct >= current_next && current_next <= 100 {
                let new_next = current_next.saturating_add(10);
                match next_pct.compare_exchange(current_next, new_next, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => {
                        let display_pct = current_next.min(100);
                        println!(
                            "Alt iterations progress: {}% ({}/{})",
                            display_pct, finished, iterations
                        );
                        break;
                    }
                    Err(actual) => current_next = actual,
                }
            }

            (lsb_f, overlap, matching_lsb)
        })
        .collect();

    if samples.is_empty() {
        return None;
    }
    let n = samples.len() as f64;

    let bits_values: Vec<f64> = samples.iter().map(|(b, _, _)| *b).collect();
    let overlap_values_pct: Vec<f64> = samples.iter().map(|(_, o, _)| o * 100.0).collect();
    let max_bits = samples.iter().map(|(_, _, mb)| *mb).max().unwrap_or(0);

    let bits_stats = compute_stats(&bits_values).unwrap();
    let overlap_stats = compute_stats(&overlap_values_pct).unwrap();

    let threshold = engine.overlap_report_threshold;
    let over_threshold_count = overlap_values_pct.iter().filter(|v| **v >= threshold).count();

    println!(
        "Alt iterations stats: bits mean {:.4}, std dev {:.4}, min {:.4}, max {:.4}; overlap % mean {:.4}, std dev {:.4}, min {:.4}, max {:.4}; overlaps >= {:.2}% count {}; max bits {}",
        bits_stats.mean,
        bits_stats.stddev,
        bits_stats.min,
        bits_stats.max,
        overlap_stats.mean,
        overlap_stats.stddev,
        overlap_stats.min,
        overlap_stats.max,
        threshold,
        over_threshold_count,
        max_bits
    );

    Some((bits_stats.mean, overlap_stats.mean, max_bits))
}

fn run_r_stress_entry(
    label: &str,
    r: &BigUint,
    ctx: &RSAContext,
    config: &Config,
    engine: &EngineConfig,
    rng: &mut StdRng,
) {
    let deadline = Instant::now() + Duration::from_secs(10);
    let Some(factors) = factor_composite_with_timeout(r, rng, deadline) else {
        return;
    };
    if factors.len() < 3 || factors.iter().any(|(p, _)| p < &BigUint::from(engine.process_min_factor)) {
        return;
    }
    let dummy_report = TestReport {
        best_r: r.clone(),
        factors: factors.clone(),
        matching_lsb: 0,
        matching_total: 0,
        message_bits: 0,
    };
    if let Some((avg_bits, avg_overlap, max_bits)) = run_fixed_r_trials(
        ctx,
        &config,
        &dummy_report,
        engine.alt_iterations.max(1),
        rng,
    ) {
        println!(
            "{} r {} -> avg bits {:.4}, avg overlap {:.4}%, max bits {}",
            label, r, avg_bits, avg_overlap, max_bits
        );
    }
}

fn count_matching_bits(a: &BigUint, b: &BigUint) -> (usize, usize) {
    let a_bits = a.to_str_radix(2);
    let b_bits = b.to_str_radix(2);
    let min_len = a_bits.len().min(b_bits.len());

    let mut matching_total = 0usize;
    for i in 0..min_len {
        if a_bits.as_bytes()[a_bits.len() - 1 - i] == b_bits.as_bytes()[b_bits.len() - 1 - i] {
            matching_total += 1;
        }
    }

    let mut matching_lsb = 0usize;
    for i in 0..min_len {
        if a_bits.as_bytes()[a_bits.len() - 1 - i] == b_bits.as_bytes()[b_bits.len() - 1 - i] {
            matching_lsb += 1;
        } else {
            break;
        }
    }

    (matching_lsb, matching_total)
}

fn compute_totient(factors: &[(BigUint, u64)]) -> BigUint {
    let mut phi = BigUint::one();
    for (p, e) in factors {
        if *e == 0 {
            continue;
        }
        let term = (p - BigUint::one()) * p.pow((*e as u32).saturating_sub(1));
        phi *= term;
    }
    phi
}

fn generate_r_candidates(
    n: &BigUint,
    engine: &EngineConfig,
    rng: &mut StdRng,
) -> Vec<(BigUint, Vec<(BigUint, u64)>)> {
    if let Some(ref override_r) = engine.override_best_r {
        if !override_r.is_empty() {
            if let Ok(r) = override_r.parse::<BigUint>() {
                if is_probable_prime_big(&r) {
                    return Vec::new();
                }
                let deadline = Instant::now() + Duration::from_secs(10);
                if let Some(factors) = factor_composite_with_timeout(&r, rng, deadline) {
                    if factors.len() >= 3
                        && factors.iter().all(|(p, _)| p >= &BigUint::from(engine.process_min_factor))
                    {
                        return vec![(r, factors)];
                    }
                }
            }
        }
    }

    let min_factor = BigUint::from(engine.process_min_factor);
    let scale = BigUint::one() << engine.process_scale;
    let count = engine.process_count.max(engine.process_min_count).max(1);
    let target_count = count as usize;

    let mut collected: Vec<(BigUint, Vec<(BigUint, u64)>)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    let load_reuse = engine.reuse_r_candidates && !engine.reuse_r_candidates_append_only;
    let append_reuse = engine.reuse_r_candidates || engine.reuse_r_candidates_append_only;

    if load_reuse {
        let reuse_path = engine.reuse_r_candidates_path.as_str();
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
    } else if engine.reuse_r_candidates_append_only {
        println!(
            "Reuse append-only enabled; will append new r candidates to {} but will not load from it",
            engine.reuse_r_candidates_path
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
        append_reuse_candidates(&engine.reuse_r_candidates_path, &new_candidates);
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

#[allow(dead_code)]
fn random_biguint_below(upper: &BigUint, rng: &mut StdRng) -> BigUint {
    if upper.is_zero() {
        return BigUint::zero();
    }
    let bits = upper.bits();
    loop {
        let candidate = random_biguint_bits(bits as u32, rng);
        if &candidate < upper {
            return candidate;
        }
    }
}

fn get_larger_number(x: &BigUint, p: &BigUint, y: u32, apply_mod: bool, use_other_root: bool) -> BigUint {
    let p_y = p.pow(y);
    let p_y_minus_one = p.pow(y.saturating_sub(1));

    let x2_mod_p = x.modpow(&BigUint::from(2u8), p);
    let x2_mod_p_y = x.modpow(&BigUint::from(2u8), &p_y);

    let test_1 = modular_sqrt(&x2_mod_p, p);
    let base_root = if use_other_root { (p - &test_1) % p } else { test_1 };
    let big_x = base_root.modpow(&p_y_minus_one, &p_y);

    let tmp_1 = (&p_y - &(BigUint::from(2u8) * &p_y_minus_one) + BigUint::one()) >> 1;
    let factor = x2_mod_p_y.modpow(&tmp_1, &p_y);
    if apply_mod {
        (big_x * factor) % p_y
    } else {
        big_x * factor
    }
}

fn modular_sqrt(a: &BigUint, p: &BigUint) -> BigUint {
    // Tonelli-Shanks for odd prime p; demo uses small-ish primes so this is fine.
    if a.is_zero() {
        return BigUint::zero();
    }
    if p == &BigUint::from(2u8) {
        return BigUint::zero();
    }
    if legendre_symbol(a, p) != BigInt::one() {
        return BigUint::one();
    }

    let mut q = p - BigUint::one();
    let mut s = 0u32;
    while (&q & BigUint::one()).is_zero() {
        q >>= 1;
        s += 1;
    }

    if s == 1 {
        return a.modpow(&((p + BigUint::one()) >> 2), p);
    }

    let mut z = BigUint::from(2u8);
    while legendre_symbol(&z, p) != BigInt::from(-1) {
        z += BigUint::one();
    }

    let mut m = s;
    let mut c = z.modpow(&q, p);
    let mut t = a.modpow(&q, p);
    let mut r = a.modpow(&((&q + BigUint::one()) >> 1), p);

    while t != BigUint::one() {
        let mut i = 1u32;
        let mut t2i = t.modpow(&BigUint::from(2u32), p);
        while t2i != BigUint::one() {
            t2i = t2i.modpow(&BigUint::from(2u32), p);
            i += 1;
            if i == m {
                break;
            }
        }
        let b = c.modpow(&BigUint::from(1u64 << (m - i - 1)), p);
        r = (&r * &b) % p;
        c = (&b * &b) % p;
        t = (&t * &c) % p;
        m = i;
    }
    r
}

fn legendre_symbol(a: &BigUint, p: &BigUint) -> BigInt {
    let ls = a.modpow(&((p - BigUint::one()) >> 1), p);
    if ls.is_zero() {
        BigInt::zero()
    } else if ls == BigUint::one() {
        BigInt::one()
    } else {
        BigInt::from(-1)
    }
}

fn homomorphic_base_conversion(x: &BigUint, r: &BigUint, p: &BigUint) -> BigUint {
    let y = x % p;
    let z = p % r;
    let q = (&y / p) * &z;
    let reduced = if &y >= p { &y - q } else { y.clone() };
    reduced % r
}

fn hbc(x: &BigUint, r: &BigUint, p: &BigUint, engine: &EngineConfig) -> BigUint {
    if engine.base_convert {
        homomorphic_base_conversion(x, r, p)
    } else {
        let num = r * x;
        num / p
    }
}

#[allow(dead_code)]
fn construct_from_factors_close_to_target_n(target_n: &BigUint, prime_factors: &[BigUint]) -> BigUint {
    if prime_factors.is_empty() {
        return BigUint::one();
    }
    let mut best = BigUint::one();
    let mut best_diff = if target_n > &best {
        target_n - &best
    } else {
        &best - target_n
    };

    let limit = 1usize << prime_factors.len().min(12);
    for mask in 1..limit {
        let mut prod = BigUint::one();
        for (i, pf) in prime_factors.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                prod *= pf;
            }
        }
        let diff = if target_n > &prod {
            target_n - &prod
        } else {
            &prod - target_n
        };
        if diff < best_diff {
            best_diff = diff;
            best = prod;
        }
    }

    best
}

#[allow(dead_code)]
fn factor_composite_with_timeout(
    n: &BigUint,
    rng: &mut StdRng,
    deadline: Instant,
) -> Option<Vec<(BigUint, u64)>> {
    let mut factors = Vec::new();
    if !factor_recursive(n.clone(), &mut factors, rng, deadline) {
        return None;
    }
    factors.sort_by(|a, b| a.0.cmp(&b.0));
    Some(coalesce_factors(factors))
}

#[allow(dead_code)]
fn factor_recursive(
    n: BigUint,
    out: &mut Vec<(BigUint, u64)>,
    rng: &mut StdRng,
    deadline: Instant,
) -> bool {
    if Instant::now() >= deadline {
        return false;
    }
    if n <= BigUint::one() {
        return true;
    }
    if is_probable_prime_big(&n) {
        out.push((n, 1));
        return true;
    }
    let Some(divisor) = pollard_rho(&n, rng, deadline) else {
        return false;
    };
    let other = &n / &divisor;
    factor_recursive(divisor, out, rng, deadline) && factor_recursive(other, out, rng, deadline)
}

#[allow(dead_code)]
fn coalesce_factors(mut factors: Vec<(BigUint, u64)>) -> Vec<(BigUint, u64)> {
    if factors.is_empty() {
        return factors;
    }
    factors.sort_by(|a, b| a.0.cmp(&b.0));
    let mut merged: Vec<(BigUint, u64)> = Vec::new();
    let mut current = factors[0].clone();
    for item in factors.into_iter().skip(1) {
        if item.0 == current.0 {
            current.1 += item.1;
        } else {
            merged.push(current);
            current = item;
        }
    }
    merged.push(current);
    merged
}

#[allow(dead_code)]
fn pollard_rho(n: &BigUint, rng: &mut StdRng, deadline: Instant) -> Option<BigUint> {
    if n.is_even() {
        return Some(BigUint::from(2u8));
    }
    let one = BigUint::one();
    let two = &one + &one;

    let mut c = random_biguint_below(n, rng);
    let mut x = random_biguint_below(n, rng);
    let mut y = x.clone();
    let f = |val: &BigUint, c: &BigUint, n: &BigUint| (val.modpow(&two, n) + c) % n;
    let mut iter: u64 = 0;

    while Instant::now() < deadline {
        iter += 1;
        x = f(&x, &c, n);
        y = f(&f(&y, &c, n), &c, n);
        let diff = if &x >= &y { &x - &y } else { &y - &x };
        let d = diff.gcd(n);
        if d != one && d != *n {
            return Some(d);
        }
        if d == *n || iter > 10_000 {
            c = random_biguint_below(n, rng);
            x = random_biguint_below(n, rng);
            y = x.clone();
            iter = 0;
        }
    }
    None
}

#[allow(dead_code)]
fn is_probable_prime_big(n: &BigUint) -> bool {
    // Tiny cases
    if n <= &BigUint::from(3u8) {
        return *n == BigUint::from(2u8) || *n == BigUint::from(3u8);
    }
    if n.is_even() {
        return false;
    }

    // Quick small-prime sieve to reject obvious composites before MR rounds.
    const SMALL_PRIMES: [u64; 16] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59];
    for p in SMALL_PRIMES {
        let p_big = BigUint::from(p);
        if n == &p_big {
            return true;
        }
        if (n % &p_big).is_zero() {
            return false;
        }
    }

    let one = BigUint::one();
    let two = &one + &one;
    let n_minus_one = n - &one;
    let (d, s) = decompose_big(n_minus_one.clone());

    // Deterministic bases sufficient for n < 2^256 (we're far below that).
    // Using a small set keeps modpow calls down for the hot path.
    const BASES: [u64; 7] = [2, 3, 5, 7, 11, 13, 17];
    'outer: for a in BASES {
        let a = BigUint::from(a);
        let mut x = a.modpow(&d, n);
        if x == one || x == n_minus_one {
            continue;
        }
        for _ in 1..s {
            x = x.modpow(&two, n);
            if x == n_minus_one {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

#[allow(dead_code)]
fn decompose_big(mut value: BigUint) -> (BigUint, u32) {
    let mut s = 0u32;
    let one = BigUint::one();
    while (&value & &one).is_zero() {
        value >>= 1;
        s += 1;
    }
    (value, s)
}
