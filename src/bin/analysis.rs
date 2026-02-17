/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>

use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use rayon::prelude::*;
use plotters::prelude::*;

use clap::Parser;
use num_bigint::BigUint;
use num_traits::{One, Zero};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use serde::Deserialize;

use rsademo::combiner::majority_vote_with_distribution;
use rsademo::dsp::{find_ramp_signals_f64, ramp_signal_strength_f64};
use rsademo::math::{
    bit_length, choose_exponent, compute_totient, factor_composite_with_timeout,
    is_probable_prime_big, mod_inverse, modular_sqrt, random_biguint_bits,
    random_prime_with_bits, to_hex,
};
use rsademo::r_candidates::{generate_r_candidates_batch, RCandidateMode, RCandidateSettings};

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

    let (p, q): (BigUint, BigUint) = if config.rsa_keypair.generate {
        let p = random_prime_with_bits(args.bits, &mut rng);
        let mut q = random_prime_with_bits(args.bits, &mut rng);
        while q == p {
            q = random_prime_with_bits(args.bits, &mut rng);
        }
        (BigUint::from(p), BigUint::from(q))
    } else {
        let p = config
            .rsa_keypair
            .p
            .clone()
            .ok_or("config.rsa_keypair.p must be set when generate is false")?;
        let q = config
            .rsa_keypair
            .q
            .clone()
            .ok_or("config.rsa_keypair.q must be set when generate is false")?;
        (p, q)
    };

    let one = BigUint::one();
    let n = &p * &q;
    let phi = (&p - &one) * (&q - &one);

    let start_e = if args.public_exponent != 65_537 {
        args.public_exponent
    } else {
        config.rsa_keypair.e
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
    let recovered = ciphertext.modpow(&d, &n);

    if recovered != message {
        return Err("RSA round trip failed".into());
    }

    println!("Prime p ({} bits): {p}", bit_length(&p));
    println!("Prime q ({} bits): {q}", bit_length(&q));
    println!("Modulus n ({} bits): {n}", n.bits());
    println!("phi(n): {phi}");
    println!("Public exponent e: {e}");
    println!("Private exponent d: {d}");
    println!("Plaintext (hex): {}", to_hex(&message));
    println!("Ciphertext (hex): {}", to_hex(&ciphertext));
    println!("Recovered (hex): {}", to_hex(&recovered));

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

    if config.engine.combiner_enable {
        let bit_width = message.bits().max(1) as usize;
        let majority_bits = biguint_to_bits_le(&message, bit_width);
        let requested_oracles = config.engine.combiner_k_oracles;
        match collect_speculative_oracle_bits(&ctx, &config.engine, &message, requested_oracles, &mut rng) {
            Ok(oracles) => match majority_vote_with_distribution(&oracles, config.engine.combiner_tie_breaker) {
                Ok(distribution) => {
                    let mut correct = 0usize;
                    for (a, b) in distribution
                        .majority_bits
                        .iter()
                        .zip(majority_bits.iter())
                    {
                        if a == b {
                            correct += 1;
                        }
                    }
                    let total = majority_bits.len();
                    let accuracy = correct as f64 / total as f64;
                    println!(
                        "Speculative combiner majority vote: accuracy {:.2}% ({} of {} bits) using {} oracles (requested {})",
                        accuracy * 100.0,
                        correct,
                        total,
                        distribution.total_oracles,
                        requested_oracles
                    );
                    if let Some(stats) = compute_stats(&distribution.probability_one) {
                        println!(
                            "Speculative combiner bit probability P(1) stats: mean {:.4}, std dev {:.4}, min {:.4}, max {:.4}, n {}",
                            stats.mean,
                            stats.stddev,
                            stats.min,
                            stats.max,
                            stats.count
                        );
                    }
                }
                Err(err) => {
                    println!("Speculative combiner majority vote failed: {}", err);
                }
            },
            Err(err) => {
                println!("Speculative combiner setup failed: {}", err);
            }
        }
    }

    if config.engine.test_iterations > 0 {
        let mut bit_hist = MatchHistogram::new();
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
                &mut bit_hist,
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
            if let Err(err) = plot_overlap_histogram(&overlaps_pct, "test_iterations") {
                println!("Failed to write overlap histogram: {}", err);
            }
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
                    "best_r",
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
                    "worst_r",
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

        if let Err(err) = bit_hist.write_histogram("test_iterations_bit_matches") {
            println!("Failed to write bit match histogram: {}", err);
        }
    }

    if config.engine.enciphered_export_enable {
        if let Err(err) = run_enciphered_export(&ctx, &config.engine, &mut rng) {
            println!("Enciphered export failed: {}", err);
        }
    }

    Ok(())
}

#[derive(Debug, Deserialize, Clone)]
struct Config {
    #[serde(default, rename = "rsa_keypair", alias = "key", alias = "keys")]
    rsa_keypair: KeyConfig,
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
    #[serde(default = "default_invert_bits")]
    invert_bits: bool,
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
    #[serde(default = "default_r_candidate_mode")]
    r_candidate_mode: RCandidateMode,
    #[serde(default = "default_r_candidate_small_primes")]
    r_candidate_small_primes: Vec<u64>,
    #[serde(default = "default_r_candidate_small_prime_factors")]
    r_candidate_small_prime_factors: usize,
    #[serde(default = "default_combiner_enable")]
    combiner_enable: bool,
    #[serde(default = "default_combiner_k_oracles")]
    combiner_k_oracles: usize,
    #[serde(default = "default_combiner_match_probability")]
    combiner_match_probability: f64,
    #[serde(default = "default_combiner_tie_breaker")]
    combiner_tie_breaker: bool,
    #[serde(default)]
    message: MessageConfig,
    #[serde(default = "default_enciphered_export_enable")]
    enciphered_export_enable: bool,
    #[serde(default = "default_enciphered_export_iterations")]
    enciphered_export_iterations: u64,
    #[serde(default = "default_enciphered_export_bins")]
    enciphered_export_bins: usize,
    #[serde(default = "default_enciphered_export_window")]
    enciphered_export_window: usize,
    #[serde(default = "default_enciphered_export_stride")]
    enciphered_export_stride: usize,
    #[serde(default = "default_enciphered_export_output_csv")]
    enciphered_export_output_csv: String,
    #[serde(default = "default_enciphered_export_ramp_length")]
    enciphered_export_ramp_length: usize,
    #[serde(default = "default_enciphered_export_ramp_step_pct")]
    enciphered_export_ramp_step_pct: f64,
    #[serde(default = "default_enciphered_export_ramp_tolerances")]
    enciphered_export_ramp_tolerances: Vec<f64>,
    #[serde(default = "default_enciphered_export_ramp_csv")]
    enciphered_export_ramp_csv: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            rsa_keypair: KeyConfig::default(),
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
            invert_bits: default_invert_bits(),
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
            r_candidate_mode: default_r_candidate_mode(),
            r_candidate_small_primes: default_r_candidate_small_primes(),
            r_candidate_small_prime_factors: default_r_candidate_small_prime_factors(),
            combiner_enable: default_combiner_enable(),
            combiner_k_oracles: default_combiner_k_oracles(),
            combiner_match_probability: default_combiner_match_probability(),
            combiner_tie_breaker: default_combiner_tie_breaker(),
            message: MessageConfig::default(),
            enciphered_export_enable: default_enciphered_export_enable(),
            enciphered_export_iterations: default_enciphered_export_iterations(),
            enciphered_export_bins: default_enciphered_export_bins(),
            enciphered_export_window: default_enciphered_export_window(),
            enciphered_export_stride: default_enciphered_export_stride(),
            enciphered_export_output_csv: default_enciphered_export_output_csv(),
            enciphered_export_ramp_length: default_enciphered_export_ramp_length(),
            enciphered_export_ramp_step_pct: default_enciphered_export_ramp_step_pct(),
            enciphered_export_ramp_tolerances: default_enciphered_export_ramp_tolerances(),
            enciphered_export_ramp_csv: default_enciphered_export_ramp_csv(),
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

fn plot_overlap_histogram(overlaps_pct: &[f64], label: &str) -> Result<(), Box<dyn Error>> {
    if overlaps_pct.is_empty() {
        return Ok(());
    }

    let images_dir = Path::new("./images");
    fs::create_dir_all(images_dir)?;

    static HIST_SEQ: AtomicUsize = AtomicUsize::new(0);
    let seq = HIST_SEQ.fetch_add(1, Ordering::Relaxed);
    let safe_label: String = label
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let file_name = format!("overlap_histogram_{}_{}.png", safe_label, seq);
    let path = images_dir.join(file_name);

    let bin_count = 150usize;
    let min_value = 0.0f64;
    let max_value = 100.0f64;
    let bin_width = (max_value - min_value) / bin_count as f64;
    let mut counts = vec![0u32; bin_count];
    for &value in overlaps_pct {
        let clamped = value.clamp(min_value, max_value);
        let mut idx = ((clamped - min_value) / bin_width) as usize;
        if idx >= bin_count {
            idx = bin_count - 1;
        }
        counts[idx] = counts[idx].saturating_add(1);
    }

    let max_count = counts.iter().copied().max().unwrap_or(0).max(1);

    let root = BitMapBackend::new(&path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Overlap Percentage Histogram ({})", label),
            ("sans-serif", 30).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(min_value..max_value, 0u32..max_count)?;

    chart
        .configure_mesh()
        .x_desc("Overlap %")
        .y_desc("Count")
        .draw()?;

    chart.draw_series((0..bin_count).map(|idx| {
        let x0 = min_value + (idx as f64) * bin_width;
        let x1 = x0 + bin_width;
        Rectangle::new([(x0, 0), (x1, counts[idx])], BLUE.filled())
    }))?;

    root.present()?;
    println!("Saved overlap histogram to {}", path.display());
    Ok(())
}

#[derive(Default, Debug, Clone, Copy)]
struct RampSummary {
    frames_with_ramp: usize,
    total_ramps: usize,
    total_strength: usize,
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().sum();
    sum / values.len() as f64
}

struct ExportSample {
    ciphertext: BigUint,
    message_bytes_le: Vec<u8>,
    decryption_bytes_le: Vec<u8>,
}

fn bit_from_bytes_le(bytes: &[u8], idx: usize) -> bool {
    let byte_idx = idx / 8;
    if byte_idx >= bytes.len() {
        return false;
    }
    let bit_idx = idx % 8;
    ((bytes[byte_idx] >> bit_idx) & 1) == 1
}

fn biguint_to_bits_le(value: &BigUint, width: usize) -> Vec<bool> {
    let bytes = value.to_bytes_le();
    (0..width)
        .map(|idx| bit_from_bytes_le(&bytes, idx))
        .collect()
}

fn collect_speculative_oracle_bits(
    ctx: &RSAContext,
    engine: &EngineConfig,
    message: &BigUint,
    k_oracles: usize,
    rng: &mut StdRng,
) -> Result<Vec<Vec<bool>>, Box<dyn Error>> {
    if k_oracles == 0 {
        return Err("combiner_k_oracles must be >= 1".into());
    }

    let bit_width = message.bits().max(1) as usize;
    let settings = build_r_candidate_settings(engine);
    let batch_size = engine.process_count.max(engine.process_min_count).max(1) as usize;
    let candidates = generate_r_candidates_batch(&ctx.n, &settings, rng, batch_size);
    if candidates.is_empty() {
        return Err("no r candidates generated for combiner".into());
    }

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);
    let ciphertext = message.modpow(&ctx.e, &ctx.n);
    let result_default = get_larger_number(&ciphertext, &ctx.n, y, true, false);

    let mut oracles = Vec::with_capacity(k_oracles.min(candidates.len()));
    for (r, factors) in candidates {
        if oracles.len() >= k_oracles {
            break;
        }

        let phi_new = compute_totient(&factors);
        let Some(d_new) = mod_inverse(&ctx.e, &phi_new) else {
            continue;
        };

        let hbc_result = hbc(&result_default, &r, &n_pow_y, engine);
        let recovered_new = if engine.use_rs_decrypt {
            hbc_result.modpow(&d_new, &r)
        } else {
            hbc_result
        };

        let r_pow_y = r.pow(y);
        let result2_default = get_larger_number(&recovered_new, &r, y, true, false);
        let hbc_default = hbc(&result2_default, &ctx.n, &r_pow_y, engine);
        let dm_raw = &hbc_default % &ctx.n;
        let width = dm_raw.bits().max(1);
        let mask = (BigUint::one() << width) - BigUint::one();
        let inverted_dm = &mask ^ &dm_raw;
        let dm = if engine.invert_bits { inverted_dm } else { dm_raw };

        oracles.push(biguint_to_bits_le(&dm, bit_width));
    }

    if oracles.is_empty() {
        return Err("no valid r candidates for combiner".into());
    }

    Ok(oracles)
}

fn run_enciphered_export(
    ctx: &RSAContext,
    engine: &EngineConfig,
    rng: &mut StdRng,
) -> Result<(), Box<dyn Error>> {
    let iterations = engine.enciphered_export_iterations.max(1) as usize;
    let fixed_message = if engine.message.is_random {
        None
    } else {
        let msg = BigUint::from_bytes_be(engine.message.fixed_message.as_bytes());
        if msg.is_zero() {
            return Err("enciphered export fixed_message cannot be empty".into());
        }
        if msg >= ctx.n {
            return Err("enciphered export fixed_message must be smaller than modulus n".into());
        }
        Some(msg)
    };
    let fixed_message_bits = fixed_message
        .as_ref()
        .map(|msg| msg.bits().max(1) as usize)
        .unwrap_or(0);
    let bit_width = engine
        .message
        .bits
        .max(1)
        .max(fixed_message_bits as u32) as usize;

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);
    let settings = build_r_candidate_settings(engine);
    let batch_size = engine.process_count.max(engine.process_min_count).max(1) as usize;
    let mut candidates = generate_r_candidates_batch(&ctx.n, &settings, rng, batch_size);
    if candidates.is_empty() {
        return Err("no r candidates generated for enciphered export".into());
    }
    let (r, factors) = candidates
        .drain(..1)
        .next()
        .ok_or("missing r candidate for enciphered export")?;
    let phi_new = compute_totient(&factors);
    let d_new = mod_inverse(&ctx.e, &phi_new)
        .ok_or("public exponent is not invertible for export r candidate")?;

    println!(
        "Enciphered export using r candidate {} with factors {:?}",
        r, factors
    );

    let r_pow_y = r.pow(y);
    let mut seeds = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        seeds.push(rng.next_u64());
    }

    let done = Arc::new(AtomicU64::new(0));
    let next_pct = Arc::new(AtomicU64::new(10));
    let iterations_u64 = iterations as u64;
    let mut samples: Vec<ExportSample> = seeds
        .into_par_iter()
        .map(|seed| {
            let msg = if let Some(ref fixed) = fixed_message {
                fixed.clone()
            } else {
                let mut local_rng = StdRng::seed_from_u64(seed);
                random_message_under_n(engine, &ctx.n, &mut local_rng)
            };
            let ciphertext = msg.modpow(&ctx.e, &ctx.n);
            let result_default = get_larger_number(&ciphertext, &ctx.n, y, true, false);
            let hbc_result = hbc(&result_default, &r, &n_pow_y, engine);
            let recovered_new = if engine.use_rs_decrypt {
                hbc_result.modpow(&d_new, &r)
            } else {
                hbc_result
            };
            let result2_default = get_larger_number(&recovered_new, &r, y, true, false);
            let hbc_default = hbc(&result2_default, &ctx.n, &r_pow_y, engine);
            let dm_raw = &hbc_default % &ctx.n;
            let width = dm_raw.bits().max(1);
            let mask = (BigUint::one() << width) - BigUint::one();
            let inverted_dm = &mask ^ &dm_raw;
            let dm = if engine.invert_bits { inverted_dm } else { dm_raw };

            let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
            let pct = finished.saturating_mul(100) / iterations_u64;
            let mut current_next = next_pct.load(Ordering::Relaxed);
            while pct >= current_next && current_next <= 100 {
                let new_next = current_next.saturating_add(10);
                match next_pct.compare_exchange(
                    current_next,
                    new_next,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        let display_pct = current_next.min(100);
                        println!(
                            "Enciphered export iterations progress: {}% ({}/{})",
                            display_pct, finished, iterations_u64
                        );
                        break;
                    }
                    Err(actual) => current_next = actual,
                }
            }

            ExportSample {
                ciphertext,
                message_bytes_le: msg.to_bytes_le(),
                decryption_bytes_le: dm.to_bytes_le(),
            }
        })
        .collect();

    if samples.is_empty() {
        return Err("no speculative decryptions generated for enciphered export".into());
    }

    samples.sort_by(|a, b| a.ciphertext.cmp(&b.ciphertext));
    let min_ct = samples
        .first()
        .map(|s| s.ciphertext.clone())
        .ok_or("missing min ciphertext")?;
    let max_ct = samples
        .last()
        .map(|s| s.ciphertext.clone())
        .ok_or("missing max ciphertext")?;

    let bins = bit_width.max(1);
    let window_size = engine
        .enciphered_export_window
        .max(1)
        .min(samples.len());
    let stride = engine.enciphered_export_stride.max(1);
    let frame_count = if samples.len() <= window_size {
        1
    } else {
        ((samples.len() - window_size) / stride) + 1
    };

    let output_path = engine.enciphered_export_output_csv.as_str();
    let mut csv = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output_path)?;

    writeln!(csv, "# enciphered_bins_export")?;
    writeln!(csv, "# iterations={}", iterations)?;
    writeln!(csv, "# bins={}", bins)?;
    writeln!(csv, "# window_size={}", window_size)?;
    writeln!(csv, "# stride={}", stride)?;
    writeln!(csv, "# min_ciphertext={}", min_ct)?;
    writeln!(csv, "# max_ciphertext={}", max_ct)?;
    writeln!(csv, "# bit_width={}", bit_width)?;
    writeln!(
        csv,
        "frame_index,frame_start,frame_end,bit_index,match_count,match_pct"
    )?;

    let ramp_tolerances = engine.enciphered_export_ramp_tolerances.clone();
    let mut ramp_csv: Option<fs::File> = None;
    if !ramp_tolerances.is_empty() {
        let ramp_path = engine.enciphered_export_ramp_csv.as_str();
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(ramp_path)?;
        writeln!(file, "# enciphered_ramps_export")?;
        writeln!(file, "# ramp_length={}", engine.enciphered_export_ramp_length)?;
        writeln!(
            file,
            "# ramp_step_pct={}",
            engine.enciphered_export_ramp_step_pct
        )?;
        writeln!(file, "# tolerances={:?}", ramp_tolerances)?;
        writeln!(
            file,
            "frame_index,tolerance,ramp_start,ramp_length,ramp_values,mean_count_pct"
        )?;
        ramp_csv = Some(file);
    }

    let mut summaries = vec![RampSummary::default(); ramp_tolerances.len()];

    for frame_idx in 0..frame_count {
        let start = frame_idx * stride;
        let end = (start + window_size).min(samples.len());
        let window = &samples[start..end];
        let mut match_counts = vec![0u32; bins];
        let window_len_f = window.len().max(1) as f64;

        for sample in window {
            for bit_idx in 0..bins {
                let dm_bit = bit_from_bytes_le(&sample.decryption_bytes_le, bit_idx);
                let msg_bit = bit_from_bytes_le(&sample.message_bytes_le, bit_idx);
                if dm_bit == msg_bit {
                    match_counts[bit_idx] = match_counts[bit_idx].saturating_add(1);
                }
            }
        }

        let mut counts_pct = vec![0.0_f64; bins];
        for (bin_idx, count) in match_counts.iter().enumerate() {
            let match_pct = (*count as f64 / window_len_f) * 100.0;
            counts_pct[bin_idx] = match_pct;
            writeln!(
                csv,
                "{},{},{},{},{},{:.8}",
                frame_idx,
                start,
                end,
                bin_idx,
                count,
                match_pct
            )?;
        }

        if !ramp_tolerances.is_empty() {
            let mean = mean_f64(&counts_pct);

            for (idx, tol) in ramp_tolerances.iter().enumerate() {
                let ramps = find_ramp_signals_f64(
                    &counts_pct,
                    engine.enciphered_export_ramp_length,
                    engine.enciphered_export_ramp_step_pct,
                    *tol,
                );
                let strength = ramp_signal_strength_f64(&ramps);
                let entry = &mut summaries[idx];
                if !ramps.is_empty() {
                    entry.frames_with_ramp += 1;
                }
                entry.total_ramps = entry.total_ramps.saturating_add(ramps.len());
                entry.total_strength = entry.total_strength.saturating_add(strength);

                if let Some(file) = ramp_csv.as_mut() {
                    for (ramp_start, ramp_len, ramp_vals) in ramps {
                        let values_str = ramp_vals
                            .iter()
                            .map(|v| format!("{:.4}", v))
                            .collect::<Vec<_>>()
                            .join("|");
                        writeln!(
                            file,
                            "{},{},{},{},{},{:.4}",
                            frame_idx,
                            tol,
                            ramp_start,
                            ramp_len,
                            values_str,
                            mean
                        )?;
                    }
                }
            }
        }
    }

    println!(
        "Enciphered export wrote {} frames to {}",
        frame_count, output_path
    );
    if !summaries.is_empty() {
        println!(
            "Ramp summary (centered around mean, step {:.4}%):",
            engine.enciphered_export_ramp_step_pct
        );
        for (tol, summary) in ramp_tolerances.iter().zip(summaries.iter()) {
            println!(
                "  tolerance {} -> frames with ramp {}, total ramps {}, total strength {}",
                tol, summary.frames_with_ramp, summary.total_ramps, summary.total_strength
            );
        }
    }

    Ok(())
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

fn build_r_candidate_settings(engine: &EngineConfig) -> RCandidateSettings {
    let override_best_r = engine.override_best_r.as_ref().and_then(|raw| {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            trimmed.parse::<BigUint>().ok()
        }
    });

    RCandidateSettings {
        mode: engine.r_candidate_mode,
        override_best_r,
        process_min_factor: BigUint::from(engine.process_min_factor),
        process_count: engine.process_count,
        process_min_count: engine.process_min_count,
        process_scale: engine.process_scale,
        reuse_r_candidates_path: engine.reuse_r_candidates_path.clone(),
        reuse_r_candidates: engine.reuse_r_candidates,
        reuse_r_candidates_append_only: engine.reuse_r_candidates_append_only,
        small_primes: engine
            .r_candidate_small_primes
            .iter()
            .map(|p| BigUint::from(*p))
            .collect(),
        small_prime_factors_per_candidate: engine.r_candidate_small_prime_factors,
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

fn default_invert_bits() -> bool {
    false
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

fn default_r_candidate_mode() -> RCandidateMode {
    RCandidateMode::Factoring
}

fn default_r_candidate_small_primes() -> Vec<u64> {
    vec![3, 5, 7, 11, 13, 17]
}

fn default_r_candidate_small_prime_factors() -> usize {
    3
}

fn default_combiner_enable() -> bool {
    false
}

fn default_combiner_k_oracles() -> usize {
    5
}

fn default_combiner_match_probability() -> f64 {
    0.75
}

fn default_combiner_tie_breaker() -> bool {
    true
}

fn default_enciphered_export_enable() -> bool {
    false
}

fn default_enciphered_export_iterations() -> u64 {
    10_000
}

fn default_enciphered_export_bins() -> usize {
    128
}

fn default_enciphered_export_window() -> usize {
    512
}

fn default_enciphered_export_stride() -> usize {
    64
}

fn default_enciphered_export_output_csv() -> String {
    "enciphered_decryption_bins.csv".to_string()
}

fn default_enciphered_export_ramp_length() -> usize {
    3
}

fn default_enciphered_export_ramp_step_pct() -> f64 {
    0.05
}

fn default_enciphered_export_ramp_tolerances() -> Vec<f64> {
    vec![0.005, 0.01, 0.02]
}

fn default_enciphered_export_ramp_csv() -> String {
    "enciphered_ramps.csv".to_string()
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

#[derive(Clone, Debug, Default)]
struct MatchHistogram {
    matches: Vec<u64>,
    samples: Vec<u64>,
}

impl MatchHistogram {
    fn new() -> Self {
        Self {
            matches: Vec::new(),
            samples: Vec::new(),
        }
    }

    fn update(&mut self, a: &BigUint, b: &BigUint) {
        let a_bits = a.to_str_radix(2);
        let b_bits = b.to_str_radix(2);
        let min_len = a_bits.len().min(b_bits.len());
        if min_len == 0 {
            return;
        }

        if self.matches.len() < min_len {
            self.matches.resize(min_len, 0);
            self.samples.resize(min_len, 0);
        }

        for i in 0..min_len {
            let a_bit = a_bits.as_bytes()[a_bits.len() - 1 - i];
            let b_bit = b_bits.as_bytes()[b_bits.len() - 1 - i];
            self.samples[i] = self.samples[i].saturating_add(1);
            if a_bit == b_bit {
                self.matches[i] = self.matches[i].saturating_add(1);
            }
        }
    }

    fn write_histogram(&self, label: &str) -> Result<(), Box<dyn Error>> {
        if self.samples.is_empty() {
            return Ok(());
        }

        let images_dir = Path::new("./images");
        fs::create_dir_all(images_dir)?;

        static BIT_HIST_SEQ: AtomicUsize = AtomicUsize::new(0);
        let seq = BIT_HIST_SEQ.fetch_add(1, Ordering::Relaxed);
        let safe_label: String = label
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        let file_name = format!("bit_match_frequency_{}_{}.png", safe_label, seq);
        let path = images_dir.join(file_name);

        let data_len = self.samples.len();
        let mut bars: Vec<(usize, f64)> = Vec::with_capacity(data_len);
        for idx in 0..data_len {
            let sample = self.samples[idx].max(1);
            let pct = (self.matches[idx] as f64) / (sample as f64) * 100.0;
            bars.push((idx, pct));
        }

        let max_pct = bars
            .iter()
            .map(|(_, pct)| *pct)
            .fold(0.0_f64, f64::max)
            .max(1.0);

        let root = BitMapBackend::new(&path, (1400, 800)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("Bit Match Frequency (%) ({})", label),
                ("sans-serif", 30).into_font(),
            )
            .margin(20)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(0usize..data_len, 0f64..max_pct)?;

        chart
            .configure_mesh()
            .x_desc("Bit position (LSB=0)")
            .y_desc("Match frequency %")
            .y_label_formatter(&|v| format!("{:.0}", v))
            .draw()?;

        chart.draw_series(bars.iter().map(|(idx, pct)| {
            Rectangle::new(
                [(*idx, 0.0), (*idx + 1, *pct)],
                BLUE.mix(0.7).filled(),
            )
        }))?;

        root.present()?;
        println!("Saved bit match frequency histogram to {}", path.display());
        Ok(())
    }
}

fn run_message_trial(
    ctx: &RSAContext,
    _config: &Config,
    engine: &EngineConfig,
    message: &BigUint,
    min_message_trials: u64,
    rng: &mut StdRng,
    histogram: &mut MatchHistogram,
) -> Result<TestReport, Box<dyn Error>> {
    let attempts = min_message_trials.max(1);
    let mut best: Option<TestReport> = None;
    let mut worst: Option<TestReport> = None;

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);

    let settings = build_r_candidate_settings(engine);
    let batch_size = engine.process_count.max(engine.process_min_count).max(1) as usize;
    let candidates = generate_r_candidates_batch(&ctx.n, &settings, rng, batch_size);
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

        let ciphertext = msg.modpow(&ctx.e, &ctx.n);
        let recovered = ciphertext.modpow(&ctx.d, &ctx.n);
        if recovered != msg {
            return Err("RSA round trip failed".into());
        }

        let result_default = get_larger_number(&ciphertext, &ctx.n, y, true, false);

        for (r, factors) in &candidates {
            let phi_new = compute_totient(factors);
            let Some(d_new) = mod_inverse(&ctx.e, &phi_new) else {
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
            let dm_raw = &hbc_default % &ctx.n;
            let width = dm_raw.bits().max(1);
            let mask = (BigUint::one() << width) - BigUint::one();
            let inverted_dm = &mask ^ &dm_raw; // Invert within current width
            let dm = if engine.invert_bits { inverted_dm } else { dm_raw };
            histogram.update(&dm, &msg);

            let (matching_lsb, matching_total) = count_matching_bits(&dm, &msg);
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
    label: &str,
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
    let d_new = mod_inverse(&ctx.e, &phi_new)?;

    let iter_count = iterations as usize;
    let mut seeds = Vec::with_capacity(iter_count);
    for _ in 0..iter_count {
        seeds.push(rng.next_u64());
    }

    let done = Arc::new(AtomicU64::new(0));
    let next_pct = Arc::new(AtomicU64::new(10));

    let overlaps_pct = Arc::new(Mutex::new(Vec::with_capacity(iter_count)));
    let bit_hist = Arc::new(Mutex::new(MatchHistogram::new()));

    let samples: Vec<(f64, f64, usize)> = seeds
        .into_par_iter()
        .map(|seed| {
            let mut local_rng = StdRng::seed_from_u64(seed);
            
            let msg = random_message_under_n(engine, &ctx.n, &mut local_rng);
            let ciphertext = msg.modpow(&ctx.e, &ctx.n);
            let result_default = get_larger_number(&ciphertext, &ctx.n, y, true, true);

            let hbc_result = hbc(&result_default, r, &n_pow_y, engine);
            let recovered_new = if engine.use_rs_decrypt {
                hbc_result.modpow(&d_new, r)
            } else {
                hbc_result
            };

            let result2_default = get_larger_number(&recovered_new, r, y, true, true);
            let hbc_default = hbc(&result2_default, &ctx.n, &r_pow_y, engine);
            let dm_raw = &hbc_default % &ctx.n;
            let width = dm_raw.bits().max(1);
            let mask = (BigUint::one() << width) - BigUint::one();
            let inverted_dm = &mask ^ &dm_raw; // Invert within current width
            let dm = if engine.invert_bits { inverted_dm } else { dm_raw };

            let (matching_lsb, matching_total) = count_matching_bits(&dm, &msg);
            if let Ok(mut hist) = bit_hist.lock() {
                hist.update(&dm, &msg);
            }
            let overlap = (matching_total as f64) / (msg.bits().max(1) as f64);
            let lsb_f = matching_lsb as f64;
            if let Ok(mut guard) = overlaps_pct.lock() {
                guard.push(overlap * 100.0);
            }

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
    let _n = samples.len() as f64;

    let bits_values: Vec<f64> = samples.iter().map(|(b, _, _)| *b).collect();
    let overlap_values_pct: Vec<f64> = samples.iter().map(|(_, o, _)| o * 100.0).collect();
    if let Ok(hist) = bit_hist.lock() {
        if let Err(err) = hist.write_histogram(&format!("{}_bit_matches", label)) {
            println!("Failed to write bit match histogram (alt): {}", err);
        }
    }
    let max_bits = samples.iter().map(|(_, _, mb)| *mb).max().unwrap_or(0);

    let bits_stats = compute_stats(&bits_values).unwrap();
    let overlap_stats = compute_stats(&overlap_values_pct).unwrap_or_else(|| StatSummary {
        mean: 0.0,
        stddev: 0.0,
        min: 0.0,
        max: 0.0,
        count: 0,
    });

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

    if let Err(err) = plot_overlap_histogram(&overlap_values_pct, label) {
        println!("Failed to write overlap histogram: {}", err);
    }

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
        label,
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

#[cfg(test)]
mod tests {
    use super::*;
    use rsademo::dsp::{find_ramp_signals, ramp_signal_strength};
    use rand::SeedableRng;

    #[test]
    fn test_ramp_detect () {
        let mut hist = MatchHistogram::new();
        let msg1 = BigUint::from(0b11110000u8);
        let msg2 = BigUint::from(0b11100000u8);
        hist.update(&msg1, &msg2);
    }

    #[test]
    fn test_analysis_detect_ramp() {
        // Sample dataset: mean is 10, ramp should be 11, 12, 13
        let bins = vec![8, 9, 10, 11, 12, 13, 7, 8];
        let ramps = find_ramp_signals(&bins, 3, 0);
        println!("Detected ramps in analysis: {:?}", ramps);
        let strength = ramp_signal_strength(&ramps);
        println!("Signal strength in analysis: {}", strength);

        // Check that at least one ramp is detected and signal strength is correct
        assert!(!ramps.is_empty());
        assert!(strength > 0);
    }

    #[test]
    fn test_r_candidates_small_primes_success() {
        let p = BigUint::from(61u8);
        let q = BigUint::from(53u8);
        let n = &p * &q;
        let phi = (&p - BigUint::one()) * (&q - BigUint::one());
        let e = choose_exponent(3, &phi);
        let d = mod_inverse(&e, &phi).expect("missing inverse");

        let ctx = RSAContext {
            p,
            q,
            n: n.clone(),
            phi,
            e,
            d,
        };

        let mut config = Config::default();
        config.engine.r_candidate_mode = RCandidateMode::SmallPrimes;
        config.engine.r_candidate_small_primes = vec![3, 5, 7];
        config.engine.r_candidate_small_prime_factors = 3;
        config.engine.process_min_factor = 3;
        config.engine.process_count = 1;
        config.engine.process_min_count = 1;
        config.engine.min_message_trials = 1;
        config.engine.rabin_exponent = 3;

        let msg = BigUint::from(42u8);
        let mut rng = StdRng::seed_from_u64(101);
        let mut hist = MatchHistogram::new();
        let result = run_message_trial(&ctx, &config, &config.engine, &msg, 1, &mut rng, &mut hist);
        if let Err(err) = &result {
            println!("r candidates success test failed: {}", err);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_r_candidates_decrypt_may_fail() {
        let p = BigUint::from(61u8);
        let q = BigUint::from(53u8);
        let n = &p * &q;
        let phi = (&p - BigUint::one()) * (&q - BigUint::one());
        let e = choose_exponent(3, &phi);
        let d = mod_inverse(&e, &phi).expect("missing inverse");

        let ctx = RSAContext {
            p,
            q,
            n,
            phi,
            e,
            d,
        };

        let mut config = Config::default();
        config.engine.r_candidate_mode = RCandidateMode::SmallPrimes;
        config.engine.r_candidate_small_primes = vec![3, 5]; // too few primes for 3-factor candidates
        config.engine.r_candidate_small_prime_factors = 3;
        config.engine.process_min_factor = 3;
        config.engine.process_count = 1;
        config.engine.process_min_count = 1;
        config.engine.min_message_trials = 1;
        config.engine.rabin_exponent = 3;

        let msg = BigUint::from(42u8);
        let mut rng = StdRng::seed_from_u64(102);
        let mut hist = MatchHistogram::new();
        let result = run_message_trial(&ctx, &config, &config.engine, &msg, 1, &mut rng, &mut hist);
        if let Err(err) = &result {
            println!("Expected r candidate failure: {}", err);
        }
        assert!(result.is_err());
    }
}
