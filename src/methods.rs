/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>

use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use rayon::prelude::*;
#[cfg(feature = "plots")]
use plotters::prelude::*;
#[cfg(feature = "plots")]
use std::sync::atomic::AtomicUsize;
#[cfg(feature = "plots")]
use std::path::Path;

#[cfg(not(feature = "plots"))]
type RGBColor = (u8, u8, u8);

#[cfg(not(feature = "plots"))]
const RED: RGBColor = (220, 20, 60);
#[cfg(not(feature = "plots"))]
const GREEN: RGBColor = (46, 139, 87);
#[cfg(not(feature = "plots"))]
const BLUE: RGBColor = (30, 144, 255);
#[cfg(not(feature = "plots"))]
const BLACK: RGBColor = (0, 0, 0);

use num_bigint::BigUint;
use num_traits::{One, Zero};
use rand::RngCore;
use serde::Serialize;
use serde_json::json;
use crate::analytics::{
    generate_r_candidates_with_analytics, RCandidateAccuracyBatch, RCandidateAccuracyEntry,
    RCandidateFactor, RCandidateTraceBatch, RCandidateTraceEntry, SessionAnalytics,
};
use crate::combiner::majority_vote_with_distribution;
use crate::config::{Config, EngineConfig};
use crate::dsp::{find_ramp_signals_f64, ramp_signal_strength_f64};
use crate::math::{
    bit_length, choose_exponent, compute_totient, factor_composite_with_timeout,
    is_probable_prime_big, mod_inverse, modular_sqrt, random_biguint_bits,
    random_prime_with_bits, shannon_entropy_bit, to_hex,
};
use crate::r_candidates::RCandidateSettings;
use crate::rng::{RngChoice, RngMode};

/// Input arguments for the RSA demo and analysis pipeline.
pub struct DemoArgs {
    pub bits: u32,
    pub message: Option<String>,
    pub public_exponent: u64,
    pub seed: Option<u64>,
    pub crypto_rng: bool,
    pub tests: bool,
    pub export: bool,
    pub shift: bool,
}

/// Executes an analytics update inside a shared session lock.
///
/// # Parameters
/// - `analytics`: Shared analytics session wrapper.
/// - `action`: Callback that mutates the analytics session.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Applies the mutation when the lock is available; no stdout/stderr output.
fn with_analytics<F>(analytics: &Arc<Mutex<SessionAnalytics>>, action: F)
where
    F: FnOnce(&mut SessionAnalytics),
{
    if let Ok(mut guard) = analytics.lock() {
        action(&mut guard);
    }
}

/// Runs the core RSA demo and analysis pipeline.
///
/// # Parameters
/// - `args`: Parsed demo arguments controlling key generation and message selection.
/// - `config`: Loaded configuration driving analysis features.
/// - `analytics`: Session analytics accumulator for timing and r candidate data.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error on failure.
///
/// # Expected Output
/// - Prints RSA parameters and analysis summaries; may emit CSV/PNG artifacts.
pub fn run_demo(
    args: DemoArgs,
    config: Config,
    analytics: &Arc<Mutex<SessionAnalytics>>,
) -> Result<(), Box<dyn Error>> {
    with_analytics(analytics, |a| {
        a.mark_feature("keypair", true);
        a.mark_feature("message_select", true);
        a.mark_feature("rsa_roundtrip", true);
        a.mark_feature("combiner", config.engine.combiner_enable);
        a.mark_feature("test_iterations", config.engine.test_iterations > 0 && args.export);
        a.mark_feature("information_sufficiency", args.tests);
        a.mark_feature("enciphered_export", config.engine.enciphered_export_enable);
        a.mark_feature("r_candidate_accuracy", config.engine.analysis_batch_enable);
        a.mark_feature(
            "r_use_list",
            config.engine.r_use_list_enable && !config.engine.r_use_list.is_empty(),
        );
        a.mark_feature("r_stress", config.engine.r_stress_test_enable);
        a.set_feature_stat("rsa_roundtrip", "shift_enabled", json!(args.shift));
    });

    let rng_start = Instant::now();
    let rng_mode = if args.crypto_rng {
        RngMode::Crypto
    } else {
        RngMode::Standard
    };
    let mut rng: RngChoice = match args.seed {
        Some(seed) => RngChoice::from_seed(rng_mode, seed),
        None => RngChoice::from_entropy(rng_mode)?,
    };
    with_analytics(analytics, |a| a.record_step("rng_init", rng_start.elapsed()));

    let key_start = Instant::now();
    let (p, q): (BigUint, BigUint) = if config.rsa_keypair.generate {
        let p = random_prime_with_bits(args.bits, &mut rng);
        let mut q = random_prime_with_bits(args.bits, &mut rng);
        while q == p {
            q = random_prime_with_bits(args.bits, &mut rng);
        }
        (p, q)
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
    with_analytics(analytics, |a| {
        a.record_step("keypair_select", key_start.elapsed());
        a.record_feature_duration("keypair", key_start.elapsed());
    });

    let one = BigUint::one();
    let n = &p * &q;
    let phi = (&p - &one) * (&q - &one);

    let exponent_start = Instant::now();
    let start_e = if args.public_exponent != 65_537 {
        args.public_exponent
    } else {
        config.rsa_keypair.e
    };
    let e = choose_exponent(start_e, &phi);
    let d = mod_inverse(&e, &phi)
        .ok_or("public exponent is not invertible; try a different size or exponent")?;
    with_analytics(analytics, |a| a.record_step("keypair_derive", exponent_start.elapsed()));

    let message_start = Instant::now();
    let message = select_message(args.message.clone(), &config.engine, &mut rng);
    if message.is_zero() {
        return Err("message cannot be empty".into());
    }
    if message >= n {
        return Err("message must be smaller than the modulus n".into());
    }
    with_analytics(analytics, |a| {
        a.record_step("message_select", message_start.elapsed());
        a.record_feature_duration("message_select", message_start.elapsed());
    });

    let roundtrip_start = Instant::now();
    let ciphertext = message.modpow(&e, &n);
    let recovered = ciphertext.modpow(&d, &n);

    if recovered != message {
        return Err("RSA round trip failed".into());
    }
    with_analytics(analytics, |a| {
        a.record_step("rsa_roundtrip", roundtrip_start.elapsed());
        a.record_feature_duration("rsa_roundtrip", roundtrip_start.elapsed());
    });

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
        let combiner_start = Instant::now();
        let bit_width = message.bits().max(1) as usize;
        let majority_bits = biguint_to_bits_le(&message, bit_width);
        let requested_oracles = config.engine.combiner_k_oracles;
        match collect_speculative_oracle_bits(
            &ctx,
            &config.engine,
            &message,
            requested_oracles,
            &mut rng,
            analytics,
            args.shift,
        ) {
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
                    with_analytics(analytics, |a| {
                        a.set_feature_stat("combiner", "requested_oracles", json!(requested_oracles));
                        a.set_feature_stat("combiner", "used_oracles", json!(distribution.total_oracles));
                        a.set_feature_stat("combiner", "bit_width", json!(total));
                        a.set_feature_stat("combiner", "accuracy_pct", json!(accuracy * 100.0));
                    });
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
                    with_analytics(analytics, |a| {
                        a.add_feature_note("combiner", &format!("majority vote failed: {}", err));
                    });
                }
            },
            Err(err) => {
                println!("Speculative combiner setup failed: {}", err);
                with_analytics(analytics, |a| {
                    a.add_feature_note("combiner", &format!("setup failed: {}", err));
                });
            }
        }
        with_analytics(analytics, |a| a.record_feature_duration("combiner", combiner_start.elapsed()));
    }

    if config.engine.test_iterations > 0 && args.export {
        let test_iterations_start = Instant::now();
        let mut bit_hist = MatchHistogram::new();
        let iterations = config.engine.test_iterations;
        let mut reports = Vec::new();
        let mut next_pct = 10u64;
        let mut iteration_total = Duration::ZERO;
        for i in 0..iterations {
            let iteration_start = Instant::now();
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
                analytics,
                args.shift,
            )?;
            reports.push(report);
            iteration_total += iteration_start.elapsed();

            log_progress_every_ten_percent(i + 1, iterations, &mut next_pct, "Test iterations");
        }
        with_analytics(analytics, |a| {
            a.record_step_summary("test_iterations_run_message_trial", iterations, iteration_total);
        });

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
                    args.shift,
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
                    args.shift,
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
            let stress_start = Instant::now();
            println!("\nRunning r_use_list stress tests...");
            for r_str in &config.engine.r_use_list {
                if let Ok(r) = r_str.parse::<BigUint>() {
                    if is_probable_prime_big(&r) {
                        continue;
                    }
                    run_r_stress_entry(
                        "r_use_list",
                        &r,
                        &ctx,
                        &config,
                        &config.engine,
                        &mut rng,
                        args.shift,
                    );
                }
            }
            with_analytics(analytics, |a| {
                a.record_feature_duration("r_use_list", stress_start.elapsed());
            });
        }

        // r_stress range testing
        if config.engine.r_stress_test_enable {
            let stress_start = Instant::now();
            if let (Some(start), Some(end)) = (&config.engine.r_stress_start, &config.engine.r_stress_end) {
                println!("\nRunning r_stress range tests...");
                let mut current = start.clone();
                while &current <= end {
                    if !is_probable_prime_big(&current) {
                        run_r_stress_entry(
                            "r_stress",
                            &current,
                            &ctx,
                            &config,
                            &config.engine,
                            &mut rng,
                            args.shift,
                        );
                    }
                    current += BigUint::one();
                }
            }
            with_analytics(analytics, |a| {
                a.record_feature_duration("r_stress", stress_start.elapsed());
            });
        }

        if let Err(err) = bit_hist.write_histogram("test_iterations_bit_matches") {
            println!("Failed to write bit match histogram: {}", err);
        }
        with_analytics(analytics, |a| {
            a.record_feature_duration("test_iterations", test_iterations_start.elapsed());
            a.set_feature_stat("test_iterations", "iterations", json!(iterations));
            a.set_feature_stat("test_iterations", "best_match_found", json!(best_match.is_some()));
        });
    }

    if args.tests {
        let info_start = Instant::now();
        match run_information_sufficiency_tests(
            &ctx,
            &config,
            &message,
            &mut rng,
            args.export,
            analytics,
            args.shift,
        ) {
            Ok(()) => {
                with_analytics(analytics, |a| {
                    a.set_feature_stat("information_sufficiency", "status", json!("pass"));
                });
            }
            Err(err) => {
                with_analytics(analytics, |a| {
                    a.set_feature_stat("information_sufficiency", "status", json!("fail"));
                    a.record_feature_duration("information_sufficiency", info_start.elapsed());
                });
                return Err(err);
            }
        }
        with_analytics(analytics, |a| {
            a.record_feature_duration("information_sufficiency", info_start.elapsed());
            a.set_feature_stat(
                "information_sufficiency",
                "analysis_iterations",
                json!(config.engine.analysis_tests_iterations),
            );
        });
    }

    if config.engine.analysis_batch_enable {
        let batch_start = Instant::now();
        if let Err(err) =
            run_r_candidate_accuracy_batches(&ctx, &config.engine, &mut rng, analytics, args.shift)
        {
            with_analytics(analytics, |a| {
                a.add_feature_note("r_candidate_accuracy", &format!("failed: {}", err));
                a.record_feature_duration("r_candidate_accuracy", batch_start.elapsed());
            });
            return Err(err);
        }
        with_analytics(analytics, |a| {
            a.record_feature_duration("r_candidate_accuracy", batch_start.elapsed());
            a.set_feature_stat(
                "r_candidate_accuracy",
                "messages_per_batch",
                json!(1u64),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "candidates_per_batch",
                json!(config.engine.analysis_batch_candidates),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "batch_count",
                json!(config.engine.analysis_batch_batches),
            );
        });
    }

    if config.engine.enciphered_export_enable {
        let export_start = Instant::now();
        if let Err(err) = run_enciphered_export(&ctx, &config.engine, &mut rng, analytics, args.shift) {
            println!("Enciphered export failed: {}", err);
            with_analytics(analytics, |a| {
                a.add_feature_note("enciphered_export", &format!("failed: {}", err));
            });
        }
        with_analytics(analytics, |a| {
            a.record_feature_duration("enciphered_export", export_start.elapsed());
            a.set_feature_stat(
                "enciphered_export",
                "iterations",
                json!(config.engine.enciphered_export_iterations),
            );
        });
    }

    Ok(())
}

/// Logs progress updates at 10% increments.
///
/// # Parameters
/// - `done`: Number of completed items.
/// - `total`: Total number of items.
/// - `next_pct`: Mutable threshold for the next log event.
/// - `label`: Human-readable label for the progress report.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Prints progress updates to stdout when thresholds are reached.
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

#[derive(Debug, Serialize)]
struct BitSimilarityEntry {
    index: usize,
    shift: usize,
    r: String,
    candidate_hex: String,
    match_pct: f64,
    matching_bits: usize,
    adjusted_match_pct: f64,
    adjusted_matching_bits: usize,
    masked_bits: usize,
    base_match_pct: f64,
    base_matching_bits: usize,
}

/// Computes mean, standard deviation, min, and max for a slice of values.
///
/// # Parameters
/// - `values`: Input values to summarize.
///
/// # Returns
/// - `Option<StatSummary>`: Summary statistics or `None` if `values` is empty.
///
/// # Expected Output
/// - Returns `None` on empty input; no side effects.
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

/// Builds per-candidate bit similarity entries for a fixed message.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `candidates`: Prepared r candidates to evaluate.
/// - `message`: Reference message used for bit comparisons.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
/// - `shift_levels`: Number of left-shift multiplications to compare per candidate.
///
/// # Returns
/// - `(Vec<BitSimilarityEntry>, Vec<u32>, usize)`: Entries, per-bit match counts, and shift levels used.
///
/// # Expected Output
/// - Returns entries describing per-candidate bit matches; no stdout/stderr output.
fn build_bit_similarity_entries(
    ctx: &RSAContext,
    engine: &EngineConfig,
    candidates: &[OracleCandidate],
    message: &BigUint,
    shift: bool,
    shift_levels: usize,
) -> (Vec<BitSimilarityEntry>, Vec<u32>, usize) {
    if candidates.is_empty() {
        return (Vec::new(), Vec::new(), 0);
    }

    let bit_width = message.bits().max(1) as usize;
    let message_bits = biguint_to_bits_le(message, bit_width);
    let base_shift = if shift { 1usize } else { 0usize };
    let max_shift = (base_shift + shift_levels).min(bit_width.saturating_sub(1));
    let mut match_counts = vec![0u32; bit_width];

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);
    let ciphertext = message.modpow(&ctx.e, &ctx.n);
    let enc_two = BigUint::from(2u8).modpow(&ctx.e, &ctx.n);
    let mut shift_results = Vec::new();
    let mut enc_two_pow = BigUint::one();
    for shift_idx in 0..=max_shift {
        if shift_idx > 0 {
            enc_two_pow = (&enc_two_pow * &enc_two) % &ctx.n;
        }
        if shift_idx < base_shift {
            continue;
        }
        let shifted_ciphertext = (&ciphertext * &enc_two_pow) % &ctx.n;
        let result_default = get_larger_number(&shifted_ciphertext, &ctx.n, y, true, false);
        shift_results.push((shift_idx, result_default));
    }

    let mut entries = Vec::with_capacity(candidates.len() * (max_shift + 1).max(1));
    for (index, candidate) in candidates.iter().enumerate() {
        let mut base_match_pct = 0.0;
        let mut base_matching_bits = 0;
        let denom = bit_width.max(1) as f64;

        for (shift_idx, result_default) in &shift_results {
            let dm = derive_candidate_message_from_result(
                ctx,
                engine,
                result_default,
                &candidate.r,
                &candidate.d_new,
                &n_pow_y,
                &candidate.r_pow_y,
                y,
                false,
            );
            let dm_bits = biguint_to_bits_le(&dm, bit_width);
            let masked_bits = (*shift_idx).min(bit_width);
            let mut matching_bits = 0usize;
            for bit_idx in 0..bit_width {
                let cand_idx = bit_idx + *shift_idx;
                if cand_idx >= bit_width {
                    continue;
                }
                if dm_bits[cand_idx] == message_bits[bit_idx] {
                    matching_bits += 1;
                    match_counts[bit_idx] = match_counts[bit_idx].saturating_add(1);
                }
            }
            let adjusted_matching_bits = matching_bits;
            let match_pct = matching_bits as f64 / denom * 100.0;
            let adjusted_denom = bit_width.saturating_sub(masked_bits).max(1) as f64;
            let adjusted_match_pct = adjusted_matching_bits as f64 / adjusted_denom * 100.0;
            if *shift_idx == base_shift {
                base_match_pct = match_pct;
                base_matching_bits = matching_bits;
            }
            entries.push(BitSimilarityEntry {
                index,
                shift: *shift_idx,
                r: candidate.r.to_string(),
                candidate_hex: to_hex(&dm),
                match_pct,
                matching_bits,
                adjusted_match_pct,
                adjusted_matching_bits,
                masked_bits,
                base_match_pct,
                base_matching_bits,
            });
        }
    }

    (entries, match_counts, max_shift)
}

/// Writes a histogram image for overlap percentages.
///
/// # Parameters
/// - `overlaps_pct`: Overlap values in percentage form.
/// - `label`: Label used in the chart caption and filename.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an I/O/plotting error.
///
/// # Expected Output
/// - Writes a PNG into `./images` and prints the output path.
#[cfg(feature = "plots")]
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

#[cfg(not(feature = "plots"))]
fn plot_overlap_histogram(_overlaps_pct: &[f64], _label: &str) -> Result<(), Box<dyn Error>> {
    Ok(())
}

#[derive(Default, Debug, Clone, Copy)]
struct RampSummary {
    frames_with_ramp: usize,
    total_ramps: usize,
    total_strength: usize,
}

/// Computes the mean of a slice of `f64` values.
///
/// # Parameters
/// - `values`: Input values.
///
/// # Returns
/// - `f64`: Arithmetic mean (0.0 for empty input).
///
/// # Expected Output
/// - Returns a floating-point mean; no side effects.
fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().sum();
    sum / values.len() as f64
}

/// Precomputed r-candidate data for oracle and timeline tests.
#[derive(Clone, Debug)]
struct OracleCandidate {
    r: BigUint,
    d_new: BigUint,
    r_pow_y: BigUint,
}

#[derive(Clone, Debug)]
struct CandidateScore {
    candidate: OracleCandidate,
    matching_lsb: usize,
    matching_total: usize,
}

#[derive(Clone, Debug)]
struct OracleEntropySeries {
    entropy_mean: Vec<f64>,
    accuracy_pct: Vec<f64>,
}

#[derive(Clone, Debug)]
struct OracleTrainingSample {
    result_default: BigUint,
    message_bits: Vec<bool>,
}

#[derive(Clone, Debug)]
struct OracleBitSelection {
    oracle_idx: usize,
    invert: bool,
}

#[derive(Clone, Debug)]
struct MatchSample {
    message_bytes_le: Vec<u8>,
    candidate_bytes_le: Vec<u8>,
}

#[derive(Clone, Debug)]
struct MatchTimelineSeries {
    entropy_mean: Vec<f64>,
    match_pct_mean: Vec<f64>,
    bit_true_prob: Vec<Vec<f64>>,
}

#[derive(Clone, Debug)]
struct SpeculativeOracleReport {
    recovered: BigUint,
    matching_lsb: usize,
    matching_total: usize,
    bit_width: usize,
    match_pct: f64,
    oracles_per_bit: usize,
    unique_oracles: usize,
}

struct ExportSample {
    ciphertext: BigUint,
    message_bytes_le: Vec<u8>,
    decryption_bytes_le: Vec<u8>,
}

#[derive(Clone, Debug)]
struct FrameExportOutput {
    frame_idx: usize,
    match_rows: String,
    ramp_rows: String,
    ramp_summary: Vec<RampSummary>,
}

/// Reads a single bit from a little-endian byte slice.
///
/// # Parameters
/// - `bytes`: Little-endian byte slice.
/// - `idx`: Bit index (LSB = 0).
///
/// # Returns
/// - `bool`: The bit value at the requested index.
///
/// # Expected Output
/// - Returns `false` for out-of-range indices; no side effects.
fn bit_from_bytes_le(bytes: &[u8], idx: usize) -> bool {
    let byte_idx = idx / 8;
    if byte_idx >= bytes.len() {
        return false;
    }
    let bit_idx = idx % 8;
    ((bytes[byte_idx] >> bit_idx) & 1) == 1
}

/// Converts a `BigUint` to a fixed-width little-endian bit vector.
///
/// # Parameters
/// - `value`: Integer to convert.
/// - `width`: Number of bits to emit.
///
/// # Returns
/// - `Vec<bool>`: Little-endian bit vector of length `width`.
///
/// # Expected Output
/// - Returns a vector padded with `false` bits if needed; no side effects.
fn biguint_to_bits_le(value: &BigUint, width: usize) -> Vec<bool> {
    let bytes = value.to_bytes_le();
    (0..width)
        .map(|idx| bit_from_bytes_le(&bytes, idx))
        .collect()
}

/// Converts a little-endian bit vector into a `BigUint`.
///
/// # Parameters
/// - `bits`: Bit slice with LSB at index 0.
///
/// # Returns
/// - `BigUint`: Value represented by the bit slice.
///
/// # Expected Output
/// - Returns the integer value; no side effects.
fn bits_le_to_biguint(bits: &[bool]) -> BigUint {
    if bits.is_empty() {
        return BigUint::zero();
    }
    let byte_len = (bits.len() + 7) / 8;
    let mut bytes = vec![0u8; byte_len];
    for (idx, bit) in bits.iter().enumerate() {
        if *bit {
            let byte_idx = idx / 8;
            let bit_idx = idx % 8;
            bytes[byte_idx] |= 1u8 << bit_idx;
        }
    }
    BigUint::from_bytes_le(&bytes)
}

/// Counts matching bits between two little-endian bit vectors.
///
/// # Parameters
/// - `a`: First bit slice to compare.
/// - `b`: Second bit slice to compare.
///
/// # Returns
/// - `(usize, usize)`: `(matching_lsb_run, matching_total)` counts.
///
/// # Expected Output
/// - Returns counts based on bitwise comparisons; no side effects.
fn count_matching_bits_le(a: &[bool], b: &[bool]) -> (usize, usize) {
    let min_len = a.len().min(b.len());
    if min_len == 0 {
        return (0, 0);
    }

    let mut matching_total = 0usize;
    for i in 0..min_len {
        if a[i] == b[i] {
            matching_total += 1;
        }
    }

    let mut matching_lsb = 0usize;
    for i in 0..min_len {
        if a[i] == b[i] {
            matching_lsb += 1;
        } else {
            break;
        }
    }

    (matching_lsb, matching_total)
}

/// Builds speculative oracle bit vectors using `r` candidates and HBC transforms.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling HBC and candidate settings.
/// - `message`: Plaintext message used to derive oracle bits.
/// - `k_oracles`: Maximum number of oracle samples to collect.
/// - `rng`: Random number generator used for candidate sampling.
/// - `analytics`: Session analytics accumulator for r candidate metadata.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `Result<Vec<Vec<bool>>, Box<dyn Error>>`: Oracle bit vectors or an error if none.
///
/// # Expected Output
/// - Returns a non-empty list of bit vectors on success; no direct stdout output.
fn collect_speculative_oracle_bits(
    ctx: &RSAContext,
    engine: &EngineConfig,
    message: &BigUint,
    k_oracles: usize,
    rng: &mut RngChoice,
    analytics: &Arc<Mutex<SessionAnalytics>>,
    shift: bool,
) -> Result<Vec<Vec<bool>>, Box<dyn Error>> {
    if k_oracles == 0 {
        return Err("combiner_k_oracles must be >= 1".into());
    }

    let bit_width = message.bits().max(1) as usize;
    let settings = build_r_candidate_settings(engine);
    let batch_size = engine.process_count.max(engine.process_min_count).max(1) as usize;
    let candidates = generate_r_candidates_with_analytics(
        "combiner_oracles",
        &ctx.n,
        &settings,
        rng,
        batch_size,
        analytics,
    );
    if candidates.is_empty() {
        return Err("no r candidates generated for combiner".into());
    }

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);
    let ciphertext = message.modpow(&ctx.e, &ctx.n);
    let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
    let result_default = get_larger_number(&shifted, &ctx.n, y, true, false);

    record_r_candidate_trace_batch_from_factors(
        ctx,
        engine,
        message,
        &candidates,
        analytics,
        "combiner_oracles",
        shift,
    );

    let mut oracles = Vec::with_capacity(k_oracles.min(candidates.len()));
    for (r, factors) in candidates.iter() {
        if oracles.len() >= k_oracles {
            break;
        }

        let phi_new = compute_totient(&factors);
        let Some(d_new) = mod_inverse(&ctx.e, &phi_new) else {
            continue;
        };

        let r_pow_y = r.pow(y);
        let dm = derive_candidate_message_from_result(
            ctx,
            engine,
            &result_default,
            &r,
            &d_new,
            &n_pow_y,
            &r_pow_y,
            y,
            false,
        );

        oracles.push(biguint_to_bits_le(&dm, bit_width));
    }

    if oracles.is_empty() {
        return Err("no valid r candidates for combiner".into());
    }

    Ok(oracles)
}

/// Resolves the fixed message (if configured) for analysis timelines.
///
/// # Parameters
/// - `engine`: Engine configuration containing message settings.
/// - `n`: RSA modulus used to bound the fixed message.
///
/// # Returns
/// - `Result<Option<BigUint>, Box<dyn Error>>`: `Some(message)` when fixed, `None` when random.
///
/// # Expected Output
/// - Returns an error if the fixed message is empty or not less than `n`.
fn resolve_fixed_message_for_tests(
    engine: &EngineConfig,
    n: &BigUint,
) -> Result<Option<BigUint>, Box<dyn Error>> {
    if engine.message.is_random {
        return Ok(None);
    }
    let msg = BigUint::from_bytes_be(engine.message.fixed_message.as_bytes());
    if msg.is_zero() {
        return Err("analysis tests fixed_message cannot be empty".into());
    }
    if !n.is_zero() && msg >= *n {
        return Err("analysis tests fixed_message must be smaller than modulus n".into());
    }
    Ok(Some(msg))
}

/// Samples a message for analysis timelines.
///
/// # Parameters
/// - `engine`: Engine configuration controlling message selection.
/// - `n`: RSA modulus used to bound random messages.
/// - `fixed_message`: Optional fixed message override.
/// - `rng`: Random number generator for random sampling.
///
/// # Returns
/// - `BigUint`: Selected message value.
///
/// # Expected Output
/// - Returns a message under `n` when random; no side effects.
fn sample_message_for_tests(
    engine: &EngineConfig,
    n: &BigUint,
    fixed_message: &Option<BigUint>,
    rng: &mut RngChoice,
) -> BigUint {
    if let Some(msg) = fixed_message {
        return msg.clone();
    }
    random_message_under_n(engine, n, rng)
}

/// Builds precomputed r-candidate data for analysis timelines.
///
/// # Parameters
/// - `ctx`: RSA context containing modulus and exponent.
/// - `engine`: Engine configuration controlling candidate generation.
/// - `rng`: Random number generator for candidate sampling.
/// - `analytics`: Session analytics accumulator for r candidate metadata.
///
/// # Returns
/// - `Result<Vec<OracleCandidate>, Box<dyn Error>>`: Candidate list or an error if none.
///
/// # Expected Output
/// - May print candidate generation logs; no file output.
fn build_oracle_candidates(
    ctx: &RSAContext,
    engine: &EngineConfig,
    rng: &mut RngChoice,
    analytics: &Arc<Mutex<SessionAnalytics>>,
) -> Result<Vec<OracleCandidate>, Box<dyn Error>> {
    let settings = build_r_candidate_settings(engine);
    let batch_size = engine.process_count.max(engine.process_min_count).max(1) as usize;
    let candidates = generate_r_candidates_with_analytics(
        "analysis_oracle_candidates",
        &ctx.n,
        &settings,
        rng,
        batch_size,
        analytics,
    );
    if candidates.is_empty() {
        return Err("no r candidates generated for analysis tests".into());
    }

    let y = engine.rabin_exponent as u32;
    let mut prepared = Vec::with_capacity(candidates.len());
    for (r, factors) in candidates {
        let phi_new = compute_totient(&factors);
        if let Some(d_new) = mod_inverse(&ctx.e, &phi_new) {
            let r_pow_y = r.pow(y);
            prepared.push(OracleCandidate { r, d_new, r_pow_y });
        }
    }

    if prepared.is_empty() {
        return Err("no valid r candidates for analysis tests".into());
    }

    Ok(prepared)
}

/// Selects the best candidate based on matching bits against a reference message.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `candidates`: Prepared r candidates to evaluate.
/// - `message`: Reference message used for scoring.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `Option<CandidateScore>`: Best candidate score or `None` if no candidates are provided.
///
/// # Expected Output
/// - Returns the top-scoring candidate; no stdout/stderr output.
fn select_best_candidate(
    ctx: &RSAContext,
    engine: &EngineConfig,
    candidates: &[OracleCandidate],
    message: &BigUint,
    shift: bool,
) -> Option<CandidateScore> {
    if candidates.is_empty() {
        return None;
    }

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);
    let ciphertext = message.modpow(&ctx.e, &ctx.n);
    let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
    let result_default = get_larger_number(&shifted, &ctx.n, y, true, false);

    let mut best: Option<CandidateScore> = None;
    for candidate in candidates {
        let dm = derive_candidate_message_from_result(
            ctx,
            engine,
            &result_default,
            &candidate.r,
            &candidate.d_new,
            &n_pow_y,
            &candidate.r_pow_y,
            y,
            false,
        );
        let (matching_lsb, matching_total) = count_matching_bits(&dm, message);
        let score = CandidateScore {
            candidate: candidate.clone(),
            matching_lsb,
            matching_total,
        };
        if best
            .as_ref()
            .map(|b| (matching_total, matching_lsb) > (b.matching_total, b.matching_lsb))
            .unwrap_or(true)
        {
            best = Some(score);
        }
    }

    best
}

/// Computes oracle entropy and accuracy timelines using prepared candidates.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling oracle behavior.
/// - `candidates`: Prepared r candidates to sample as oracles.
/// - `iterations`: Number of message samples to evaluate.
/// - `rng`: Random number generator for message sampling.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `Result<OracleEntropySeries, Box<dyn Error>>`: Entropy/accuracy series or an error.
///
/// # Expected Output
/// - Prints progress updates; no file output.
fn run_oracle_entropy_timeline(
    ctx: &RSAContext,
    engine: &EngineConfig,
    candidates: &[OracleCandidate],
    iterations: usize,
    rng: &mut RngChoice,
    shift: bool,
) -> Result<OracleEntropySeries, Box<dyn Error>> {
    if iterations == 0 {
        return Err("analysis tests iterations must be >= 1".into());
    }
    if engine.combiner_k_oracles == 0 {
        return Err("combiner_k_oracles must be >= 1 for analysis tests".into());
    }
    let requested_oracles = engine.combiner_k_oracles;
    let oracle_count = candidates.len().min(requested_oracles);
    if oracle_count == 0 {
        return Err("not enough r candidates for oracle entropy timeline".into());
    }
    if oracle_count < requested_oracles {
        println!(
            "Oracle entropy timeline using {} oracles (requested {})",
            oracle_count, requested_oracles
        );
    }

    let fixed_message = resolve_fixed_message_for_tests(engine, &ctx.n)?;
    let bit_width = analysis_bit_width(engine, &ctx.n, &fixed_message);

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);

    let mut entropy_mean = Vec::with_capacity(iterations);
    let mut accuracy_pct = Vec::with_capacity(iterations);
    let mut next_pct = 10u64;
    for idx in 0..iterations {
        let msg = sample_message_for_tests(engine, &ctx.n, &fixed_message, rng);
        let ciphertext = msg.modpow(&ctx.e, &ctx.n);
        let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
        let result_default = get_larger_number(&shifted, &ctx.n, y, true, false);

        let mut oracles: Vec<Vec<bool>> = Vec::with_capacity(oracle_count);
        for candidate in candidates.iter().take(oracle_count) {
            let dm = derive_candidate_message_from_result(
                ctx,
                engine,
                &result_default,
                &candidate.r,
                &candidate.d_new,
                &n_pow_y,
                &candidate.r_pow_y,
                y,
                false,
            );
            oracles.push(biguint_to_bits_le(&dm, bit_width));
        }

        let distribution =
            majority_vote_with_distribution(&oracles, engine.combiner_tie_breaker)?;

        let mut entropy_sum = 0.0;
        for p in &distribution.probability_one {
            entropy_sum += shannon_entropy_bit(*p);
        }
        let entropy = if distribution.probability_one.is_empty() {
            0.0
        } else {
            entropy_sum / distribution.probability_one.len() as f64
        };

        let message_bits = biguint_to_bits_le(&msg, distribution.majority_bits.len());
        let mut correct = 0usize;
        for (a, b) in distribution.majority_bits.iter().zip(message_bits.iter()) {
            if a == b {
                correct += 1;
            }
        }
        let total = distribution.majority_bits.len().max(1);
        let accuracy = correct as f64 / total as f64 * 100.0;

        entropy_mean.push(entropy);
        accuracy_pct.push(accuracy);

        log_progress_every_ten_percent(
            (idx + 1) as u64,
            iterations as u64,
            &mut next_pct,
            "Oracle entropy timeline",
        );
    }

    Ok(OracleEntropySeries {
        entropy_mean,
        accuracy_pct,
    })
}

/// Computes match entropy and percentage timelines for a fixed r candidate.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `candidate`: Prepared r candidate to evaluate.
/// - `iterations`: Number of message samples to evaluate.
/// - `window`: Sliding window size for timeline frames.
/// - `stride`: Step size between timeline frames.
/// - `rng`: Random number generator for message sampling.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `Result<MatchTimelineSeries, Box<dyn Error>>`: Timeline series or an error.
///
/// # Expected Output
/// - Prints progress updates; no file output.
fn run_match_entropy_timeline(
    ctx: &RSAContext,
    engine: &EngineConfig,
    candidate: &OracleCandidate,
    iterations: usize,
    window: usize,
    stride: usize,
    rng: &mut RngChoice,
    shift: bool,
) -> Result<MatchTimelineSeries, Box<dyn Error>> {
    if iterations == 0 {
        return Err("analysis tests iterations must be >= 1".into());
    }

    let fixed_message = resolve_fixed_message_for_tests(engine, &ctx.n)?;
    let bit_width = analysis_bit_width(engine, &ctx.n, &fixed_message);

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);

    let mut seeds = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        seeds.push(rng.next_u64());
    }

    let done = Arc::new(AtomicU64::new(0));
    let next_pct = Arc::new(AtomicU64::new(10));
    let iterations_u64 = iterations as u64;

    let samples: Vec<MatchSample> = seeds
        .into_par_iter()
        .map(|seed| {
            let mut local_rng = RngChoice::from_seed(rng.mode(), seed);
            let msg = sample_message_for_tests(engine, &ctx.n, &fixed_message, &mut local_rng);
            let ciphertext = msg.modpow(&ctx.e, &ctx.n);
            let dm = derive_candidate_message(
                ctx,
                engine,
                &ciphertext,
                &candidate.r,
                &candidate.d_new,
                &n_pow_y,
                &candidate.r_pow_y,
                y,
                false,
                shift,
            );

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
                            "Match entropy timeline progress: {}% ({}/{})",
                            display_pct, finished, iterations_u64
                        );
                        break;
                    }
                    Err(actual) => current_next = actual,
                }
            }

            MatchSample {
                message_bytes_le: msg.to_bytes_le(),
                candidate_bytes_le: dm.to_bytes_le(),
            }
        })
        .collect();

    if samples.is_empty() {
        return Err("no samples generated for match entropy timeline".into());
    }

    let window = window.max(1).min(samples.len());
    let stride = stride.max(1);
    let frame_count = if samples.len() <= window {
        1
    } else {
        ((samples.len() - window) / stride) + 1
    };

    let samples = Arc::new(samples);
    let mut frames = (0..frame_count)
        .into_par_iter()
        .map(|frame_idx| {
            let start = frame_idx * stride;
            let end = (start + window).min(samples.len());
            let frame_samples = &samples[start..end];
            let mut match_counts = vec![0u32; bit_width];
            let mut one_counts = vec![0u32; bit_width];
            let window_len_f = frame_samples.len().max(1) as f64;

            for sample in frame_samples {
                for bit_idx in 0..bit_width {
                    let a = bit_from_bytes_le(&sample.message_bytes_le, bit_idx);
                    let b = bit_from_bytes_le(&sample.candidate_bytes_le, bit_idx);
                    if a == b {
                        match_counts[bit_idx] = match_counts[bit_idx].saturating_add(1);
                    }
                    if b {
                        one_counts[bit_idx] = one_counts[bit_idx].saturating_add(1);
                    }
                }
            }

            let mut entropy_sum = 0.0;
            let mut match_sum = 0.0;
            let mut prob_one = Vec::with_capacity(bit_width);
            for (count, ones) in match_counts.into_iter().zip(one_counts.into_iter()) {
                let p = count as f64 / window_len_f;
                entropy_sum += shannon_entropy_bit(p);
                match_sum += p * 100.0;
                prob_one.push(ones as f64 / window_len_f);
            }

            let denom = bit_width.max(1) as f64;
            (frame_idx, entropy_sum / denom, match_sum / denom, prob_one)
        })
        .collect::<Vec<_>>();

    frames.sort_by_key(|(idx, _, _, _)| *idx);
    let mut entropy_mean = Vec::with_capacity(frame_count);
    let mut match_pct_mean = Vec::with_capacity(frame_count);
    let mut bit_true_prob = Vec::with_capacity(frame_count);
    for (_, entropy, match_pct, prob_one) in frames {
        entropy_mean.push(entropy);
        match_pct_mean.push(match_pct);
        bit_true_prob.push(prob_one);
    }

    Ok(MatchTimelineSeries {
        entropy_mean,
        match_pct_mean,
        bit_true_prob,
    })
}

/// Screens r candidates to select top oracles per bit based on random-message matches.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling oracle behavior.
/// - `candidates`: Prepared r candidates to evaluate.
/// - `iterations`: Number of random messages to use for screening.
/// - `top_k`: Number of top oracles to select per bit.
/// - `rng`: Random number generator for message sampling.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `Result<(Vec<Vec<OracleBitSelection>>, Vec<f64>), Box<dyn Error>>`: Per-bit oracle selection and top match %.
///
/// # Expected Output
/// - Prints screening progress; no file output.
fn screen_oracles_per_bit(
    ctx: &RSAContext,
    engine: &EngineConfig,
    candidates: &[OracleCandidate],
    iterations: usize,
    top_k: usize,
    rng: &mut RngChoice,
    shift: bool,
) -> Result<(Vec<Vec<OracleBitSelection>>, Vec<f64>), Box<dyn Error>> {
    if iterations == 0 {
        return Err("oracle_screen_iterations must be >= 1".into());
    }
    if candidates.is_empty() {
        return Err("no r candidates available for oracle screening".into());
    }
    let top_k = top_k.max(1).min(candidates.len());

    let bit_width = analysis_bit_width(engine, &ctx.n, &None);
    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);

    let mut samples: Vec<OracleTrainingSample> = Vec::with_capacity(iterations);
    let mut next_pct = 10u64;
    for idx in 0..iterations {
        let msg = random_message_under_n(engine, &ctx.n, rng);
        let ciphertext = msg.modpow(&ctx.e, &ctx.n);
        let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
        let result_default = get_larger_number(&shifted, &ctx.n, y, true, false);
        let message_bits = biguint_to_bits_le(&msg, bit_width);
        samples.push(OracleTrainingSample {
            result_default,
            message_bits,
        });

        log_progress_every_ten_percent(
            (idx + 1) as u64,
            iterations as u64,
            &mut next_pct,
            "Oracle screening",
        );
    }

    let samples = Arc::new(samples);
    let counts: Vec<Vec<u32>> = candidates
        .par_iter()
        .map(|candidate| {
            let mut match_counts = vec![0u32; bit_width];
            for sample in samples.iter() {
                let dm = derive_candidate_message_from_result(
                    ctx,
                    engine,
                    &sample.result_default,
                    &candidate.r,
                    &candidate.d_new,
                    &n_pow_y,
                    &candidate.r_pow_y,
                    y,
                    false,
                );
                let dm_bits = biguint_to_bits_le(&dm, bit_width);
                for (bit_idx, bit) in dm_bits.iter().enumerate() {
                    if *bit == sample.message_bits[bit_idx] {
                        match_counts[bit_idx] = match_counts[bit_idx].saturating_add(1);
                    }
                }
            }
            match_counts
        })
        .collect();

    let mut per_bit_oracles = vec![Vec::with_capacity(top_k); bit_width];
    let mut top_match_pct = vec![0.0; bit_width];
    for bit_idx in 0..bit_width {
        let mut ranked: Vec<(usize, f64, bool)> = counts
            .iter()
            .enumerate()
            .map(|(oracle_idx, counts)| {
                let match_pct = counts[bit_idx] as f64 / iterations as f64 * 100.0;
                if match_pct < 50.0 {
                    (oracle_idx, 100.0 - match_pct, true)
                } else {
                    (oracle_idx, match_pct, false)
                }
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let best = ranked.first().map(|(_, pct, _)| *pct).unwrap_or(0.0);
        top_match_pct[bit_idx] = best;

        for (oracle_idx, _, invert) in ranked.into_iter().take(top_k) {
            per_bit_oracles[bit_idx].push(OracleBitSelection { oracle_idx, invert });
        }
    }

    Ok((per_bit_oracles, top_match_pct))
}

/// Runs a bitwise speculative oracle attempt using per-bit oracle batches.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `candidates`: Prepared r candidates to use as oracles.
/// - `per_bit_oracles`: Per-bit oracle selection ranked by screening.
/// - `message`: Reference message to compare against.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `Result<SpeculativeOracleReport, Box<dyn Error>>`: Reconstruction report or an error.
///
/// # Expected Output
/// - Returns reconstruction metrics; no side effects.
fn run_bitwise_speculative_oracle_attempt(
    ctx: &RSAContext,
    engine: &EngineConfig,
    candidates: &[OracleCandidate],
    per_bit_oracles: &[Vec<OracleBitSelection>],
    message: &BigUint,
    shift: bool,
) -> Result<SpeculativeOracleReport, Box<dyn Error>> {
    if per_bit_oracles.is_empty() {
        return Err("per-bit oracle selection is empty".into());
    }
    let bit_width = message.bits().max(1) as usize;
    let oracles_per_bit = per_bit_oracles[0].len().max(1);

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);
    let ciphertext = message.modpow(&ctx.e, &ctx.n);
    let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
    let result_default = get_larger_number(&shifted, &ctx.n, y, true, false);

    let mut unique_oracle_indices = std::collections::HashSet::new();
    for selections in per_bit_oracles {
        for selection in selections {
            unique_oracle_indices.insert(selection.oracle_idx);
        }
    }

    let mut oracle_bits: Vec<Option<Vec<bool>>> = vec![None; candidates.len()];
    for oracle_idx in unique_oracle_indices.iter().copied() {
        let candidate = &candidates[oracle_idx];
        let dm = derive_candidate_message_from_result(
            ctx,
            engine,
            &result_default,
            &candidate.r,
            &candidate.d_new,
            &n_pow_y,
            &candidate.r_pow_y,
            y,
            false,
        );
        oracle_bits[oracle_idx] = Some(biguint_to_bits_le(&dm, bit_width));
    }

    let mut recovered_bits = vec![false; bit_width];
    for (bit_idx, selections) in per_bit_oracles.iter().enumerate().take(bit_width) {
        let mut ones = 0usize;
        let mut zeros = 0usize;
        for selection in selections {
            if let Some(bits) = oracle_bits
                .get(selection.oracle_idx)
                .and_then(|entry| entry.as_ref())
            {
                let bit = if selection.invert { !bits[bit_idx] } else { bits[bit_idx] };
                if bit {
                    ones += 1;
                } else {
                    zeros += 1;
                }
            }
        }
        recovered_bits[bit_idx] = if ones == zeros {
            engine.combiner_tie_breaker
        } else {
            ones > zeros
        };
    }

    let recovered = bits_le_to_biguint(&recovered_bits);
    let message_bits = biguint_to_bits_le(message, bit_width);
    let (matching_lsb, matching_total) = count_matching_bits_le(&recovered_bits, &message_bits);
    let mut match_pct = matching_total as f64 / bit_width.max(1) as f64 * 100.0;
    let match_pct_inverted = match_pct.max(100.0 - match_pct);
    let match_pct_reported = if match_pct < match_pct_inverted {match_pct_inverted} else {match_pct};
    match_pct = match_pct_reported;

    Ok(SpeculativeOracleReport {
        recovered,
        matching_lsb,
        matching_total,
        bit_width,
        match_pct,
        oracles_per_bit,
        unique_oracles: unique_oracle_indices.len(),
    })
}

/// Computes per-bit best-case match percentages for a target message.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `candidates`: Prepared r candidates to use as oracles.
/// - `per_bit_oracles`: Per-bit oracle selection ranked by screening.
/// - `message`: Reference message to compare against.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `Result<(Vec<f64>, Vec<bool>), Box<dyn Error>>`: Per-bit match percentages and best-case bits.
///
/// # Expected Output
/// - Returns per-bit match percentages; no side effects.
fn compute_per_bit_best_case_match(
    ctx: &RSAContext,
    engine: &EngineConfig,
    candidates: &[OracleCandidate],
    per_bit_oracles: &[Vec<OracleBitSelection>],
    message: &BigUint,
    shift: bool,
) -> Result<(Vec<f64>, Vec<bool>), Box<dyn Error>> {
    if per_bit_oracles.is_empty() {
        return Err("per-bit oracle selection is empty".into());
    }

    let bit_width = message.bits().max(1) as usize;
    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);
    let ciphertext = message.modpow(&ctx.e, &ctx.n);
    let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
    let result_default = get_larger_number(&shifted, &ctx.n, y, true, false);
    let message_bits = biguint_to_bits_le(message, bit_width);

    let mut unique_oracle_indices = std::collections::HashSet::new();
    for selections in per_bit_oracles {
        for selection in selections {
            unique_oracle_indices.insert(selection.oracle_idx);
        }
    }

    let mut oracle_bits: Vec<Option<Vec<bool>>> = vec![None; candidates.len()];
    for oracle_idx in unique_oracle_indices.iter().copied() {
        let candidate = &candidates[oracle_idx];
        let dm = derive_candidate_message_from_result(
            ctx,
            engine,
            &result_default,
            &candidate.r,
            &candidate.d_new,
            &n_pow_y,
            &candidate.r_pow_y,
            y,
            false,
        );
        oracle_bits[oracle_idx] = Some(biguint_to_bits_le(&dm, bit_width));
    }

    let mut per_bit_pct = Vec::with_capacity(bit_width);
    let mut best_case_bits = Vec::with_capacity(bit_width);
    for (bit_idx, selections) in per_bit_oracles.iter().enumerate().take(bit_width) {
        let target = message_bits[bit_idx];
        let mut matched = false;
        let mut selected_bit = false;
        for selection in selections {
            if let Some(bits) = oracle_bits
                .get(selection.oracle_idx)
                .and_then(|entry| entry.as_ref())
            {
                let bit = if selection.invert { !bits[bit_idx] } else { bits[bit_idx] };
                if bit == target {
                    matched = true;
                    selected_bit = bit;
                    break;
                }
                if !matched {
                    selected_bit = bit;
                }
            }
        }
        per_bit_pct.push(if matched { 100.0 } else { 0.0 });
        best_case_bits.push(selected_bit);
    }

    Ok((per_bit_pct, best_case_bits))
}

/// Prints hex strings for the original and best-case messages with color-coded matches.
///
/// # Parameters
/// - `message`: Original message value.
/// - `best_case_bits`: Best-case bit vector aligned to the original message.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Prints two hex strings with color highlighting; no file output.
fn print_best_case_hex(message: &BigUint, best_case_bits: &[bool]) {
    let best_case = bits_le_to_biguint(best_case_bits);
    let original_hex = to_hex(message);
    let best_hex = to_hex(&best_case);

    let max_len = original_hex.len().max(best_hex.len());
    let original_padded = pad_left_hex(&original_hex, max_len);
    let best_padded = pad_left_hex(&best_hex, max_len);

    let original_colored = colorize_hex_matches(&original_padded, &best_padded);
    let best_colored = colorize_hex_matches(&best_padded, &original_padded);

    println!("Original message (hex): {}", original_colored);
    println!("Best-case message (hex): {}", best_colored);
    println!("Hex match key: green = match, red = mismatch");
}

/// Pads a hex string with leading zeros to the requested length.
///
/// # Parameters
/// - `value`: Hex string to pad (no 0x prefix).
/// - `width`: Target width for the output.
///
/// # Returns
/// - `String`: Left-padded hex string.
///
/// # Expected Output
/// - Returns a padded string; no side effects.
fn pad_left_hex(value: &str, width: usize) -> String {
    if value.len() >= width {
        return value.to_string();
    }
    let mut padded = String::with_capacity(width);
    for _ in 0..(width - value.len()) {
        padded.push('0');
    }
    padded.push_str(value);
    padded
}

/// Applies ANSI color highlighting for matching hex characters.
///
/// # Parameters
/// - `value`: Hex string to colorize.
/// - `reference`: Hex string to compare against.
///
/// # Returns
/// - `String`: Colorized string with ANSI escapes.
///
/// # Expected Output
/// - Returns a colorized string; no side effects.
fn colorize_hex_matches(value: &str, reference: &str) -> String {
    const GREEN: &str = "\u{1b}[32m";
    const RED: &str = "\u{1b}[31m";
    const RESET: &str = "\u{1b}[0m";

    let mut out = String::new();
    for (a, b) in value.chars().zip(reference.chars()) {
        if a == b {
            out.push_str(GREEN);
            out.push(a);
            out.push_str(RESET);
        } else {
            out.push_str(RED);
            out.push(a);
            out.push_str(RESET);
        }
    }
    out
}

/// Sanitizes a label for use in output filenames.
///
/// # Parameters
/// - `label`: Raw label string.
///
/// # Returns
/// - `String`: Sanitized label containing only alphanumeric, `-`, and `_`.
///
/// # Expected Output
/// - Returns a sanitized string; no side effects.
#[cfg(feature = "plots")]
fn sanitize_label(label: &str) -> String {
    label
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

/// Writes a timeline line chart for a floating-point series.
///
/// # Parameters
/// - `series`: Time-ordered values to plot.
/// - `label`: Label used in the chart caption and filename.
/// - `caption`: Chart caption prefix.
/// - `y_desc`: Y-axis description.
/// - `y_range`: Tuple of `(min, max)` for the Y-axis.
/// - `file_prefix`: Prefix for the output filename.
/// - `color`: Line color for the series.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an I/O/plotting error.
///
/// # Expected Output
/// - Writes a PNG into `./images` and prints the output path.
#[cfg(feature = "plots")]
fn plot_timeline_series(
    series: &[f64],
    label: &str,
    caption: &str,
    y_desc: &str,
    y_range: (f64, f64),
    file_prefix: &str,
    color: RGBColor,
) -> Result<(), Box<dyn Error>> {
    if series.is_empty() {
        return Ok(());
    }

    let images_dir = Path::new("./images");
    fs::create_dir_all(images_dir)?;

    static TIMELINE_SEQ: AtomicUsize = AtomicUsize::new(0);
    let seq = TIMELINE_SEQ.fetch_add(1, Ordering::Relaxed);
    let safe_label = sanitize_label(label);
    let file_name = format!("{}_{}_{}.png", file_prefix, safe_label, seq);
    let path = images_dir.join(file_name);

    let x_end = series.len().max(2);
    let (y_min, y_max) = y_range;
    let adjusted_max = if y_max > y_min { y_max } else { y_min + 1.0 };

    let root = BitMapBackend::new(&path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} ({})", caption, label),
            ("sans-serif", 30).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0usize..x_end, y_min..adjusted_max)?;

    chart
        .configure_mesh()
        .x_desc("Frame")
        .y_desc(y_desc)
        .draw()?;

    chart.draw_series(LineSeries::new(
        series.iter().enumerate().map(|(idx, val)| (idx, *val)),
        color,
    ))?;

    root.present()?;
    println!("Saved timeline chart to {}", path.display());
    Ok(())
}

#[cfg(not(feature = "plots"))]
fn plot_timeline_series(
    _series: &[f64],
    _label: &str,
    _caption: &str,
    _y_desc: &str,
    _y_range: (f64, f64),
    _file_prefix: &str,
    _color: RGBColor,
) -> Result<(), Box<dyn Error>> {
    Ok(())
}

/// Runs analysis tests and reports information sufficiency for speculative oracles.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `config`: Full configuration used for analysis settings.
/// - `message`: Reference message used to select the best candidate.
/// - `rng`: Random number generator for candidate/message sampling.
/// - `export`: Whether to emit oracle entropy timeline charts.
/// - `analytics`: Session analytics accumulator for timing and candidate metadata.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` if sufficient, otherwise an error.
///
/// # Expected Output
/// - Prints summaries and writes timeline PNGs into `./images`.
fn run_information_sufficiency_tests(
    ctx: &RSAContext,
    config: &Config,
    message: &BigUint,
    rng: &mut RngChoice,
    export: bool,
    analytics: &Arc<Mutex<SessionAnalytics>>,
    shift: bool,
) -> Result<(), Box<dyn Error>> {
    let engine = &config.engine;
    let iterations = engine.analysis_tests_iterations as usize;
    if iterations == 0 {
        return Err("analysis_tests_iterations must be >= 1".into());
    }

    let window = engine.analysis_tests_window.max(1);
    let stride = engine.analysis_tests_stride.max(1);

    println!("\nRunning analysis sufficiency tests...");
    let candidates = build_oracle_candidates(ctx, engine, rng, analytics)?;
    record_r_candidate_trace_batch_from_prepared(
        ctx,
        engine,
        message,
        &candidates,
        analytics,
        "analysis_oracle_candidates",
        shift,
    );
    let Some(best) = select_best_candidate(ctx, engine, &candidates, message, shift) else {
        return Err("failed to select a best r candidate for tests".into());
    };
    println!(
        "Selected analysis r candidate {} with matching bits LSB {} / total {}",
        best.candidate.r, best.matching_lsb, best.matching_total
    );
    with_analytics(analytics, |a| {
        a.set_feature_stat(
            "information_sufficiency",
            "selected_r",
            json!(best.candidate.r.to_string()),
        );
        a.set_feature_stat(
            "information_sufficiency",
            "selected_r_matching_lsb",
            json!(best.matching_lsb),
        );
        a.set_feature_stat(
            "information_sufficiency",
            "selected_r_matching_total",
            json!(best.matching_total),
        );
    });

    let (oracle_entropy_stats, oracle_accuracy_stats) = if export {
        let oracle_series =
            run_oracle_entropy_timeline(ctx, engine, &candidates, iterations, rng, shift)?;
        if let Err(err) = plot_timeline_series(
            &oracle_series.entropy_mean,
            "analysis_tests",
            "Oracle Entropy Timeline",
            "Entropy (bits)",
            (0.0, 1.0),
            "oracle_entropy_timeline",
            RED,
        ) {
            println!("Failed to write oracle entropy timeline: {}", err);
        }
        if let Err(err) = plot_timeline_series(
            &oracle_series.accuracy_pct,
            "analysis_tests",
            "Oracle Accuracy Timeline",
            "Accuracy %",
            (0.0, 100.0),
            "oracle_accuracy_timeline",
            BLUE,
        ) {
            println!("Failed to write oracle accuracy timeline: {}", err);
        }

        let oracle_entropy_stats = compute_stats(&oracle_series.entropy_mean)
            .ok_or("no oracle entropy samples")?;
        let oracle_accuracy_stats = compute_stats(&oracle_series.accuracy_pct)
            .ok_or("no oracle accuracy samples")?;

        println!(
            "Oracle entropy stats: mean {:.4}, std dev {:.4}, min {:.4}, max {:.4}, n {}",
            oracle_entropy_stats.mean,
            oracle_entropy_stats.stddev,
            oracle_entropy_stats.min,
            oracle_entropy_stats.max,
            oracle_entropy_stats.count
        );
        println!(
            "Oracle accuracy stats: mean {:.2}%, std dev {:.2}, min {:.2}%, max {:.2}%, n {}",
            oracle_accuracy_stats.mean,
            oracle_accuracy_stats.stddev,
            oracle_accuracy_stats.min,
            oracle_accuracy_stats.max,
            oracle_accuracy_stats.count
        );
        with_analytics(analytics, |a| {
            a.set_feature_stat(
                "information_sufficiency",
                "oracle_entropy_mean",
                json!(oracle_entropy_stats.mean),
            );
            a.set_feature_stat(
                "information_sufficiency",
                "oracle_accuracy_mean_pct",
                json!(oracle_accuracy_stats.mean),
            );
        });
        (Some(oracle_entropy_stats), Some(oracle_accuracy_stats))
    } else {
        println!("Oracle entropy/accuracy timelines skipped (use --export to enable)");
        with_analytics(analytics, |a| {
            a.add_feature_note(
                "information_sufficiency",
                "oracle entropy/accuracy timelines skipped",
            );
        });
        (None, None)
    };

    let match_series = run_match_entropy_timeline(
        ctx,
        engine,
        &best.candidate,
        iterations,
        window,
        stride,
        rng,
        shift,
    )?;
    if export {
        if let Err(err) = plot_timeline_series(
            &match_series.entropy_mean,
            "analysis_tests",
            "Match Entropy Timeline",
            "Entropy (bits)",
            (0.0, 1.0),
            "match_entropy_timeline",
            GREEN,
        ) {
            println!("Failed to write match entropy timeline: {}", err);
        }
        if let Err(err) = plot_timeline_series(
            &match_series.match_pct_mean,
            "analysis_tests",
            "Match Percentage Timeline",
            "Match %",
            (0.0, 100.0),
            "match_pct_timeline",
            BLACK,
        ) {
            println!("Failed to write match percentage timeline: {}", err);
        }
    } else {
        println!("Match timeline charts skipped (use --export to enable)");
        with_analytics(analytics, |a| {
            a.add_feature_note("information_sufficiency", "match timeline charts skipped");
        });
    }

    let match_entropy_stats = compute_stats(&match_series.entropy_mean)
        .ok_or("no match entropy samples")?;
    let match_pct_stats = compute_stats(&match_series.match_pct_mean)
        .ok_or("no match percentage samples")?;

    println!(
        "Match entropy stats: mean {:.4}, std dev {:.4}, min {:.4}, max {:.4}, n {}",
        match_entropy_stats.mean,
        match_entropy_stats.stddev,
        match_entropy_stats.min,
        match_entropy_stats.max,
        match_entropy_stats.count
    );
    println!(
        "Match percentage stats: mean {:.2}%, std dev {:.2}, min {:.2}%, max {:.2}%, n {}",
        match_pct_stats.mean,
        match_pct_stats.stddev,
        match_pct_stats.min,
        match_pct_stats.max,
        match_pct_stats.count
    );
    let bit_true_width = match_series
        .bit_true_prob
        .first()
        .map(|row| row.len())
        .unwrap_or(0);
    with_analytics(analytics, |a| {
        a.set_feature_stat(
            "information_sufficiency",
            "match_entropy_mean",
            json!(match_entropy_stats.mean),
        );
        a.set_feature_stat(
            "information_sufficiency",
            "match_pct_mean",
            json!(match_pct_stats.mean),
        );
        a.set_feature_stat(
            "information_sufficiency",
            "bit_true_timeline",
            json!({
                "bit_order": "lsb0",
                "bit_width": bit_true_width,
                "window": window,
                "stride": stride,
                "frames": match_series.bit_true_prob,
            }),
        );
    });

    let mut screen_iterations = engine.oracle_screen_iterations as usize;
    if screen_iterations >= 1000 {
        println!(
            "Oracle screen iterations {} >= 1000; clamping to 999",
            screen_iterations
        );
        screen_iterations = 999;
    }
    if screen_iterations == 0 {
        screen_iterations = iterations.min(999).max(1);
    }
    let top_k = engine.combiner_k_oracles.max(1);
    println!(
        "Screening {} r candidates with {} random messages; selecting top {} oracles per bit",
        candidates.len(),
        screen_iterations,
        top_k.min(candidates.len())
    );
    let (per_bit_oracles, top_match_pct) = screen_oracles_per_bit(
        ctx,
        engine,
        &candidates,
        screen_iterations,
        top_k,
        rng,
        shift,
    )?;
    let mut inverted_total = 0usize;
    for selections in &per_bit_oracles {
        inverted_total += selections.iter().filter(|sel| sel.invert).count();
    }
    if let Some(stats) = compute_stats(&top_match_pct) {
        println!(
            "Per-bit top oracle adjusted match % stats: mean {:.2}, std dev {:.2}, min {:.2}, max {:.2}, n {}; inverted selections {}",
            stats.mean,
            stats.stddev,
            stats.min,
            stats.max,
            stats.count,
            inverted_total
        );
    }

    let (per_bit_best_pct, best_case_bits) = compute_per_bit_best_case_match(
        ctx,
        engine,
        &candidates,
        &per_bit_oracles,
        message,
        shift,
    )?;
    if let Some(stats) = compute_stats(&per_bit_best_pct) {
        println!(
            "Per-bit best-case match % on original message: mean {:.2}, std dev {:.2}, min {:.2}, max {:.2}, n {}",
            stats.mean,
            stats.stddev,
            stats.min,
            stats.max,
            stats.count
        );
    }
    print_best_case_hex(message, &best_case_bits);

    let (bit_similarity_entries, match_counts_per_bit, shift_levels_used) =
        build_bit_similarity_entries(
            ctx,
            engine,
            &candidates,
            message,
            shift,
            engine.analysis_shift_multiplications,
        );
    with_analytics(analytics, |a| {
        a.set_feature_stat(
            "information_sufficiency",
            "bit_similarity",
            json!({
                "bit_order": "lsb0",
                "bit_width": message.bits().max(1),
                "original_hex": to_hex(message),
                "shift_levels_configured": engine.analysis_shift_multiplications,
                "shift_levels_used": shift_levels_used,
                "match_counts_per_bit": match_counts_per_bit,
                "candidates": bit_similarity_entries,
            }),
        );
    });

    let speculative_report = run_bitwise_speculative_oracle_attempt(
        ctx,
        engine,
        &candidates,
        &per_bit_oracles,
        message,
        shift,
    )?;
    
    println!(
        "Bitwise speculative oracle recovered (hex): {}",
        to_hex(&speculative_report.recovered)
    );

    println!(
        "Bitwise speculative oracle match: LSB run {} / overlap {} of {} bits ({:.2}%) using {} oracles per bit ({} unique)",
        speculative_report.matching_lsb,
        speculative_report.matching_total,
        speculative_report.bit_width,
        speculative_report.match_pct,
        speculative_report.oracles_per_bit,
        speculative_report.unique_oracles
    );
    with_analytics(analytics, |a| {
        a.set_feature_stat(
            "information_sufficiency",
            "speculative_match_pct",
            json!(speculative_report.match_pct),
        );
    });

    let entropy_threshold = engine.entropy_report_threshold;
    let match_threshold = engine.overlap_report_threshold;
    let oracle_accuracy_threshold = engine.oracle_accuracy_threshold;

    let oracle_entropy_ok = oracle_entropy_stats
        .as_ref()
        .map(|stats| stats.mean <= entropy_threshold)
        .unwrap_or(true);
    let match_entropy_ok = match_entropy_stats.mean <= entropy_threshold;

    //This mean is times 100.0.
    let match_pct_ok = (match_pct_stats.mean >= match_threshold) || (match_pct_stats.mean <= (100.0 - match_threshold));
    //let match_pct_inverted = match_pct_stats.mean < 50.0;

    let oracle_accuracy_ok = oracle_accuracy_stats
        .as_ref()
        .map(|stats| stats.mean >= oracle_accuracy_threshold)
        .unwrap_or(true);
    let speculative_match_ok = (speculative_report.match_pct >= match_threshold) || (speculative_report.match_pct <= (100.0 - match_threshold));

    println!(
        "Sufficiency thresholds: entropy <= {:.4}, match % >= {:.2}, oracle accuracy % >= {:.2}",
        entropy_threshold, match_threshold, oracle_accuracy_threshold
    );
    println!(
        "Sufficiency checks: oracle entropy {}, match entropy {}, match % {}, oracle accuracy {}, speculative match {}",
        if oracle_entropy_ok { "OK" } else { "FAIL" },
        if match_entropy_ok { "OK" } else { "FAIL" },
        if match_pct_ok { "OK" } else { "FAIL" },
        if oracle_accuracy_ok { "OK" } else { "FAIL" },
        if speculative_match_ok { "OK" } else { "FAIL" }
    );

    if oracle_entropy_ok
        && match_entropy_ok
        && match_pct_ok
        && oracle_accuracy_ok
        && speculative_match_ok
    {
        println!("Sufficiency verdict: PASS (enough signal for speculative oracle attempts)");
        Ok(())
    } else {
        Err("analysis tests indicate insufficient information for speculative oracle attempts".into())
    }
}

/// Runs the enciphered export pipeline and writes per-bit match statistics.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling export behavior.
/// - `rng`: Random number generator used for sampling messages.
/// - `analytics`: Session analytics accumulator for r candidate metadata.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error on failure.
///
/// # Expected Output
/// - Writes CSV outputs (and optional ramp CSV), prints progress and summary lines.
fn run_enciphered_export(
    ctx: &RSAContext,
    engine: &EngineConfig,
    rng: &mut RngChoice,
    analytics: &Arc<Mutex<SessionAnalytics>>,
    shift: bool,
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
    let mut candidates = generate_r_candidates_with_analytics(
        "enciphered_export",
        &ctx.n,
        &settings,
        rng,
        batch_size,
        analytics,
    );
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
                let mut local_rng = RngChoice::from_seed(rng.mode(), seed);
                random_message_under_n(engine, &ctx.n, &mut local_rng)
            };
            let ciphertext = msg.modpow(&ctx.e, &ctx.n);
            let dm = derive_candidate_message(
                ctx,
                engine,
                &ciphertext,
                &r,
                &d_new,
                &n_pow_y,
                &r_pow_y,
                y,
                false,
                shift,
            );

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

    use std::fmt::Write as FmtWrite;

    let frame_outputs: Vec<FrameExportOutput> = (0..frame_count)
        .into_par_iter()
        .map(|frame_idx| {
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
            let mut match_rows = String::with_capacity(bins * 64);
            for (bin_idx, count) in match_counts.iter().enumerate() {
                let match_pct = (*count as f64 / window_len_f) * 100.0;
                counts_pct[bin_idx] = match_pct;
                let _ = writeln!(
                    match_rows,
                    "{},{},{},{},{},{:.8}",
                    frame_idx,
                    start,
                    end,
                    bin_idx,
                    count,
                    match_pct
                );
            }

            let mut ramp_rows = String::new();
            let mut ramp_summary = vec![RampSummary::default(); ramp_tolerances.len()];
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
                    let entry = &mut ramp_summary[idx];
                    if !ramps.is_empty() {
                        entry.frames_with_ramp += 1;
                    }
                    entry.total_ramps = entry.total_ramps.saturating_add(ramps.len());
                    entry.total_strength = entry.total_strength.saturating_add(strength);

                    for (ramp_start, ramp_len, ramp_vals) in ramps {
                        let values_str = ramp_vals
                            .iter()
                            .map(|v| format!("{:.4}", v))
                            .collect::<Vec<_>>()
                            .join("|");
                        let _ = writeln!(
                            ramp_rows,
                            "{},{},{},{},{},{:.4}",
                            frame_idx,
                            tol,
                            ramp_start,
                            ramp_len,
                            values_str,
                            mean
                        );
                    }
                }
            }

            FrameExportOutput {
                frame_idx,
                match_rows,
                ramp_rows,
                ramp_summary,
            }
        })
        .collect();

    let mut frame_outputs = frame_outputs;
    frame_outputs.sort_by_key(|entry| entry.frame_idx);

    let mut summaries = vec![RampSummary::default(); ramp_tolerances.len()];
    for output in &frame_outputs {
        if let Some(file) = ramp_csv.as_mut() {
            if !output.ramp_rows.is_empty() {
                file.write_all(output.ramp_rows.as_bytes())?;
            }
        }
        csv.write_all(output.match_rows.as_bytes())?;

        for (idx, entry) in output.ramp_summary.iter().enumerate() {
            summaries[idx].frames_with_ramp =
                summaries[idx].frames_with_ramp.saturating_add(entry.frames_with_ramp);
            summaries[idx].total_ramps =
                summaries[idx].total_ramps.saturating_add(entry.total_ramps);
            summaries[idx].total_strength =
                summaries[idx].total_strength.saturating_add(entry.total_strength);
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

/// Selects the plaintext message according to CLI args and configuration.
///
/// # Parameters
/// - `args_message`: Optional CLI-provided message override.
/// - `engine`: Engine configuration with message settings.
/// - `rng`: Random number generator for random message selection.
///
/// # Returns
/// - `BigUint`: Selected message as a big integer.
///
/// # Expected Output
/// - Returns the selected message; no side effects.
fn select_message(args_message: Option<String>, engine: &EngineConfig, rng: &mut RngChoice) -> BigUint {
    if let Some(explicit) = args_message {
        return BigUint::from_bytes_be(explicit.as_bytes());
    }
    if engine.message.is_random {
        return random_message_under_n(engine, &BigUint::zero(), rng);
    }
    BigUint::from_bytes_be(engine.message.fixed_message.as_bytes())
}

/// Samples a random message that is non-zero and less than `n` (when provided).
///
/// # Parameters
/// - `engine`: Engine configuration with message bit-length settings.
/// - `n`: Optional modulus bound; use zero to skip the bound.
/// - `rng`: Random number generator for sampling.
///
/// # Returns
/// - `BigUint`: Random message value.
///
/// # Expected Output
/// - Returns a non-zero value under `n` when `n` is non-zero; no side effects.
fn random_message_under_n(engine: &EngineConfig, n: &BigUint, rng: &mut RngChoice) -> BigUint {
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

/// Computes the analysis bit width based on configuration and message bounds.
///
/// # Parameters
/// - `engine`: Engine configuration containing message bit-length hints.
/// - `n`: RSA modulus for upper bound sizing.
/// - `fixed_message`: Optional fixed message to include in the width calculation.
///
/// # Returns
/// - `usize`: Bit width used for analysis bit vectors.
///
/// # Expected Output
/// - Returns a positive width; no side effects.
fn analysis_bit_width(
    engine: &EngineConfig,
    n: &BigUint,
    fixed_message: &Option<BigUint>,
) -> usize {
    let mut bit_width = engine.message.bits.max(1) as usize;
    if let Some(msg) = fixed_message {
        bit_width = bit_width.max(msg.bits().max(1) as usize);
    }
    if !n.is_zero() {
        bit_width = bit_width.min(n.bits().max(1) as usize);
    }
    bit_width.max(1)
}

/// Builds `RCandidateSettings` from the engine configuration.
///
/// # Parameters
/// - `engine`: Engine configuration containing candidate fields.
///
/// # Returns
/// - `RCandidateSettings`: Fully populated candidate settings.
///
/// # Expected Output
/// - Returns a settings struct; no side effects.
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
        max_factors_per_candidate: engine.r_candidate_max_factors,
        target_bit_length: engine.r_candidate_bit_length,
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
    /// Creates an empty match histogram.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `MatchHistogram`: A histogram with empty counters.
    ///
    /// # Expected Output
    /// - Returns a new histogram; no side effects.
    fn new() -> Self {
        Self {
            matches: Vec::new(),
            samples: Vec::new(),
        }
    }

    /// Updates match counts for corresponding bits between two values.
    ///
    /// # Parameters
    /// - `a`: First value to compare.
    /// - `b`: Second value to compare.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Updates internal counts; no stdout/stderr output.
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

    /// Merges another histogram into this one.
    ///
    /// # Parameters
    /// - `other`: Histogram whose counts are added into `self`.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Updates internal counts; no stdout/stderr output.
    fn merge(&mut self, other: &MatchHistogram) {
        if other.matches.len() > self.matches.len() {
            self.matches.resize(other.matches.len(), 0);
            self.samples.resize(other.samples.len(), 0);
        }

        for i in 0..other.matches.len() {
            self.matches[i] = self.matches[i].saturating_add(other.matches[i]);
            self.samples[i] = self.samples[i].saturating_add(other.samples[i]);
        }
    }

    /// Writes a PNG histogram showing per-bit match frequency.
    ///
    /// # Parameters
    /// - `label`: Label used in the chart caption and output filename.
    ///
    /// # Returns
    /// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an I/O/plotting error.
    ///
    /// # Expected Output
    /// - Writes a PNG into `./images` and prints the output path.
    #[cfg(feature = "plots")]
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

    /// Writes a PNG histogram showing per-bit match frequency.
    ///
    /// # Parameters
    /// - `label`: Label used in the chart caption and output filename.
    ///
    /// # Returns
    /// - `Result<(), Box<dyn Error>>`: `Ok(())` when plotting is disabled.
    ///
    /// # Expected Output
    /// - No side effects when plotting is disabled.
    #[cfg(not(feature = "plots"))]
    fn write_histogram(&self, _label: &str) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}

/// Runs message trials against generated r candidates and returns the best match report.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `_config`: Full config (currently unused).
/// - `engine`: Engine configuration controlling trial behavior.
/// - `message`: Base message to test (used on first trial).
/// - `min_message_trials`: Minimum number of trial messages to run.
/// - `rng`: Random number generator for sampling messages/candidates.
/// - `histogram`: Histogram updated with match frequencies.
/// - `analytics`: Session analytics accumulator for r candidate metadata.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `Result<TestReport, Box<dyn Error>>`: Best matching report or an error.
///
/// # Expected Output
/// - Prints candidate generation info; updates `histogram` in-place.
fn run_message_trial(
    ctx: &RSAContext,
    _config: &Config,
    engine: &EngineConfig,
    message: &BigUint,
    min_message_trials: u64,
    rng: &mut RngChoice,
    histogram: &mut MatchHistogram,
    analytics: &Arc<Mutex<SessionAnalytics>>,
    shift: bool,
) -> Result<TestReport, Box<dyn Error>> {
    let attempts = min_message_trials.max(1);
    let mut best: Option<TestReport> = None;
    let mut worst: Option<TestReport> = None;

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);

    let settings = build_r_candidate_settings(engine);
    let batch_size = engine.process_count.max(engine.process_min_count).max(1) as usize;
    let candidates = generate_r_candidates_with_analytics(
        "test_iterations",
        &ctx.n,
        &settings,
        rng,
        batch_size,
        analytics,
    );
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

        let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
        let result_default = get_larger_number(&shifted, &ctx.n, y, true, false);

        let (attempt_best, attempt_worst, attempt_hist) = candidates
            .par_iter()
            .fold(
                || {
                    (
                        Option::<TestReport>::None,
                        Option::<TestReport>::None,
                        MatchHistogram::new(),
                    )
                },
                |mut acc, (r, factors)| {
                    let phi_new = compute_totient(factors);
                    let Some(d_new) = mod_inverse(&ctx.e, &phi_new) else {
                        return acc;
                    };

                    let r_pow_y = r.pow(y);
                    let dm = derive_candidate_message_from_result(
                        ctx,
                        engine,
                        &result_default,
                        r,
                        &d_new,
                        &n_pow_y,
                        &r_pow_y,
                        y,
                        false,
                    );
                    acc.2.update(&dm, &msg);

                    let (matching_lsb, matching_total) = count_matching_bits(&dm, &msg);
                    let report = TestReport {
                        best_r: r.clone(),
                        factors: factors.clone(),
                        matching_lsb,
                        matching_total,
                        message_bits: msg.bits() as usize,
                    };

                    if acc
                        .0
                        .as_ref()
                        .map(|b| {
                            (matching_total, matching_lsb) > (b.matching_total, b.matching_lsb)
                        })
                        .unwrap_or(true)
                    {
                        acc.0 = Some(report.clone());
                    }
                    if acc
                        .1
                        .as_ref()
                        .map(|b| {
                            (matching_total, matching_lsb) < (b.matching_total, b.matching_lsb)
                        })
                        .unwrap_or(true)
                    {
                        acc.1 = Some(report);
                    }

                    acc
                },
            )
            .reduce(
                || {
                    (
                        Option::<TestReport>::None,
                        Option::<TestReport>::None,
                        MatchHistogram::new(),
                    )
                },
                |mut left, right| {
                    if let Some(candidate) = right.0.as_ref() {
                        if left
                            .0
                            .as_ref()
                            .map(|b| {
                                (candidate.matching_total, candidate.matching_lsb)
                                    > (b.matching_total, b.matching_lsb)
                            })
                            .unwrap_or(true)
                        {
                            left.0 = right.0;
                        }
                    }

                    if let Some(candidate) = right.1.as_ref() {
                        if left
                            .1
                            .as_ref()
                            .map(|b| {
                                (candidate.matching_total, candidate.matching_lsb)
                                    < (b.matching_total, b.matching_lsb)
                            })
                            .unwrap_or(true)
                        {
                            left.1 = right.1;
                        }
                    }

                    left.2.merge(&right.2);
                    left
                },
            );

        histogram.merge(&attempt_hist);
        if let Some(report) = attempt_best {
            if best
                .as_ref()
                .map(|b| {
                    (report.matching_total, report.matching_lsb)
                        > (b.matching_total, b.matching_lsb)
                })
                .unwrap_or(true)
            {
                best = Some(report);
            }
        }
        if let Some(report) = attempt_worst {
            if worst
                .as_ref()
                .map(|b| {
                    (report.matching_total, report.matching_lsb)
                        < (b.matching_total, b.matching_lsb)
                })
                .unwrap_or(true)
            {
                worst = Some(report);
            }
        }
    }

    best.ok_or_else(|| "no valid r candidates after filtering".into())
}

/// Runs multiple trials for a fixed r candidate and summarizes statistics.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `config`: Full config with engine settings.
/// - `r_report`: Report describing the fixed r candidate to test.
/// - `label`: Label used for logging and output filenames.
/// - `iterations`: Number of iterations to run.
/// - `rng`: Random number generator for sampling messages.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `Option<(f64, f64, usize)>`: `(avg_bits, avg_overlap_pct, max_bits)` or `None` if skipped.
///
/// # Expected Output
/// - Prints progress and statistics; may write histogram and overlap plots.
fn run_fixed_r_trials(
    ctx: &RSAContext,
    config: &Config,
    r_report: &TestReport,
    label: &str,
    iterations: u64,
    rng: &mut RngChoice,
    shift: bool,
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
            let mut local_rng = RngChoice::from_seed(rng.mode(), seed);
            
            let msg = random_message_under_n(engine, &ctx.n, &mut local_rng);
            let ciphertext = msg.modpow(&ctx.e, &ctx.n);
            let dm = derive_candidate_message(
                ctx,
                engine,
                &ciphertext,
                r,
                &d_new,
                &n_pow_y,
                &r_pow_y,
                y,
                true,
                shift,
            );

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

/// Runs a stress test for a single r value and prints summary stats.
///
/// # Parameters
/// - `label`: Label identifying the stress-test source.
/// - `r`: Candidate r value to test.
/// - `ctx`: RSA context containing key material.
/// - `config`: Full configuration.
/// - `engine`: Engine configuration controlling trial behavior.
/// - `rng`: Random number generator for factorization/trials.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Prints summary stats when factorization and trials succeed.
fn run_r_stress_entry(
    label: &str,
    r: &BigUint,
    ctx: &RSAContext,
    config: &Config,
    engine: &EngineConfig,
    rng: &mut RngChoice,
    shift: bool,
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
        shift,
    ) {
        println!(
            "{} r {} -> avg bits {:.4}, avg overlap {:.4}%, max bits {}",
            label, r, avg_bits, avg_overlap, max_bits
        );
    }
}

/// Counts matching bits between two values (total and LSB run).
///
/// # Parameters
/// - `a`: First value to compare.
/// - `b`: Second value to compare.
///
/// # Returns
/// - `(usize, usize)`: `(matching_lsb_run, matching_total)` counts.
///
/// # Expected Output
/// - Returns counts based on binary string comparisons; no side effects.
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

/// Computes a derived value used in homomorphic base conversion flows.
///
/// # Parameters
/// - `x`: Input value.
/// - `p`: Modulus base.
/// - `y`: Exponent parameter.
/// - `apply_mod`: Whether to apply modulus at the end.
/// - `use_other_root`: Whether to use the alternate square root branch.
///
/// # Returns
/// - `BigUint`: Derived value based on modular square roots and exponentiation.
///
/// # Expected Output
/// - Returns a computed `BigUint`; no side effects.
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

/// Applies the homomorphic base conversion formula.
///
/// # Parameters
/// - `x`: Input value to convert.
/// - `r`: Target modulus.
/// - `p`: Source modulus.
///
/// # Returns
/// - `BigUint`: Converted value reduced modulo `r`.
///
/// # Expected Output
/// - Returns the base-converted value; no side effects.
fn homomorphic_base_conversion(x: &BigUint, r: &BigUint, p: &BigUint) -> BigUint {
    let y = x % p;
    let z = p % r;
    let q = (&y / p) * &z;
    let reduced = if &y >= p { &y - q } else { y.clone() };
    reduced % r
}

/// Dispatches between base conversion and division-based conversion.
///
/// # Parameters
/// - `x`: Input value to convert.
/// - `r`: Target modulus.
/// - `p`: Source modulus.
/// - `engine`: Engine configuration controlling conversion mode.
///
/// # Returns
/// - `BigUint`: Converted value.
///
/// # Expected Output
/// - Returns a converted value based on configuration; no side effects.
fn hbc(x: &BigUint, r: &BigUint, p: &BigUint, engine: &EngineConfig) -> BigUint {
    if engine.base_convert {
        homomorphic_base_conversion(x, r, p)
    } else {
        let num = r * x;
        num / p
    }
}

/// Derives the candidate message for a given ciphertext and r candidate.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `ciphertext`: Ciphertext to transform through the HBC flow.
/// - `r`: Candidate modulus for alternate decryption.
/// - `d_new`: Private exponent corresponding to `r`.
/// - `n_pow_y`: Precomputed `n^y` value.
/// - `r_pow_y`: Precomputed `r^y` value.
/// - `y`: Rabin exponent used for modular transforms.
/// - `use_other_root`: Whether to use the alternate square root branch.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `BigUint`: Derived candidate message modulo `n`.
///
/// # Expected Output
/// - Returns the derived message; no side effects.
fn derive_candidate_message(
    ctx: &RSAContext,
    engine: &EngineConfig,
    ciphertext: &BigUint,
    r: &BigUint,
    d_new: &BigUint,
    n_pow_y: &BigUint,
    r_pow_y: &BigUint,
    y: u32,
    use_other_root: bool,
    shift: bool,
) -> BigUint {
    let shifted = maybe_shift_ciphertext(ctx, ciphertext, shift);
    let result_default = get_larger_number(&shifted, &ctx.n, y, true, use_other_root);
    derive_candidate_message_from_result(
        ctx,
        engine,
        &result_default,
        r,
        d_new,
        n_pow_y,
        r_pow_y,
        y,
        use_other_root,
    )
}

/// Derives the candidate message given a precomputed first-stage result.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `result_default`: Output from the first `get_larger_number` stage.
/// - `r`: Candidate modulus for alternate decryption.
/// - `d_new`: Private exponent corresponding to `r`.
/// - `n_pow_y`: Precomputed `n^y` value.
/// - `r_pow_y`: Precomputed `r^y` value.
/// - `y`: Rabin exponent used for modular transforms.
/// - `use_other_root`: Whether to use the alternate square root branch.
///
/// # Returns
/// - `BigUint`: Derived candidate message modulo `n`.
///
/// # Expected Output
/// - Returns the derived message; no side effects.
fn derive_candidate_message_from_result(
    ctx: &RSAContext,
    engine: &EngineConfig,
    result_default: &BigUint,
    r: &BigUint,
    d_new: &BigUint,
    n_pow_y: &BigUint,
    r_pow_y: &BigUint,
    y: u32,
    use_other_root: bool,
) -> BigUint {
    let hbc_result = hbc(result_default, r, n_pow_y, engine);
    let recovered_new = if engine.use_rs_decrypt {
        hbc_result.modpow(d_new, r)
    } else {
        hbc_result
    };

    let result2_default = get_larger_number(&recovered_new, r, y, true, use_other_root);
    let hbc_default = hbc(&result2_default, &ctx.n, r_pow_y, engine);
    let dm_raw = &hbc_default % &ctx.n;
    let width = dm_raw.bits().max(1);
    let mask = (BigUint::one() << width) - BigUint::one();
    let inverted_dm = &mask ^ &dm_raw;
    if engine.invert_bits { inverted_dm } else { dm_raw }
}

/// Applies the optional ciphertext shift by homomorphically multiplying by encrypted 2.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `ciphertext`: Ciphertext to optionally shift.
/// - `shift`: Whether to apply the shift.
///
/// # Returns
/// - `BigUint`: Shifted ciphertext when enabled, otherwise the original ciphertext.
///
/// # Expected Output
/// - Returns a ciphertext value; no side effects.
fn maybe_shift_ciphertext(ctx: &RSAContext, ciphertext: &BigUint, shift: bool) -> BigUint {
    if !shift {
        return ciphertext.clone();
    }
    let enc_two = BigUint::from(2u8).modpow(&ctx.e, &ctx.n);
    (ciphertext * enc_two) % &ctx.n
}

/// Records per-candidate trace data for a raw r-candidate batch.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `message`: Plaintext message used to derive the ciphertext.
/// - `candidates`: Raw r candidates and factor lists.
/// - `analytics`: Session analytics accumulator to receive the trace batch.
/// - `context`: Label matching the candidate batch context.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Appends a trace batch to the analytics session; no stdout/stderr output.
fn record_r_candidate_trace_batch_from_factors(
    ctx: &RSAContext,
    engine: &EngineConfig,
    message: &BigUint,
    candidates: &[(BigUint, Vec<(BigUint, u64)>)],
    analytics: &Arc<Mutex<SessionAnalytics>>,
    context: &str,
    shift: bool,
) {
    if candidates.is_empty() {
        return;
    }

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);
    let ciphertext = message.modpow(&ctx.e, &ctx.n);
    let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
    let result_default = get_larger_number(&shifted, &ctx.n, y, true, false);

    let mut entries = Vec::with_capacity(candidates.len());
    for (r, factors) in candidates {
        let phi_new = compute_totient(factors);
        let Some(d_new) = mod_inverse(&ctx.e, &phi_new) else {
            continue;
        };
        let r_pow_y = r.pow(y);
        let hbc_result = hbc(&result_default, r, &n_pow_y, engine);
        let dm = derive_candidate_message_from_result(
            ctx,
            engine,
            &result_default,
            r,
            &d_new,
            &n_pow_y,
            &r_pow_y,
            y,
            false,
        );
        entries.push(RCandidateTraceEntry {
            r: r.to_string(),
            r_bits: r.bits(),
            hbc_ciphertext_r: hbc_result.to_string(),
            candidate_decryption: dm.to_string(),
        });
    }

    if entries.is_empty() {
        return;
    }

    with_analytics(analytics, |a| {
        a.push_r_candidate_trace_batch(RCandidateTraceBatch {
            context: context.to_string(),
            message: message.to_string(),
            ciphertext: ciphertext.to_string(),
            shifted_ciphertext: shifted.to_string(),
            rabin_exponent: y,
            tonelli_shanks_modulus: n_pow_y.to_string(),
            tonelli_shanks_ciphertext: result_default.to_string(),
            candidates: entries,
        });
    });
}

/// Records per-candidate trace data for prepared oracle candidates.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `message`: Plaintext message used to derive the ciphertext.
/// - `candidates`: Prepared r candidates with precomputed exponents.
/// - `analytics`: Session analytics accumulator to receive the trace batch.
/// - `context`: Label matching the candidate batch context.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Appends a trace batch to the analytics session; no stdout/stderr output.
fn record_r_candidate_trace_batch_from_prepared(
    ctx: &RSAContext,
    engine: &EngineConfig,
    message: &BigUint,
    candidates: &[OracleCandidate],
    analytics: &Arc<Mutex<SessionAnalytics>>,
    context: &str,
    shift: bool,
) {
    if candidates.is_empty() {
        return;
    }

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);
    let ciphertext = message.modpow(&ctx.e, &ctx.n);
    let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
    let result_default = get_larger_number(&shifted, &ctx.n, y, true, false);

    let mut entries = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let hbc_result = hbc(&result_default, &candidate.r, &n_pow_y, engine);
        let dm = derive_candidate_message_from_result(
            ctx,
            engine,
            &result_default,
            &candidate.r,
            &candidate.d_new,
            &n_pow_y,
            &candidate.r_pow_y,
            y,
            false,
        );
        entries.push(RCandidateTraceEntry {
            r: candidate.r.to_string(),
            r_bits: candidate.r.bits(),
            hbc_ciphertext_r: hbc_result.to_string(),
            candidate_decryption: dm.to_string(),
        });
    }

    if entries.is_empty() {
        return;
    }

    with_analytics(analytics, |a| {
        a.push_r_candidate_trace_batch(RCandidateTraceBatch {
            context: context.to_string(),
            message: message.to_string(),
            ciphertext: ciphertext.to_string(),
            shifted_ciphertext: shifted.to_string(),
            rabin_exponent: y,
            tonelli_shanks_modulus: n_pow_y.to_string(),
            tonelli_shanks_ciphertext: result_default.to_string(),
            candidates: entries,
        });
    });
}

#[derive(Debug, Clone)]
struct AccuracyCandidate {
    r: BigUint,
    factors: Vec<(BigUint, u64)>,
    d_new: BigUint,
    r_pow_y: BigUint,
}

/// Runs r-candidate accuracy batches with one random message per batch.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling candidate and message settings.
/// - `rng`: Random number generator for candidate and message sampling.
/// - `analytics`: Session analytics accumulator receiving batch data.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error on invalid configuration.
///
/// # Expected Output
/// - Appends accuracy batches to the session analytics; no stdout/stderr output.
fn run_r_candidate_accuracy_batches(
    ctx: &RSAContext,
    engine: &EngineConfig,
    rng: &mut RngChoice,
    analytics: &Arc<Mutex<SessionAnalytics>>,
    shift: bool,
) -> Result<(), Box<dyn Error>> {
    if !engine.analysis_batch_enable {
        return Ok(());
    }

    let candidates_per_batch_raw = engine.analysis_batch_candidates;
    let batch_count_raw = engine.analysis_batch_batches;
    if candidates_per_batch_raw == 0 {
        return Err("analysis_batch_candidates must be >= 1".into());
    }
    if batch_count_raw == 0 {
        return Err("analysis_batch_batches must be >= 1".into());
    }

    let candidates_per_batch = candidates_per_batch_raw as usize;
    let batch_count = batch_count_raw as usize;

    let total_candidates = candidates_per_batch * batch_count;
    let settings = build_r_candidate_settings(engine);
    let mut candidates = generate_r_candidates_with_analytics(
        "analysis_batch_accuracy",
        &ctx.n,
        &settings,
        rng,
        total_candidates,
        analytics,
    );
    if candidates.is_empty() {
        return Err("no r candidates generated for accuracy batches".into());
    }

    let y = engine.rabin_exponent as u32;
    let mut prepared = Vec::with_capacity(candidates.len());
    for (r, factors) in candidates.drain(..) {
        let phi_new = compute_totient(&factors);
        let Some(d_new) = mod_inverse(&ctx.e, &phi_new) else {
            continue;
        };
        let r_pow_y = r.pow(y);
        prepared.push(AccuracyCandidate {
            r,
            factors,
            d_new,
            r_pow_y,
        });
    }

    if prepared.len() < total_candidates {
        return Err(format!(
            "only {} valid r candidates available for accuracy batches (need {})",
            prepared.len(),
            total_candidates
        )
        .into());
    }

    let n_pow_y = ctx.n.pow(y);
    let mut candidate_offset = 0usize;
    for batch_idx in 0..batch_count {
        let start = candidate_offset;
        let end = candidate_offset + candidates_per_batch;
        let batch_candidates = &prepared[start..end];
        candidate_offset = end;

        let message = random_message_under_n(engine, &ctx.n, rng);
        let messages = vec![message.clone()];
        let ciphertext = message.modpow(&ctx.e, &ctx.n);
        let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
        let result_default = get_larger_number(&shifted, &ctx.n, y, true, false);
        let ciphertexts = vec![ciphertext];
        let shifted_ciphertexts = vec![shifted];
        let tonelli_ciphertexts = vec![result_default.clone()];

        let mut entries = Vec::with_capacity(batch_candidates.len());
        for candidate in batch_candidates {
            let hbc_result = hbc(&result_default, &candidate.r, &n_pow_y, engine);
            let dm = derive_candidate_message_from_result(
                ctx,
                engine,
                &result_default,
                &candidate.r,
                &candidate.d_new,
                &n_pow_y,
                &candidate.r_pow_y,
                y,
                false,
            );
            let message_bits = message.bits().max(1) as f64;
            let (_, matching_total) = count_matching_bits(&dm, &message);
            let accuracy_pct = (matching_total as f64 / message_bits) * 100.0;
            let entry = RCandidateAccuracyEntry {
                r: candidate.r.to_string(),
                r_bits: candidate.r.bits(),
                factors: candidate
                    .factors
                    .iter()
                    .map(|(p, e)| RCandidateFactor {
                        prime: p.to_string(),
                        exponent: *e,
                        prime_bits: p.bits(),
                    })
                    .collect(),
                accuracy_pct,
                hbc_ciphertexts_r: vec![hbc_result.to_string()],
                candidate_decryptions: vec![dm.to_string()],
            };
            entries.push(entry);
        }

        with_analytics(analytics, |a| {
            a.push_r_candidate_accuracy_batch(RCandidateAccuracyBatch {
                context: format!("analysis_batch_accuracy_{}", batch_idx + 1),
                messages: messages.iter().map(|m| m.to_string()).collect(),
                ciphertexts: ciphertexts.iter().map(|c| c.to_string()).collect(),
                shifted_ciphertexts: shifted_ciphertexts.iter().map(|c| c.to_string()).collect(),
                rabin_exponent: y,
                tonelli_shanks_modulus: n_pow_y.to_string(),
                tonelli_shanks_ciphertexts: tonelli_ciphertexts
                    .iter()
                    .map(|c| c.to_string())
                    .collect(),
                candidates: entries,
            });
        });
    }

    Ok(())
}

#[allow(dead_code)]
/// Picks a subset product of prime factors closest to a target value.
///
/// # Parameters
/// - `target_n`: Target value to approximate.
/// - `prime_factors`: Candidate prime factors.
///
/// # Returns
/// - `BigUint`: Product of a subset of factors closest to `target_n`.
///
/// # Expected Output
/// - Returns `1` if `prime_factors` is empty; no side effects.
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
    use crate::analytics::AnalyticsCliArgs;
    use crate::dsp::{find_ramp_signals, ramp_signal_strength};
    use crate::r_candidates::RCandidateMode;

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
        config.engine.process_count = 6;
        config.engine.process_min_count = 6;
        config.engine.min_message_trials = 1;
        config.engine.rabin_exponent = 3;

        let msg = BigUint::from(42u8);
        let mut rng = RngChoice::from_seed(RngMode::Standard, 101);
        let mut hist = MatchHistogram::new();
        let analytics = Arc::new(Mutex::new(SessionAnalytics::new(AnalyticsCliArgs {
            bits: 56,
            message_override: None,
            public_exponent: 65_537,
            seed: None,
            crypto_rng: false,
            config_path: "config/rsa_config.json".to_string(),
            tests: false,
            export: false,
            session_json: "session.json".to_string(),
            shift: false,
        })));
        let result = run_message_trial(
            &ctx,
            &config,
            &config.engine,
            &msg,
            1,
            &mut rng,
            &mut hist,
            &analytics,
            false,
        );
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
        let mut rng = RngChoice::from_seed(RngMode::Standard, 102);
        let mut hist = MatchHistogram::new();
        let analytics = Arc::new(Mutex::new(SessionAnalytics::new(AnalyticsCliArgs {
            bits: 56,
            message_override: None,
            public_exponent: 65_537,
            seed: None,
            crypto_rng: false,
            config_path: "config/rsa_config.json".to_string(),
            tests: false,
            export: false,
            session_json: "session.json".to_string(),
            shift: false,
        })));
        let result = run_message_trial(
            &ctx,
            &config,
            &config.engine,
            &msg,
            1,
            &mut rng,
            &mut hist,
            &analytics,
            false,
        );
        if let Err(err) = &result {
            println!("Expected r candidate failure: {}", err);
        }
        assert!(result.is_err());
    }
}
