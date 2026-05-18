/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use bigdecimal::BigDecimal;
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
#[cfg(any(test, feature = "plots"))]
use std::{fs, path::PathBuf};

#[cfg(feature = "plots")]
use plotters::prelude::*;
use rayon::prelude::*;
#[cfg(feature = "plots")]
use std::sync::atomic::AtomicUsize;

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

use crate::analytics::{
    AvalancheBestCenterBiasReport, AvalancheCenterBiasEntry, AvalancheCombinationBeamResult,
    AvalancheCombinationSample, AvalancheCombinationSampleInput, AvalancheFinalTierBiasReport,
    AvalancheTierSampleStat, AvalancheTierStatistics, RCandidateAccuracyBatch,
    RCandidateTraceBatch, RCandidateTraceEntry, SessionAnalytics,
    generate_r_candidates_with_analytics,
};
use crate::avalanche::{
    AvalancheBuilder, AvalancheInput, AvalancheNode, mirror_inverted_candidates,
    search_avalanche_tree_with_progress, search_avalanche_tree_with_scores_progress,
    sort_candidates_by_hamming_distance,
};
use crate::combiner::majority_vote_with_distribution;
use crate::config::{
    Config, EngineConfig, RsaKeyFileFormat, load_rsa_key_material_from_config_keyfile,
};
use crate::database::{
    AvalancheCacheGuard, approximate_scored_avalanche_input_bytes,
    build_cached_avalanche_tier_statistics, count_cached_scored_inputs,
    count_cached_selected_samples, deserialize_selected_avalanche_sample_row,
    insert_cached_scored_inputs, insert_cached_selected_samples,
    load_cached_recursive_sample_summaries, load_cached_scored_inputs_by_ids,
    load_cached_selected_sample_rows_by_ids, load_cached_selected_sample_rows_page,
};
use crate::fitness::{
    AvalancheFitnessScore, CachedScoredInputSummary, HammingDistancePrunedPool,
    RankedScoredAvalancheInput, StreamingScoredAvalancheFitnessPool,
    apply_cached_scored_avalanche_fitness_pass, apply_ranked_scored_avalanche_fitness_pass,
    build_candidate_message_transform, build_scored_avalanche_fitness_pass,
    extract_payload_bits_for_accuracy, enforce_global_unique_cached_scored_inputs,
    enforce_global_unique_scored_inputs,
    group_cached_scored_inputs_by_r_candidate_with_progress,
    group_scored_inputs_by_r_candidate_with_progress, load_cached_scored_input_summaries,
    lsb_zero_count_fitness, normalize_avalanche_fitness_mean_score,
    normalize_avalanche_fitness_score, payload_message_bits,
    prune_cached_scored_inputs_by_hamming_distance_percentile_with_progress,
    prune_scored_inputs_by_hamming_distance_percentile_with_progress, resolve_avalanche_bit_width,
    resolve_avalanche_fitness_bit_width, resolve_avalanche_fitness_retained_input_limit,
    resolve_plaintext_message_bit_width, single_message_avalanche_fitness_score,
    transform_message_for_candidate_scoring, validate_avalanche_fitness_log_top_pct,
    validate_avalanche_fitness_threshold, validate_message_width_under_modulus,
};
use crate::helpers::{
    PackedBits, format_beam_float, matching_bit_counts_bytes_le, normalize_avalanche_biases,
    spread_normalized_avalanche_biases, stored_beam_value_is_one,
};
use crate::math::{
    bit_length, choose_exponent, compute_totient, mod_inverse, random_biguint_bits,
    random_prime_with_bits, shannon_entropy_bit, to_hex,
};
use crate::r_candidates::{RCandidate, RCandidateSettings, resolve_retargeted_r_candidates_path};
use crate::rng::{RngChoice, RngMode};
use crate::search::{beam_search_top_k, beam_search_top_k_with_progress, viterbi_decode};
use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};
use rand::RngCore;
use serde::Serialize;
use serde_json::json;

#[cfg(test)]
use crate::database::resolve_avalanche_cache_db_path;
#[cfg(test)]
use crate::fitness::apply_scored_avalanche_fitness_pass;
#[cfg(test)]
use crate::fitness::select_scored_inputs_for_mixed_r_candidates;
#[cfg(test)]
use diesel::QueryableByName;
#[cfg(test)]
use diesel::prelude::*;
#[cfg(test)]
use diesel::sql_query;

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
    pub true_match: bool,
    pub bits_decrypt: Option<u32>,
}

const AVALANCHE_CACHE_FLUSH_BYTES: usize = 1 << 30;

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

/// Resolves the bit width for decryptions that use ciphertext exponent variants.
///
/// # Parameters
/// - `message`: Reference message used for sizing defaults.
/// - `expected_bits`: Optional expected bit width override.
///
/// # Returns
/// - `Result<usize, Box<dyn Error>>`: Bit width to use for decryption bit vectors.
///
/// # Expected Output
/// - Returns the resolved bit width; no stdout/stderr output.
fn resolve_decrypt_bit_width(
    message: &BigUint,
    expected_bits: Option<u32>,
) -> Result<usize, Box<dyn Error>> {
    if let Some(bits) = expected_bits {
        return usize::try_from(bits).map_err(|_| "decrypt bit width exceeds usize range".into());
    }
    Ok(message.bits().max(1) as usize)
}

#[derive(Clone, Debug)]
enum RoundtripVerification {
    Full {
        p: BigUint,
        q: BigUint,
        phi: BigUint,
        d: BigUint,
    },
    Peek {
        d: BigUint,
    },
}

/// Returns whether the active RSA key configuration uses a public keyfile without inline primes.
///
/// # Parameters
/// - `config`: Loaded configuration driving the current analysis or demo run.
///
/// # Returns
/// - `bool`: `true` when the active key material comes from a public keyfile without hydrated primes.
///
/// # Expected Output
/// - Returns the derived mode flag; no stdout/stderr output.
fn uses_public_keyfile_only(config: &Config) -> bool {
    !config.rsa_keypair.generate
        && config.rsa_keypair.modulus.is_some()
        && config.rsa_keypair.p.is_none()
        && config.rsa_keypair.q.is_none()
}

/// Loads the optional private verification key referenced by a public-key config.
///
/// # Parameters
/// - `config`: Loaded configuration that may reference `rsa_keypair.private_keyfile`.
/// - `expected_modulus`: Public modulus that the private peek file must match.
/// - `expected_exponent`: Public exponent that the private peek file must match.
///
/// # Returns
/// - `Result<Option<RoundtripVerification>, Box<dyn Error>>`: Optional private verification handle.
///
/// # Expected Output
/// - Reads the configured private keyfile when present; no stdout/stderr output on success.
fn load_private_keyfile_peek(
    config: &Config,
    expected_modulus: &BigUint,
    expected_exponent: &BigUint,
) -> Result<Option<RoundtripVerification>, Box<dyn Error>> {
    let private_keyfile = config.rsa_keypair.private_keyfile.trim();
    if private_keyfile.is_empty() {
        return Ok(None);
    }

    let config_path = config
        .source_path
        .as_deref()
        .ok_or("relative rsa_keypair.private_keyfile requires a loaded config path")?;
    let material = load_rsa_key_material_from_config_keyfile(config_path, private_keyfile)?;
    if material.format != RsaKeyFileFormat::PrivateKeyV1 {
        return Err(format!(
            "rsa_keypair.private_keyfile must reference an rsa-private-key-v1 file, got {:?}",
            material.format
        )
        .into());
    }
    if material.modulus != *expected_modulus {
        return Err(
            "rsa_keypair.private_keyfile modulus does not match rsa_keypair.keyfile".into(),
        );
    }

    if BigUint::from(material.public_exponent) != *expected_exponent {
        return Err(
            "rsa_keypair.private_keyfile public exponent does not match rsa_keypair.keyfile".into(),
        );
    }

    let d = material
        .private_exponent
        .ok_or("rsa_keypair.private_keyfile is missing private_exponent")?;
    Ok(Some(RoundtripVerification::Peek { d }))
}

/// Returns whether public-key-only analysis should prefer beam score over private-match ordering.
///
/// # Parameters
/// - `config`: Loaded configuration driving the current analysis or demo run.
///
/// # Returns
/// - `bool`: `true` when the run uses only a public keyfile without a private verification peek.
///
/// # Expected Output
/// - Returns the derived ordering flag; no stdout/stderr output.
fn prefer_public_key_beam_score_ordering(config: &Config) -> bool {
    uses_public_keyfile_only(config) && config.rsa_keypair.private_keyfile.trim().is_empty()
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
    let public_key_mode = uses_public_keyfile_only(&config);
    let prefer_beam_score_ordering = prefer_public_key_beam_score_ordering(&config);
    with_analytics(analytics, |a| {
        a.mark_feature("keypair", true);
        a.mark_feature("message_select", true);
        a.mark_feature("rsa_roundtrip", !public_key_mode);
        a.mark_feature("information_sufficiency", args.tests);
        a.mark_feature("r_candidate_accuracy", config.engine.analysis_batch_enable);
        a.set_feature_stat("rsa_roundtrip", "shift_enabled", json!(args.shift));
        a.set_feature_stat(
            "rsa_roundtrip",
            "mode",
            json!(if public_key_mode {
                if config.rsa_keypair.private_keyfile.trim().is_empty() {
                    "public_keyfile_only"
                } else {
                    "public_keyfile_with_private_peek"
                }
            } else {
                "private_roundtrip"
            }),
        );
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
    with_analytics(analytics, |a| {
        a.record_step("rng_init", rng_start.elapsed())
    });

    let key_start = Instant::now();
    let start_e = if args.public_exponent != 65_537 {
        args.public_exponent
    } else {
        config.rsa_keypair.e
    };
    let (n, e, key_bit_width, roundtrip_verification): (
        BigUint,
        BigUint,
        u64,
        Option<RoundtripVerification>,
    ) = if config.rsa_keypair.generate {
        let p = random_prime_with_bits(args.bits, &mut rng);
        let mut q = random_prime_with_bits(args.bits, &mut rng);
        while q == p {
            q = random_prime_with_bits(args.bits, &mut rng);
        }
        let one = BigUint::one();
        let n = &p * &q;
        let phi = (&p - &one) * (&q - &one);
        let e = choose_exponent(start_e, &phi);
        let d = mod_inverse(&e, &phi)
            .ok_or("public exponent is not invertible; try a different size or exponent")?;
        (
            n,
            e,
            p.bits().saturating_add(q.bits()),
            Some(RoundtripVerification::Full { p, q, phi, d }),
        )
    } else if let (Some(p), Some(q)) = (config.rsa_keypair.p.clone(), config.rsa_keypair.q.clone())
    {
        let one = BigUint::one();
        let n = &p * &q;
        let phi = (&p - &one) * (&q - &one);
        let e = choose_exponent(start_e, &phi);
        let d = mod_inverse(&e, &phi)
            .ok_or("public exponent is not invertible; try a different size or exponent")?;
        (
            n,
            e,
            p.bits().saturating_add(q.bits()),
            Some(RoundtripVerification::Full { p, q, phi, d }),
        )
    } else {
        let n = config
            .rsa_keypair
            .modulus
            .clone()
            .ok_or("config.rsa_keypair.keyfile must provide a modulus when generate is false and no inline primes are configured")?;
        let e = BigUint::from(start_e);
        let roundtrip_verification = load_private_keyfile_peek(&config, &n, &e)?;
        (n.clone(), e, n.bits(), roundtrip_verification)
    };
    with_analytics(analytics, |a| {
        a.record_step("keypair_select", key_start.elapsed());
        a.record_feature_duration("keypair", key_start.elapsed());
    });

    let exponent_start = Instant::now();
    with_analytics(analytics, |a| {
        a.record_step("keypair_derive", exponent_start.elapsed())
    });

    let message_start = Instant::now();
    let message = select_message(args.message.clone(), &config.engine, &n, &mut rng)?;
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

    let ciphertext = message.modpow(&e, &n);
    let recovered = if let Some(verification) = roundtrip_verification.as_ref() {
        let roundtrip_start = Instant::now();
        let recovered = match verification {
            RoundtripVerification::Full { d, .. } | RoundtripVerification::Peek { d } => {
                ciphertext.modpow(d, &n)
            }
        };
        if recovered != message {
            return Err("RSA round trip failed".into());
        }
        if !public_key_mode {
            with_analytics(analytics, |a| {
                a.record_step("rsa_roundtrip", roundtrip_start.elapsed());
                a.record_feature_duration("rsa_roundtrip", roundtrip_start.elapsed());
            });
        }
        Some(recovered)
    } else {
        with_analytics(analytics, |a| {
            a.add_feature_note(
                "rsa_roundtrip",
                "skipped round-trip verification because the active rsa_keypair.keyfile is public-only",
            );
        });
        None
    };

    if let Some(RoundtripVerification::Full { p, q, phi, d }) = roundtrip_verification.as_ref() {
        println!("Prime p ({} bits): {p}", bit_length(p));
        println!("Prime q ({} bits): {q}", bit_length(q));
        println!("Modulus n ({} bits): {n}", n.bits());
        println!("phi(n): {phi}");
        println!("Public exponent e: {e}");
        println!("Private exponent d: {d}");
    } else {
        println!("Modulus n ({} bits): {n}", n.bits());
        println!("Public exponent e: {e}");
    }
    println!("Plaintext (hex): {}", to_hex(&message));
    println!("Ciphertext (hex): {}", to_hex(&ciphertext));
    if let Some(recovered) = recovered.as_ref() {
        if public_key_mode {
            println!("Peek recovered (hex): {}", to_hex(recovered));
        } else {
            println!("Recovered (hex): {}", to_hex(recovered));
        }
    } else {
        println!("RSA round-trip verification: skipped (public keyfile only)");
    }

    if let Some(seed) = args.seed {
        println!("RNG seed: {seed}");
    }

    let ctx = RSAContext {
        n: n.clone(),
        e: e.clone(),
        key_bit_width,
    };
    let avalanche_cache = if config.engine.analysis_batch_enable {
        let cache = AvalancheCacheGuard::new(args.seed, &config.engine)?;
        println!("Avalanche cache database: {}", cache.path.to_string_lossy());
        Some(cache)
    } else {
        None
    };

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
            args.true_match,
            args.bits_decrypt,
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
        if let Err(err) = run_r_candidate_accuracy_batches(
            &ctx,
            &config.engine,
            &mut rng,
            analytics,
            args.shift,
            prefer_beam_score_ordering,
            avalanche_cache.as_ref(),
        ) {
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
                json!(config.engine.analysis_batch_messages),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "candidates_per_batch",
                json!(if config.engine.same_r_batch {
                    1
                } else {
                    config.engine.analysis_batch_candidates
                }),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "batch_count",
                json!(config.engine.analysis_batch_batches),
            );
        });
    }

    Ok(())
}

/// Logs progress updates at a fixed percentage increment.
///
/// # Parameters
/// - `done`: Number of completed items.
/// - `total`: Total number of items.
/// - `next_pct`: Mutable threshold for the next log event.
/// - `label`: Human-readable label for the progress report.
/// - `step_pct`: Percentage increment used for log emission.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Prints progress updates to stdout when thresholds are reached.
fn log_progress_every_percent_step(
    done: u64,
    total: u64,
    next_pct: &mut u64,
    label: &str,
    step_pct: u64,
) {
    if total == 0 {
        return;
    }
    let step_pct = step_pct.clamp(1, 100);

    let pct = done.saturating_mul(100) / total;
    if pct >= *next_pct || done == total {
        let display_pct = if done == total {
            100
        } else {
            ((pct / step_pct) * step_pct).min(100)
        };
        println!("{label} progress: {}% ({}/{})", display_pct, done, total);

        while *next_pct <= pct && *next_pct < 100 {
            *next_pct += step_pct;
        }
        if done == total {
            *next_pct = 100;
        }
    }
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
    log_progress_every_percent_step(done, total, next_pct, label, 10);
}

/// Logs progress for parallel work every ten percent using atomics.
///
/// # Parameters
/// - `done`: Number of completed items after the latest atomic increment.
/// - `total`: Total number of items expected.
/// - `next_pct`: Shared next-percentage threshold for the progress report.
/// - `label`: Human-readable label for the progress report.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Prints progress updates to stdout when thresholds are reached across parallel workers.
fn log_parallel_progress_every_ten_percent(
    done: u64,
    total: u64,
    next_pct: &AtomicU64,
    label: &str,
) {
    if total == 0 {
        return;
    }

    let pct = done.saturating_mul(100) / total;
    loop {
        let threshold = next_pct.load(Ordering::Relaxed);
        if pct < threshold && done != total {
            return;
        }

        let display_pct = if done == total {
            100
        } else {
            ((pct / 10) * 10).min(100)
        };
        let mut updated_threshold = threshold;
        while updated_threshold <= pct && updated_threshold < 100 {
            updated_threshold += 10;
        }
        if done == total {
            updated_threshold = 100;
        }

        if next_pct
            .compare_exchange(
                threshold,
                updated_threshold,
                Ordering::SeqCst,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            println!("{label} progress: {}% ({}/{})", display_pct, done, total);
            return;
        }
    }
}

/// Logs progress for parallel work at a fixed wall-clock interval using atomics.
///
/// # Parameters
/// - `done`: Number of completed items after the latest atomic increment.
/// - `total`: Total number of items expected.
/// - `start`: Start time for the parallel region.
/// - `next_log_at_ms`: Shared elapsed-milliseconds deadline for the next progress report.
/// - `label`: Human-readable label for the progress report.
/// - `interval`: Minimum duration between progress reports.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Prints progress updates when the interval elapses or when work reaches completion.
pub(crate) fn log_parallel_progress_every_interval(
    done: u64,
    total: u64,
    start: &Instant,
    next_log_at_ms: &AtomicU64,
    label: &str,
    interval: Duration,
) {
    if total == 0 {
        return;
    }

    let elapsed_ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    let interval_ms = interval.as_millis().min(u128::from(u64::MAX)) as u64;
    loop {
        let scheduled_ms = next_log_at_ms.load(Ordering::Relaxed);
        if done != total && elapsed_ms < scheduled_ms {
            return;
        }

        let next_deadline_ms = if done == total {
            u64::MAX
        } else {
            elapsed_ms.saturating_add(interval_ms)
        };

        if next_log_at_ms
            .compare_exchange(
                scheduled_ms,
                next_deadline_ms,
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            let pct = ((done as f64) * 100.0 / (total as f64)).min(100.0);
            println!("{label} progress: {:.5}% ({}/{})", pct, done, total);
            return;
        }
    }
}

/// Computes a chunk size that keeps enough parallel work in flight for progress-aware scans.
///
/// # Parameters
/// - `total_items`: Number of items that will be processed in parallel.
///
/// # Returns
/// - `usize`: Chunk size for Rayon chunk-based work; always at least `1`.
///
/// # Expected Output
/// - Returns a chunk size only; no stdout/stderr output.
pub(crate) fn parallel_progress_chunk_size(total_items: usize) -> usize {
    if total_items == 0 {
        return 1;
    }

    let target_chunks = rayon::current_num_threads().saturating_mul(8).max(1);
    total_items.div_ceil(target_chunks).max(1)
}

/// Computes an increasing odd exponent `x` per batch instance so that `e * x` remains odd.
///
/// # Parameters
/// - `e`: RSA public exponent.
/// - `instance_idx`: Zero-based batch instance index.
/// - `context`: Label for error messages.
///
/// # Returns
/// - `Result<BigUint, Box<dyn Error>>`: Odd exponent value for the instance.
///
/// # Expected Output
/// - Returns the computed exponent; no side effects.
fn odd_ciphertext_exponent(
    e: &BigUint,
    instance_idx: usize,
    context: &str,
) -> Result<BigUint, Box<dyn Error>> {
    if e.is_even() {
        return Err(format!("{context} requires an odd public exponent to keep e*x odd").into());
    }
    let idx = u64::try_from(instance_idx)
        .map_err(|_| format!("{context} message index exceeds u64 range"))?;
    let x_value = idx
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| format!("{context} message index exceeds u64 range"))?;
    Ok(BigUint::from(x_value))
}

/// Prepared ciphertext exponent variant guaranteed invertible for targeted candidates.
#[derive(Clone, Debug)]
struct CiphertextVariant {
    x: BigUint,
    e_x: BigUint,
    ciphertext: BigUint,
    shifted: BigUint,
}

/// Collects ciphertext variants whose `e * x` values are invertible for every target modulus.
///
/// # Parameters
/// - `ctx`: RSA context containing the modulus and public exponent.
/// - `base_ciphertext`: Base ciphertext to exponentiate.
/// - `phi_values`: Candidate totients that must admit inverses for every accepted `e * x`.
/// - `count`: Number of ciphertext variants to collect.
/// - `shift`: Whether to shift the accepted ciphertexts by encrypted `2`.
/// - `context`: Label used in overflow/error messages.
///
/// # Returns
/// - `Result<Vec<CiphertextVariant>, Box<dyn Error>>`: Accepted ciphertext variants in generation order.
///
/// # Expected Output
/// - Returns exactly `count` accepted variants or an error on exponent-index overflow; no stdout/stderr output.
fn collect_invertible_ciphertext_variants(
    ctx: &RSAContext,
    base_ciphertext: &BigUint,
    phi_values: &[&BigUint],
    count: usize,
    shift: bool,
    context: &str,
) -> Result<Vec<CiphertextVariant>, Box<dyn Error>> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let e_big = ctx.e.clone();
    let mut variants = Vec::with_capacity(count);
    let mut next_instance_idx = 0usize;
    let thread_chunk_floor = rayon::current_num_threads().saturating_mul(8).max(32);
    let accepted_total =
        u64::try_from(count).map_err(|_| format!("{context} count exceeds u64 range"))?;
    let accepted_done = AtomicU64::new(0);
    let progress_started_at = Instant::now();
    let progress_next_log_at_ms =
        AtomicU64::new(Duration::from_secs(5).as_millis().min(u128::from(u64::MAX)) as u64);
    let progress_label = format!("{context} ciphertext variants");
    while variants.len() < count {
        let remaining = count.saturating_sub(variants.len());
        let search_width = remaining.saturating_mul(4).max(thread_chunk_floor).max(1);
        let end_instance_idx = next_instance_idx
            .checked_add(search_width)
            .ok_or_else(|| format!("{context} exponent index overflow"))?;
        let chunk_results = (next_instance_idx..end_instance_idx)
            .into_par_iter()
            .map(|instance_idx| {
                let x = odd_ciphertext_exponent(&e_big, instance_idx, context)
                    .map_err(|err| err.to_string())?;
                let e_x = &e_big * &x;
                if !phi_values.iter().all(|phi| e_x.gcd(phi).is_one()) {
                    return Ok(None);
                }

                let ciphertext = if x.is_one() {
                    base_ciphertext.clone()
                } else {
                    base_ciphertext.modpow(&x, &ctx.n)
                };
                let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
                let done = accepted_done.fetch_add(1, Ordering::Relaxed) + 1;
                log_parallel_progress_every_interval(
                    done.min(accepted_total),
                    accepted_total,
                    &progress_started_at,
                    &progress_next_log_at_ms,
                    &progress_label,
                    Duration::from_secs(5),
                );
                Ok::<_, String>(Some(CiphertextVariant {
                    x,
                    e_x,
                    ciphertext,
                    shifted,
                }))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| -> Box<dyn Error> { err.into() })?;
        next_instance_idx = end_instance_idx;

        for variant in chunk_results.into_iter().flatten() {
            variants.push(variant);
            if variants.len() >= count {
                break;
            }
        }
    }
    log_parallel_progress_every_interval(
        accepted_total,
        accepted_total,
        &progress_started_at,
        &progress_next_log_at_ms,
        &progress_label,
        Duration::from_secs(5),
    );

    Ok(variants)
}

/// Computes the ciphertext variant for a candidate using its exponent `x`.
///
/// # Parameters
/// - `ctx`: RSA context containing modulus information.
/// - `base_ciphertext`: Ciphertext computed as `m^e mod n`.
/// - `candidate`: Oracle candidate providing the `x` exponent.
///
/// # Returns
/// - `BigUint`: Ciphertext for the candidate (`base_ciphertext^x mod n` when `x != 1`).
///
/// # Expected Output
/// - Returns the ciphertext variant; no side effects.
fn ciphertext_for_candidate(
    ctx: &RSAContext,
    base_ciphertext: &BigUint,
    candidate: &OracleCandidate,
) -> BigUint {
    if candidate.x.is_one() {
        base_ciphertext.clone()
    } else {
        base_ciphertext.modpow(&candidate.x, &ctx.n)
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
    r: BigUint,
    e: BigUint,
    x: BigUint,
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
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

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
/// - `true_match`: Whether to report the true match percentage without inversion.
/// - `selected_candidate`: Optional single r-candidate index to use for the batch.
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

    let e_value = ctx.e.clone();
    let base_ciphertext = message.modpow(&ctx.e, &ctx.n);
    let enc_two = BigUint::from(2u8).modpow(&ctx.e, &ctx.n);

    let mut entries = Vec::with_capacity(candidates.len() * (max_shift + 1).max(1));
    for (index, candidate) in candidates.iter().enumerate() {
        let mut base_match_pct = 0.0;
        let mut base_matching_bits = 0;
        let denom = bit_width.max(1) as f64;
        let x_value = candidate.x.clone();
        let candidate_ciphertext = ciphertext_for_candidate(ctx, &base_ciphertext, candidate);
        let mut shift_results = Vec::new();
        let mut enc_two_pow = BigUint::one();
        for shift_idx in 0..=max_shift {
            if shift_idx > 0 {
                enc_two_pow = (&enc_two_pow * &enc_two) % &ctx.n;
            }
            if shift_idx < base_shift {
                continue;
            }
            let shifted_ciphertext = (&candidate_ciphertext * &enc_two_pow) % &ctx.n;
            let prepared_ciphertext =
                prepare_candidate_ciphertext(engine, &shifted_ciphertext, &candidate.r, &ctx.n);
            shift_results.push((shift_idx, prepared_ciphertext));
        }

        for (shift_idx, prepared_ciphertext) in &shift_results {
            let dm = derive_candidate_message_from_result(
                ctx,
                engine,
                prepared_ciphertext,
                &candidate.r,
                &candidate.d_new,
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
                r: candidate.r.clone(),
                e: e_value.clone(),
                x: x_value.clone(),
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

/// Applies Jeffreys smoothing to per-bit `P(1)` estimates.
///
/// # Parameters
/// - `ones_count`: Count of `1` votes at each bit position.
/// - `total_oracles`: Number of sampled oracle bit vectors contributing to each count.
///
/// # Returns
/// - `Vec<f64>`: Smoothed per-bit probabilities of observing `1`.
///
/// # Expected Output
/// - Returns probabilities in `(0, 1)` when `total_oracles > 0`; no stdout/stderr output.
fn smooth_probability_one_jeffreys(ones_count: &[usize], total_oracles: usize) -> Vec<f64> {
    if total_oracles == 0 {
        return vec![0.5; ones_count.len()];
    }
    let denominator = total_oracles as f64 + 1.0;
    ones_count
        .iter()
        .map(|ones| (*ones as f64 + 0.5) / denominator)
        .collect()
}

/// Precomputed r-candidate data for oracle and timeline tests.
#[derive(Clone, Debug)]
struct OracleCandidate {
    r: BigUint,
    d_new: BigUint,
    phi_new: BigUint,
    x: BigUint,
    target_exponent: BigDecimal,
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
    ciphertext: BigUint,
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
pub(crate) fn biguint_to_bits_le(value: &BigUint, width: usize) -> Vec<bool> {
    let bytes = value.to_bytes_le();
    (0..width)
        .map(|idx| bit_from_bytes_le(&bytes, idx))
        .collect()
}

/// Converts a `BigUint` to fixed-width packed little-endian bit storage.
///
/// # Parameters
/// - `value`: Integer to convert.
/// - `width`: Number of logical bits to keep.
///
/// # Returns
/// - `PackedBits`: Packed little-endian bit storage of length `width`.
///
/// # Expected Output
/// - Returns packed bits padded with zero bytes if needed; no side effects.
fn biguint_to_packed_bits_le(value: &BigUint, width: usize) -> PackedBits {
    PackedBits::from_bytes_le(&value.to_bytes_le(), width)
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

    let packed_a = PackedBits::from_bools(&a[..min_len]);
    let packed_b = PackedBits::from_bools(&b[..min_len]);
    matching_bit_counts_bytes_le(packed_a.bytes_le(), packed_b.bytes_le(), min_len)
}

/// Computes a truncated match percentage using the provided reference bit width.
///
/// # Parameters
/// - `candidate`: Candidate value to score after truncation.
/// - `reference_bits`: Reference bit vector whose length defines the truncation width.
///
/// # Returns
/// - `(Vec<bool>, f64)`: Truncated candidate bits and their total match percentage.
///
/// # Expected Output
/// - Returns truncated bits and a percentage score; no stdout/stderr output.
fn truncated_match_percentage(candidate: &BigUint, reference_bits: &[bool]) -> (Vec<bool>, f64) {
    let bit_width = reference_bits.len().max(1);
    let candidate_bits = biguint_to_bits_le(candidate, bit_width);
    let (_, matching_total) = count_matching_bits_le(&candidate_bits, reference_bits);
    let match_pct = matching_total as f64 / bit_width as f64 * 100.0;
    (candidate_bits, match_pct)
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
/// - `Result<BigUint, Box<dyn Error>>`: Selected message value.
///
/// # Expected Output
/// - Returns a message under `n` when random or an error when the configured widened width cannot fit.
fn sample_message_for_tests(
    engine: &EngineConfig,
    n: &BigUint,
    fixed_message: &Option<BigUint>,
    rng: &mut RngChoice,
) -> Result<BigUint, Box<dyn Error>> {
    if let Some(msg) = fixed_message {
        return Ok(msg.clone());
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
    let settings = build_r_candidate_settings(engine, ctx.key_bit_width);
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

    let e_big = ctx.e.clone();
    let target_count = candidates.len();
    let mut prepared = Vec::with_capacity(target_count);

    if engine.same_r_batch {
        let mut selected: Option<(RCandidate, BigUint)> = None;
        for candidate in candidates {
            let phi_new = compute_totient(&candidate.factors);
            if mod_inverse(&e_big, &phi_new).is_some() {
                selected = Some((candidate, phi_new));
                break;
            }
        }

        let (candidate, phi_new) =
            selected.ok_or("no valid r candidates available for same-r analysis tests")?;

        let mut instance_idx = 0usize;
        let mut attempts = 0usize;
        let attempt_limit = target_count.saturating_mul(50).max(100);
        while prepared.len() < target_count {
            let x = odd_ciphertext_exponent(&e_big, instance_idx, "analysis_oracle_candidates")?;
            let e_x = &e_big * &x;
            if let Some(d_new) = mod_inverse(&e_x, &phi_new) {
                prepared.push(OracleCandidate {
                    r: candidate.r.clone(),
                    d_new,
                    phi_new: phi_new.clone(),
                    x,
                    target_exponent: candidate.target_exponent.clone(),
                });
            }
            instance_idx = instance_idx.saturating_add(1);
            attempts = attempts.saturating_add(1);
            if attempts > attempt_limit {
                return Err(
                    "insufficient invertible exponents for same-r analysis candidates".into(),
                );
            }
        }
    } else {
        for candidate in candidates {
            let phi_new = compute_totient(&candidate.factors);
            if let Some(d_new) = mod_inverse(&e_big, &phi_new) {
                prepared.push(OracleCandidate {
                    r: candidate.r,
                    d_new,
                    phi_new,
                    x: BigUint::one(),
                    target_exponent: candidate.target_exponent,
                });
            }
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

    let base_ciphertext = message.modpow(&ctx.e, &ctx.n);

    let mut best: Option<CandidateScore> = None;
    for candidate in candidates {
        let ciphertext = ciphertext_for_candidate(ctx, &base_ciphertext, candidate);
        let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
        let prepared_ciphertext =
            prepare_candidate_ciphertext(engine, &shifted, &candidate.r, &ctx.n);
        let dm = derive_candidate_message_from_result(
            ctx,
            engine,
            &prepared_ciphertext,
            &candidate.r,
            &candidate.d_new,
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

    let mut seeds = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        seeds.push(rng.next_u64());
    }
    let done = Arc::new(AtomicU64::new(0));
    let next_pct = Arc::new(AtomicU64::new(10));
    let iterations_u64 = iterations as u64;

    let mut results = seeds
        .into_par_iter()
        .map(|seed| {
            let mut local_rng = RngChoice::from_seed(rng.mode(), seed);
            let msg = sample_message_for_tests(engine, &ctx.n, &fixed_message, &mut local_rng)
                .map_err(|err| err.to_string())?;
            let base_ciphertext = msg.modpow(&ctx.e, &ctx.n);

            let mut oracles: Vec<Vec<bool>> = Vec::with_capacity(oracle_count);
            for candidate in candidates.iter().take(oracle_count) {
                let ciphertext = ciphertext_for_candidate(ctx, &base_ciphertext, candidate);
                let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
                let prepared_ciphertext =
                    prepare_candidate_ciphertext(engine, &shifted, &candidate.r, &ctx.n);
                let dm = derive_candidate_message_from_result(
                    ctx,
                    engine,
                    &prepared_ciphertext,
                    &candidate.r,
                    &candidate.d_new,
                );
                oracles.push(biguint_to_bits_le(&dm, bit_width));
            }

            let distribution =
                majority_vote_with_distribution(&oracles, engine.combiner_tie_breaker)
                    .map_err(|err| err.to_string())?;

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
                        let display_pct = if finished == iterations_u64 {
                            100
                        } else {
                            ((pct / 10) * 10).min(100)
                        };
                        println!(
                            "Oracle entropy timeline progress: {}% ({}/{})",
                            display_pct, finished, iterations_u64
                        );
                        current_next = new_next;
                    }
                    Err(actual) => current_next = actual,
                }
            }

            Ok::<_, String>((entropy, accuracy))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| -> Box<dyn Error> { err.into() })?;

    let mut entropy_mean = Vec::with_capacity(iterations);
    let mut accuracy_pct = Vec::with_capacity(iterations);
    for (entropy, accuracy) in results.drain(..) {
        entropy_mean.push(entropy);
        accuracy_pct.push(accuracy);
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
            let msg = sample_message_for_tests(engine, &ctx.n, &fixed_message, &mut local_rng)
                .map_err(|err| err.to_string())?;
            let base_ciphertext = msg.modpow(&ctx.e, &ctx.n);
            let ciphertext = ciphertext_for_candidate(ctx, &base_ciphertext, candidate);
            let dm = derive_candidate_message(
                ctx,
                engine,
                &ciphertext,
                &candidate.r,
                &candidate.d_new,
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

            Ok::<_, String>(MatchSample {
                message_bytes_le: msg.to_bytes_le(),
                candidate_bytes_le: dm.to_bytes_le(),
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| -> Box<dyn Error> { err.into() })?;

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

    let mut seeds = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        seeds.push(rng.next_u64());
    }
    let done = Arc::new(AtomicU64::new(0));
    let next_pct = Arc::new(AtomicU64::new(10));
    let iterations_u64 = iterations as u64;

    let samples: Vec<OracleTrainingSample> = seeds
        .into_par_iter()
        .map(|seed| {
            let mut local_rng = RngChoice::from_seed(rng.mode(), seed);
            let msg = random_message_under_n(engine, &ctx.n, &mut local_rng)
                .map_err(|err| err.to_string())?;
            let ciphertext = msg.modpow(&ctx.e, &ctx.n);
            let message_bits = biguint_to_bits_le(&msg, bit_width);

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
                        let display_pct = if finished == iterations_u64 {
                            100
                        } else {
                            ((pct / 10) * 10).min(100)
                        };
                        println!(
                            "Oracle screening progress: {}% ({}/{})",
                            display_pct, finished, iterations_u64
                        );
                        current_next = new_next;
                    }
                    Err(actual) => current_next = actual,
                }
            }

            Ok::<_, String>(OracleTrainingSample {
                ciphertext,
                message_bits,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| -> Box<dyn Error> { err.into() })?;

    let samples = Arc::new(samples);
    let counts: Vec<Vec<u32>> = candidates
        .par_iter()
        .map(|candidate| {
            let mut match_counts = vec![0u32; bit_width];
            for sample in samples.iter() {
                let ciphertext = ciphertext_for_candidate(ctx, &sample.ciphertext, candidate);
                let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
                let prepared_ciphertext =
                    prepare_candidate_ciphertext(engine, &shifted, &candidate.r, &ctx.n);
                let dm = derive_candidate_message_from_result(
                    ctx,
                    engine,
                    &prepared_ciphertext,
                    &candidate.r,
                    &candidate.d_new,
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
/// - `batch_size`: Number of ciphertext exponent variants in the batch.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
/// - `bits_decrypt`: Optional expected bit width override.
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
    batch_size: usize,
    shift: bool,
    true_match: bool,
    selected_candidate: Option<usize>,
    bits_decrypt: Option<u32>,
) -> Result<SpeculativeOracleReport, Box<dyn Error>> {
    if per_bit_oracles.is_empty() {
        return Err("per-bit oracle selection is empty".into());
    }
    if batch_size == 0 {
        return Err("analysis batch size must be >= 1".into());
    }
    let bit_width = resolve_decrypt_bit_width(message, bits_decrypt)?;
    let selected_candidate = selected_candidate.filter(|&idx| idx < candidates.len());
    let oracles_per_bit = if selected_candidate.is_some() {
        1
    } else {
        per_bit_oracles[0].len().max(1)
    };

    let base_ciphertext = message.modpow(&ctx.e, &ctx.n);

    let mut unique_oracle_indices = std::collections::HashSet::new();
    if let Some(candidate_idx) = selected_candidate {
        unique_oracle_indices.insert(candidate_idx);
    } else {
        for selections in per_bit_oracles {
            for selection in selections {
                unique_oracle_indices.insert(selection.oracle_idx);
            }
        }
    }
    let single_selection = selected_candidate.map(|idx| {
        vec![OracleBitSelection {
            oracle_idx: idx,
            invert: false,
        }]
    });

    let mut oracle_index_list: Vec<usize> = unique_oracle_indices.iter().copied().collect();
    oracle_index_list.sort_unstable();
    let phi_values: Vec<&BigUint> = oracle_index_list
        .iter()
        .map(|&oracle_idx| &candidates[oracle_idx].phi_new)
        .collect();
    let ciphertext_variants = collect_invertible_ciphertext_variants(
        ctx,
        &base_ciphertext,
        &phi_values,
        batch_size,
        shift,
        "analysis batch",
    )?;
    let oracle_bits_by_instance: Vec<Vec<Option<Vec<bool>>>> = ciphertext_variants
        .into_par_iter()
        .map(|variant| {
            let x_label = variant.x.to_string();

            let mut oracle_bits: Vec<Option<Vec<bool>>> = vec![None; candidates.len()];
            for oracle_idx in oracle_index_list.iter().copied() {
                let candidate = &candidates[oracle_idx];
                let d_new = mod_inverse(&variant.e_x, &candidate.phi_new).ok_or_else(|| {
                    format!(
                        "analysis batch missing modular inverse for oracle {} and x {}",
                        oracle_idx, x_label
                    )
                })?;
                let prepared_ciphertext =
                    prepare_candidate_ciphertext(engine, &variant.shifted, &candidate.r, &ctx.n);
                let dm = derive_candidate_message_from_result(
                    ctx,
                    engine,
                    &prepared_ciphertext,
                    &candidate.r,
                    &d_new,
                );
                oracle_bits[oracle_idx] = Some(biguint_to_bits_le(&dm, bit_width));
            }
            Ok::<_, String>(oracle_bits)
        })
        .collect::<Result<_, _>>()
        .map_err(|err| -> Box<dyn Error> { err.into() })?;

    let mut recovered_bits = vec![false; bit_width];
    if let Some(single) = single_selection.as_ref() {
        for bit_idx in 0..bit_width {
            let selections = single.as_slice();
            let mut ones = 0usize;
            let mut zeros = 0usize;
            for selection in selections {
                for oracle_bits in &oracle_bits_by_instance {
                    if let Some(bits) = oracle_bits
                        .get(selection.oracle_idx)
                        .and_then(|entry| entry.as_ref())
                    {
                        let bit = if selection.invert {
                            !bits[bit_idx]
                        } else {
                            bits[bit_idx]
                        };
                        if bit {
                            ones += 1;
                        } else {
                            zeros += 1;
                        }
                    }
                }
            }
            recovered_bits[bit_idx] = if ones == zeros {
                engine.combiner_tie_breaker
            } else {
                ones > zeros
            };
        }
    } else {
        for (bit_idx, selections) in per_bit_oracles.iter().enumerate().take(bit_width) {
            let mut ones = 0usize;
            let mut zeros = 0usize;
            for selection in selections {
                for oracle_bits in &oracle_bits_by_instance {
                    if let Some(bits) = oracle_bits
                        .get(selection.oracle_idx)
                        .and_then(|entry| entry.as_ref())
                    {
                        let bit = if selection.invert {
                            !bits[bit_idx]
                        } else {
                            bits[bit_idx]
                        };
                        if bit {
                            ones += 1;
                        } else {
                            zeros += 1;
                        }
                    }
                }
            }
            recovered_bits[bit_idx] = if ones == zeros {
                engine.combiner_tie_breaker
            } else {
                ones > zeros
            };
        }
    }

    let recovered = bits_le_to_biguint(&recovered_bits);
    let message_bits = biguint_to_bits_le(message, bit_width);
    let (matching_lsb, matching_total) = count_matching_bits_le(&recovered_bits, &message_bits);
    let mut match_pct = matching_total as f64 / bit_width.max(1) as f64 * 100.0;
    if !true_match {
        match_pct = match_pct.max(100.0 - match_pct);
    }

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

/// Builds avalanche candidates from unique `(e * x)^{-1} mod phi(r)` decryptions.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `candidates`: Prepared r candidates to use as oracles.
/// - `message`: Reference message used for bit width sizing.
/// - `batch_size`: Number of ciphertext exponent variants in the batch.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
/// # Returns
/// - `Result<Vec<AvalancheNode>, Box<dyn Error>>`: Avalanche nodes for tree search.
///
/// # Expected Output
/// - Returns candidate nodes truncated to `engine.message.bits`; no stdout/stderr output.
fn build_avalanche_nodes_unique_d(
    ctx: &RSAContext,
    engine: &EngineConfig,
    candidates: &[OracleCandidate],
    message: &BigUint,
    batch_size: usize,
    shift: bool,
) -> Result<Vec<AvalancheNode>, Box<dyn Error>> {
    if batch_size == 0 {
        return Err("analysis batch size must be >= 1".into());
    }
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let bit_width = resolve_avalanche_bit_width(engine);
    let avalanche_message =
        transform_message_for_candidate_scoring(engine, message, &ctx.n, "analysis avalanche")?;
    let base_ciphertext = avalanche_message.modpow(&ctx.e, &ctx.n);

    let use_distance = engine.use_hamming_distance;
    let mut seen: Vec<HashSet<BigUint>> = vec![HashSet::new(); candidates.len()];
    let target_bits = use_distance.then(|| biguint_to_bits_le(&avalanche_message, bit_width));
    let mut collected_nodes = Vec::new();

    struct CandidateInstanceNode {
        candidate_idx: usize,
        d_new: BigUint,
        node: AvalancheNode,
    }

    let phi_values: Vec<&BigUint> = candidates
        .iter()
        .map(|candidate| &candidate.phi_new)
        .collect();
    let ciphertext_variants = collect_invertible_ciphertext_variants(
        ctx,
        &base_ciphertext,
        &phi_values,
        batch_size,
        shift,
        "analysis avalanche",
    )?;
    let per_instance_nodes: Vec<Vec<CandidateInstanceNode>> = ciphertext_variants
        .into_par_iter()
        .map(|variant| {
            let x_label = variant.x.to_string();

            let mut nodes = Vec::with_capacity(candidates.len());
            for (candidate_idx, candidate) in candidates.iter().enumerate() {
                let d_new = mod_inverse(&variant.e_x, &candidate.phi_new).ok_or_else(|| {
                    format!(
                        "analysis avalanche missing modular inverse for candidate {} and x {}",
                        candidate_idx, x_label
                    )
                })?;
                let prepared_ciphertext =
                    prepare_candidate_ciphertext(engine, &variant.shifted, &candidate.r, &ctx.n);
                let dm = derive_candidate_message_from_result(
                    ctx,
                    engine,
                    &prepared_ciphertext,
                    &candidate.r,
                    &d_new,
                );
                let message_bits = biguint_to_packed_bits_le(&dm, bit_width);
                let node = AvalancheNode::from_packed_bits(message_bits, vec![0.0; bit_width]);
                nodes.push(CandidateInstanceNode {
                    candidate_idx,
                    d_new,
                    node,
                });
            }
            Ok::<_, String>(nodes)
        })
        .collect::<Result<_, _>>()
        .map_err(|err| -> Box<dyn Error> { err.into() })?;

    for instance_nodes in per_instance_nodes {
        for entry in instance_nodes {
            let seen_set = &mut seen[entry.candidate_idx];
            if !seen_set.insert(entry.d_new) {
                continue;
            }
            collected_nodes.push(entry.node);
        }
    }

    if engine.mirror_invert_candidates {
        collected_nodes = mirror_inverted_candidates(collected_nodes)
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
    }

    if use_distance {
        let reference_bits = target_bits
            .as_deref()
            .expect("distance ordering requires a reference bit vector");
        return sort_candidates_by_hamming_distance(collected_nodes, reference_bits)
            .map_err(|err| -> Box<dyn Error> { Box::new(err) });
    }

    if !collected_nodes.is_empty() {
        let mut nodes_with_value: Vec<(BigUint, AvalancheNode)> = collected_nodes
            .into_iter()
            .map(|node| (BigUint::from_bytes_le(node.packed_message_bits()), node))
            .collect();
        nodes_with_value.sort_by(|a, b| a.0.cmp(&b.0));
        collected_nodes = nodes_with_value.into_iter().map(|(_, node)| node).collect();
    }

    Ok(collected_nodes)
}

const BEAM_SCORE_DECIMALS: usize = 8;
pub(crate) const BEAM_PCT_DECIMALS: usize = 8;

/// Runs the avalanche tree search for unique `(e*x)^{-1} mod phi(r)` decryptions.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `candidates`: Prepared r candidates to use as oracles.
/// - `message`: Reference message used for bit width sizing.
/// - `batch_size`: Number of ciphertext exponent variants in the batch.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
/// - `analytics`: Session analytics accumulator for reporting.
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` when search runs or is skipped.
///
/// # Expected Output
/// - Prints avalanche bias diagnostics using `engine.message.bits` as the avalanche width and,
///   when batch sampling is disabled, detailed beam-search output.
fn run_avalanche_search(
    ctx: &RSAContext,
    engine: &EngineConfig,
    candidates: &[OracleCandidate],
    message: &BigUint,
    batch_size: usize,
    shift: bool,
    analytics: &Arc<Mutex<SessionAnalytics>>,
) -> Result<(), Box<dyn Error>> {
    let avalanche_nodes =
        build_avalanche_nodes_unique_d(ctx, engine, candidates, message, batch_size, shift)?;
    if avalanche_nodes.is_empty() {
        with_analytics(analytics, |a| {
            a.add_feature_note(
                "information_sufficiency",
                "avalanche tree skipped: no unique decryptions",
            );
        });
        return Ok(());
    }

    println!("Avalanche tree instances: {}", avalanche_nodes.len());
    let msb_one_count = avalanche_nodes
        .iter()
        .filter(|node| node.msb().unwrap_or(false))
        .count();
    let msb_zero_count = avalanche_nodes.len().saturating_sub(msb_one_count);
    let avalanche_count = avalanche_nodes.len();
    let avalanche_search = if engine.avalanche_statistics_collection {
        search_avalanche_tree_with_scores_progress(avalanche_nodes, "Avalanche tree reduction")?
    } else {
        crate::avalanche::AvalancheSearchResult {
            node: search_avalanche_tree_with_progress(avalanche_nodes, "Avalanche tree reduction")?,
            level_similarity_pct: Vec::new(),
            level_pair_counts: Vec::new(),
        }
    };
    let avalanche_result = avalanche_search.node;
    // dbg!(&avalanche_result);
    with_analytics(analytics, |a| {
        a.set_feature_stat(
            "information_sufficiency",
            "avalanche_tree",
            json!({
                "bit_order": "lsb0",
                "bit_width": avalanche_result.bit_len(),
                "unique_messages": avalanche_count,
                "biases": avalanche_result.biases,
                "message_bits": avalanche_result.message_bits_vec(),
                "level_similarity_pct": avalanche_search.level_similarity_pct,
                "level_pair_counts": avalanche_search.level_pair_counts,
            }),
        );
    });

    let bit_width = avalanche_result.bit_len().max(1);
    let beam_bit_one_threshold = engine.beam_bit_one_threshold;
    let avalanche_beam_top_k = engine.avalanche_beam_top_k.max(1);
    let avalanche_probability_spread_exponent = engine.avalanche_probability_spread_exponent;
    let normalized_biases = normalize_avalanche_biases(&avalanche_result.biases);
    let beam_probabilities = spread_normalized_avalanche_biases(
        &normalized_biases,
        avalanche_probability_spread_exponent,
    );
    println!(
        "Avalanche beam bias diagnostics: raw_len {} bit_width {} raw_last {}",
        avalanche_result.biases.len(),
        bit_width,
        avalanche_result.biases.last().copied().unwrap_or(0.0)
    );
    println!(
        "Avalanche beam MSB count: ones {} zeros {}",
        msb_one_count, msb_zero_count
    );
    println!(
        "Avalanche beam scoring thresholds: bit_one >= {} spread_exponent {}",
        format_beam_float(beam_bit_one_threshold, BEAM_SCORE_DECIMALS),
        format_beam_float(avalanche_probability_spread_exponent, BEAM_SCORE_DECIMALS),
    );
    let beam_result = beam_search_top_k_with_progress(
        vec![Vec::new()],
        avalanche_beam_top_k,
        bit_width,
        "Avalanche beam search",
        |candidate| {
            if candidate.len() >= bit_width {
                return Vec::new();
            }
            let mut zero = candidate.to_vec();
            let mut one = candidate.to_vec();
            zero.push(0.0);
            one.push(1.0);
            vec![zero, one]
        },
        |candidate| {
            candidate
                .iter()
                .enumerate()
                .map(|(idx, bit)| {
                    let bias = beam_probabilities.get(idx).copied().unwrap_or(0.0);
                    if stored_beam_value_is_one(*bit, beam_bit_one_threshold) {
                        bias
                    } else {
                        1.0 - bias
                    }
                })
                .sum()
        },
    )?;

    let message_bits = payload_message_bits(engine, message);
    if !engine.analysis_batch_enable {
        println!(
            "Avalanche beam search top {} candidates (lsb0 order):",
            beam_result.beam.len()
        );
        for (idx, candidate) in beam_result.beam.iter().enumerate() {
            let candidate_bits: Vec<bool> = candidate
                .vector
                .iter()
                .map(|value| stored_beam_value_is_one(*value, beam_bit_one_threshold))
                .collect();
            let payload_candidate_bits = extract_payload_bits_for_accuracy(engine, &candidate_bits);
            let (match_pct, ones_match_pct) =
                compute_bit_match_percentages(&payload_candidate_bits, &message_bits);
            let candidate_hex = format_bits_hex_le(&payload_candidate_bits);
            println!(
                "Beam {} score {} match {}% ones-match {}% hex {}",
                idx + 1,
                format_beam_float(candidate.score, BEAM_SCORE_DECIMALS),
                format_beam_float(match_pct, BEAM_PCT_DECIMALS),
                format_beam_float(ones_match_pct, BEAM_PCT_DECIMALS),
                candidate_hex
            );
            let candidate_value = bits_le_to_biguint(&payload_candidate_bits);
            println!(
                "Beam {} bits: total {} biguint {}",
                idx + 1,
                payload_candidate_bits.len(),
                candidate_value.bits()
            );
        }
    }
    if let Some(top) = beam_result.beam.first() {
        let top_bits: Vec<bool> = top
            .vector
            .iter()
            .map(|value| stored_beam_value_is_one(*value, beam_bit_one_threshold))
            .collect();
        let payload_top_bits = extract_payload_bits_for_accuracy(engine, &top_bits);
        let msb = payload_top_bits.last().copied().unwrap_or(false);
        println!("Avalanche beam top MSB: {}", if msb { 1 } else { 0 });
    }

    let viterbi_bits = {
        let observations: Vec<usize> = (0..bit_width).collect();
        let start_log_probs = vec![0.5f64.ln(), 0.5f64.ln()];
        let transition_log_probs = vec![
            vec![0.5f64.ln(), 0.5f64.ln()],
            vec![0.5f64.ln(), 0.5f64.ln()],
        ];
        let emission_zero: Vec<f64> = beam_probabilities
            .iter()
            .map(|bias| {
                let p = bias.clamp(1e-12, 1.0 - 1e-12);
                (1.0 - p).ln()
            })
            .collect();
        let emission_one: Vec<f64> = beam_probabilities
            .iter()
            .map(|bias| {
                let p = bias.clamp(1e-12, 1.0 - 1e-12);
                p.ln()
            })
            .collect();
        let emission_log_probs = vec![emission_zero, emission_one];
        let result = viterbi_decode(
            &observations,
            &start_log_probs,
            &transition_log_probs,
            &emission_log_probs,
        )?;
        let bits: Vec<bool> = result.path.iter().map(|state| *state == 1).collect();
        (bits, result.log_prob)
    };

    let payload_viterbi_bits = extract_payload_bits_for_accuracy(engine, &viterbi_bits.0);
    let mut viterbi_hex = to_hex(&bits_le_to_biguint(&payload_viterbi_bits));
    let hex_len = payload_viterbi_bits.len().div_ceil(4);
    if viterbi_hex.len() < hex_len {
        let padding = "0".repeat(hex_len - viterbi_hex.len());
        viterbi_hex = format!("{}{}", padding, viterbi_hex);
    }
    println!(
        "Avalanche viterbi decode (lsb0 order): log_prob {} hex {}",
        format_beam_float(viterbi_bits.1, BEAM_SCORE_DECIMALS),
        viterbi_hex
    );
    let viterbi_msb = payload_viterbi_bits.last().copied().unwrap_or(false);
    println!("Avalanche viterbi MSB: {}", if viterbi_msb { 1 } else { 0 });

    Ok(())
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
/// - `bits_decrypt`: Optional expected bit width override.
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
    bits_decrypt: Option<u32>,
) -> Result<(Vec<f64>, Vec<bool>), Box<dyn Error>> {
    if per_bit_oracles.is_empty() {
        return Err("per-bit oracle selection is empty".into());
    }

    let bit_width = resolve_decrypt_bit_width(message, bits_decrypt)?;
    let base_ciphertext = message.modpow(&ctx.e, &ctx.n);
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
        let ciphertext = ciphertext_for_candidate(ctx, &base_ciphertext, candidate);
        let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
        let prepared_ciphertext =
            prepare_candidate_ciphertext(engine, &shifted, &candidate.r, &ctx.n);
        let dm = derive_candidate_message_from_result(
            ctx,
            engine,
            &prepared_ciphertext,
            &candidate.r,
            &candidate.d_new,
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
                let bit = if selection.invert {
                    !bits[bit_idx]
                } else {
                    bits[bit_idx]
                };
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
    let message_bits = biguint_to_bits_le(message, best_case_bits.len().max(1));
    print_colored_hex_comparison(
        "Original message",
        &message_bits,
        "Best-case message",
        best_case_bits,
    );
}

/// Prints two bit vectors as color-coded hex strings.
///
/// # Parameters
/// - `reference_label`: Label printed for the reference bit vector.
/// - `reference_bits`: Reference bit vector shown as the comparison target.
/// - `candidate_label`: Label printed for the candidate bit vector.
/// - `candidate_bits`: Candidate bit vector shown with match highlighting.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Prints two hex strings with color highlighting; no file output.
fn print_colored_hex_comparison(
    reference_label: &str,
    reference_bits: &[bool],
    candidate_label: &str,
    candidate_bits: &[bool],
) {
    let reference_hex = format_bits_hex_le(reference_bits);
    let candidate_hex = format_bits_hex_le(candidate_bits);
    let max_len = reference_hex.len().max(candidate_hex.len());
    let reference_padded = pad_left_hex(&reference_hex, max_len);
    let candidate_padded = pad_left_hex(&candidate_hex, max_len);
    let reference_output_label = format!("{} (hex)", reference_label);
    let candidate_output_label = format!("{} (hex)", candidate_label);
    let label_width = reference_output_label
        .len()
        .max(candidate_output_label.len());

    let reference_colored = colorize_hex_matches(&reference_padded, &candidate_padded);
    let candidate_colored = colorize_hex_matches(&candidate_padded, &reference_padded);

    println!(
        "{:<label_width$}: {}",
        reference_output_label, reference_colored
    );
    println!(
        "{:<label_width$}: {}",
        candidate_output_label, candidate_colored
    );
    println!("Hex match key: green = match, red = mismatch");
}

/// Computes bit-match percentages for a candidate bit vector against a reference.
///
/// # Parameters
/// - `candidate_bits`: Candidate bit vector to score.
/// - `message_bits`: Reference bit vector to compare against.
///
/// # Returns
/// - `(f64, f64)`: `(match_pct, ones_match_pct)` in percent.
///
/// # Expected Output
/// - Returns percentage scores; no side effects.
fn compute_bit_match_percentages(candidate_bits: &[bool], message_bits: &[bool]) -> (f64, f64) {
    let (_, matching_total) = count_matching_bits_le(candidate_bits, message_bits);
    let match_pct = matching_total as f64 / candidate_bits.len().max(1) as f64 * 100.0;
    let candidate_ones = candidate_bits.iter().filter(|bit| **bit).count();
    let matched_ones = candidate_bits
        .iter()
        .zip(message_bits.iter())
        .filter(|(cand, msg)| **cand && **msg)
        .count();
    let ones_match_pct = if candidate_ones == 0 {
        0.0
    } else {
        matched_ones as f64 / candidate_ones as f64 * 100.0
    };
    (match_pct, ones_match_pct)
}

/// Validates that reported Avalanche display data matches the underlying payload bits.
///
/// # Parameters
/// - `label`: Human-readable label describing the checked candidate.
/// - `message_bits`: Reference payload bits shown in the log output.
/// - `candidate_bits`: Candidate payload bits shown in the log output.
/// - `reported_match_pct`: Stored overall match percentage to validate.
/// - `reported_ones_match_pct`: Stored predicted-one match percentage to validate.
/// - `reported_hex`: Optional stored hex string to validate against the candidate bits.
///
/// # Returns
/// - `Result<(), String>`: `Ok(())` when the display fields are self-consistent.
///
/// # Expected Output
/// - Returns a descriptive error when the stored percentages or hex disagree with the displayed bits.
fn validate_displayed_candidate_consistency(
    label: &str,
    message_bits: &[bool],
    candidate_bits: &[bool],
    reported_match_pct: f64,
    reported_ones_match_pct: f64,
    reported_hex: Option<&str>,
) -> Result<(), String> {
    const PCT_TOLERANCE: f64 = 1e-9;

    let (computed_match_pct, computed_ones_match_pct) =
        compute_bit_match_percentages(candidate_bits, message_bits);
    if (computed_match_pct - reported_match_pct).abs() > PCT_TOLERANCE {
        return Err(format!(
            "{label} match percentage mismatch: stored={} computed={} candidate_bits={} message_bits={}",
            reported_match_pct,
            computed_match_pct,
            candidate_bits.len(),
            message_bits.len()
        ));
    }
    if (computed_ones_match_pct - reported_ones_match_pct).abs() > PCT_TOLERANCE {
        return Err(format!(
            "{label} ones-match percentage mismatch: stored={} computed={} candidate_bits={} message_bits={}",
            reported_ones_match_pct,
            computed_ones_match_pct,
            candidate_bits.len(),
            message_bits.len()
        ));
    }
    if let Some(reported_hex) = reported_hex {
        let computed_hex = format_bits_hex_le(candidate_bits);
        if computed_hex != reported_hex {
            return Err(format!(
                "{label} hex mismatch: stored={} computed={}",
                reported_hex, computed_hex
            ));
        }
    }

    Ok(())
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
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
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

    let images_dir = PathBuf::from("./images");
    fs::create_dir_all(&images_dir)?;

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
/// - `true_match`: Whether to report the true match percentage without inversion.
/// - `bits_decrypt`: Optional expected bit width override.
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
    true_match: bool,
    bits_decrypt: Option<u32>,
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

        let oracle_entropy_stats =
            compute_stats(&oracle_series.entropy_mean).ok_or("no oracle entropy samples")?;
        let oracle_accuracy_stats =
            compute_stats(&oracle_series.accuracy_pct).ok_or("no oracle accuracy samples")?;

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

    let match_entropy_stats =
        compute_stats(&match_series.entropy_mean).ok_or("no match entropy samples")?;
    let match_pct_stats =
        compute_stats(&match_series.match_pct_mean).ok_or("no match percentage samples")?;

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
            stats.mean, stats.stddev, stats.min, stats.max, stats.count, inverted_total
        );
    }

    let (per_bit_best_pct, best_case_bits) = compute_per_bit_best_case_match(
        ctx,
        engine,
        &candidates,
        &per_bit_oracles,
        message,
        shift,
        bits_decrypt,
    )?;
    if let Some(stats) = compute_stats(&per_bit_best_pct) {
        println!(
            "Per-bit best-case match % on original message: mean {:.2}, std dev {:.2}, min {:.2}, max {:.2}, n {}",
            stats.mean, stats.stddev, stats.min, stats.max, stats.count
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

    let batch_size = engine.analysis_batch_messages as usize;
    if batch_size == 0 {
        return Err("analysis_batch_messages must be >= 1 for speculative batch".into());
    }
    let selected_candidate = if engine.same_r_batch {
        Some((rng.next_u64() as usize) % candidates.len())
    } else {
        None
    };
    let speculative_report = run_bitwise_speculative_oracle_attempt(
        ctx,
        engine,
        &candidates,
        &per_bit_oracles,
        message,
        batch_size,
        shift,
        true_match,
        selected_candidate,
        bits_decrypt,
    )?;
    if !engine.analysis_batch_enable {
        run_avalanche_search(
            ctx,
            engine,
            &candidates,
            message,
            batch_size,
            shift,
            analytics,
        )?;
    } else {
        with_analytics(analytics, |a| {
            a.add_feature_note(
                "information_sufficiency",
                "standalone avalanche search skipped because sampled batch avalanche is enabled",
            );
        });
    }

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
    let match_pct_ok = (match_pct_stats.mean >= match_threshold)
        || (match_pct_stats.mean <= (100.0 - match_threshold));
    //let match_pct_inverted = match_pct_stats.mean < 50.0;

    let oracle_accuracy_ok = oracle_accuracy_stats
        .as_ref()
        .map(|stats| stats.mean >= oracle_accuracy_threshold)
        .unwrap_or(true);
    let speculative_match_ok = (speculative_report.match_pct >= match_threshold)
        || (speculative_report.match_pct <= (100.0 - match_threshold));

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
        Err(
            "analysis tests indicate insufficient information for speculative oracle attempts"
                .into(),
        )
    }
}

/// Selects the plaintext message according to CLI args and configuration.
///
/// # Parameters
/// - `args_message`: Optional CLI-provided message override.
/// - `engine`: Engine configuration with message settings.
/// - `rng`: Random number generator for random message selection.
///
/// # Returns
/// - `Result<BigUint, Box<dyn Error>>`: Selected message as a big integer.
///
/// # Expected Output
/// - Returns the selected message or a validation error; no side effects.
fn select_message(
    args_message: Option<String>,
    engine: &EngineConfig,
    n: &BigUint,
    rng: &mut RngChoice,
) -> Result<BigUint, Box<dyn Error>> {
    if let Some(explicit) = args_message {
        return Ok(BigUint::from_bytes_be(explicit.as_bytes()));
    }
    if engine.message.is_random {
        return random_message_under_n(engine, n, rng);
    }
    Ok(BigUint::from_bytes_be(
        engine.message.fixed_message.as_bytes(),
    ))
}

/// Samples a random message that is non-zero and less than `n` (when provided).
///
/// # Parameters
/// - `engine`: Engine configuration with message bit-length settings.
/// - `n`: Optional modulus bound; use zero to skip the bound.
/// - `rng`: Random number generator for sampling.
///
/// # Returns
/// - `Result<BigUint, Box<dyn Error>>`: Random message value.
///
/// # Expected Output
/// - Returns a non-zero exact-width value under `n` or a validation error when the widened message cannot fit.
fn random_message_under_n(
    engine: &EngineConfig,
    n: &BigUint,
    rng: &mut RngChoice,
) -> Result<BigUint, Box<dyn Error>> {
    validate_message_width_under_modulus(engine, n, "random message sampling")?;
    let transform = build_candidate_message_transform(engine);
    let target_bits = engine.message.bits.max(1);

    loop {
        let candidate = random_biguint_bits(target_bits, rng);
        if candidate.is_zero() {
            continue;
        }
        if n.is_zero() || transform(&candidate) < *n {
            return Ok(candidate);
        }
    }
}

/// Builds the extra random-message ciphertexts used for candidate fitness scoring.
///
/// # Parameters
/// - `ctx`: RSA context containing the modulus and public exponent.
/// - `engine`: Engine configuration controlling message width and fitness shifting.
/// - `count`: Number of additional random messages to generate.
/// - `rng`: Random number generator used to sample the additional payload messages.
///
/// # Returns
/// - `Result<Vec<AdditionalFitnessMessage>, Box<dyn Error>>`: Additional fitness messages in
///   generation order.
///
/// # Expected Output
/// - Returns the encrypted shifted messages used for padding-bit fitness checks; no stdout/stderr
///   output.
fn build_additional_fitness_messages(
    ctx: &RSAContext,
    engine: &EngineConfig,
    count: usize,
    rng: &mut RngChoice,
) -> Result<Vec<AdditionalFitnessMessage>, Box<dyn Error>> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let mut messages = Vec::with_capacity(count);
    for _ in 0..count {
        let payload_message = random_message_under_n(engine, &ctx.n, rng)?;
        let transformed_message = transform_message_for_candidate_scoring(
            engine,
            &payload_message,
            &ctx.n,
            "analysis_batch_fitness",
        )?;
        messages.push(AdditionalFitnessMessage {
            base_ciphertext: transformed_message.modpow(&ctx.e, &ctx.n),
        });
    }
    Ok(messages)
}

/// Computes the padding-bit fitness score for one scored Avalanche input across multiple messages.
///
/// # Parameters
/// - `ctx`: RSA context containing the source modulus and exponent.
/// - `engine`: Engine configuration controlling HBC behavior and fitness width.
/// - `candidate_r`: Candidate modulus associated with the scored input.
/// - `d_new`: Candidate private exponent corresponding to the current `r/x` pairing.
/// - `x_value`: Ciphertext exponent associated with the scored input.
/// - `current_message_bits`: Candidate decryption bits for the batch's main message.
/// - `additional_messages`: Extra random-message ciphertexts used for fitness scoring.
/// - `shift`: Whether ciphertexts should be shifted by encrypted `2` before candidate conversion.
///
/// # Returns
/// - `AvalancheFitnessScore`: Minimum and cumulative padding-bit fitness across all evaluated
///   messages.
///
/// # Expected Output
/// - Returns the candidate fitness metrics with no stdout/stderr output.
fn compute_padding_fitness_score(
    ctx: &RSAContext,
    engine: &EngineConfig,
    candidate_r: &BigUint,
    d_new: &BigUint,
    x_value: &BigUint,
    current_message_bits: &PackedBits,
    additional_messages: &[AdditionalFitnessMessage],
    shift: bool,
) -> AvalancheFitnessScore {
    let fitness_bit_width = resolve_avalanche_fitness_bit_width(engine);
    let avalanche_bit_width = current_message_bits.len();
    let current_score = lsb_zero_count_fitness(current_message_bits, fitness_bit_width);
    let mut minimum_score = current_score;
    let mut total_score = current_score;

    for additional_message in additional_messages {
        let ciphertext = if x_value.is_one() {
            additional_message.base_ciphertext.clone()
        } else {
            additional_message.base_ciphertext.modpow(x_value, &ctx.n)
        };
        let shifted_ciphertext = maybe_shift_ciphertext(ctx, &ciphertext, shift);
        let candidate_ciphertext = prepare_candidate_ciphertext(
            engine,
            &shifted_ciphertext,
            candidate_r,
            &ctx.n,
        );
        let candidate_message = derive_candidate_message_from_result(
            ctx,
            engine,
            &candidate_ciphertext,
            candidate_r,
            d_new,
        );
        let candidate_message_bits =
            biguint_to_packed_bits_le(&candidate_message, avalanche_bit_width);
        let score = lsb_zero_count_fitness(&candidate_message_bits, fitness_bit_width);
        minimum_score = minimum_score.min(score);
        total_score += score;
    }

    AvalancheFitnessScore {
        fitness_score: minimum_score,
        fitness_total_score: total_score,
        fitness_message_count: additional_messages.len() + 1,
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

/// Resolves the minimum candidate-modulus size required by the shifted Avalanche message width.
///
/// # Parameters
/// - `engine`: Engine configuration containing the message and fitness-shift settings.
///
/// # Returns
/// - `u64`: Minimum target bit length for generated r candidates.
///
/// # Expected Output
/// - Returns a deterministic lower bound; no stdout/stderr output.
fn minimum_r_candidate_bit_length(engine: &EngineConfig) -> u64 {
    let doubled_width = resolve_avalanche_bit_width(engine).saturating_mul(2);
    u64::try_from(doubled_width).unwrap_or(u64::MAX)
}

/// Builds `RCandidateSettings` from the engine configuration.
///
/// # Parameters
/// - `engine`: Engine configuration containing candidate fields.
/// - `configured_key_bit_width`: Bit width of the original RSA key used to key retargeted-cache files.
///
/// # Returns
/// - `RCandidateSettings`: Fully populated candidate settings.
///
/// # Expected Output
/// - Returns a settings struct; no side effects.
pub fn build_r_candidate_settings(
    engine: &EngineConfig,
    configured_key_bit_width: u64,
) -> RCandidateSettings {
    let override_best_r = engine.override_best_r.as_ref().and_then(|raw| {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            trimmed.parse::<BigUint>().ok()
        }
    });
    let minimum_target_bit_length = minimum_r_candidate_bit_length(engine);
    let target_bit_length = Some(
        engine
            .r_candidate_bit_length
            .unwrap_or(minimum_target_bit_length)
            .max(minimum_target_bit_length),
    );

    RCandidateSettings {
        mode: engine.r_candidate_mode,
        override_best_r,
        process_min_factor: BigUint::from(engine.process_min_factor),
        process_count: engine.process_count,
        process_min_count: engine.process_min_count,
        process_scale: engine.process_scale,
        reuse_retargeted_r_candidates: engine.reuse_retargeted_r_candidates,
        reuse_retargeted_r_candidates_path: resolve_retargeted_r_candidates_path(
            &engine.reuse_retargeted_r_candidates_path_prefix,
            configured_key_bit_width,
        ),
        small_primes: engine
            .r_candidate_small_primes
            .iter()
            .map(|p| BigUint::from(*p))
            .collect(),
        small_prime_factors_per_candidate: engine.r_candidate_small_prime_factors,
        max_factors_per_candidate: engine.r_candidate_max_factors,
        target_bit_length,
        random_power_window: engine.r_candidate_random_power_window,
        target_exponent_minimum: engine.r_candidate_target_exponent_minimum.clone(),
        target_exponent: engine.r_candidate_target_exponent.clone(),
        retarget_partition_count: engine.r_candidate_retarget_partition_count,
        retarget_minimum_exponent: engine.r_candidate_retarget_minimum_exponent.clone(),
    }
}

#[derive(Clone, Debug)]
struct RSAContext {
    n: BigUint,
    e: BigUint,
    key_bit_width: u64,
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
/// - Returns counts based on little-endian byte comparisons; no side effects.
fn count_matching_bits(a: &BigUint, b: &BigUint) -> (usize, usize) {
    let a_bit_len = a.bits().max(1) as usize;
    let b_bit_len = b.bits().max(1) as usize;
    let min_len = a_bit_len.min(b_bit_len);
    let a_bytes = a.to_bytes_le();
    let b_bytes = b.to_bytes_le();
    matching_bit_counts_bytes_le(&a_bytes, &b_bytes, min_len)
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
    (x % p) % r
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

/// Prepares a candidate-modulus ciphertext by applying HBC from the source modulus.
///
/// # Parameters
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `ciphertext`: Ciphertext to convert into the candidate modulus.
/// - `target_modulus`: Candidate modulus that receives the converted ciphertext.
/// - `source_modulus`: Source modulus currently associated with `ciphertext`.
///
/// # Returns
/// - `BigUint`: HBC-converted ciphertext reduced modulo `target_modulus`.
///
/// # Expected Output
/// - Returns the prepared ciphertext; no side effects.
fn prepare_candidate_ciphertext(
    engine: &EngineConfig,
    ciphertext: &BigUint,
    target_modulus: &BigUint,
    source_modulus: &BigUint,
) -> BigUint {
    hbc(ciphertext, target_modulus, source_modulus, engine)
}

/// Derives the candidate message for a given ciphertext and r candidate.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `ciphertext`: Ciphertext to transform through the HBC flow.
/// - `r`: Candidate modulus for alternate decryption.
/// - `d_new`: Private exponent corresponding to `r`.
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
    shift: bool,
) -> BigUint {
    let shifted = maybe_shift_ciphertext(ctx, ciphertext, shift);
    let prepared_ciphertext = prepare_candidate_ciphertext(engine, &shifted, r, &ctx.n);
    derive_candidate_message_from_result(ctx, engine, &prepared_ciphertext, r, d_new)
}

/// Derives the candidate message given a candidate-modulus ciphertext.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `candidate_ciphertext_r`: Ciphertext prepared in the candidate modulus via HBC.
/// - `r`: Candidate modulus for alternate decryption.
/// - `d_new`: Private exponent corresponding to `r`.
///
/// # Returns
/// - `BigUint`: Derived candidate message modulo `n`.
///
/// # Expected Output
/// - Returns the derived message; no side effects.
fn derive_candidate_message_from_result(
    ctx: &RSAContext,
    engine: &EngineConfig,
    candidate_ciphertext_r: &BigUint,
    r: &BigUint,
    d_new: &BigUint,
) -> BigUint {
    let recovered_new = if engine.use_rs_decrypt {
        candidate_ciphertext_r.modpow(d_new, r)
    } else {
        candidate_ciphertext_r.clone()
    };

    let hbc_default = hbc(&recovered_new, &ctx.n, r, engine);
    let dm_raw = &hbc_default % &ctx.n;

    let width = dm_raw.bits().max(1);
    let mask = (BigUint::one() << width) - BigUint::one();
    let inverted_dm = &mask ^ &dm_raw;
    if engine.invert_bits {
        inverted_dm
    } else {
        dm_raw
    }
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
    let base_ciphertext = message.modpow(&ctx.e, &ctx.n);
    let shifted_base_ciphertext = maybe_shift_ciphertext(ctx, &base_ciphertext, shift);

    let mut entries = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let ciphertext = ciphertext_for_candidate(ctx, &base_ciphertext, candidate);
        let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
        let hbc_result = prepare_candidate_ciphertext(engine, &shifted, &candidate.r, &ctx.n);
        let dm = derive_candidate_message_from_result(
            ctx,
            engine,
            &hbc_result,
            &candidate.r,
            &candidate.d_new,
        );
        entries.push(RCandidateTraceEntry {
            r: candidate.r.clone(),
            r_bits: candidate.r.bits(),
            target_exponent: candidate.target_exponent.normalized(),
            hbc_ciphertext_r: hbc_result,
            candidate_decryption: dm,
        });
    }

    if entries.is_empty() {
        return;
    }

    with_analytics(analytics, |a| {
        a.push_r_candidate_trace_batch(RCandidateTraceBatch {
            context: context.to_string(),
            message: message.clone(),
            ciphertext: base_ciphertext.clone(),
            shifted_ciphertext: shifted_base_ciphertext.clone(),
            rabin_exponent: y,
            tonelli_shanks_modulus: ctx.n.clone(),
            tonelli_shanks_ciphertext: shifted_base_ciphertext,
            candidate_count: entries.len(),
            candidates: entries,
        });
    });
}

#[derive(Debug, Clone)]
struct AccuracyCandidate {
    r: BigUint,
    phi_new: BigUint,
    d_new: BigUint,
    target_exponent: BigDecimal,
}

#[derive(Debug, Clone)]
struct AdditionalFitnessMessage {
    base_ciphertext: BigUint,
}

#[derive(Clone, Debug)]
struct BeamMaxCandidate {
    beam_match_pct: f64,
    average_score_pct: f64,
    top_beam_score: f64,
    beam_results: Vec<AvalancheCombinationBeamResult>,
    center_biases: Vec<AvalancheCenterBiasEntry>,
    best_bits: Vec<bool>,
    message_bits: Vec<bool>,
    batch_number: usize,
    sample_index: usize,
    tier_index: usize,
}

#[derive(Clone, Debug)]
struct MajorityVoteMaxCandidate {
    average_score_pct: f64,
    majority_vote_bits: Vec<bool>,
    majority_vote_match_pct: f64,
    majority_vote_ones_match_pct: f64,
    message_bits: Vec<bool>,
    batch_number: usize,
    sample_index: usize,
    tier_index: usize,
}

/// Determines whether a beam-search candidate should replace the current run maximum.
///
/// # Parameters
/// - `current`: Current run maximum candidate, if any.
/// - `top_beam`: Top-ranked beam result from the candidate sample.
/// - `sample`: Selected sample that produced `top_beam`.
/// - `prefer_beam_score_ordering`: Whether beam score should outrank known match percentage.
///
/// # Returns
/// - `bool`: `true` when the candidate should become the new run maximum.
///
/// # Expected Output
/// - Returns the selection decision; no stdout/stderr output.
fn should_replace_beam_max_candidate(
    current: Option<&BeamMaxCandidate>,
    top_beam: &AvalancheCombinationBeamResult,
    sample: &SelectedAvalancheSample,
    prefer_beam_score_ordering: bool,
) -> bool {
    match current {
        Some(current) if prefer_beam_score_ordering => {
            sample.top_beam_score > current.top_beam_score
                || (sample.top_beam_score == current.top_beam_score
                    && sample.average_score_pct > current.average_score_pct)
                || (sample.top_beam_score == current.top_beam_score
                    && sample.average_score_pct == current.average_score_pct
                    && top_beam.match_pct > current.beam_match_pct)
        }
        Some(current) => {
            top_beam.match_pct > current.beam_match_pct
                || (top_beam.match_pct == current.beam_match_pct
                    && sample.top_beam_score > current.top_beam_score)
                || (top_beam.match_pct == current.beam_match_pct
                    && sample.top_beam_score == current.top_beam_score
                    && sample.average_score_pct > current.average_score_pct)
        }
        None => true,
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ScoredAvalancheInputDetail {
    pub(crate) target_exponent: BigDecimal,
    pub(crate) hbc_ciphertext_r: BigUint,
    pub(crate) candidate_decryption: BigUint,
}

#[derive(Clone, Debug)]
pub(crate) struct ScoredAvalancheInput {
    pub(crate) batch_candidate_index: usize,
    pub(crate) message_index: usize,
    pub(crate) r: BigUint,
    pub(crate) x: BigUint,
    pub(crate) score_match_pct: f64,
    pub(crate) message_bits: PackedBits,
    pub(crate) detail: Option<ScoredAvalancheInputDetail>,
}

impl AvalancheInput for ScoredAvalancheInput {
    fn avalanche_node(&self) -> Result<AvalancheNode, crate::avalanche::AvalancheError> {
        Ok(AvalancheNode::from_packed_bits(
            self.message_bits.clone(),
            vec![0.0; self.message_bits.len()],
        ))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ScoredAvalancheInputGroup {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) batch_candidate_index: usize,
    pub(crate) inputs: Vec<ScoredAvalancheInput>,
}

#[derive(Clone, Debug)]
pub(crate) struct SelectedAvalancheSample {
    pub(crate) sample_index: usize,
    pub(crate) tier_index: usize,
    pub(crate) input_count: usize,
    pub(crate) average_score_pct: f64,
    pub(crate) beam_results: Vec<AvalancheCombinationBeamResult>,
    pub(crate) majority_vote_bits: Vec<bool>,
    pub(crate) majority_vote_match_pct: f64,
    pub(crate) majority_vote_ones_match_pct: f64,
    pub(crate) best_bits: Vec<bool>,
    pub(crate) top_beam_score: f64,
    pub(crate) top_beam_match_pct: Option<f64>,
    pub(crate) best_match_pct: f64,
    pub(crate) center_biases: Vec<AvalancheCenterBiasEntry>,
    pub(crate) node: AvalancheNode,
}

#[derive(Clone, Debug)]
struct RecursiveAvalancheSourceSample {
    best_match_pct: f64,
    message_bits: PackedBits,
}

#[derive(Clone, Debug)]
struct RecursiveAvalancheInput {
    message_bits: PackedBits,
}

impl AvalancheInput for RecursiveAvalancheInput {
    fn avalanche_node(&self) -> Result<AvalancheNode, crate::avalanche::AvalancheError> {
        Ok(AvalancheNode::from_packed_bits(
            self.message_bits.clone(),
            vec![0.0; self.message_bits.len()],
        ))
    }
}

/// Selects which finalized sample bits should feed the next recursive Avalanche tier.
///
/// # Parameters
/// - `sample`: Prior-tier selected sample being forwarded into the next recursive tier.
/// - `engine`: Engine configuration controlling whether recursion uses top-beam or majority-vote outputs.
///
/// # Returns
/// - `&[bool]`: Borrowed bit slice forwarded into the next recursive tier.
///
/// # Expected Output
/// - Returns a borrowed bit slice; no stdout/stderr output.
pub(crate) fn recursive_tier_bits<'a>(
    sample: &'a SelectedAvalancheSample,
    engine: &EngineConfig,
) -> &'a [bool] {
    if engine.avalanche_use_top_beam {
        &sample.best_bits
    } else {
        &sample.majority_vote_bits
    }
}

/// Compacts prior-tier selected samples into the minimal recursive source payload.
///
/// # Parameters
/// - `samples`: Prior-tier selected samples whose beam metadata and finalized nodes can be discarded.
/// - `engine`: Engine configuration controlling whether recursive tiers use top-beam or majority-vote bits.
///
/// # Returns
/// - `Vec<RecursiveAvalancheSourceSample>`: Recursive-tier source records with only match score and packed message bits.
///
/// # Expected Output
/// - Returns compact recursive-tier source samples; no stdout/stderr output.
fn compact_recursive_avalanche_source_samples(
    samples: Vec<SelectedAvalancheSample>,
    engine: &EngineConfig,
) -> Vec<RecursiveAvalancheSourceSample> {
    samples
        .into_iter()
        .map(|sample| RecursiveAvalancheSourceSample {
            best_match_pct: sample.best_match_pct,
            message_bits: PackedBits::from_bools(recursive_tier_bits(&sample, engine)),
        })
        .collect()
}

/// Builds the recursive Avalanche inputs for a next-tier run from prior-tier selected samples.
///
/// # Parameters
/// - `samples`: Compact prior-tier source samples chosen for one recursive Avalanche execution.
///
/// # Returns
/// - `Vec<RecursiveAvalancheInput>`: Recursive-tier inputs containing only the selected bit vectors.
///
/// # Expected Output
/// - Returns compact recursive-tier inputs; no stdout/stderr output.
fn build_recursive_avalanche_inputs(
    samples: &[&RecursiveAvalancheSourceSample],
) -> Vec<RecursiveAvalancheInput> {
    samples
        .iter()
        .map(|sample| RecursiveAvalancheInput {
            message_bits: sample.message_bits.clone(),
        })
        .collect()
}

#[derive(Debug)]
struct SampledAvalancheSampleOutcome {
    retained_sample: Option<AvalancheCombinationSample>,
    sample: Option<SelectedAvalancheSample>,
    evaluated_candidates: usize,
    produced_sample: bool,
}

#[derive(Debug)]
struct SampledAvalancheBatchResult {
    selected_sample: Option<SelectedAvalancheSample>,
    final_tier_samples: Vec<SelectedAvalancheSample>,
    retained_samples: Vec<AvalancheCombinationSample>,
    tier_statistics: Vec<AvalancheTierStatistics>,
    sample_count: usize,
    evaluated_candidates: usize,
}

impl Default for SampledAvalancheBatchResult {
    fn default() -> Self {
        Self {
            selected_sample: None,
            final_tier_samples: Vec::new(),
            retained_samples: Vec::new(),
            tier_statistics: Vec::new(),
            sample_count: 0,
            evaluated_candidates: 0,
        }
    }
}

/// Determines whether a recursive Avalanche sample should replace the current selected sample.
///
/// # Parameters
/// - `current`: Current selected sample, if any.
/// - `candidate`: Candidate sample being considered for selection.
/// - `prefer_beam_score_ordering`: Whether selection should prioritize beam score over match percentage.
///
/// # Returns
/// - `bool`: `true` when the candidate should become the new selected sample.
///
/// # Expected Output
/// - Returns the selection decision; no stdout/stderr output.
fn should_replace_selected_sample(
    current: Option<&SelectedAvalancheSample>,
    candidate: &SelectedAvalancheSample,
    prefer_beam_score_ordering: bool,
) -> bool {
    match current {
        Some(current) if prefer_beam_score_ordering => {
            candidate.top_beam_score > current.top_beam_score
                || (candidate.top_beam_score == current.top_beam_score
                    && candidate.average_score_pct > current.average_score_pct)
                || (candidate.top_beam_score == current.top_beam_score
                    && candidate.average_score_pct == current.average_score_pct
                    && candidate.best_match_pct > current.best_match_pct)
        }
        Some(current) => {
            candidate.best_match_pct > current.best_match_pct
                || (candidate.best_match_pct == current.best_match_pct
                    && candidate.top_beam_score > current.top_beam_score)
                || (candidate.best_match_pct == current.best_match_pct
                    && candidate.top_beam_score == current.top_beam_score
                    && candidate.average_score_pct > current.average_score_pct)
        }
        None => true,
    }
}

impl SampledAvalancheBatchResult {
    fn update_selected_sample_ref(
        &mut self,
        candidate: &SelectedAvalancheSample,
        prefer_beam_score_ordering: bool,
    ) {
        if should_replace_selected_sample(
            self.selected_sample.as_ref(),
            candidate,
            prefer_beam_score_ordering,
        ) {
            self.selected_sample = Some(candidate.clone());
        }
    }
}

#[derive(Debug)]
struct BatchCxMax {
    match_pct: f64,
    x: BigUint,
    r: BigUint,
    batch_candidate_index: usize,
}

#[derive(Debug, Default)]
struct AccuracyBatchAccumulator {
    candidate_count: usize,
    cx_max: Option<BatchCxMax>,
    cx_evaluated_candidates: usize,
    ranked_samples: Vec<RankedScoredAvalancheInput>,
}

impl AccuracyBatchAccumulator {
    fn set_cx_max(&mut self, candidate: BatchCxMax) {
        let replace = match self.cx_max.as_ref() {
            Some(current) => {
                candidate.match_pct > current.match_pct
                    || (candidate.match_pct == current.match_pct
                        && candidate.batch_candidate_index < current.batch_candidate_index)
            }
            None => true,
        };
        if replace {
            self.cx_max = Some(candidate);
        }
    }

    fn merge(mut self, mut other: Self) -> Self {
        self.candidate_count += other.candidate_count;
        self.cx_evaluated_candidates += other.cx_evaluated_candidates;
        if let Some(candidate) = other.cx_max.take() {
            self.set_cx_max(candidate);
        }
        self.ranked_samples.append(&mut other.ranked_samples);
        self
    }
}

#[derive(Clone, Debug)]
struct CxMatchCandidate {
    match_pct: f64,
    x: BigUint,
    r: BigUint,
    batch_number: usize,
}

/// Formats little-endian bits as a zero-padded hexadecimal string.
///
/// # Parameters
/// - `bits`: Little-endian bit vector to format.
///
/// # Returns
/// - `String`: Zero-padded hexadecimal representation.
///
/// # Expected Output
/// - Returns the formatted hex string; no stdout/stderr output.
fn format_bits_hex_le(bits: &[bool]) -> String {
    let mut hex = to_hex(&bits_le_to_biguint(bits));
    let hex_len = bits.len().div_ceil(4);
    if hex.len() < hex_len {
        let padding = "0".repeat(hex_len - hex.len());
        hex = format!("{}{}", padding, hex);
    }
    hex
}

/// Selects `sample_size` unique indices from `0..pool_size` without replacement.
///
/// # Parameters
/// - `pool_size`: Number of candidate indices available.
/// - `sample_size`: Number of indices to return.
/// - `rng`: Random number generator used for the partial shuffle state.
///
/// # Returns
/// - `Vec<usize>`: Unique sampled indices in shuffled order.
///
/// # Expected Output
/// - Returns sampled indices without allocating a full `0..pool_size` index vector; no stdout/stderr output.
pub(crate) fn sample_unique_indices(
    pool_size: usize,
    sample_size: usize,
    rng: &mut RngChoice,
) -> Vec<usize> {
    let sample_len = sample_size.min(pool_size);
    let mut swaps = HashMap::with_capacity(sample_len.saturating_mul(2));
    let mut sampled = Vec::with_capacity(sample_len);
    for offset in 0..sample_len {
        let remaining = pool_size - offset;
        let swap_offset = (rng.next_u64() as usize) % remaining;
        let swap_index = offset + swap_offset;
        let chosen_index = *swaps.get(&swap_index).unwrap_or(&swap_index);
        let offset_index = *swaps.get(&offset).unwrap_or(&offset);
        sampled.push(chosen_index);
        if swap_index != offset {
            swaps.insert(swap_index, offset_index);
        }
        swaps.remove(&offset);
    }
    sampled
}

/// Selects raw scored avalanche inputs directly from the flattened pool.
///
/// # Parameters
/// - `inputs`: Flattened scored avalanche inputs available for sampling.
/// - `sample_size`: Maximum number of inputs to keep.
/// - `rng`: Random number generator used for index sampling.
///
/// # Returns
/// - `Vec<ScoredAvalancheInput>`: Randomly selected scored inputs without replacement.
///
/// # Expected Output
/// - Returns up to `sample_size` unique scored inputs; no stdout/stderr output.
#[cfg_attr(not(test), allow(dead_code))]
fn select_random_scored_inputs(
    inputs: &[ScoredAvalancheInput],
    sample_size: usize,
    rng: &mut RngChoice,
) -> Vec<ScoredAvalancheInput> {
    if sample_size == 0 || inputs.is_empty() {
        return Vec::new();
    }

    sample_unique_indices(inputs.len(), sample_size, rng)
        .into_iter()
        .filter_map(|index| inputs.get(index).cloned())
        .collect()
}

/// Builds distinct sample plans from a flat item list.
///
/// # Parameters
/// - `items`: Flat source list whose entries may be reused across returned plans.
/// - `group_size`: Maximum number of entries assigned to each plan.
/// - `target_group_count`: Maximum number of plans to produce.
/// - `rng`: Random number generator used to sample candidate plans.
///
/// # Returns
/// - `Vec<Vec<T>>`: Distinct plans whose per-plan item sets are unique.
///
/// # Expected Output
/// - Returns plans that never repeat an item within one plan and never duplicate the same item set
///   across plans; source items may still appear in multiple different plans.
fn build_unique_flat_sample_plans<T: Clone>(
    items: &[T],
    group_size: usize,
    target_group_count: usize,
    rng: &mut RngChoice,
) -> Vec<Vec<T>> {
    if group_size == 0 || target_group_count == 0 || items.is_empty() {
        return Vec::new();
    }

    let sample_size = group_size.min(items.len());
    let mut sample_plans = Vec::with_capacity(target_group_count);
    let mut seen_signatures = HashSet::with_capacity(target_group_count);
    let max_attempts = target_group_count
        .saturating_mul(items.len().max(1))
        .saturating_mul(8)
        .max(64);
    let mut attempts = 0usize;

    while sample_plans.len() < target_group_count && attempts < max_attempts {
        attempts += 1;
        let sampled_indices = sample_unique_indices(items.len(), sample_size, rng);
        if sampled_indices.is_empty() {
            break;
        }

        let mut signature = sampled_indices.clone();
        signature.sort_unstable();
        if !seen_signatures.insert(signature) {
            continue;
        }

        let plan = sampled_indices
            .into_iter()
            .filter_map(|index| items.get(index).cloned())
            .collect::<Vec<_>>();
        if !plan.is_empty() {
            sample_plans.push(plan);
        }
    }

    sample_plans
}

/// Builds distinct sample plans while capping the number of distinct source groups per plan.
///
/// # Parameters
/// - `grouped_items`: Source items already grouped by originating `r` candidate.
/// - `mixed_group_count`: Maximum number of distinct source groups allowed in any one plan.
/// - `combination_size`: Maximum number of items assigned to each plan.
/// - `target_group_count`: Maximum number of plans to produce.
/// - `rng`: Random number generator used to sample candidate plans.
///
/// # Returns
/// - `Vec<Vec<T>>`: Distinct plans that respect the per-plan source-group cap.
///
/// # Expected Output
/// - Returns plans that never reuse an item within one plan, never duplicate the same item set
///   across plans, and never exceed the per-plan group cap; source items may still appear in
///   multiple different plans.
fn build_unique_grouped_sample_plans<T: Clone>(
    grouped_items: &[Vec<T>],
    mixed_group_count: usize,
    combination_size: usize,
    target_group_count: usize,
    rng: &mut RngChoice,
) -> Vec<Vec<T>> {
    if combination_size == 0
        || target_group_count == 0
        || grouped_items.is_empty()
        || mixed_group_count == 0
    {
        return Vec::new();
    }

    let distinct_group_limit = mixed_group_count.min(combination_size);
    let mut sample_plans = Vec::new();
    let mut seen_signatures = HashSet::with_capacity(target_group_count);
    let max_attempts = target_group_count
        .saturating_mul(grouped_items.len().max(1))
        .saturating_mul(combination_size.max(1))
        .saturating_mul(8)
        .max(64);
    let mut attempts = 0usize;

    while sample_plans.len() < target_group_count && attempts < max_attempts {
        attempts += 1;
        let sampled_group_indices =
            sample_unique_indices(grouped_items.len(), distinct_group_limit, rng);
        if sampled_group_indices.is_empty() {
            break;
        }

        let sampled_groups = sampled_group_indices
            .into_iter()
            .filter_map(|group_idx| {
                grouped_items
                    .get(group_idx)
                    .filter(|group| !group.is_empty())
                    .map(|group| (group_idx, group))
            })
            .collect::<Vec<_>>();
        if sampled_groups.is_empty() {
            continue;
        }

        let available_input_count = sampled_groups
            .iter()
            .map(|(_, group)| group.len())
            .sum::<usize>();
        if available_input_count == 0 {
            continue;
        }

        let mut selected_item_indices =
            Vec::with_capacity(available_input_count.min(combination_size));
        if available_input_count <= combination_size {
            for (group_idx, group) in &sampled_groups {
                selected_item_indices
                    .extend((0..group.len()).map(|item_idx| (*group_idx, item_idx)));
            }
        } else {
            let required_group_slots = sampled_groups.len().min(combination_size);
            let mut leftover_item_indices =
                Vec::with_capacity(available_input_count - required_group_slots);

            for (group_order, (group_idx, group)) in sampled_groups.iter().enumerate() {
                let pick_indices = sample_unique_indices(group.len(), 1, rng);
                if group_order < required_group_slots {
                    if let Some(&picked_index) = pick_indices.first() {
                        selected_item_indices.push((*group_idx, picked_index));
                        for item_idx in 0..group.len() {
                            if item_idx != picked_index {
                                leftover_item_indices.push((*group_idx, item_idx));
                            }
                        }
                        continue;
                    }
                }

                leftover_item_indices
                    .extend((0..group.len()).map(|item_idx| (*group_idx, item_idx)));
            }

            let remaining_slots = combination_size.saturating_sub(selected_item_indices.len());
            let leftover_indices =
                sample_unique_indices(leftover_item_indices.len(), remaining_slots, rng);
            for leftover_idx in leftover_indices {
                if let Some(&(group_idx, item_idx)) = leftover_item_indices.get(leftover_idx) {
                    selected_item_indices.push((group_idx, item_idx));
                }
            }
        }

        if selected_item_indices.is_empty() {
            continue;
        }

        let mut signature = selected_item_indices.clone();
        signature.sort_unstable();
        if !seen_signatures.insert(signature) {
            continue;
        }

        let sample_plan = selected_item_indices
            .into_iter()
            .filter_map(|(group_idx, item_idx)| {
                grouped_items
                    .get(group_idx)
                    .and_then(|group| group.get(item_idx))
                    .cloned()
            })
            .collect::<Vec<_>>();
        if !sample_plan.is_empty() {
            sample_plans.push(sample_plan);
        }
    }

    sample_plans
}

#[derive(Debug, Clone, Copy)]
struct RecursiveAvalancheTierConfig {
    group_size: usize,
    resample_count: usize,
}

/// Resolves the effective config for one recursive Avalanche tier.
///
/// # Parameters
/// - `engine`: Engine configuration containing the per-tier recursive Avalanche arrays.
/// - `recursive_level`: One-based recursive level where `1` is the first tier after the sampled-input tier.
///
/// # Returns
/// - `RecursiveAvalancheTierConfig`: Effective group-size and resample-count values for that recursive level.
///
/// # Expected Output
/// - Returns the resolved tier config, reusing the last configured array entry when recursion exceeds the configured depth.
fn resolve_recursive_avalanche_tier_config(
    engine: &EngineConfig,
    recursive_level: usize,
) -> RecursiveAvalancheTierConfig {
    let tier_index = recursive_level.saturating_sub(1);
    let group_size = engine
        .avalanche_combination_recursive_group_size
        .get(tier_index)
        .copied()
        .or_else(|| {
            engine
                .avalanche_combination_recursive_group_size
                .last()
                .copied()
        })
        .unwrap_or(8)
        .max(1);
    let resample_count = engine
        .avalanche_combination_recursive_resample_count
        .get(tier_index)
        .copied()
        .or_else(|| {
            engine
                .avalanche_combination_recursive_resample_count
                .last()
                .copied()
        })
        .unwrap_or(0);

    RecursiveAvalancheTierConfig {
        group_size,
        resample_count,
    }
}

#[derive(Debug)]
struct ComputedAvalancheSample {
    sample: SelectedAvalancheSample,
    majority_vote_ones_count: Vec<usize>,
    majority_vote_zeros_count: Vec<usize>,
    majority_vote_probability_one: Vec<f64>,
    normalized_bias_probabilities: Vec<f64>,
    beam_search_probabilities: Vec<f64>,
    level_similarity_pct: Vec<f64>,
    level_pair_counts: Vec<usize>,
}

/// Sorts Avalanche nodes by their numeric little-endian bit value.
///
/// # Parameters
/// - `nodes`: Avalanche nodes to order.
///
/// # Returns
/// - `Vec<AvalancheNode>`: Nodes ordered by their represented integer value.
///
/// # Expected Output
/// - Returns a reordered vector; no stdout/stderr output.
fn sort_avalanche_nodes_by_value(mut nodes: Vec<AvalancheNode>) -> Vec<AvalancheNode> {
    let mut nodes_with_value: Vec<(BigUint, AvalancheNode)> = nodes
        .drain(..)
        .map(|node| (BigUint::from_bytes_le(node.packed_message_bits()), node))
        .collect();
    nodes_with_value.sort_by(|a, b| a.0.cmp(&b.0));
    nodes_with_value.into_iter().map(|(_, node)| node).collect()
}

/// Prepares an Avalanche object from generic inputs using the shared Avalanche builder.
///
/// # Parameters
/// - `inputs`: Inputs convertible into Avalanche nodes.
/// - `engine`: Engine configuration controlling candidate preprocessing.
/// - `message_bits`: Reference message bits used for optional Hamming-distance ordering.
/// - `collect_scores`: Whether per-level similarity scores should be recorded.
/// - `progress_label`: Optional progress label used during execution.
///
/// # Returns
/// - `Result<crate::avalanche::Avalanche, Box<dyn Error>>`: Prepared Avalanche reducer.
///
/// # Expected Output
/// - Returns a prepared Avalanche reducer; no stdout/stderr output.
fn build_prepared_avalanche<T: AvalancheInput>(
    inputs: &[T],
    engine: &EngineConfig,
    message_bits: &[bool],
    collect_scores: bool,
    progress_label: Option<&str>,
) -> Result<crate::avalanche::Avalanche, Box<dyn Error>> {
    let mut nodes = inputs
        .iter()
        .map(|input| input.avalanche_node())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
    if !engine.use_hamming_distance {
        nodes = sort_avalanche_nodes_by_value(nodes);
    }

    let mut builder = AvalancheBuilder::new()
        .candidates(nodes)
        .map_err(|err| -> Box<dyn Error> { Box::new(err) })?
        .mirror_invert_candidates(engine.mirror_invert_candidates)
        .collect_scores(collect_scores);
    if engine.use_hamming_distance {
        builder = builder.reference_bits(Some(message_bits.to_vec()));
    }
    if let Some(label) = progress_label {
        builder = builder.progress_label(Some(label.to_string()));
    }
    builder
        .build()
        .map_err(|err| -> Box<dyn Error> { Box::new(err) })
}

/// Builds avalanche nodes from scored candidate decryptions.
///
/// # Parameters
/// - `inputs`: Scored candidate decryptions selected for the sample.
/// - `engine`: Engine configuration controlling optional Hamming-distance ordering.
/// - `message_bits`: Reference message bits used for Hamming-distance ordering.
///
/// # Returns
/// - `Result<Vec<AvalancheNode>, Box<dyn Error>>`: Ordered avalanche nodes for the sample.
///
/// # Expected Output
/// - Returns avalanche nodes; no stdout/stderr output.
#[cfg(test)]
fn build_avalanche_nodes_from_scored_inputs(
    inputs: &[ScoredAvalancheInput],
    engine: &EngineConfig,
    message_bits: &[bool],
) -> Result<Vec<AvalancheNode>, Box<dyn Error>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    Ok(
        build_prepared_avalanche(inputs, engine, message_bits, false, None)?
            .candidates()
            .to_vec(),
    )
}

/// Finalizes beam-search and majority-vote outputs for a prepared Avalanche search.
///
/// # Parameters
/// - `engine`: Engine configuration controlling beam-search behavior.
/// - `comparison_message_bits`: Original plaintext payload bits used for scoring.
/// - `sample_index`: Zero-based sample index within the current tier.
/// - `tier_index`: One-based Avalanche tier index.
/// - `input_count`: Number of source items that produced the sample.
/// - `average_score_pct`: Mean source-score percentage for the sample.
/// - `avalanche_search`: Executed Avalanche result for the sample.
/// - `selected_oracles`: Bit vectors contributing to the sample majority vote.
///
/// # Returns
/// - `Result<ComputedAvalancheSample, String>`: Finalized sample plus intermediate analytics payloads.
///
/// # Expected Output
/// - Returns the finalized sample data; no stdout/stderr output.
fn finalize_avalanche_sample(
    engine: &EngineConfig,
    comparison_message_bits: &[bool],
    sample_index: usize,
    tier_index: usize,
    input_count: usize,
    average_score_pct: f64,
    avalanche_search: crate::avalanche::AvalancheSearchResult,
    selected_oracles: Vec<PackedBits>,
) -> Result<ComputedAvalancheSample, String> {
    let majority_distribution = crate::combiner::majority_vote_with_distribution_packed(
        &selected_oracles,
        engine.combiner_tie_breaker,
    )
    .map_err(|err| err.to_string())?;
    let majority_probabilities = if engine.avalanche_combination_sample_smoothing {
        smooth_probability_one_jeffreys(
            &majority_distribution.ones_count,
            majority_distribution.total_oracles,
        )
    } else {
        majority_distribution.probability_one.clone()
    };
    let normalized_biases = if engine.avalanche_combination_majority_vote {
        majority_probabilities.clone()
    } else {
        normalize_avalanche_biases(&avalanche_search.node.biases)
    };
    let beam_probabilities = spread_normalized_avalanche_biases(
        &normalized_biases,
        engine.avalanche_probability_spread_exponent,
    );
    let beam_bit_one_threshold = engine.beam_bit_one_threshold;
    let bit_width = avalanche_search.node.bit_len().max(1);
    let beam_result = beam_search_top_k(
        vec![Vec::new()],
        engine.avalanche_beam_top_k.max(1),
        bit_width,
        |candidate| {
            if candidate.len() >= bit_width {
                return Vec::new();
            }
            let mut zero = candidate.to_vec();
            let mut one = candidate.to_vec();
            zero.push(0.0);
            one.push(1.0);
            vec![zero, one]
        },
        |candidate| {
            candidate
                .iter()
                .enumerate()
                .map(|(idx, bit)| {
                    let bias = beam_probabilities.get(idx).copied().unwrap_or(0.0);
                    if stored_beam_value_is_one(*bit, beam_bit_one_threshold) {
                        bias
                    } else {
                        1.0 - bias
                    }
                })
                .sum()
        },
    )
    .map_err(|err| err.to_string())?;

    let mut best_bits = Vec::new();
    let beam_results = beam_result
        .beam
        .iter()
        .enumerate()
        .map(|(rank, candidate)| {
            let candidate_bits: Vec<bool> = candidate
                .vector
                .iter()
                .map(|value| stored_beam_value_is_one(*value, beam_bit_one_threshold))
                .collect();
            if rank == 0 {
                best_bits = candidate_bits.clone();
            }
            let payload_candidate_bits = extract_payload_bits_for_accuracy(engine, &candidate_bits);
            let (match_pct, ones_match_pct) =
                compute_bit_match_percentages(&payload_candidate_bits, comparison_message_bits);
            AvalancheCombinationBeamResult {
                rank: rank + 1,
                score: candidate.score,
                match_pct,
                ones_match_pct,
                hex: format_bits_hex_le(&payload_candidate_bits),
                bit_width: payload_candidate_bits.len(),
            }
        })
        .collect::<Vec<_>>();

    let top_beam_score = beam_results.first().map(|beam| beam.score).unwrap_or(0.0);
    let top_beam_match_pct = beam_results.first().map(|beam| beam.match_pct);
    let majority_vote_bits = majority_distribution.majority_bits;
    let payload_majority_vote_bits = extract_payload_bits_for_accuracy(engine, &majority_vote_bits);
    let (majority_vote_match_pct, majority_vote_ones_match_pct) =
        compute_bit_match_percentages(&payload_majority_vote_bits, comparison_message_bits);
    let best_match_pct = top_beam_match_pct
        .unwrap_or(0.0)
        .max(majority_vote_match_pct);
    let sample_index = sample_index + 1;
    let center_biases = if engine.avalanche_report_biases {
        build_center_bias_entries(&beam_probabilities, engine.avalanche_center_threshold)
    } else {
        Vec::new()
    };
    let stored_node = compact_stored_avalanche_node(&avalanche_search.node);

    Ok(ComputedAvalancheSample {
        sample: SelectedAvalancheSample {
            sample_index,
            tier_index,
            input_count,
            average_score_pct,
            beam_results: beam_results.clone(),
            majority_vote_bits: majority_vote_bits.clone(),
            majority_vote_match_pct,
            majority_vote_ones_match_pct,
            best_bits,
            top_beam_score,
            top_beam_match_pct,
            best_match_pct,
            center_biases,
            node: stored_node,
        },
        majority_vote_ones_count: majority_distribution.ones_count,
        majority_vote_zeros_count: majority_distribution.zeros_count,
        majority_vote_probability_one: majority_probabilities,
        normalized_bias_probabilities: normalized_biases,
        beam_search_probabilities: beam_probabilities,
        level_similarity_pct: avalanche_search.level_similarity_pct,
        level_pair_counts: avalanche_search.level_pair_counts,
    })
}

/// Converts a tier's finalized samples into session analytics suitable for a heatmap-style viewer.
///
/// # Parameters
/// - `tier_index`: One-based Avalanche tier index.
/// - `group_size`: Number of source items grouped into each sample for the tier.
/// - `source_kind`: Human-readable description of the source data for the tier.
/// - `samples`: Finalized sample outputs for the tier.
///
/// # Returns
/// - `AvalancheTierStatistics`: Per-tier sample accuracy summary.
///
/// # Expected Output
/// - Returns tier analytics; no stdout/stderr output.
fn build_avalanche_tier_statistics(
    tier_index: usize,
    group_size: usize,
    source_kind: &str,
    samples: &[SelectedAvalancheSample],
) -> AvalancheTierStatistics {
    AvalancheTierStatistics {
        tier_index,
        sample_count: samples.len(),
        group_size,
        source_kind: source_kind.to_string(),
        sample_stats: samples
            .iter()
            .map(|sample| AvalancheTierSampleStat {
                sample_index: sample.sample_index,
                input_count: sample.input_count,
                average_score_pct: sample.average_score_pct,
                beam_match_pct: sample.top_beam_match_pct,
                majority_vote_match_pct: Some(sample.majority_vote_match_pct),
                best_match_pct: sample.best_match_pct,
            })
            .collect(),
    }
}

/// Builds the filtered near-center bias report for one finalized sampled-Avalanche output.
///
/// # Parameters
/// - `probabilities`: Final beam probabilities after normalization and spreading.
/// - `center_threshold`: Inclusive maximum absolute distance from `0.5`.
///
/// # Returns
/// - `Vec<AvalancheCenterBiasEntry>`: Filtered bit positions that remain near the decision boundary.
///
/// # Expected Output
/// - Returns filtered report entries without stdout/stderr output.
fn build_center_bias_entries(
    probabilities: &[f64],
    center_threshold: f64,
) -> Vec<AvalancheCenterBiasEntry> {
    probabilities
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(bit_index_lsb0, probability_one)| {
            let signed_distance_from_half = probability_one - 0.5;
            (signed_distance_from_half.abs() <= center_threshold + f64::EPSILON).then_some(
                AvalancheCenterBiasEntry {
                    bit_index_lsb0,
                    probability_one,
                    signed_distance_from_half,
                },
            )
        })
        .collect()
}

/// Builds final-tier near-center bias reports for sampled-Avalanche outputs.
///
/// # Parameters
/// - `samples`: Final-tier samples produced for one analysis batch.
///
/// # Returns
/// - `Vec<AvalancheFinalTierBiasReport>`: Filtered per-sample reports ready for session logging.
///
/// # Expected Output
/// - Returns report entries without stdout/stderr output.
fn build_final_tier_bias_reports(
    samples: &[SelectedAvalancheSample],
) -> Vec<AvalancheFinalTierBiasReport> {
    samples
        .iter()
        .map(|sample| AvalancheFinalTierBiasReport {
            tier_index: sample.tier_index,
            sample_index: sample.sample_index,
            center_biases: sample.center_biases.clone(),
        })
        .collect()
}

/// Builds the best-overall center-bias report for session logging.
///
/// # Parameters
/// - `candidate`: Best overall Avalanche beam candidate selected across all batches.
///
/// # Returns
/// - `AvalancheBestCenterBiasReport`: Session-ready best-only center-bias payload.
///
/// # Expected Output
/// - Returns one report value without stdout/stderr output.
fn build_best_center_bias_report(candidate: &BeamMaxCandidate) -> AvalancheBestCenterBiasReport {
    AvalancheBestCenterBiasReport {
        batch_number: candidate.batch_number,
        tier_index: candidate.tier_index,
        sample_index: candidate.sample_index,
        center_biases: candidate.center_biases.clone(),
    }
}

/// Validates the configured final-tier near-center bias-report threshold.
///
/// # Parameters
/// - `engine`: Engine configuration providing the report threshold.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` when the threshold lies in `[0.0, 0.5]`.
///
/// # Expected Output
/// - Returns validation status without stdout/stderr output.
fn validate_avalanche_center_threshold(engine: &EngineConfig) -> Result<(), Box<dyn Error>> {
    if (0.0..=0.5).contains(&engine.avalanche_center_threshold) {
        Ok(())
    } else {
        Err("avalanche_center_threshold must be in [0, 0.5]".into())
    }
}

/// Strips persisted bias vectors from a finalized Avalanche node while preserving its bits.
///
/// # Parameters
/// - `node`: Finalized Avalanche node whose message bits should be retained.
///
/// # Returns
/// - `AvalancheNode`: Node containing the same message bits with zeroed stored biases.
///
/// # Expected Output
/// - Returns a compact node suitable for recursive-tier reuse; no stdout/stderr output.
fn compact_stored_avalanche_node(node: &AvalancheNode) -> AvalancheNode {
    AvalancheNode::from_packed_bits(
        PackedBits::from_bools(&node.message_bits_vec()),
        vec![0.0; node.bit_len()],
    )
}

/// Removes original sampled `c^x`/`r` inputs from retained tier-one sample payloads before recursion.
///
/// # Parameters
/// - `samples`: Retained sampled-avalanche records that may still own original tier-one inputs.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Clears retained source-input payloads in place while preserving sample-level summary fields.
fn compact_retained_avalanche_sample_inputs(samples: &mut [AvalancheCombinationSample]) {
    for sample in samples {
        sample.inputs.clear();
    }
}

/// Resolves how many fitness-ranked entries belong to the configured top cohort.
///
/// # Parameters
/// - `retained_count`: Number of candidates retained after thresholding and ranking.
/// - `top_pct`: Fraction of retained candidates to include in the logged cohort.
///
/// # Returns
/// - `usize`: Count of retained candidates included in the logged top cohort.
///
/// # Expected Output
/// - Returns `0` for an empty retained pool or at least `1` for any non-empty retained pool; no stdout/stderr output.
fn resolve_fitness_top_cohort_count(retained_count: usize, top_pct: f64) -> usize {
    if retained_count == 0 {
        return 0;
    }

    let top_count = (retained_count as f64 * top_pct).ceil();
    if !top_count.is_finite() || top_count <= 0.0 {
        return 1;
    }

    (top_count as usize).clamp(1, retained_count)
}

/// Prints the configured top percentage of retained in-memory fitness-ranked candidates for one batch.
///
/// # Parameters
/// - `batch_number`: One-based analysis batch number.
/// - `retained_inputs`: Fitness-ranked scored inputs retained for sampled Avalanche.
/// - `fitness_bit_width`: Number of least-significant bits used to compute the normalized fitness score.
/// - `top_pct`: Fraction of retained candidates to include in the logged cohort.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Prints one header plus one line per logged candidate showing the normalized fitness percentage and match percentage.
fn log_top_scored_avalanche_fitness_entries(
    batch_number: usize,
    retained_inputs: &[RankedScoredAvalancheInput],
    fitness_bit_width: usize,
    top_pct: f64,
) {
    let top_count = resolve_fitness_top_cohort_count(retained_inputs.len(), top_pct);
    if top_count == 0 {
        println!(
            "Avalanche fitness top cohort for batch {}: no scored inputs remained after thresholding and ranking",
            batch_number
        );
        return;
    }

    println!(
        "Avalanche fitness top cohort for batch {}: logging top {} of {} retained scored inputs ({}%)",
        batch_number,
        top_count,
        retained_inputs.len(),
        format_beam_float(top_pct * 100.0, BEAM_PCT_DECIMALS)
    );
    for (rank, input) in retained_inputs.iter().take(top_count).enumerate() {
        let normalized_fitness_pct =
            normalize_avalanche_fitness_score(input.fitness.fitness_score, fitness_bit_width)
                * 100.0;
        let normalized_mean_fitness_pct = normalize_avalanche_fitness_mean_score(
            input.fitness.fitness_total_score,
            fitness_bit_width,
            input.fitness.fitness_message_count,
        ) * 100.0;
        println!(
            "Avalanche fitness top cohort for batch {} [{}]: batch-index {} message-index {} x {} minimum-padding-fitness {} ({}%) mean-padding-fitness {}% across {} message(s) match {}%",
            batch_number,
            rank + 1,
            input.input.batch_candidate_index,
            input.input.message_index,
            input.input.x,
            input.fitness.fitness_score,
            format_beam_float(normalized_fitness_pct, BEAM_PCT_DECIMALS),
            format_beam_float(normalized_mean_fitness_pct, BEAM_PCT_DECIMALS),
            input.fitness.fitness_message_count,
            format_beam_float(input.input.score_match_pct, BEAM_PCT_DECIMALS),
        );
    }
}

/// Prints the configured top percentage of retained cached fitness-ranked candidates for one batch.
///
/// # Parameters
/// - `batch_number`: One-based analysis batch number.
/// - `retained_inputs`: Fitness-ranked cached scored-input summaries retained for sampled Avalanche.
/// - `fitness_bit_width`: Number of least-significant bits used to compute the normalized fitness score.
/// - `top_pct`: Fraction of retained candidates to include in the logged cohort.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Prints one header plus one line per logged cached candidate showing the normalized fitness percentage and match percentage.
fn log_top_cached_avalanche_fitness_entries(
    batch_number: usize,
    retained_inputs: &[CachedScoredInputSummary],
    fitness_bit_width: usize,
    top_pct: f64,
) {
    let top_count = resolve_fitness_top_cohort_count(retained_inputs.len(), top_pct);
    if top_count == 0 {
        println!(
            "Avalanche fitness top cohort for batch {}: no cached scored inputs remained after thresholding and ranking",
            batch_number
        );
        return;
    }

    println!(
        "Avalanche fitness top cohort for batch {}: logging top {} of {} retained cached scored inputs ({}%)",
        batch_number,
        top_count,
        retained_inputs.len(),
        format_beam_float(top_pct * 100.0, BEAM_PCT_DECIMALS)
    );
    for (rank, input) in retained_inputs.iter().take(top_count).enumerate() {
        let normalized_fitness_pct =
            normalize_avalanche_fitness_score(input.fitness.fitness_score, fitness_bit_width)
                * 100.0;
        let normalized_mean_fitness_pct = normalize_avalanche_fitness_mean_score(
            input.fitness.fitness_total_score,
            fitness_bit_width,
            input.fitness.fitness_message_count,
        ) * 100.0;
        println!(
            "Avalanche fitness top cohort for batch {} [{}]: batch-index {} message-index {} x {} minimum-padding-fitness {} ({}%) mean-padding-fitness {}% across {} message(s) match {}%",
            batch_number,
            rank + 1,
            input.batch_candidate_index,
            input.message_index,
            input.x,
            input.fitness.fitness_score,
            format_beam_float(normalized_fitness_pct, BEAM_PCT_DECIMALS),
            format_beam_float(normalized_mean_fitness_pct, BEAM_PCT_DECIMALS),
            input.fitness.fitness_message_count,
            format_beam_float(input.score_match_pct, BEAM_PCT_DECIMALS),
        );
    }
}

/// Executes one sampled Avalanche combination from a preselected input set.
///
/// # Parameters
/// - `engine`: Engine configuration controlling combination sampling and beam scoring.
/// - `reference_bits`: Full-width shifted reference bits used for ordering and reduction.
/// - `comparison_message_bits`: Original plaintext payload bits used for beam-match scoring.
/// - `selected_inputs`: Preselected scored inputs assigned exclusively to this sample.
/// - `pool_size`: Total number of scored inputs available in the batch.
/// - `r_candidate_pool_size`: Total number of distinct `r` candidates available in the batch.
/// - `tier_index`: One-based Avalanche tier index being executed.
/// - `sample_index`: Zero-based sample index for analytics ordering.
///
/// # Returns
/// - `Result<SampledAvalancheSampleOutcome, String>`: Sample analytics, selected execution, and evaluated-node count.
///
/// # Expected Output
/// - Returns sample analytics for one combination; no stdout/stderr output.
fn execute_sampled_avalanche_sample(
    engine: &EngineConfig,
    reference_bits: &[bool],
    comparison_message_bits: &[bool],
    selected_inputs: Vec<ScoredAvalancheInput>,
    pool_size: usize,
    r_candidate_pool_size: usize,
    tier_index: usize,
    sample_index: usize,
) -> Result<SampledAvalancheSampleOutcome, String> {
    let keep_all_samples = engine.avalanche_statistics_collection
        && engine.avalanche_combination_keep_all_samples_in_memory;
    let selected_group_count = selected_inputs
        .iter()
        .map(|input| input.batch_candidate_index)
        .collect::<HashSet<_>>()
        .len();
    let average_score_pct = mean_f64(
        &selected_inputs
            .iter()
            .map(|entry| entry.score_match_pct)
            .collect::<Vec<_>>(),
    );
    let evaluated_candidates = selected_inputs.len();
    if selected_inputs.is_empty() {
        return Ok(SampledAvalancheSampleOutcome {
            retained_sample: None,
            sample: None,
            evaluated_candidates,
            produced_sample: false,
        });
    }

    let avalanche_search = build_prepared_avalanche(
        &selected_inputs,
        engine,
        reference_bits,
        engine.avalanche_statistics_collection,
        None,
    )
    .map_err(|err| err.to_string())?
    .execute()
    .map_err(|err| err.to_string())?;
    let selected_oracles = selected_inputs
        .iter()
        .map(|input| input.message_bits.clone())
        .collect::<Vec<_>>();
    let computed = finalize_avalanche_sample(
        engine,
        comparison_message_bits,
        sample_index,
        tier_index,
        selected_inputs.len(),
        average_score_pct,
        avalanche_search,
        selected_oracles,
    )?;
    let selected_sample = SelectedAvalancheSample {
        sample_index: computed.sample.sample_index,
        tier_index: computed.sample.tier_index,
        input_count: computed.sample.input_count,
        average_score_pct: computed.sample.average_score_pct,
        beam_results: computed.sample.beam_results.clone(),
        majority_vote_bits: computed.sample.majority_vote_bits.clone(),
        majority_vote_match_pct: computed.sample.majority_vote_match_pct,
        majority_vote_ones_match_pct: computed.sample.majority_vote_ones_match_pct,
        best_bits: computed.sample.best_bits.clone(),
        top_beam_score: computed.sample.top_beam_score,
        top_beam_match_pct: computed.sample.top_beam_match_pct,
        best_match_pct: computed.sample.best_match_pct,
        center_biases: computed.sample.center_biases.clone(),
        node: computed.sample.node.clone(),
    };
    let retained_sample = if keep_all_samples {
        Some(AvalancheCombinationSample {
            sample_index: selected_sample.sample_index,
            pool_size,
            r_candidate_pool_size,
            combination_size: selected_inputs.len(),
            mixed_r_candidate_count: selected_group_count,
            average_score_pct,
            majority_vote_enabled: engine.avalanche_combination_majority_vote,
            sample_smoothing_enabled: engine.avalanche_combination_sample_smoothing,
            inputs: selected_inputs
                .iter()
                .map(|input| {
                    let detail = input
                        .detail
                        .as_ref()
                        .expect("sample details must exist when storing all avalanche samples");
                    AvalancheCombinationSampleInput {
                        batch_candidate_index: input.batch_candidate_index,
                        message_index: input.message_index,
                        r: input.r.clone(),
                        r_bits: input.r.bits(),
                        target_exponent: detail.target_exponent.clone(),
                        x: input.x.clone(),
                        score_match_pct: input.score_match_pct,
                        hbc_ciphertext_r: detail.hbc_ciphertext_r.clone(),
                        candidate_decryption: detail.candidate_decryption.clone(),
                    }
                })
                .collect(),
            majority_vote_bits: extract_payload_bits_for_accuracy(
                engine,
                &selected_sample.majority_vote_bits,
            ),
            majority_vote_ones_count: computed.majority_vote_ones_count,
            majority_vote_zeros_count: computed.majority_vote_zeros_count,
            majority_vote_probability_one: computed.majority_vote_probability_one,
            level_similarity_pct: computed.level_similarity_pct,
            level_pair_counts: computed.level_pair_counts,
            normalized_bias_probabilities: computed.normalized_bias_probabilities,
            beam_search_probabilities: computed.beam_search_probabilities,
            beam_results: selected_sample.beam_results.clone(),
        })
    } else {
        None
    };

    Ok(SampledAvalancheSampleOutcome {
        retained_sample,
        sample: Some(selected_sample),
        evaluated_candidates,
        produced_sample: true,
    })
}

/// Executes one sampled Avalanche combination by loading only the selected cached inputs.
///
/// # Parameters
/// - `engine`: Engine configuration controlling combination sampling and beam scoring.
/// - `reference_bits`: Full-width shifted reference bits used for ordering and reduction.
/// - `comparison_message_bits`: Original plaintext payload bits used for beam-match scoring.
/// - `cache`: Shared SQLite cache wrapper containing tier-one scored inputs.
/// - `selected_input_ids`: Cached row ids assigned exclusively to this sample.
/// - `pool_size`: Total number of scored inputs available in the batch.
/// - `r_candidate_pool_size`: Total number of distinct `r` candidates available in the batch.
/// - `tier_index`: One-based Avalanche tier index being executed.
/// - `sample_index`: Zero-based sample index for analytics ordering.
///
/// # Returns
/// - `Result<SampledAvalancheSampleOutcome, String>`: Sample analytics, selected execution, and evaluated-node count.
///
/// # Expected Output
/// - Loads only the selected cached rows from SQLite and returns sample analytics for one combination.
fn execute_sampled_avalanche_sample_from_cache(
    engine: &EngineConfig,
    reference_bits: &[bool],
    comparison_message_bits: &[bool],
    cache: &AvalancheCacheGuard,
    selected_input_ids: &[i64],
    pool_size: usize,
    r_candidate_pool_size: usize,
    tier_index: usize,
    sample_index: usize,
) -> Result<SampledAvalancheSampleOutcome, String> {
    let keep_all_samples = engine.avalanche_statistics_collection
        && engine.avalanche_combination_keep_all_samples_in_memory;
    let selected_inputs = load_cached_scored_inputs_by_ids(cache, selected_input_ids)
        .map_err(|err| err.to_string())?;
    let selected_group_count = selected_inputs
        .iter()
        .map(|input| input.batch_candidate_index)
        .collect::<HashSet<_>>()
        .len();
    let average_score_pct = mean_f64(
        &selected_inputs
            .iter()
            .map(|entry| entry.score_match_pct)
            .collect::<Vec<_>>(),
    );
    let evaluated_candidates = selected_inputs.len();
    if selected_inputs.is_empty() {
        return Ok(SampledAvalancheSampleOutcome {
            retained_sample: None,
            sample: None,
            evaluated_candidates,
            produced_sample: false,
        });
    }

    let avalanche_search = build_prepared_avalanche(
        &selected_inputs,
        engine,
        reference_bits,
        engine.avalanche_statistics_collection,
        None,
    )
    .map_err(|err| err.to_string())?
    .execute()
    .map_err(|err| err.to_string())?;
    let selected_oracles = selected_inputs
        .iter()
        .map(|input| input.message_bits.clone())
        .collect::<Vec<_>>();
    let computed = finalize_avalanche_sample(
        engine,
        comparison_message_bits,
        sample_index,
        tier_index,
        selected_inputs.len(),
        average_score_pct,
        avalanche_search,
        selected_oracles,
    )?;
    let selected_sample = SelectedAvalancheSample {
        sample_index: computed.sample.sample_index,
        tier_index: computed.sample.tier_index,
        input_count: computed.sample.input_count,
        average_score_pct: computed.sample.average_score_pct,
        beam_results: computed.sample.beam_results.clone(),
        majority_vote_bits: computed.sample.majority_vote_bits.clone(),
        majority_vote_match_pct: computed.sample.majority_vote_match_pct,
        majority_vote_ones_match_pct: computed.sample.majority_vote_ones_match_pct,
        best_bits: computed.sample.best_bits.clone(),
        top_beam_score: computed.sample.top_beam_score,
        top_beam_match_pct: computed.sample.top_beam_match_pct,
        best_match_pct: computed.sample.best_match_pct,
        center_biases: computed.sample.center_biases.clone(),
        node: computed.sample.node.clone(),
    };
    let retained_sample = if keep_all_samples {
        Some(AvalancheCombinationSample {
            sample_index: selected_sample.sample_index,
            pool_size,
            r_candidate_pool_size,
            combination_size: selected_inputs.len(),
            mixed_r_candidate_count: selected_group_count,
            average_score_pct,
            majority_vote_enabled: engine.avalanche_combination_majority_vote,
            sample_smoothing_enabled: engine.avalanche_combination_sample_smoothing,
            inputs: selected_inputs
                .iter()
                .map(|input| {
                    let detail = input
                        .detail
                        .as_ref()
                        .expect("sample details must exist when storing all avalanche samples");
                    AvalancheCombinationSampleInput {
                        batch_candidate_index: input.batch_candidate_index,
                        message_index: input.message_index,
                        r: input.r.clone(),
                        r_bits: input.r.bits(),
                        target_exponent: detail.target_exponent.clone(),
                        x: input.x.clone(),
                        score_match_pct: input.score_match_pct,
                        hbc_ciphertext_r: detail.hbc_ciphertext_r.clone(),
                        candidate_decryption: detail.candidate_decryption.clone(),
                    }
                })
                .collect(),
            majority_vote_bits: extract_payload_bits_for_accuracy(
                engine,
                &selected_sample.majority_vote_bits,
            ),
            majority_vote_ones_count: computed.majority_vote_ones_count,
            majority_vote_zeros_count: computed.majority_vote_zeros_count,
            majority_vote_probability_one: computed.majority_vote_probability_one,
            level_similarity_pct: computed.level_similarity_pct,
            level_pair_counts: computed.level_pair_counts,
            normalized_bias_probabilities: computed.normalized_bias_probabilities,
            beam_search_probabilities: computed.beam_search_probabilities,
            beam_results: selected_sample.beam_results.clone(),
        })
    } else {
        None
    };

    Ok(SampledAvalancheSampleOutcome {
        retained_sample,
        sample: Some(selected_sample),
        evaluated_candidates,
        produced_sample: true,
    })
}

/// Loads lightweight cached recursive sample summaries for one batch/tier.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// Executes one recursive Avalanche sample by loading prior-tier sample rows from SQLite.
///
/// # Parameters
/// - `engine`: Engine configuration controlling beam-search behavior.
/// - `reference_bits`: Full-width shifted reference bits used for optional ordering.
/// - `comparison_message_bits`: Original plaintext payload bits used for scoring.
/// - `cache`: Shared SQLite cache wrapper containing prior-tier sample rows.
/// - `source_sample_ids`: Cache row ids selecting the prior-tier samples used by this recursive sample.
/// - `tier_index`: One-based tier index being executed.
/// - `sample_index`: Zero-based sample index within the current tier.
///
/// # Returns
/// - `Result<SelectedAvalancheSample, String>`: Finalized recursive sample result.
///
/// # Expected Output
/// - Loads the requested prior-tier sample rows from SQLite and returns the recursive sample output.
fn execute_recursive_avalanche_sample_from_cache_ids(
    engine: &EngineConfig,
    reference_bits: &[bool],
    comparison_message_bits: &[bool],
    cache: &AvalancheCacheGuard,
    source_sample_ids: &[i64],
    tier_index: usize,
    sample_index: usize,
) -> Result<SelectedAvalancheSample, String> {
    let rows = load_cached_selected_sample_rows_by_ids(cache, source_sample_ids)
        .map_err(|err| err.to_string())?;
    let mut by_id = HashMap::with_capacity(rows.len());
    for row in rows {
        by_id.insert(row.id, row);
    }
    let selected_samples = source_sample_ids
        .iter()
        .map(|id| {
            let row = by_id
                .remove(id)
                .ok_or_else(|| format!("missing cached recursive sample row id {}", id))?;
            Ok::<_, String>(RecursiveAvalancheSourceSample {
                best_match_pct: row.best_match_pct,
                message_bits: PackedBits::from_bytes_le(
                    &row.recursive_bits,
                    usize::try_from(row.recursive_bits_bit_len)
                        .map_err(|_| "cached recursive bit length exceeds usize range")?,
                ),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let selected_sample_refs = selected_samples.iter().collect::<Vec<_>>();
    let average_score_pct = mean_f64(
        &selected_sample_refs
            .iter()
            .map(|sample| sample.best_match_pct)
            .collect::<Vec<_>>(),
    );
    let recursive_inputs = build_recursive_avalanche_inputs(&selected_sample_refs);
    let avalanche_search = build_prepared_avalanche(
        &recursive_inputs,
        engine,
        reference_bits,
        engine.avalanche_statistics_collection,
        None,
    )
    .map_err(|err| err.to_string())?
    .execute()
    .map_err(|err| err.to_string())?;
    let selected_oracles = recursive_inputs
        .iter()
        .map(|input| input.message_bits.clone())
        .collect::<Vec<_>>();
    finalize_avalanche_sample(
        engine,
        comparison_message_bits,
        sample_index,
        tier_index,
        selected_sample_refs.len(),
        average_score_pct,
        avalanche_search,
        selected_oracles,
    )
    .map(|computed| computed.sample)
}

/// Executes one recursive Avalanche sample from prior-tier outputs selected by index.
///
/// # Parameters
/// - `engine`: Engine configuration controlling beam-search behavior.
/// - `reference_bits`: Full-width shifted reference bits used for optional ordering.
/// - `comparison_message_bits`: Original plaintext payload bits used for scoring.
/// - `source_samples`: Compact prior-tier recursive source samples available for indexed lookup.
/// - `source_sample_indices`: Indices selecting the prior-tier samples used by this recursive sample.
/// - `tier_index`: One-based tier index being executed.
/// - `sample_index`: Zero-based sample index within the current tier.
///
/// # Returns
/// - `Result<SelectedAvalancheSample, String>`: Finalized recursive sample result.
///
/// # Expected Output
/// - Returns the recursive sample output using borrowed prior-tier samples; no stdout/stderr output.
fn execute_recursive_avalanche_sample_from_indices(
    engine: &EngineConfig,
    reference_bits: &[bool],
    comparison_message_bits: &[bool],
    source_samples: &[RecursiveAvalancheSourceSample],
    source_sample_indices: &[usize],
    tier_index: usize,
    sample_index: usize,
) -> Result<SelectedAvalancheSample, String> {
    let selected_samples = source_sample_indices
        .iter()
        .map(|&index| {
            source_samples
                .get(index)
                .ok_or_else(|| format!("missing recursive source sample index {}", index))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let average_score_pct = mean_f64(
        &selected_samples
            .iter()
            .map(|sample| sample.best_match_pct)
            .collect::<Vec<_>>(),
    );
    let recursive_inputs = build_recursive_avalanche_inputs(&selected_samples);
    let avalanche_search = build_prepared_avalanche(
        &recursive_inputs,
        engine,
        reference_bits,
        engine.avalanche_statistics_collection,
        None,
    )
    .map_err(|err| err.to_string())?
    .execute()
    .map_err(|err| err.to_string())?;
    let selected_oracles = recursive_inputs
        .iter()
        .map(|input| input.message_bits.clone())
        .collect::<Vec<_>>();
    finalize_avalanche_sample(
        engine,
        comparison_message_bits,
        sample_index,
        tier_index,
        selected_samples.len(),
        average_score_pct,
        avalanche_search,
        selected_oracles,
    )
    .map(|computed| computed.sample)
}

/// Runs sampled avalanche combinations over cached scored batch outputs.
///
/// # Parameters
/// - `engine`: Engine configuration controlling combination sampling and beam scoring.
/// - `payload_message`: Original plaintext payload used for scoring and to derive the widened reference.
/// - `cache`: Shared SQLite cache wrapper containing the batch-scored inputs.
/// - `batch_number`: One-based batch index used for progress logging.
/// - `prefer_beam_score_ordering`: Whether selection should prefer beam score over known match percentage.
/// - `rng`: Random number generator for combination sampling.
///
/// # Returns
/// - `Result<SampledAvalancheBatchResult, Box<dyn Error>>`: Sample logs plus the selected best sample.
///
/// # Expected Output
/// - Prints sampled-avalanche progress and returns results using SQLite-backed tier inputs and recursive tiers.
fn run_sampled_avalanche_beam_search_cached(
    engine: &EngineConfig,
    payload_message: &BigUint,
    cache: &AvalancheCacheGuard,
    batch_number: usize,
    prefer_beam_score_ordering: bool,
    rng: &mut RngChoice,
) -> Result<SampledAvalancheBatchResult, Box<dyn Error>> {
    if engine.avalanche_combination_samples == 0 {
        return Err("avalanche_combination_samples must be >= 1".into());
    }
    if !engine.avalanche_random_chacha20_inputs
        && engine.avalanche_combination_mixed_r_candidates == 0
    {
        return Err("avalanche_combination_mixed_r_candidates must be >= 1".into());
    }
    if engine.avalanche_combination_size == 0 {
        return Err("avalanche_combination_size must be >= 1".into());
    }
    if engine.avalanche_combination_hamming_distance_prune
        && !(0.0 < engine.avalanche_combination_hamming_distance_keep_percentile
            && engine.avalanche_combination_hamming_distance_keep_percentile <= 100.0)
    {
        return Err(
            "avalanche_combination_hamming_distance_keep_percentile must be in (0, 100]".into(),
        );
    }
    if !(0.0..=100.0)
        .contains(&engine.avalanche_combination_hamming_distance_outlier_preference_pct)
    {
        return Err(
            "avalanche_combination_hamming_distance_outlier_preference_pct must be in [0, 100]"
                .into(),
        );
    }
    validate_avalanche_fitness_threshold(engine)?;
    validate_avalanche_fitness_log_top_pct(engine)?;
    validate_avalanche_center_threshold(engine)?;

    let cached_input_total = count_cached_scored_inputs(cache, batch_number)?;
    if cached_input_total == 0 {
        return Ok(SampledAvalancheBatchResult::default());
    }

    let fitness_bit_width = resolve_avalanche_fitness_bit_width(engine);
    let mut scored_inputs = if engine.avalanche_fitness_scoring_pass {
        let preprocessed = apply_cached_scored_avalanche_fitness_pass(
            cache,
            batch_number,
            fitness_bit_width,
            engine.avalanche_fitness_r_candidate_limit,
            engine.avalanche_fitness_cx_candidate_limit,
            engine.avalanche_fitness_use_threshold,
            engine.avalanche_fitness_threshold,
        )?;
        println!(
            "Avalanche fitness pass for batch {}: retained {} of {} cached scored inputs in a globally ranked pool using {} LSB fitness bits with effective retained-input cap {} (r-limit {} cx-limit {}, threshold-enabled {} threshold {}, additional-random-messages {})",
            batch_number,
            preprocessed.len(),
            cached_input_total,
            fitness_bit_width,
            resolve_avalanche_fitness_retained_input_limit(
                engine.avalanche_fitness_r_candidate_limit,
                engine.avalanche_fitness_cx_candidate_limit,
            ),
            engine.avalanche_fitness_r_candidate_limit,
            engine.avalanche_fitness_cx_candidate_limit,
            if engine.avalanche_fitness_use_threshold {
                "on"
            } else {
                "off"
            },
            format_beam_float(engine.avalanche_fitness_threshold, 3),
            engine.avalanche_fitness_additional_random_messages
        );
        log_top_cached_avalanche_fitness_entries(
            batch_number,
            &preprocessed,
            fitness_bit_width,
            engine.avalanche_fitness_log_top_pct,
        );
        preprocessed
    } else {
        load_cached_scored_input_summaries(cache, batch_number)?
    };
    if engine.avalanche_unique_r_cx_inputs {
        if !engine.avalanche_fitness_scoring_pass {
            scored_inputs.sort_by(|left, right| {
                right
                    .score_match_pct
                    .total_cmp(&left.score_match_pct)
                    .then_with(|| left.batch_candidate_index.cmp(&right.batch_candidate_index))
                    .then_with(|| left.message_index.cmp(&right.message_index))
                    .then_with(|| left.x.cmp(&right.x))
            });
        }
        let original_count = scored_inputs.len();
        let (unique_inputs, rejected_overlap_count) =
            enforce_global_unique_cached_scored_inputs(scored_inputs);
        scored_inputs = unique_inputs;
        println!(
            "Avalanche unique-input filter for batch {}: retained {} of {} cached scored inputs after dropping {} overlapping r/x candidates",
            batch_number,
            scored_inputs.len(),
            original_count,
            rejected_overlap_count
        );
    }
    if scored_inputs.is_empty() {
        return Err(
            "avalanche_fitness_threshold removed all cached scored inputs for sampled avalanche"
                .into(),
        );
    }

    let comparison_message_bits =
        biguint_to_bits_le(payload_message, resolve_plaintext_message_bit_width(engine));
    let transformed_message = build_candidate_message_transform(engine)(payload_message);
    let reference_bits = biguint_to_bits_le(
        &transformed_message,
        load_cached_scored_inputs_by_ids(cache, &[scored_inputs[0].id])?
            .first()
            .map(|input| input.message_bits.len())
            .unwrap_or_else(|| resolve_avalanche_bit_width(engine)),
    );
    let packed_message_bits = PackedBits::from_bools(&reference_bits);
    let hamming_prune_label = format!("Accuracy batch {} cached Hamming prune", batch_number);
    let (
        pruned_scored_inputs,
        retained_inlier_count,
        available_outlier_count,
        preferred_outlier_count,
    ) = if engine.avalanche_combination_hamming_distance_prune {
        prune_cached_scored_inputs_by_hamming_distance_percentile_with_progress(
            cache,
            &scored_inputs,
            &packed_message_bits,
            engine.avalanche_combination_hamming_distance_keep_percentile,
            engine.avalanche_combination_hamming_distance_outlier_preference_pct,
            Some(&hamming_prune_label),
        )?
    } else {
        (scored_inputs.clone(), scored_inputs.len(), 0, 0)
    };
    let grouping_label = format!(
        "Accuracy batch {} cached avalanche input grouping",
        batch_number
    );
    let grouped_inputs = group_cached_scored_inputs_by_r_candidate_with_progress(
        &pruned_scored_inputs,
        Some(&grouping_label),
    );
    let pool_size = pruned_scored_inputs.len();
    let r_candidate_pool_size = grouped_inputs.len();
    if r_candidate_pool_size == 0 {
        return Ok(SampledAvalancheBatchResult::default());
    }
    let mixed_r_candidate_count = if engine.avalanche_random_chacha20_inputs {
        0
    } else {
        engine
            .avalanche_combination_mixed_r_candidates
            .min(engine.avalanche_combination_size)
            .min(r_candidate_pool_size)
    };

    let sample_count = engine.avalanche_combination_samples as usize;
    let recursion_depth = engine.avalanche_combination_recursion_depth.max(1);
    let majority_vote_enabled = engine.avalanche_combination_majority_vote;
    let sample_smoothing_enabled = engine.avalanche_combination_sample_smoothing;
    let majority_vote_print_enabled = engine.avalanche_combination_majority_vote_print;
    let recursive_input_mode = if engine.avalanche_use_top_beam {
        "top-beam"
    } else {
        "majority-vote"
    };
    let statistics_collection_enabled = engine.avalanche_statistics_collection;
    let keep_all_samples_enabled =
        statistics_collection_enabled && engine.avalanche_combination_keep_all_samples_in_memory;
    let selection_mode = if engine.avalanche_random_chacha20_inputs {
        "random-chacha20-inputs"
    } else {
        "mixed-r-combinations"
    };
    let sample_plans = if engine.avalanche_random_chacha20_inputs {
        let pruned_input_ids = pruned_scored_inputs
            .iter()
            .map(|summary| summary.id)
            .collect::<Vec<_>>();
        build_unique_flat_sample_plans(
            &pruned_input_ids,
            engine.avalanche_combination_size,
            sample_count,
            rng,
        )
    } else {
        let grouped_input_ids = grouped_inputs
            .iter()
            .map(|group| group.input_ids.clone())
            .collect::<Vec<_>>();
        build_unique_grouped_sample_plans(
            &grouped_input_ids,
            mixed_r_candidate_count,
            engine.avalanche_combination_size,
            sample_count,
            rng,
        )
    };
    let effective_sample_count = sample_plans.len();

    println!(
        "Avalanche combination setup for batch {}: scored inputs {} r-candidate-pool {} selection-mode {} configured-mixed-r-candidates {} effective-mixed-r-candidates {} configured-samples {} effective-samples {} recursion-depth {} recursive-group-sizes {:?} recursive-resample-counts {:?} majority-vote {} sample-smoothing {} majority-print {} recursive-input {} statistics-collection {} keep-all-samples {} hamming-prune {} kept-percentile {} outlier-preference-pct {}",
        batch_number,
        pool_size,
        r_candidate_pool_size,
        selection_mode,
        engine.avalanche_combination_mixed_r_candidates,
        mixed_r_candidate_count,
        sample_count,
        effective_sample_count,
        recursion_depth,
        &engine.avalanche_combination_recursive_group_size,
        &engine.avalanche_combination_recursive_resample_count,
        if majority_vote_enabled { "on" } else { "off" },
        if sample_smoothing_enabled {
            "on"
        } else {
            "off"
        },
        if majority_vote_print_enabled {
            "on"
        } else {
            "off"
        },
        recursive_input_mode,
        if statistics_collection_enabled {
            "on"
        } else {
            "off"
        },
        if keep_all_samples_enabled {
            "on"
        } else {
            "off"
        },
        if engine.avalanche_combination_hamming_distance_prune {
            "on"
        } else {
            "off"
        },
        engine.avalanche_combination_hamming_distance_keep_percentile,
        engine.avalanche_combination_hamming_distance_outlier_preference_pct
    );
    if engine.avalanche_combination_hamming_distance_prune && pool_size < scored_inputs.len() {
        println!(
            "Avalanche combination batch {} pruned cached scored inputs by Hamming distance from {} to {} before sampling (retained-inliers {} available-outliers {} preferred-outliers {})",
            batch_number,
            scored_inputs.len(),
            pool_size,
            retained_inlier_count,
            available_outlier_count,
            preferred_outlier_count
        );
    }
    if !engine.avalanche_random_chacha20_inputs
        && mixed_r_candidate_count < engine.avalanche_combination_mixed_r_candidates
    {
        println!(
            "Avalanche combination batch {} capped mixed r-candidates from {} to {} because only {} distinct r candidates were available in the batch",
            batch_number,
            engine.avalanche_combination_mixed_r_candidates,
            mixed_r_candidate_count,
            r_candidate_pool_size
        );
    }
    if effective_sample_count < sample_count {
        println!(
            "Avalanche combination batch {} capped unique sampled combinations from {} to {} because only that many distinct tier-one input sets were produced",
            batch_number, sample_count, effective_sample_count
        );
    }
    if effective_sample_count == 0 {
        return Ok(SampledAvalancheBatchResult::default());
    }

    let sample_label = format!("Avalanche sample batch {}", batch_number);
    let sample_done = AtomicU64::new(0);
    let sample_log_start = Instant::now();
    let sample_log_interval = Duration::from_secs(5);
    let sample_next_log_at_ms =
        AtomicU64::new(sample_log_interval.as_millis().min(u128::from(u64::MAX)) as u64);
    let mut base_outcomes = sample_plans
        .into_par_iter()
        .enumerate()
        .map(|(sample_index, selected_input_ids)| {
            let outcome = execute_sampled_avalanche_sample_from_cache(
                engine,
                &reference_bits,
                &comparison_message_bits,
                cache,
                &selected_input_ids,
                pool_size,
                r_candidate_pool_size,
                1,
                sample_index,
            )?;
            let done = sample_done.fetch_add(1, Ordering::Relaxed) + 1;
            log_parallel_progress_every_interval(
                done,
                effective_sample_count as u64,
                &sample_log_start,
                &sample_next_log_at_ms,
                &sample_label,
                sample_log_interval,
            );
            Ok::<_, String>(outcome)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| -> Box<dyn Error> { err.into() })?;

    let mut reduced = SampledAvalancheBatchResult::default();
    reduced.sample_count = base_outcomes
        .iter()
        .filter(|outcome| outcome.produced_sample)
        .count();
    reduced.evaluated_candidates = base_outcomes
        .iter()
        .map(|outcome| outcome.evaluated_candidates)
        .sum();
    reduced.retained_samples = base_outcomes
        .iter_mut()
        .filter_map(|outcome| outcome.retained_sample.take())
        .collect();

    let current_tier_samples = base_outcomes
        .iter_mut()
        .filter_map(|outcome| outcome.sample.take())
        .collect::<Vec<_>>();
    let will_recurse = recursion_depth > 1 && current_tier_samples.len() > 1;
    if will_recurse {
        compact_retained_avalanche_sample_inputs(&mut reduced.retained_samples);
    }
    insert_cached_selected_samples(cache, batch_number, &current_tier_samples, engine)?;
    drop(base_outcomes);
    drop(grouped_inputs);
    drop(pruned_scored_inputs);
    if statistics_collection_enabled {
        reduced
            .tier_statistics
            .push(build_avalanche_tier_statistics(
                1,
                engine.avalanche_combination_size,
                selection_mode,
                &current_tier_samples,
            ));
    }

    let mut tier_index = 1usize;
    let mut current_tier_summaries =
        load_cached_recursive_sample_summaries(cache, batch_number, tier_index)?;
    drop(current_tier_samples);
    while tier_index < recursion_depth && current_tier_summaries.len() > 1 {
        let next_tier_index = tier_index + 1;
        let recursive_tier_config = resolve_recursive_avalanche_tier_config(engine, tier_index);
        let recursive_group_size = recursive_tier_config.group_size;
        let recursive_resample_count = recursive_tier_config.resample_count;
        let source_sample_count = current_tier_summaries.len();
        let recursive_done = AtomicU64::new(0);
        let recursive_evaluated_candidates = AtomicU64::new(0);
        let recursive_log_start = Instant::now();
        let recursive_log_interval = Duration::from_secs(5);
        let recursive_next_log_at_ms =
            AtomicU64::new(recursive_log_interval.as_millis().min(u128::from(u64::MAX)) as u64);
        let progress_label = format!(
            "Avalanche recursive tier {} batch {}",
            next_tier_index, batch_number
        );
        let target_group_count = if recursive_resample_count > 0 {
            recursive_resample_count
        } else {
            source_sample_count.div_ceil(recursive_group_size)
        };
        let source_sample_indices = (0..source_sample_count).collect::<Vec<_>>();
        let recursive_sample_plans = build_unique_flat_sample_plans(
            &source_sample_indices,
            recursive_group_size,
            target_group_count,
            rng,
        );
        let source_kind = if recursive_resample_count > 0 {
            "recursive-resampled-samples"
        } else {
            "recursive-samples"
        };
        let group_count = recursive_sample_plans.len();
        println!(
            "Avalanche recursive tier {} for batch {}: source-samples {} group-size {} configured-groups {} groups {} mode {}",
            next_tier_index,
            batch_number,
            source_sample_count,
            recursive_group_size,
            target_group_count,
            group_count,
            source_kind
        );
        if group_count < target_group_count {
            println!(
                "Avalanche recursive tier {} for batch {} capped unique groups from {} to {} because only that many distinct prior-tier input sets were produced",
                next_tier_index, batch_number, target_group_count, group_count
            );
        }
        let next_samples = recursive_sample_plans
            .into_par_iter()
            .enumerate()
            .map(|(group_index, source_sample_indices)| {
                let source_sample_ids = source_sample_indices
                    .iter()
                    .filter_map(|&index| {
                        current_tier_summaries.get(index).map(|summary| summary.id)
                    })
                    .collect::<Vec<_>>();
                recursive_evaluated_candidates
                    .fetch_add(source_sample_ids.len() as u64, Ordering::Relaxed);
                let sample = execute_recursive_avalanche_sample_from_cache_ids(
                    engine,
                    &reference_bits,
                    &comparison_message_bits,
                    cache,
                    &source_sample_ids,
                    next_tier_index,
                    group_index,
                )?;
                let done = recursive_done.fetch_add(1, Ordering::Relaxed) + 1;
                log_parallel_progress_every_interval(
                    done,
                    group_count as u64,
                    &recursive_log_start,
                    &recursive_next_log_at_ms,
                    &progress_label,
                    recursive_log_interval,
                );
                Ok::<_, String>(sample)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| -> Box<dyn Error> { err.into() })?;
        reduced.evaluated_candidates +=
            recursive_evaluated_candidates.load(Ordering::Relaxed) as usize;
        insert_cached_selected_samples(cache, batch_number, &next_samples, engine)?;
        if statistics_collection_enabled {
            reduced
                .tier_statistics
                .push(build_cached_avalanche_tier_statistics(
                    cache,
                    batch_number,
                    next_tier_index,
                    recursive_group_size,
                    source_kind,
                )?);
        }
        debug_assert_eq!(next_samples.len(), group_count);
        current_tier_summaries =
            load_cached_recursive_sample_summaries(cache, batch_number, next_tier_index)?;
        tier_index = next_tier_index;
    }

    let final_tier_total = count_cached_selected_samples(cache, batch_number, tier_index)?;
    let mut final_tier_offset_rows = 0i64;
    let mut final_tier_samples = Vec::with_capacity(final_tier_total);
    while final_tier_samples.len() < final_tier_total {
        let rows = load_cached_selected_sample_rows_page(
            cache,
            batch_number,
            tier_index,
            final_tier_offset_rows,
            cache.page_rows_i64(),
        )?;
        if rows.is_empty() {
            break;
        }
        final_tier_offset_rows +=
            i64::try_from(rows.len()).map_err(|_| "final tier page length exceeds i64 range")?;
        for row in rows {
            let sample = deserialize_selected_avalanche_sample_row(row)?;
            reduced.update_selected_sample_ref(&sample, prefer_beam_score_ordering);
            final_tier_samples.push(sample);
        }
    }
    reduced.final_tier_samples = final_tier_samples;
    Ok(reduced)
}

/// Scores one analysis batch and spills the resulting Avalanche tier-one inputs to SQLite in chunks.
///
/// # Parameters
/// - `ctx`: RSA context containing the public modulus and exponent.
/// - `engine`: Engine configuration controlling HBC behavior and detail retention.
/// - `cache`: Shared SQLite cache wrapper that owns the batch rows.
/// - `batch_number`: One-based analysis batch number.
/// - `batch_candidates`: Prepared `r` candidates participating in this batch.
/// - `shifted_ciphertexts`: Candidate ciphertext variants already shifted when required.
/// - `x_values`: Ciphertext exponents corresponding to `shifted_ciphertexts`.
/// - `e_x_values`: Optional `e * x` values used when ciphertext modification is enabled.
/// - `avalanche_message_bits`: Shifted reference bits used for match scoring.
/// - `additional_fitness_messages`: Extra random-message ciphertexts used for padding-bit fitness checks.
/// - `shift`: Whether ciphertexts should be shifted by encrypted `2` before candidate conversion.
/// - `batch_cx_total`: Total number of `c^x` candidates evaluated in the batch.
/// - `batch_cx_done`: Shared progress counter for `c^x` evaluation logs.
/// - `batch_cx_next_pct`: Shared percentage-step threshold for `c^x` evaluation logs.
/// - `batch_cx_started_at`: Progress start time for interval logging.
/// - `batch_cx_next_log_at_ms`: Shared interval threshold for `c^x` evaluation logs.
/// - `batch_cx_label`: Human-readable progress label for `c^x` evaluation logs.
/// - `keep_sample_details`: Whether to retain expensive sample detail payloads for analytics.
///
/// # Returns
/// - `Result<(usize, usize, Option<BatchCxMax>), Box<dyn Error>>`: `(candidate_count, cx_evaluated_candidates, cx_max)` for the batch.
///
/// # Expected Output
/// - Evaluates candidates in parallel chunks, writes scored inputs to SQLite in batches, and prints cache flush progress.
fn cache_batch_scored_avalanche_inputs(
    ctx: &RSAContext,
    engine: &EngineConfig,
    cache: &AvalancheCacheGuard,
    batch_number: usize,
    batch_candidates: &[AccuracyCandidate],
    shifted_ciphertexts: &[BigUint],
    x_values: &[BigUint],
    e_x_values: &[BigUint],
    avalanche_message_bits: &[bool],
    additional_fitness_messages: &[AdditionalFitnessMessage],
    shift: bool,
    batch_cx_total: u64,
    batch_cx_done: &AtomicU64,
    batch_cx_next_pct: &AtomicU64,
    batch_cx_started_at: &Instant,
    batch_cx_next_log_at_ms: &AtomicU64,
    batch_cx_label: &str,
    keep_sample_details: bool,
) -> Result<(usize, usize, Option<BatchCxMax>), Box<dyn Error>> {
    #[derive(Debug, Default)]
    struct CachedChunkAccumulator {
        candidate_count: usize,
        cx_max: Option<BatchCxMax>,
        cx_evaluated_candidates: usize,
        ranked_samples: Vec<RankedScoredAvalancheInput>,
        estimated_bytes: usize,
    }

    impl CachedChunkAccumulator {
        fn set_cx_max(&mut self, candidate: BatchCxMax) {
            let replace = match self.cx_max.as_ref() {
                Some(current) => {
                    candidate.match_pct > current.match_pct
                        || (candidate.match_pct == current.match_pct
                            && candidate.batch_candidate_index < current.batch_candidate_index)
                }
                None => true,
            };
            if replace {
                self.cx_max = Some(candidate);
            }
        }

        fn merge(mut self, mut other: Self) -> Self {
            self.candidate_count += other.candidate_count;
            self.cx_evaluated_candidates += other.cx_evaluated_candidates;
            self.estimated_bytes += other.estimated_bytes;
            if let Some(candidate) = other.cx_max.take() {
                self.set_cx_max(candidate);
            }
            self.ranked_samples.append(&mut other.ranked_samples);
            self
        }
    }

    let chunk_size = rayon::current_num_threads().saturating_mul(8).max(1);
    let mut total_candidate_count = 0usize;
    let mut total_cx_evaluated_candidates = 0usize;
    let mut batch_cx_max: Option<BatchCxMax> = None;
    let mut streaming_fitness_pool = (engine.avalanche_fitness_scoring_pass
        && engine.avalanche_fitness_streaming_prune)
        .then(|| {
            StreamingScoredAvalancheFitnessPool::new(
                resolve_avalanche_fitness_bit_width(engine),
                engine.avalanche_fitness_r_candidate_limit,
                engine.avalanche_fitness_cx_candidate_limit,
                engine.avalanche_fitness_use_threshold,
                engine.avalanche_fitness_threshold,
            )
        });
    let mut pending_rows = Vec::new();
    let mut pending_bytes = 0usize;
    let mut next_flush_threshold = AVALANCHE_CACHE_FLUSH_BYTES;

    for (chunk_offset, candidate_chunk) in batch_candidates.chunks(chunk_size).enumerate() {
        let global_offset = chunk_offset.saturating_mul(chunk_size);
        let mut chunk_aggregate = candidate_chunk
            .par_iter()
            .enumerate()
            .try_fold(
                CachedChunkAccumulator::default,
                |mut acc, (local_index, candidate)| {
                    let batch_candidate_index = global_offset + local_index;
                    let mut cx_max = None;
                    let mut cx_evaluated_candidates = 0usize;
                    let mut ranked_samples = Vec::with_capacity(shifted_ciphertexts.len());
                    let target_exponent =
                        keep_sample_details.then(|| candidate.target_exponent.normalized());

                    for idx in 0..shifted_ciphertexts.len() {
                        let shifted = &shifted_ciphertexts[idx];
                        let hbc_result =
                            prepare_candidate_ciphertext(engine, shifted, &candidate.r, &ctx.n);
                        let x_value = x_values.get(idx).cloned().ok_or_else(|| {
                            "missing ciphertext exponent for message index".to_string()
                        })?;
                        let d_new_owned = if engine.ciphertext_modify {
                            let e_x = e_x_values.get(idx).ok_or_else(|| {
                                "missing ciphertext exponent for message index".to_string()
                            })?;
                            Some(mod_inverse(e_x, &candidate.phi_new).ok_or_else(|| {
                                format!("analysis_batch missing modular inverse for x {}", x_value)
                            })?)
                        } else {
                            None
                        };
                        let d_new = d_new_owned.as_ref().unwrap_or(&candidate.d_new);
                        let dm = derive_candidate_message_from_result(
                            ctx,
                            engine,
                            &hbc_result,
                            &candidate.r,
                            d_new,
                        );
                        let (dm_bits, match_pct) =
                            truncated_match_percentage(&dm, avalanche_message_bits);
                        cx_evaluated_candidates += 1;
                        if cx_max
                            .as_ref()
                            .is_none_or(|current: &BatchCxMax| match_pct > current.match_pct)
                        {
                            cx_max = Some(BatchCxMax {
                                match_pct,
                                x: x_value.clone(),
                                r: candidate.r.clone(),
                                batch_candidate_index,
                            });
                        }

                        let message_bits = PackedBits::from_bools(&dm_bits);
                        let scored_input = ScoredAvalancheInput {
                            batch_candidate_index,
                            message_index: idx,
                            r: candidate.r.clone(),
                            x: x_value,
                            score_match_pct: match_pct,
                            message_bits: message_bits.clone(),
                            detail: target_exponent.as_ref().map(|target_exponent| {
                                ScoredAvalancheInputDetail {
                                    target_exponent: target_exponent.clone(),
                                    hbc_ciphertext_r: hbc_result.clone(),
                                    candidate_decryption: dm.clone(),
                                }
                            }),
                        };
                        let fitness = compute_padding_fitness_score(
                            ctx,
                            engine,
                            &candidate.r,
                            d_new,
                            &scored_input.x,
                            &message_bits,
                            additional_fitness_messages,
                            shift,
                        );
                        acc.estimated_bytes +=
                            approximate_scored_avalanche_input_bytes(&scored_input);
                        ranked_samples.push(RankedScoredAvalancheInput {
                            input: scored_input,
                            fitness,
                        });
                        let done = batch_cx_done.fetch_add(1, Ordering::Relaxed) + 1;
                        log_parallel_progress_every_ten_percent(
                            done,
                            batch_cx_total,
                            batch_cx_next_pct,
                            batch_cx_label,
                        );
                        log_parallel_progress_every_interval(
                            done,
                            batch_cx_total,
                            batch_cx_started_at,
                            batch_cx_next_log_at_ms,
                            batch_cx_label,
                            Duration::from_secs(5),
                        );
                    }

                    acc.candidate_count += 1;
                    acc.cx_evaluated_candidates += cx_evaluated_candidates;
                    if let Some(candidate) = cx_max {
                        acc.set_cx_max(candidate);
                    }
                    acc.ranked_samples.extend(ranked_samples);
                    Ok::<_, String>(acc)
                },
            )
            .try_reduce(CachedChunkAccumulator::default, |left, right| {
                Ok::<_, String>(left.merge(right))
            })
            .map_err(|err| -> Box<dyn Error> { err.into() })?;

        total_candidate_count += chunk_aggregate.candidate_count;
        total_cx_evaluated_candidates += chunk_aggregate.cx_evaluated_candidates;
        if let Some(candidate) = chunk_aggregate.cx_max.take() {
            let replace = match batch_cx_max.as_ref() {
                Some(current) => candidate.match_pct > current.match_pct,
                None => true,
            };
            if replace {
                batch_cx_max = Some(candidate);
            }
        }
        pending_bytes += chunk_aggregate.estimated_bytes;
        if let Some(pool) = streaming_fitness_pool.as_mut() {
            pool.extend_with_scores(std::mem::take(&mut chunk_aggregate.ranked_samples));
        } else {
            pending_rows.append(&mut chunk_aggregate.ranked_samples);
            if pending_bytes >= next_flush_threshold {
                insert_cached_scored_inputs(cache, batch_number, &pending_rows)?;
                println!(
                    "Avalanche cache flush for batch {}: wrote {} scored inputs at approximately {:.2} GiB pending",
                    batch_number,
                    pending_rows.len(),
                    pending_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                pending_rows.clear();
                pending_bytes = 0;
                next_flush_threshold = AVALANCHE_CACHE_FLUSH_BYTES;
            }
        }
    }

    if let Some(pool) = streaming_fitness_pool.take() {
        let retained_inputs = pool.finalize(engine.avalanche_unique_r_cx_inputs);
        if !retained_inputs.is_empty() {
            insert_cached_scored_inputs(cache, batch_number, &retained_inputs)?;
        }
        println!(
            "Avalanche cache streaming prune for batch {}: wrote final retained pool of {} scored inputs",
            batch_number,
            retained_inputs.len()
        );
    } else if !pending_rows.is_empty() {
        insert_cached_scored_inputs(cache, batch_number, &pending_rows)?;
        println!(
            "Avalanche cache flush for batch {}: wrote final {} scored inputs at approximately {:.2} GiB pending",
            batch_number,
            pending_rows.len(),
            pending_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    Ok((
        total_candidate_count,
        total_cx_evaluated_candidates,
        batch_cx_max,
    ))
}

/// Runs sampled avalanche combinations over the scored batch outputs.
///
/// # Parameters
/// - `engine`: Engine configuration controlling combination sampling and beam scoring.
/// - `message`: Original plaintext payload used for scoring and to derive the widened reference.
/// - `scored_inputs`: Scored candidate decryptions available for sampling.
/// - `batch_number`: One-based batch index used for progress logging.
/// - `prefer_beam_score_ordering`: Whether selection should prefer beam score over known match percentage.
/// - `rng`: Random number generator for combination sampling.
///
/// # Returns
/// - `Result<SampledAvalancheBatchResult, Box<dyn Error>>`: Sample logs plus the selected best sample.
///
/// # Expected Output
/// - Prints sampled-avalanche progress and returns sampled avalanche results.
#[cfg_attr(not(test), allow(dead_code))]
fn run_sampled_avalanche_beam_search(
    engine: &EngineConfig,
    payload_message: &BigUint,
    scored_inputs: &[ScoredAvalancheInput],
    batch_number: usize,
    prefer_beam_score_ordering: bool,
    rng: &mut RngChoice,
) -> Result<SampledAvalancheBatchResult, Box<dyn Error>> {
    run_sampled_avalanche_beam_search_with_ranked_inputs(
        engine,
        payload_message,
        scored_inputs,
        None,
        batch_number,
        prefer_beam_score_ordering,
        rng,
    )
}

/// Runs sampled avalanche combinations over the scored batch outputs with optional pre-ranked
/// fitness metadata.
///
/// # Parameters
/// - `engine`: Engine configuration controlling combination sampling and beam scoring.
/// - `payload_message`: Original plaintext payload used for scoring and to derive the widened reference.
/// - `scored_inputs`: Scored candidate decryptions available for sampling.
/// - `ranked_inputs`: Optional scored inputs plus precomputed padding-bit fitness metrics.
/// - `batch_number`: One-based batch index used for progress logging.
/// - `prefer_beam_score_ordering`: Whether selection should prefer beam score over known match percentage.
/// - `rng`: Random number generator for combination sampling.
///
/// # Returns
/// - `Result<SampledAvalancheBatchResult, Box<dyn Error>>`: Sample logs plus the selected best sample.
///
/// # Expected Output
/// - Prints sampled-avalanche progress and returns sampled avalanche results.
fn run_sampled_avalanche_beam_search_with_ranked_inputs(
    engine: &EngineConfig,
    payload_message: &BigUint,
    scored_inputs: &[ScoredAvalancheInput],
    ranked_inputs: Option<&[RankedScoredAvalancheInput]>,
    batch_number: usize,
    prefer_beam_score_ordering: bool,
    rng: &mut RngChoice,
) -> Result<SampledAvalancheBatchResult, Box<dyn Error>> {
    if engine.avalanche_combination_samples == 0 {
        return Err("avalanche_combination_samples must be >= 1".into());
    }
    if !engine.avalanche_random_chacha20_inputs
        && engine.avalanche_combination_mixed_r_candidates == 0
    {
        return Err("avalanche_combination_mixed_r_candidates must be >= 1".into());
    }
    if engine.avalanche_combination_size == 0 {
        return Err("avalanche_combination_size must be >= 1".into());
    }
    if engine.avalanche_combination_hamming_distance_prune
        && !(0.0 < engine.avalanche_combination_hamming_distance_keep_percentile
            && engine.avalanche_combination_hamming_distance_keep_percentile <= 100.0)
    {
        return Err(
            "avalanche_combination_hamming_distance_keep_percentile must be in (0, 100]".into(),
        );
    }
    if !(0.0..=100.0)
        .contains(&engine.avalanche_combination_hamming_distance_outlier_preference_pct)
    {
        return Err(
            "avalanche_combination_hamming_distance_outlier_preference_pct must be in [0, 100]"
                .into(),
        );
    }
    validate_avalanche_fitness_threshold(engine)?;
    validate_avalanche_fitness_log_top_pct(engine)?;
    validate_avalanche_center_threshold(engine)?;
    if scored_inputs.is_empty() {
        return Ok(SampledAvalancheBatchResult::default());
    }

    let fitness_bit_width = resolve_avalanche_fitness_bit_width(engine);
    let mut scored_inputs = if engine.avalanche_fitness_scoring_pass {
        let preprocessed_ranked_inputs = if let Some(ranked_inputs) = ranked_inputs {
            apply_ranked_scored_avalanche_fitness_pass(
                ranked_inputs.to_vec(),
                fitness_bit_width,
                engine.avalanche_fitness_r_candidate_limit,
                engine.avalanche_fitness_cx_candidate_limit,
                engine.avalanche_fitness_use_threshold,
                engine.avalanche_fitness_threshold,
            )
        } else if let Some(pass) = build_scored_avalanche_fitness_pass(engine) {
            let preprocessed = pass(scored_inputs.to_vec());
            preprocessed
                .into_iter()
                .map(|input| RankedScoredAvalancheInput {
                    fitness: single_message_avalanche_fitness_score(
                        &input.message_bits,
                        fitness_bit_width,
                    ),
                    input,
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let preprocessed = preprocessed_ranked_inputs
            .iter()
            .map(|input| input.input.clone())
            .collect::<Vec<_>>();
        println!(
            "Avalanche fitness pass for batch {}: retained {} of {} scored inputs in a globally ranked pool using {} LSB fitness bits with effective retained-input cap {} (r-limit {} cx-limit {}, threshold-enabled {} threshold {}, additional-random-messages {})",
            batch_number,
            preprocessed.len(),
            scored_inputs.len(),
            fitness_bit_width,
            resolve_avalanche_fitness_retained_input_limit(
                engine.avalanche_fitness_r_candidate_limit,
                engine.avalanche_fitness_cx_candidate_limit,
            ),
            engine.avalanche_fitness_r_candidate_limit,
            engine.avalanche_fitness_cx_candidate_limit,
            if engine.avalanche_fitness_use_threshold {
                "on"
            } else {
                "off"
            },
            format_beam_float(engine.avalanche_fitness_threshold, 3),
            engine.avalanche_fitness_additional_random_messages
        );
        log_top_scored_avalanche_fitness_entries(
            batch_number,
            &preprocessed_ranked_inputs,
            fitness_bit_width,
            engine.avalanche_fitness_log_top_pct,
        );
        preprocessed
    } else {
        scored_inputs.to_vec()
    };
    if engine.avalanche_unique_r_cx_inputs {
        if !engine.avalanche_fitness_scoring_pass {
            scored_inputs.par_sort_unstable_by(|left, right| {
                right
                    .score_match_pct
                    .total_cmp(&left.score_match_pct)
                    .then_with(|| left.batch_candidate_index.cmp(&right.batch_candidate_index))
                    .then_with(|| left.message_index.cmp(&right.message_index))
                    .then_with(|| left.x.cmp(&right.x))
            });
        }
        let original_count = scored_inputs.len();
        let (unique_inputs, rejected_overlap_count) =
            enforce_global_unique_scored_inputs(scored_inputs);
        scored_inputs = unique_inputs;
        println!(
            "Avalanche unique-input filter for batch {}: retained {} of {} scored inputs after dropping {} overlapping r/x candidates",
            batch_number,
            scored_inputs.len(),
            original_count,
            rejected_overlap_count
        );
    }
    if scored_inputs.is_empty() {
        return Err(
            "avalanche_fitness_threshold removed all scored inputs for sampled avalanche".into(),
        );
    }

    let comparison_message_bits =
        biguint_to_bits_le(payload_message, resolve_plaintext_message_bit_width(engine));
    let transformed_message = build_candidate_message_transform(engine)(payload_message);
    let reference_bits =
        biguint_to_bits_le(&transformed_message, scored_inputs[0].message_bits.len());
    let packed_message_bits = PackedBits::from_bools(&reference_bits);
    let hamming_prune_label = format!("Accuracy batch {} Hamming prune", batch_number);
    let pruned_pool = if engine.avalanche_combination_hamming_distance_prune {
        prune_scored_inputs_by_hamming_distance_percentile_with_progress(
            &scored_inputs,
            &packed_message_bits,
            engine.avalanche_combination_hamming_distance_keep_percentile,
            engine.avalanche_combination_hamming_distance_outlier_preference_pct,
            Some(&hamming_prune_label),
        )
    } else {
        HammingDistancePrunedPool {
            selected_inputs: scored_inputs.clone(),
            retained_inlier_count: scored_inputs.len(),
            available_outlier_count: 0,
            preferred_outlier_count: 0,
        }
    };
    let retained_inlier_count = pruned_pool.retained_inlier_count;
    let available_outlier_count = pruned_pool.available_outlier_count;
    let preferred_outlier_count = pruned_pool.preferred_outlier_count;
    let pruned_scored_inputs = pruned_pool.selected_inputs;
    let grouping_label = format!("Accuracy batch {} avalanche input grouping", batch_number);
    let grouped_inputs = group_scored_inputs_by_r_candidate_with_progress(
        &pruned_scored_inputs,
        Some(&grouping_label),
    );
    let pool_size = pruned_scored_inputs.len();
    let r_candidate_pool_size = grouped_inputs.len();
    if r_candidate_pool_size == 0 {
        return Ok(SampledAvalancheBatchResult::default());
    }
    let mixed_r_candidate_count = if engine.avalanche_random_chacha20_inputs {
        0
    } else {
        engine
            .avalanche_combination_mixed_r_candidates
            .min(engine.avalanche_combination_size)
            .min(r_candidate_pool_size)
    };

    let sample_count = engine.avalanche_combination_samples as usize;
    let recursion_depth = engine.avalanche_combination_recursion_depth.max(1);
    let majority_vote_enabled = engine.avalanche_combination_majority_vote;
    let sample_smoothing_enabled = engine.avalanche_combination_sample_smoothing;
    let majority_vote_print_enabled = engine.avalanche_combination_majority_vote_print;
    let recursive_input_mode = if engine.avalanche_use_top_beam {
        "top-beam"
    } else {
        "majority-vote"
    };
    let statistics_collection_enabled = engine.avalanche_statistics_collection;
    let keep_all_samples_enabled =
        statistics_collection_enabled && engine.avalanche_combination_keep_all_samples_in_memory;
    let selection_mode = if engine.avalanche_random_chacha20_inputs {
        "random-chacha20-inputs"
    } else {
        "mixed-r-combinations"
    };
    let sample_plans = if engine.avalanche_random_chacha20_inputs {
        build_unique_flat_sample_plans(
            &pruned_scored_inputs,
            engine.avalanche_combination_size,
            sample_count,
            rng,
        )
    } else {
        let grouped_sample_inputs = grouped_inputs
            .iter()
            .map(|group| group.inputs.clone())
            .collect::<Vec<_>>();
        build_unique_grouped_sample_plans(
            &grouped_sample_inputs,
            mixed_r_candidate_count,
            engine.avalanche_combination_size,
            sample_count,
            rng,
        )
    };
    let effective_sample_count = sample_plans.len();

    println!(
        "Avalanche combination setup for batch {}: scored inputs {} r-candidate-pool {} selection-mode {} configured-mixed-r-candidates {} effective-mixed-r-candidates {} configured-samples {} effective-samples {} recursion-depth {} recursive-group-sizes {:?} recursive-resample-counts {:?} majority-vote {} sample-smoothing {} majority-print {} recursive-input {} statistics-collection {} keep-all-samples {} hamming-prune {} kept-percentile {} outlier-preference-pct {}",
        batch_number,
        pool_size,
        r_candidate_pool_size,
        selection_mode,
        engine.avalanche_combination_mixed_r_candidates,
        mixed_r_candidate_count,
        sample_count,
        effective_sample_count,
        recursion_depth,
        &engine.avalanche_combination_recursive_group_size,
        &engine.avalanche_combination_recursive_resample_count,
        if majority_vote_enabled { "on" } else { "off" },
        if sample_smoothing_enabled {
            "on"
        } else {
            "off"
        },
        if majority_vote_print_enabled {
            "on"
        } else {
            "off"
        },
        recursive_input_mode,
        if statistics_collection_enabled {
            "on"
        } else {
            "off"
        },
        if keep_all_samples_enabled {
            "on"
        } else {
            "off"
        },
        if engine.avalanche_combination_hamming_distance_prune {
            "on"
        } else {
            "off"
        },
        engine.avalanche_combination_hamming_distance_keep_percentile,
        engine.avalanche_combination_hamming_distance_outlier_preference_pct
    );
    if engine.avalanche_combination_hamming_distance_prune && pool_size < scored_inputs.len() {
        println!(
            "Avalanche combination batch {} pruned scored inputs by Hamming distance from {} to {} before sampling (retained-inliers {} available-outliers {} preferred-outliers {})",
            batch_number,
            scored_inputs.len(),
            pool_size,
            retained_inlier_count,
            available_outlier_count,
            preferred_outlier_count
        );
    }
    if !engine.avalanche_random_chacha20_inputs
        && mixed_r_candidate_count < engine.avalanche_combination_mixed_r_candidates
    {
        println!(
            "Avalanche combination batch {} capped mixed r-candidates from {} to {} because only {} distinct r candidates were available in the batch",
            batch_number,
            engine.avalanche_combination_mixed_r_candidates,
            mixed_r_candidate_count,
            r_candidate_pool_size
        );
    }
    if effective_sample_count < sample_count {
        println!(
            "Avalanche combination batch {} capped unique sampled combinations from {} to {} because only that many distinct tier-one input sets were produced",
            batch_number, sample_count, effective_sample_count
        );
    }
    if effective_sample_count == 0 {
        return Ok(SampledAvalancheBatchResult::default());
    }

    let sample_label = format!("Avalanche sample batch {}", batch_number);
    let sample_done = AtomicU64::new(0);
    let sample_log_start = Instant::now();
    let sample_log_interval = Duration::from_secs(5);
    let sample_next_log_at_ms =
        AtomicU64::new(sample_log_interval.as_millis().min(u128::from(u64::MAX)) as u64);
    let mut base_outcomes = sample_plans
        .into_par_iter()
        .enumerate()
        .map(|(sample_index, selected_inputs)| {
            let outcome = execute_sampled_avalanche_sample(
                engine,
                &reference_bits,
                &comparison_message_bits,
                selected_inputs,
                pool_size,
                r_candidate_pool_size,
                1,
                sample_index,
            )?;
            let done = sample_done.fetch_add(1, Ordering::Relaxed) + 1;
            log_parallel_progress_every_interval(
                done,
                effective_sample_count as u64,
                &sample_log_start,
                &sample_next_log_at_ms,
                &sample_label,
                sample_log_interval,
            );
            Ok::<_, String>(outcome)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| -> Box<dyn Error> { err.into() })?;

    let mut reduced = SampledAvalancheBatchResult::default();
    reduced.sample_count = base_outcomes
        .iter()
        .filter(|outcome| outcome.produced_sample)
        .count();
    reduced.evaluated_candidates = base_outcomes
        .iter()
        .map(|outcome| outcome.evaluated_candidates)
        .sum();
    reduced.retained_samples = base_outcomes
        .iter_mut()
        .filter_map(|outcome| outcome.retained_sample.take())
        .collect();

    let mut current_tier_samples = base_outcomes
        .iter_mut()
        .filter_map(|outcome| outcome.sample.take())
        .collect::<Vec<_>>();
    let will_recurse = recursion_depth > 1 && current_tier_samples.len() > 1;
    if will_recurse {
        compact_retained_avalanche_sample_inputs(&mut reduced.retained_samples);
    }
    drop(base_outcomes);
    drop(grouped_inputs);
    drop(pruned_scored_inputs);
    if statistics_collection_enabled {
        reduced
            .tier_statistics
            .push(build_avalanche_tier_statistics(
                1,
                engine.avalanche_combination_size,
                selection_mode,
                &current_tier_samples,
            ));
    }

    let mut tier_index = 1usize;
    while tier_index < recursion_depth && current_tier_samples.len() > 1 {
        let next_tier_index = tier_index + 1;
        let recursive_tier_config = resolve_recursive_avalanche_tier_config(engine, tier_index);
        let recursive_group_size = recursive_tier_config.group_size;
        let recursive_resample_count = recursive_tier_config.resample_count;
        let source_samples = compact_recursive_avalanche_source_samples(
            std::mem::take(&mut current_tier_samples),
            engine,
        );
        let source_sample_count = source_samples.len();
        let recursive_done = AtomicU64::new(0);
        let recursive_evaluated_candidates = AtomicU64::new(0);
        let recursive_log_start = Instant::now();
        let recursive_log_interval = Duration::from_secs(5);
        let recursive_next_log_at_ms =
            AtomicU64::new(recursive_log_interval.as_millis().min(u128::from(u64::MAX)) as u64);
        let progress_label = format!(
            "Avalanche recursive tier {} batch {}",
            next_tier_index, batch_number
        );
        let target_group_count = if recursive_resample_count > 0 {
            recursive_resample_count
        } else {
            source_sample_count.div_ceil(recursive_group_size)
        };
        let source_sample_indices = (0..source_sample_count).collect::<Vec<_>>();
        let recursive_sample_plans = build_unique_flat_sample_plans(
            &source_sample_indices,
            recursive_group_size,
            target_group_count,
            rng,
        );
        let source_kind = if recursive_resample_count > 0 {
            "recursive-resampled-samples"
        } else {
            "recursive-samples"
        };
        let group_count = recursive_sample_plans.len();
        println!(
            "Avalanche recursive tier {} for batch {}: source-samples {} group-size {} configured-groups {} groups {} mode {}",
            next_tier_index,
            batch_number,
            source_sample_count,
            recursive_group_size,
            target_group_count,
            group_count,
            source_kind
        );
        if group_count < target_group_count {
            println!(
                "Avalanche recursive tier {} for batch {} capped unique groups from {} to {} because only that many distinct prior-tier input sets were produced",
                next_tier_index, batch_number, target_group_count, group_count
            );
        }
        let next_samples = recursive_sample_plans
            .into_par_iter()
            .enumerate()
            .map(|(group_index, source_sample_indices)| {
                recursive_evaluated_candidates
                    .fetch_add(source_sample_indices.len() as u64, Ordering::Relaxed);
                let sample = execute_recursive_avalanche_sample_from_indices(
                    engine,
                    &reference_bits,
                    &comparison_message_bits,
                    &source_samples,
                    &source_sample_indices,
                    next_tier_index,
                    group_index,
                )?;
                let done = recursive_done.fetch_add(1, Ordering::Relaxed) + 1;
                log_parallel_progress_every_interval(
                    done,
                    group_count as u64,
                    &recursive_log_start,
                    &recursive_next_log_at_ms,
                    &progress_label,
                    recursive_log_interval,
                );
                Ok::<_, String>(sample)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| -> Box<dyn Error> { err.into() })?;
        reduced.evaluated_candidates +=
            recursive_evaluated_candidates.load(Ordering::Relaxed) as usize;
        drop(source_samples);

        if statistics_collection_enabled {
            reduced
                .tier_statistics
                .push(build_avalanche_tier_statistics(
                    next_tier_index,
                    recursive_group_size,
                    source_kind,
                    &next_samples,
                ));
        }
        debug_assert_eq!(next_samples.len(), group_count);
        current_tier_samples = next_samples;
        tier_index = next_tier_index;
    }

    for sample in &current_tier_samples {
        reduced.update_selected_sample_ref(sample, prefer_beam_score_ordering);
    }
    reduced.final_tier_samples = current_tier_samples;
    Ok(reduced)
}

/// Runs r-candidate accuracy batches with one random message per batch.
///
/// # Parameters
/// - `ctx`: RSA context containing key material.
/// - `engine`: Engine configuration controlling candidate and message settings.
/// - `rng`: Random number generator for candidate and message sampling.
/// - `analytics`: Session analytics accumulator receiving batch data.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
/// - `prefer_beam_score_ordering`: Whether public-key-only analysis should rank selected results by beam score.
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error on invalid configuration.
///
/// # Expected Output
/// - Prints batch-progress summaries and appends accuracy plus avalanche sample data to the
///   session analytics, truncating avalanche inputs to `engine.message.bits`.
fn run_r_candidate_accuracy_batches(
    ctx: &RSAContext,
    engine: &EngineConfig,
    rng: &mut RngChoice,
    analytics: &Arc<Mutex<SessionAnalytics>>,
    shift: bool,
    prefer_beam_score_ordering: bool,
    avalanche_cache: Option<&AvalancheCacheGuard>,
) -> Result<(), Box<dyn Error>> {
    if !engine.analysis_batch_enable {
        return Ok(());
    }

    let message_count_raw = engine.analysis_batch_messages;
    let candidates_per_batch_raw = engine.analysis_batch_candidates;
    let batch_count_raw = engine.analysis_batch_batches;
    if message_count_raw == 0 || (!engine.same_r_batch && candidates_per_batch_raw == 0) {
        return Err("analysis_batch_messages and analysis_batch_candidates must be >= 1".into());
    }
    if batch_count_raw == 0 {
        return Err("analysis_batch_batches must be >= 1".into());
    }

    let message_count = message_count_raw as usize;
    let candidates_per_batch = if engine.same_r_batch {
        1usize
    } else {
        candidates_per_batch_raw as usize
    };
    let batch_count = batch_count_raw as usize;

    let total_candidates = candidates_per_batch * batch_count;
    println!(
        "Starting r-candidate accuracy batches: batches {} candidates-per-batch {} messages-per-batch {} avalanche-samples {} configured-combination-size {} configured-mixed-r-candidates {} same-r-batch {} pool-source full-batch majority-vote {} sample-smoothing {} majority-print {} statistics-collection {} keep-all-samples {}",
        batch_count,
        candidates_per_batch,
        message_count,
        engine.avalanche_combination_samples,
        engine.avalanche_combination_size,
        engine.avalanche_combination_mixed_r_candidates,
        if engine.same_r_batch { "on" } else { "off" },
        if engine.avalanche_combination_majority_vote {
            "on"
        } else {
            "off"
        },
        if engine.avalanche_combination_sample_smoothing {
            "on"
        } else {
            "off"
        },
        if engine.avalanche_combination_majority_vote_print {
            "on"
        } else {
            "off"
        },
        if engine.avalanche_statistics_collection {
            "on"
        } else {
            "off"
        },
        if engine.avalanche_statistics_collection
            && engine.avalanche_combination_keep_all_samples_in_memory
        {
            "on"
        } else {
            "off"
        },
    );
    let settings = build_r_candidate_settings(engine, ctx.key_bit_width);
    let candidates = generate_r_candidates_with_analytics(
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
    let e_big = ctx.e.clone();
    let prepared_total =
        u64::try_from(candidates.len()).map_err(|_| "candidate count exceeds u64 range")?;
    let prepared_started_at = Instant::now();
    let prepared_done = AtomicU64::new(0);
    let prepared_next_log_at_ms =
        AtomicU64::new(Duration::from_secs(5).as_millis().min(u128::from(u64::MAX)) as u64);
    println!(
        "Preparing {} retargeted r candidates for accuracy batch scoring",
        candidates.len()
    );
    let prepared = candidates
        .into_par_iter()
        .filter_map(|candidate| {
            let phi_new = compute_totient(&candidate.factors);
            let prepared_candidate = mod_inverse(&e_big, &phi_new).map(|d_new| AccuracyCandidate {
                r: candidate.r,
                phi_new,
                d_new,
                target_exponent: candidate.target_exponent,
            });
            let done = prepared_done.fetch_add(1, Ordering::Relaxed) + 1;
            log_parallel_progress_every_interval(
                done,
                prepared_total,
                &prepared_started_at,
                &prepared_next_log_at_ms,
                "Accuracy batch candidate preparation",
                Duration::from_secs(5),
            );
            prepared_candidate
        })
        .collect::<Vec<_>>();

    if prepared.len() < total_candidates {
        return Err(format!(
            "only {} valid r candidates available for accuracy batches (need {})",
            prepared.len(),
            total_candidates
        )
        .into());
    }

    let mut candidate_offset = 0usize;
    let mut beam_max: Option<BeamMaxCandidate> = None;
    let mut majority_vote_max: Option<MajorityVoteMaxCandidate> = None;
    let mut total_avalanche_evaluated_candidates = 0usize;
    let mut cx_run_max: Option<CxMatchCandidate> = None;
    let mut total_cx_evaluated_candidates = 0usize;
    let mut batch_top_match_percentages = Vec::new();
    let mut next_batch_pct = 10u64;
    for batch_idx in 0..batch_count {
        let batch_number = batch_idx + 1;
        let start = candidate_offset;
        let end = candidate_offset + candidates_per_batch;
        let batch_candidates = &prepared[start..end];
        candidate_offset = end;
        println!(
            "Accuracy batch {} of {}: scoring {} r candidates across {} ciphertext variants",
            batch_number,
            batch_count,
            batch_candidates.len(),
            message_count
        );

        let message = if engine.message.is_random {
            random_message_under_n(engine, &ctx.n, rng)?
        } else {
            let msg = BigUint::from_bytes_be(engine.message.fixed_message.as_bytes());
            if msg.is_zero() {
                return Err("analysis_batch fixed_message cannot be empty".into());
            }
            transform_message_for_candidate_scoring(engine, &msg, &ctx.n, "analysis_batch")?;
            msg
        };
        let avalanche_message =
            transform_message_for_candidate_scoring(engine, &message, &ctx.n, "analysis_batch")?;
        let messages = vec![message.clone(); message_count];
        let base_ciphertext = avalanche_message.modpow(&ctx.e, &ctx.n);
        let mut ciphertexts = Vec::with_capacity(message_count);
        let mut shifted_ciphertexts = Vec::with_capacity(message_count);
        let mut e_x_values = Vec::with_capacity(message_count);
        let mut x_values = Vec::with_capacity(message_count);

        if engine.ciphertext_modify {
            let phi_values: Vec<&BigUint> = batch_candidates
                .iter()
                .map(|candidate| &candidate.phi_new)
                .collect();
            let ciphertext_variants = collect_invertible_ciphertext_variants(
                ctx,
                &base_ciphertext,
                &phi_values,
                message_count,
                shift,
                "analysis_batch",
            )?;
            for variant in ciphertext_variants {
                e_x_values.push(variant.e_x);
                x_values.push(variant.x);
                ciphertexts.push(variant.ciphertext);
                shifted_ciphertexts.push(variant.shifted);
            }
        } else {
            let shifted = maybe_shift_ciphertext(ctx, &base_ciphertext, shift);
            for _ in 0..message_count {
                x_values.push(BigUint::one());
                ciphertexts.push(base_ciphertext.clone());
                shifted_ciphertexts.push(shifted.clone());
            }
        }

        let avalanche_bit_width = resolve_avalanche_bit_width(engine);
        let avalanche_message_bits = biguint_to_bits_le(&avalanche_message, avalanche_bit_width);
        let additional_fitness_messages = build_additional_fitness_messages(
            ctx,
            engine,
            engine.avalanche_fitness_additional_random_messages,
            rng,
        )?;
        if !additional_fitness_messages.is_empty() {
            println!(
                "Accuracy batch {} fitness scoring: testing each c^x/r candidate across {} additional random message(s) for padding-bit consistency",
                batch_number,
                additional_fitness_messages.len()
            );
        }
        let batch_cx_total = u64::try_from(batch_candidates.len())
            .map_err(|_| "batch candidate count exceeds u64 range")?
            .checked_mul(
                u64::try_from(message_count).map_err(|_| "message count exceeds u64 range")?,
            )
            .ok_or("c^x progress total overflowed u64")?;
        let batch_cx_done = AtomicU64::new(0);
        let batch_cx_next_pct = AtomicU64::new(10);
        let batch_cx_started_at = Instant::now();
        let batch_cx_next_log_at_ms =
            AtomicU64::new(Duration::from_secs(5).as_millis().min(u128::from(u64::MAX)) as u64);
        let batch_cx_label = format!("Accuracy batch {} c^x candidates", batch_number);
        let keep_sample_details = engine.avalanche_statistics_collection
            && engine.avalanche_combination_keep_all_samples_in_memory;
        if let Some(cache) = avalanche_cache {
            cache.clear_batch(batch_number)?;
        }
        let (
            batch_candidate_count,
            batch_cx_evaluated_candidates,
            mut batch_cx_max,
            sampled_avalanche_result,
        ) = if let Some(cache) = avalanche_cache {
            let (candidate_count, cx_evaluated_candidates, cx_max) =
                cache_batch_scored_avalanche_inputs(
                    ctx,
                    engine,
                    cache,
                    batch_number,
                    batch_candidates,
                    &shifted_ciphertexts,
                    &x_values,
                    &e_x_values,
                    &avalanche_message_bits,
                    &additional_fitness_messages,
                    shift,
                    batch_cx_total,
                    &batch_cx_done,
                    &batch_cx_next_pct,
                    &batch_cx_started_at,
                    &batch_cx_next_log_at_ms,
                    &batch_cx_label,
                    keep_sample_details,
                )?;
            let sampled = run_sampled_avalanche_beam_search_cached(
                engine,
                &message,
                cache,
                batch_number,
                prefer_beam_score_ordering,
                rng,
            )?;
            (candidate_count, cx_evaluated_candidates, cx_max, sampled)
        } else {
            let chunk_size = rayon::current_num_threads().saturating_mul(8).max(1);
            let mut batch_aggregate = AccuracyBatchAccumulator::default();
            let mut streaming_fitness_pool = (engine.avalanche_fitness_scoring_pass
                && engine.avalanche_fitness_streaming_prune)
                .then(|| {
                    StreamingScoredAvalancheFitnessPool::new(
                        resolve_avalanche_fitness_bit_width(engine),
                        engine.avalanche_fitness_r_candidate_limit,
                        engine.avalanche_fitness_cx_candidate_limit,
                        engine.avalanche_fitness_use_threshold,
                        engine.avalanche_fitness_threshold,
                    )
                });

            for (chunk_offset, candidate_chunk) in batch_candidates.chunks(chunk_size).enumerate() {
                let global_offset = chunk_offset.saturating_mul(chunk_size);
                let mut chunk_aggregate = candidate_chunk
                    .par_iter()
                    .enumerate()
                    .try_fold(
                        AccuracyBatchAccumulator::default,
                        |mut acc, (local_index, candidate)| {
                            let batch_candidate_index = global_offset + local_index;
                            let mut cx_max = None;
                            let mut cx_evaluated_candidates = 0usize;
                            let mut ranked_samples = Vec::with_capacity(message_count);
                            let target_exponent =
                                keep_sample_details.then(|| candidate.target_exponent.normalized());

                            for idx in 0..messages.len() {
                                let shifted = &shifted_ciphertexts[idx];
                                let hbc_result = prepare_candidate_ciphertext(
                                    engine,
                                    shifted,
                                    &candidate.r,
                                    &ctx.n,
                                );
                                let x_value = x_values.get(idx).cloned().ok_or_else(|| {
                                    "missing ciphertext exponent for message index".to_string()
                                })?;
                                let d_new_owned = if engine.ciphertext_modify {
                                    let e_x = e_x_values.get(idx).ok_or_else(|| {
                                        "missing ciphertext exponent for message index".to_string()
                                    })?;
                                    Some(mod_inverse(e_x, &candidate.phi_new).ok_or_else(
                                        || {
                                            format!(
                                                "analysis_batch missing modular inverse for x {}",
                                                x_value
                                            )
                                        },
                                    )?)
                                } else {
                                    None
                                };
                                let d_new = d_new_owned.as_ref().unwrap_or(&candidate.d_new);

                                let dm = derive_candidate_message_from_result(
                                    ctx,
                                    engine,
                                    &hbc_result,
                                    &candidate.r,
                                    d_new,
                                );
                                let (dm_bits, match_pct) =
                                    truncated_match_percentage(&dm, &avalanche_message_bits);
                                cx_evaluated_candidates += 1;
                                if cx_max.as_ref().is_none_or(|current: &BatchCxMax| {
                                    match_pct > current.match_pct
                                }) {
                                    cx_max = Some(BatchCxMax {
                                        match_pct,
                                        x: x_value.clone(),
                                        r: candidate.r.clone(),
                                        batch_candidate_index,
                                    });
                                }

                                let message_bits = PackedBits::from_bools(&dm_bits);
                                let scored_input = ScoredAvalancheInput {
                                    batch_candidate_index,
                                    message_index: idx,
                                    r: candidate.r.clone(),
                                    x: x_value,
                                    score_match_pct: match_pct,
                                    message_bits: message_bits.clone(),
                                    detail: target_exponent.as_ref().map(|target_exponent| {
                                        ScoredAvalancheInputDetail {
                                            target_exponent: target_exponent.clone(),
                                            hbc_ciphertext_r: hbc_result.clone(),
                                            candidate_decryption: dm.clone(),
                                        }
                                    }),
                                };
                                let fitness = compute_padding_fitness_score(
                                    ctx,
                                    engine,
                                    &candidate.r,
                                    d_new,
                                    &scored_input.x,
                                    &message_bits,
                                    &additional_fitness_messages,
                                    shift,
                                );
                                ranked_samples.push(RankedScoredAvalancheInput {
                                    input: scored_input,
                                    fitness,
                                });
                                let done = batch_cx_done.fetch_add(1, Ordering::Relaxed) + 1;
                                log_parallel_progress_every_ten_percent(
                                    done,
                                    batch_cx_total,
                                    &batch_cx_next_pct,
                                    &batch_cx_label,
                                );
                                log_parallel_progress_every_interval(
                                    done,
                                    batch_cx_total,
                                    &batch_cx_started_at,
                                    &batch_cx_next_log_at_ms,
                                    &batch_cx_label,
                                    Duration::from_secs(5),
                                );
                            }

                            acc.candidate_count += 1;
                            acc.cx_evaluated_candidates += cx_evaluated_candidates;
                            if let Some(candidate) = cx_max {
                                acc.set_cx_max(candidate);
                            }
                            acc.ranked_samples.extend(ranked_samples);
                            Ok::<_, String>(acc)
                        },
                    )
                    .try_reduce(AccuracyBatchAccumulator::default, |left, right| {
                        Ok::<_, String>(left.merge(right))
                    })
                    .map_err(|err| -> Box<dyn Error> { err.into() })?;

                batch_aggregate.candidate_count += chunk_aggregate.candidate_count;
                batch_aggregate.cx_evaluated_candidates += chunk_aggregate.cx_evaluated_candidates;
                if let Some(candidate) = chunk_aggregate.cx_max.take() {
                    batch_aggregate.set_cx_max(candidate);
                }
                if let Some(pool) = streaming_fitness_pool.as_mut() {
                    pool.extend_with_scores(std::mem::take(&mut chunk_aggregate.ranked_samples));
                } else {
                    batch_aggregate
                        .ranked_samples
                        .append(&mut chunk_aggregate.ranked_samples);
                }
            }

            let batch_ranked_inputs = if let Some(pool) = streaming_fitness_pool.take() {
                pool.finalize(engine.avalanche_unique_r_cx_inputs)
            } else {
                std::mem::take(&mut batch_aggregate.ranked_samples)
            };
            let batch_scored_inputs = batch_ranked_inputs
                .iter()
                .map(|input| input.input.clone())
                .collect::<Vec<_>>();
            let sampled = run_sampled_avalanche_beam_search_with_ranked_inputs(
                engine,
                &message,
                &batch_scored_inputs,
                Some(&batch_ranked_inputs),
                batch_number,
                prefer_beam_score_ordering,
                rng,
            )?;
            (
                batch_aggregate.candidate_count,
                batch_aggregate.cx_evaluated_candidates,
                batch_aggregate.cx_max.take(),
                sampled,
            )
        };
        let mut batch_cx_max_match_pct = None;
        let mut batch_cx_max_x = None;
        if let Some(best) = batch_cx_max.take() {
            batch_cx_max_match_pct = Some(best.match_pct);
            batch_cx_max_x = Some(best.x.clone());
            let replace = match cx_run_max {
                Some(ref current) => best.match_pct > current.match_pct,
                None => true,
            };
            if replace {
                cx_run_max = Some(CxMatchCandidate {
                    match_pct: best.match_pct,
                    x: best.x,
                    r: best.r,
                    batch_number,
                });
            }
        }
        total_cx_evaluated_candidates += batch_cx_evaluated_candidates;

        let mut beam_match_pct = None;
        let mut beam_ones_match_pct = None;
        let mut majority_vote_match_pct = None;
        let mut majority_vote_ones_match_pct = None;
        let mut beam_score = None;
        let mut beam_bit_width = None;
        let mut batch_selected_sample_index = None;
        let mut batch_selected_sample_average_score_pct = None;
        total_avalanche_evaluated_candidates += sampled_avalanche_result.evaluated_candidates;
        if let Some(selected_sample) = sampled_avalanche_result.selected_sample.as_ref() {
            batch_selected_sample_index = Some(selected_sample.sample_index);
            batch_selected_sample_average_score_pct = Some(selected_sample.average_score_pct);
            println!(
                "Accuracy batch {} selected avalanche tier {} sample {} with average source score {}% and best match {}%",
                batch_number,
                selected_sample.tier_index,
                selected_sample.sample_index,
                format_beam_float(selected_sample.average_score_pct, BEAM_PCT_DECIMALS),
                format_beam_float(selected_sample.best_match_pct, BEAM_PCT_DECIMALS)
            );
            if let Some(top_beam) = selected_sample.beam_results.first() {
                beam_match_pct = Some(top_beam.match_pct);
                beam_ones_match_pct = Some(top_beam.ones_match_pct);
                beam_score = Some(top_beam.score);
                beam_bit_width = Some(top_beam.bit_width);
            }
            majority_vote_match_pct = Some(selected_sample.majority_vote_match_pct);
            majority_vote_ones_match_pct = Some(selected_sample.majority_vote_ones_match_pct);
            batch_top_match_percentages.push(selected_sample.best_match_pct);

            let message_bits = payload_message_bits(engine, &message);
            if let Some(top_beam) = selected_sample.beam_results.first() {
                let beam_best_bits =
                    extract_payload_bits_for_accuracy(engine, &selected_sample.best_bits);
                validate_displayed_candidate_consistency(
                    "avalanche beam top candidate",
                    &message_bits,
                    &beam_best_bits,
                    top_beam.match_pct,
                    top_beam.ones_match_pct,
                    Some(&top_beam.hex),
                )
                .map_err(|err| -> Box<dyn Error> { err.into() })?;
                if should_replace_beam_max_candidate(
                    beam_max.as_ref(),
                    top_beam,
                    selected_sample,
                    prefer_beam_score_ordering,
                ) {
                    beam_max = Some(BeamMaxCandidate {
                        beam_match_pct: top_beam.match_pct,
                        average_score_pct: selected_sample.average_score_pct,
                        top_beam_score: selected_sample.top_beam_score,
                        beam_results: selected_sample.beam_results.clone(),
                        center_biases: selected_sample.center_biases.clone(),
                        best_bits: beam_best_bits,
                        message_bits: message_bits.clone(),
                        batch_number,
                        sample_index: selected_sample.sample_index,
                        tier_index: selected_sample.tier_index,
                    });
                }
            }
            let majority_vote_bits =
                extract_payload_bits_for_accuracy(engine, &selected_sample.majority_vote_bits);
            validate_displayed_candidate_consistency(
                "avalanche majority-vote candidate",
                &message_bits,
                &majority_vote_bits,
                selected_sample.majority_vote_match_pct,
                selected_sample.majority_vote_ones_match_pct,
                None,
            )
            .map_err(|err| -> Box<dyn Error> { err.into() })?;
            let expected_best_match_pct = selected_sample
                .top_beam_match_pct
                .unwrap_or(0.0)
                .max(selected_sample.majority_vote_match_pct);
            if (selected_sample.best_match_pct - expected_best_match_pct).abs() > 1e-9 {
                return Err(format!(
                    "avalanche selected sample best-match mismatch: stored={} expected={}",
                    selected_sample.best_match_pct, expected_best_match_pct
                )
                .into());
            }
            let replace_majority = match majority_vote_max {
                Some(ref current) => {
                    selected_sample.majority_vote_match_pct > current.majority_vote_match_pct
                        || (selected_sample.majority_vote_match_pct
                            == current.majority_vote_match_pct
                            && selected_sample.average_score_pct > current.average_score_pct)
                }
                None => true,
            };
            if replace_majority {
                majority_vote_max = Some(MajorityVoteMaxCandidate {
                    average_score_pct: selected_sample.average_score_pct,
                    majority_vote_bits,
                    majority_vote_match_pct: selected_sample.majority_vote_match_pct,
                    majority_vote_ones_match_pct: selected_sample.majority_vote_ones_match_pct,
                    message_bits,
                    batch_number,
                    sample_index: selected_sample.sample_index,
                    tier_index: selected_sample.tier_index,
                });
            }
        } else {
            println!(
                "Accuracy batch {} produced no valid avalanche samples",
                batch_number
            );
        }
        log_progress_every_ten_percent(
            batch_number as u64,
            batch_count as u64,
            &mut next_batch_pct,
            "Accuracy batch",
        );
        if batch_number == batch_count {
            if let Some(ref max) = beam_max {
                let top_beam = max
                    .beam_results
                    .first()
                    .cloned()
                    .ok_or("missing top beam for selected avalanche sample")?;
                println!(
                    "Avalanche beam run max: avg-score {}% beam-score {} batch {} tier {} sample {} match {}% ones-match {}% hex {}",
                    format_beam_float(max.average_score_pct, BEAM_PCT_DECIMALS),
                    format_beam_float(max.top_beam_score, BEAM_SCORE_DECIMALS),
                    max.batch_number,
                    max.tier_index,
                    max.sample_index,
                    format_beam_float(top_beam.match_pct, BEAM_PCT_DECIMALS),
                    format_beam_float(top_beam.ones_match_pct, BEAM_PCT_DECIMALS),
                    top_beam.hex
                );
                println!(
                    "Avalanche beam max after {} batches: avg-score {}% beam-score {} batch {} tier {} sample {} match {}% ones-match {}% hex {}",
                    batch_count,
                    format_beam_float(max.average_score_pct, BEAM_PCT_DECIMALS),
                    format_beam_float(max.top_beam_score, BEAM_SCORE_DECIMALS),
                    max.batch_number,
                    max.tier_index,
                    max.sample_index,
                    format_beam_float(top_beam.match_pct, BEAM_PCT_DECIMALS),
                    format_beam_float(top_beam.ones_match_pct, BEAM_PCT_DECIMALS),
                    top_beam.hex
                );
                println!(
                    "Avalanche beam search top {} candidates (best sample avg {}%, batch {}, tier {}, sample {}, lsb0 order):",
                    max.beam_results.len(),
                    format_beam_float(max.average_score_pct, BEAM_PCT_DECIMALS),
                    max.batch_number,
                    max.tier_index,
                    max.sample_index
                );
                for beam in &max.beam_results {
                    println!(
                        "Beam {} score {} match {}% ones-match {}% hex {}",
                        beam.rank,
                        format_beam_float(beam.score, BEAM_SCORE_DECIMALS),
                        format_beam_float(beam.match_pct, BEAM_PCT_DECIMALS),
                        format_beam_float(beam.ones_match_pct, BEAM_PCT_DECIMALS),
                        beam.hex
                    );
                }
                let max_value = bits_le_to_biguint(&max.best_bits);
                println!(
                    "Avalanche beam max bits: total {} biguint {}",
                    max.best_bits.len(),
                    max_value.bits()
                );
                let msb = max.best_bits.last().copied().unwrap_or(false);
                println!("Avalanche beam max MSB: {}", if msb { 1 } else { 0 });
                println!(
                    "Avalanche beam colored hex (best sample avg {}%, batch {}, tier {}, sample {}, lsb0 order):",
                    format_beam_float(max.average_score_pct, BEAM_PCT_DECIMALS),
                    max.batch_number,
                    max.tier_index,
                    max.sample_index
                );
                print_colored_hex_comparison(
                    "Original message",
                    &max.message_bits,
                    "Beam-search message",
                    &max.best_bits,
                );
            } else {
                println!("Avalanche beam run max: N/A");
                println!("Avalanche beam max after {} batches: N/A", batch_count);
                println!("Avalanche beam search results: N/A");
            }
            if engine.avalanche_combination_majority_vote_print {
                if let Some(ref max) = majority_vote_max {
                    let majority_hex = format_bits_hex_le(&max.majority_vote_bits);
                    println!(
                        "Avalanche majority vote run max: avg-score {}% batch {} tier {} sample {} match {}% ones-match {}% hex {}",
                        format_beam_float(max.average_score_pct, BEAM_PCT_DECIMALS),
                        max.batch_number,
                        max.tier_index,
                        max.sample_index,
                        format_beam_float(max.majority_vote_match_pct, BEAM_PCT_DECIMALS),
                        format_beam_float(max.majority_vote_ones_match_pct, BEAM_PCT_DECIMALS),
                        majority_hex
                    );
                    println!(
                        "Avalanche majority vote colored hex (best sample avg {}%, batch {}, tier {}, sample {}, lsb0 order):",
                        format_beam_float(max.average_score_pct, BEAM_PCT_DECIMALS),
                        max.batch_number,
                        max.tier_index,
                        max.sample_index
                    );
                    print_colored_hex_comparison(
                        "Original message",
                        &max.message_bits,
                        "Majority-vote message",
                        &max.majority_vote_bits,
                    );
                } else {
                    println!("Avalanche majority vote results: N/A");
                }
            }
            if let Some(ref max) = cx_run_max {
                println!(
                    "Avalanche c^x run max: match {}% batch {} x {} r {}",
                    format_beam_float(max.match_pct, BEAM_PCT_DECIMALS),
                    max.batch_number,
                    max.x,
                    max.r
                );
            } else {
                println!("Avalanche c^x run max: N/A");
            }
            println!(
                "Avalanche c^x evaluated total: {}",
                total_cx_evaluated_candidates
            );
            println!(
                "Avalanche evaluated candidates total: {}",
                total_avalanche_evaluated_candidates
            );
        }

        with_analytics(analytics, |a| {
            a.push_r_candidate_accuracy_batch(RCandidateAccuracyBatch {
                context: format!("analysis_batch_accuracy_{}", batch_number),
                messages: messages.clone(),
                ciphertexts: ciphertexts.clone(),
                shifted_ciphertexts: shifted_ciphertexts.clone(),
                rabin_exponent: y,
                tonelli_shanks_modulus: ctx.n.clone(),
                tonelli_shanks_ciphertexts: shifted_ciphertexts.clone(),
                candidate_count: batch_candidate_count,
                candidates: Vec::new(),
                cx_max_match_pct: batch_cx_max_match_pct,
                cx_max_x: batch_cx_max_x,
                cx_evaluated_candidates: batch_cx_evaluated_candidates,
                avalanche_evaluated_candidates: sampled_avalanche_result.evaluated_candidates,
                beam_match_pct,
                beam_ones_match_pct,
                majority_vote_match_pct,
                majority_vote_ones_match_pct,
                beam_score,
                beam_bit_width,
                avalanche_selected_sample_index: batch_selected_sample_index,
                avalanche_selected_sample_average_score_pct:
                    batch_selected_sample_average_score_pct,
                avalanche_sampled_candidates_evaluated: sampled_avalanche_result
                    .evaluated_candidates,
                avalanche_combination_sample_count: sampled_avalanche_result.sample_count,
                avalanche_tier_statistics: sampled_avalanche_result.tier_statistics,
                avalanche_final_tier_bias_reports: if engine.avalanche_report_biases
                    && !engine.avalanche_center_threshold_best
                {
                    build_final_tier_bias_reports(&sampled_avalanche_result.final_tier_samples)
                } else {
                    Vec::new()
                },
                avalanche_combination_samples: sampled_avalanche_result.retained_samples,
            });
        });
    }

    if let Some(ref max) = beam_max {
        with_analytics(analytics, |a| {
            a.set_feature_stat(
                "r_candidate_accuracy",
                "avalanche_max_score",
                json!(max.top_beam_score),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "avalanche_max_match_pct",
                json!(max.beam_match_pct),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "avalanche_max_sample_average_score_pct",
                json!(max.average_score_pct),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "avalanche_max_batch_number",
                json!(max.batch_number),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "avalanche_max_sample_index",
                json!(max.sample_index),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "avalanche_max_tier_index",
                json!(max.tier_index),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "avalanche_total_evaluated_candidates",
                json!(total_avalanche_evaluated_candidates),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "avalanche_batch_top_match_percentages",
                json!(batch_top_match_percentages),
            );
            if engine.avalanche_report_biases && engine.avalanche_center_threshold_best {
                a.set_feature_stat(
                    "r_candidate_accuracy",
                    "avalanche_best_center_bias_report",
                    json!(build_best_center_bias_report(max)),
                );
            }
        });
    } else {
        with_analytics(analytics, |a| {
            a.set_feature_stat(
                "r_candidate_accuracy",
                "avalanche_total_evaluated_candidates",
                json!(total_avalanche_evaluated_candidates),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "avalanche_batch_top_match_percentages",
                json!(batch_top_match_percentages),
            );
        });
    }
    if let Some(ref max) = majority_vote_max {
        with_analytics(analytics, |a| {
            a.set_feature_stat(
                "r_candidate_accuracy",
                "avalanche_majority_max_match_pct",
                json!(max.majority_vote_match_pct),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "avalanche_majority_max_batch_number",
                json!(max.batch_number),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "avalanche_majority_max_sample_index",
                json!(max.sample_index),
            );
            a.set_feature_stat(
                "r_candidate_accuracy",
                "avalanche_majority_max_tier_index",
                json!(max.tier_index),
            );
        });
    }
    if let Some(ref max) = cx_run_max {
        with_analytics(analytics, |a| {
            a.set_feature_stat(
                "r_candidate_accuracy",
                "cx_max_match_pct",
                json!(max.match_pct),
            );
            a.set_feature_stat("r_candidate_accuracy", "cx_max_x", json!(max.x));
            a.set_feature_stat(
                "r_candidate_accuracy",
                "cx_max_batch_number",
                json!(max.batch_number),
            );
            a.set_feature_stat("r_candidate_accuracy", "cx_max_r", json!(max.r));
            a.set_feature_stat(
                "r_candidate_accuracy",
                "cx_total_evaluated_candidates",
                json!(total_cx_evaluated_candidates),
            );
        });
    } else {
        with_analytics(analytics, |a| {
            a.set_feature_stat(
                "r_candidate_accuracy",
                "cx_total_evaluated_candidates",
                json!(total_cx_evaluated_candidates),
            );
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
fn construct_from_factors_close_to_target_n(
    target_n: &BigUint,
    prime_factors: &[BigUint],
) -> BigUint {
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
    use crate::dsp::{find_ramp_signals, ramp_signal_strength};
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Builds a unique temporary path for Avalanche cache tests.
    ///
    /// # Parameters
    /// - `label`: Short suffix describing the test artifact.
    ///
    /// # Returns
    /// - `PathBuf`: Unique filesystem path under the process temp directory.
    ///
    /// # Expected Output
    /// - Returns a path value without touching the filesystem.
    fn temp_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("rsademo_avalanche_{label}_{nanos}"))
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
    fn test_collect_invertible_ciphertext_variants_retries_noninvertible_x() {
        let p = BigUint::from(61u8);
        let q = BigUint::from(53u8);
        let n = &p * &q;
        let phi = (&p - BigUint::one()) * (&q - BigUint::one());
        let e = choose_exponent(3, &phi);

        let ctx = RSAContext {
            key_bit_width: p.bits().saturating_add(q.bits()),
            n,
            e,
        };
        let base_ciphertext = BigUint::from(17u8);
        let candidate_phi = BigUint::from(10u8);
        let phi_values = vec![&candidate_phi];

        let variants = collect_invertible_ciphertext_variants(
            &ctx,
            &base_ciphertext,
            &phi_values,
            4,
            false,
            "test",
        )
        .expect("missing variants");

        let x_values: Vec<BigUint> = variants.iter().map(|variant| variant.x.clone()).collect();
        assert_eq!(
            x_values,
            vec![
                BigUint::from(1u8),
                BigUint::from(3u8),
                BigUint::from(7u8),
                BigUint::from(9u8)
            ]
        );
        assert!(
            variants
                .iter()
                .all(|variant| mod_inverse(&variant.e_x, &candidate_phi).is_some())
        );
    }

    #[test]
    fn test_count_matching_bits_handles_zero_width_mismatch_without_strings() {
        let left = BigUint::zero();
        let right = BigUint::from(8u8);
        assert_eq!(count_matching_bits(&left, &right), (1, 1));
    }

    #[test]
    fn test_count_matching_bits_counts_total_and_lsb_run() {
        let left = BigUint::from(0b1111_0000u8);
        let right = BigUint::from(0b1110_0000u8);
        assert_eq!(count_matching_bits(&left, &right), (4, 7));
    }

    #[test]
    fn test_count_matching_bits_le_uses_packed_comparison() {
        let left = [true, false, true, true, false, false, true, false];
        let right = [true, false, false, true, true, false, true, false];
        assert_eq!(count_matching_bits_le(&left, &right), (2, 6));
    }

    #[test]
    fn test_resolve_avalanche_bit_width_uses_configured_message_bits() {
        let mut config = Config::default();
        config.engine.message.bits = 13;
        assert_eq!(resolve_avalanche_bit_width(&config.engine), 13);

        config.engine.message.bits = 0;
        assert_eq!(resolve_avalanche_bit_width(&config.engine), 1);
    }

    #[test]
    fn test_resolve_avalanche_bit_width_adds_fitness_shift_bits() {
        let mut config = Config::default();
        config.engine.message.bits = 64;
        config.engine.avalanche_fitness_shift_bytes = 4;

        assert_eq!(resolve_avalanche_bit_width(&config.engine), 96);
        assert_eq!(resolve_avalanche_fitness_bit_width(&config.engine), 32);
        assert_eq!(minimum_r_candidate_bit_length(&config.engine), 192);
    }

    #[test]
    fn test_build_candidate_message_transform_shifts_message_and_preserves_payload_bits() {
        let mut config = Config::default();
        config.engine.avalanche_fitness_shift_bytes = 4;
        let transform = build_candidate_message_transform(&config.engine);
        let message = BigUint::from(0x1234_5678u64);
        let transformed = transform(&message);

        assert_eq!(transformed, &message << 32usize);
        assert_eq!(&transformed >> 32usize, message);
        assert_eq!(
            &transformed & ((BigUint::one() << 32usize) - BigUint::one()),
            BigUint::zero()
        );
    }

    #[test]
    fn test_validate_message_width_under_modulus_rejects_impossible_widened_payload() {
        let mut config = Config::default();
        config.engine.message.bits = 128;
        config.engine.avalanche_fitness_shift_bytes = 4;
        let modulus = BigUint::one() << 143usize;

        let error = validate_message_width_under_modulus(
            &config.engine,
            &modulus,
            "test random message sampling",
        )
        .expect_err("widened payload should exceed the modulus width");

        assert!(error
            .to_string()
            .contains("configured payload width 128 bits plus fitness shift 32 bits exceeds modulus width 144 bits"));
    }

    #[test]
    fn test_random_message_under_n_preserves_configured_payload_width() {
        let mut config = Config::default();
        config.engine.message.bits = 64;
        config.engine.avalanche_fitness_shift_bytes = 4;
        let modulus = BigUint::one() << 127usize;
        let mut rng = RngChoice::from_seed(RngMode::Standard, 23);

        let message = random_message_under_n(&config.engine, &modulus, &mut rng)
            .expect("random message sampling should preserve the configured width");

        assert_eq!(message.bits(), 64);
        let widened = transform_message_for_candidate_scoring(
            &config.engine,
            &message,
            &modulus,
            "test random message sampling",
        )
        .expect("widened payload should fit");
        assert!(widened < modulus);
    }

    #[test]
    fn test_extract_payload_bits_for_accuracy_removes_fitness_slice() {
        let mut config = Config::default();
        config.engine.avalanche_fitness_shift_bytes = 1;
        config.engine.message.bits = 2;

        let payload = extract_payload_bits_for_accuracy(
            &config.engine,
            &[true, true, true, true, true, true, true, true, false, true],
        );
        assert_eq!(payload, vec![false, true]);
    }

    #[test]
    fn test_validate_displayed_candidate_consistency_accepts_matching_fields() {
        let message_bits = vec![true, false, true, false];
        let candidate_bits = vec![true, true, true, false];
        let (match_pct, ones_match_pct) =
            compute_bit_match_percentages(&candidate_bits, &message_bits);

        validate_displayed_candidate_consistency(
            "test candidate",
            &message_bits,
            &candidate_bits,
            match_pct,
            ones_match_pct,
            Some("07"),
        )
        .expect("matching display fields should validate");
    }

    #[test]
    fn test_validate_displayed_candidate_consistency_rejects_mismatched_percentages() {
        let message_bits = vec![true, false, true, false];
        let candidate_bits = vec![true, true, true, false];

        let error = validate_displayed_candidate_consistency(
            "test candidate",
            &message_bits,
            &candidate_bits,
            100.0,
            100.0,
            Some("07"),
        )
        .expect_err("mismatched display percentages should fail validation");

        assert!(error.contains("match percentage mismatch"));
    }

    #[test]
    fn test_shifted_payload_reference_can_artificially_inflate_match_percentage() {
        let mut config = Config::default();
        config.engine.message.bits = 64;
        config.engine.avalanche_fitness_shift_bytes = 4;

        let payload_message =
            BigUint::parse_bytes(b"e859a7c01a265845", 16).expect("payload hex should parse");
        let shifted_message = transform_message_for_candidate_scoring(
            &config.engine,
            &payload_message,
            &BigUint::zero(),
            "test",
        )
        .expect("shifted payload should build");
        let candidate_bits = biguint_to_bits_le(
            &BigUint::parse_bytes(b"1e02cd4531000001", 16).expect("candidate hex should parse"),
            64,
        );

        let payload_reference_bits = payload_message_bits(&config.engine, &payload_message);
        let shifted_reference_bits = biguint_to_bits_le(&shifted_message, 64);
        let (payload_match_pct, payload_ones_match_pct) =
            compute_bit_match_percentages(&candidate_bits, &payload_reference_bits);
        let (shifted_match_pct, shifted_ones_match_pct) =
            compute_bit_match_percentages(&candidate_bits, &shifted_reference_bits);

        assert!((payload_match_pct - 53.125).abs() < 1e-9);
        assert!((payload_ones_match_pct - 41.17647058823529).abs() < 1e-9);
        assert!((shifted_match_pct - 82.8125).abs() < 1e-9);
        assert!((shifted_ones_match_pct - 52.94117647058823).abs() < 1e-9);
    }

    #[test]
    fn test_lsb_zero_count_fitness_counts_zero_bits_in_window() {
        let bits = PackedBits::from_bools(&[false, true, false, false, true, false]);
        assert_eq!(lsb_zero_count_fitness(&bits, 5), 3);
        assert_eq!(lsb_zero_count_fitness(&bits, 2), 1);
    }

    #[test]
    fn test_resolve_avalanche_fitness_retained_input_limit_combines_limits() {
        assert_eq!(resolve_avalanche_fitness_retained_input_limit(0, 0), 0);
        assert_eq!(resolve_avalanche_fitness_retained_input_limit(3, 0), 3);
        assert_eq!(resolve_avalanche_fitness_retained_input_limit(0, 4), 4);
        assert_eq!(resolve_avalanche_fitness_retained_input_limit(3, 4), 12);
    }

    #[test]
    fn test_truncated_match_percentage_uses_reference_width() {
        let candidate = BigUint::from(0b1111_0000u8);
        let reference = biguint_to_bits_le(&BigUint::from(0b1110_0000u8), 4);
        let (candidate_bits, match_pct) = truncated_match_percentage(&candidate, &reference);

        assert_eq!(candidate_bits, vec![false, false, false, false]);
        assert!((match_pct - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_scored_avalanche_fitness_pass_downselects_r_and_cx_candidates() {
        let retained = apply_scored_avalanche_fitness_pass(
            vec![
                ScoredAvalancheInput {
                    batch_candidate_index: 0,
                    message_index: 0,
                    r: BigUint::from(3u8),
                    x: BigUint::from(1u8),
                    score_match_pct: 70.0,
                    message_bits: PackedBits::from_bools(&[false, false, false, false, true]),
                    detail: None,
                },
                ScoredAvalancheInput {
                    batch_candidate_index: 0,
                    message_index: 1,
                    r: BigUint::from(3u8),
                    x: BigUint::from(3u8),
                    score_match_pct: 60.0,
                    message_bits: PackedBits::from_bools(&[false, true, false, false, true]),
                    detail: None,
                },
                ScoredAvalancheInput {
                    batch_candidate_index: 1,
                    message_index: 0,
                    r: BigUint::from(5u8),
                    x: BigUint::from(1u8),
                    score_match_pct: 65.0,
                    message_bits: PackedBits::from_bools(&[false, false, false, true, true]),
                    detail: None,
                },
                ScoredAvalancheInput {
                    batch_candidate_index: 1,
                    message_index: 1,
                    r: BigUint::from(5u8),
                    x: BigUint::from(3u8),
                    score_match_pct: 55.0,
                    message_bits: PackedBits::from_bools(&[false, false, true, true, true]),
                    detail: None,
                },
                ScoredAvalancheInput {
                    batch_candidate_index: 2,
                    message_index: 0,
                    r: BigUint::from(7u8),
                    x: BigUint::from(1u8),
                    score_match_pct: 90.0,
                    message_bits: PackedBits::from_bools(&[true, true, true, true, true]),
                    detail: None,
                },
            ],
            4,
            2,
            1,
            false,
            0.580,
        );

        let retained_keys = retained
            .iter()
            .map(|input| (input.batch_candidate_index, input.message_index))
            .collect::<Vec<_>>();
        assert_eq!(retained_keys, vec![(0, 0), (1, 0)]);
    }

    #[test]
    fn test_apply_scored_avalanche_fitness_pass_prefers_total_zero_count_over_trailing_run() {
        let retained = apply_scored_avalanche_fitness_pass(
            vec![
                ScoredAvalancheInput {
                    batch_candidate_index: 0,
                    message_index: 0,
                    r: BigUint::from(3u8),
                    x: BigUint::from(1u8),
                    score_match_pct: 80.0,
                    message_bits: PackedBits::from_bools(&[false, true, false, false, true]),
                    detail: None,
                },
                ScoredAvalancheInput {
                    batch_candidate_index: 1,
                    message_index: 0,
                    r: BigUint::from(5u8),
                    x: BigUint::from(1u8),
                    score_match_pct: 70.0,
                    message_bits: PackedBits::from_bools(&[false, false, true, true, true]),
                    detail: None,
                },
            ],
            4,
            1,
            1,
            false,
            0.580,
        );

        let retained_keys = retained
            .iter()
            .map(|input| (input.batch_candidate_index, input.message_index))
            .collect::<Vec<_>>();
        assert_eq!(retained_keys, vec![(0, 0)]);
    }

    #[test]
    fn test_apply_scored_avalanche_fitness_pass_ranks_in_one_global_pool() {
        let retained = apply_scored_avalanche_fitness_pass(
            vec![
                ScoredAvalancheInput {
                    batch_candidate_index: 0,
                    message_index: 0,
                    r: BigUint::from(3u8),
                    x: BigUint::from(1u8),
                    score_match_pct: 92.0,
                    message_bits: PackedBits::from_bools(&[false, false, false, false, true]),
                    detail: None,
                },
                ScoredAvalancheInput {
                    batch_candidate_index: 0,
                    message_index: 1,
                    r: BigUint::from(3u8),
                    x: BigUint::from(3u8),
                    score_match_pct: 88.0,
                    message_bits: PackedBits::from_bools(&[false, false, false, true, true]),
                    detail: None,
                },
                ScoredAvalancheInput {
                    batch_candidate_index: 1,
                    message_index: 0,
                    r: BigUint::from(5u8),
                    x: BigUint::from(1u8),
                    score_match_pct: 40.0,
                    message_bits: PackedBits::from_bools(&[false, true, true, true, true]),
                    detail: None,
                },
            ],
            4,
            2,
            1,
            false,
            0.580,
        );

        let retained_keys = retained
            .iter()
            .map(|input| (input.batch_candidate_index, input.message_index))
            .collect::<Vec<_>>();
        assert_eq!(retained_keys, vec![(0, 0), (0, 1)]);
    }

    #[test]
    fn test_enforce_global_unique_scored_inputs_rejects_overlapping_r_and_x_values() {
        let (retained, rejected_overlap_count) = enforce_global_unique_scored_inputs(vec![
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 0,
                r: BigUint::from(3u8),
                x: BigUint::from(5u8),
                score_match_pct: 90.0,
                message_bits: PackedBits::from_bools(&[false, false, false, false]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 0,
                r: BigUint::from(3u8),
                x: BigUint::from(7u8),
                score_match_pct: 80.0,
                message_bits: PackedBits::from_bools(&[false, false, false, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 2,
                message_index: 0,
                r: BigUint::from(11u8),
                x: BigUint::from(5u8),
                score_match_pct: 70.0,
                message_bits: PackedBits::from_bools(&[false, false, true, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 3,
                message_index: 0,
                r: BigUint::from(13u8),
                x: BigUint::from(17u8),
                score_match_pct: 60.0,
                message_bits: PackedBits::from_bools(&[true, true, true, true]),
                detail: None,
            },
        ]);

        assert_eq!(rejected_overlap_count, 2);
        let retained_keys = retained
            .iter()
            .map(|input| (input.batch_candidate_index, input.message_index))
            .collect::<Vec<_>>();
        assert_eq!(retained_keys, vec![(0, 0), (3, 0)]);
    }

    #[test]
    fn test_streaming_scored_avalanche_fitness_pool_matches_batch_fitness_selection() {
        let inputs = vec![
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 0,
                r: BigUint::from(3u8),
                x: BigUint::from(1u8),
                score_match_pct: 70.0,
                message_bits: PackedBits::from_bools(&[false, false, false, false, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 1,
                r: BigUint::from(3u8),
                x: BigUint::from(3u8),
                score_match_pct: 60.0,
                message_bits: PackedBits::from_bools(&[false, true, false, false, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 0,
                r: BigUint::from(5u8),
                x: BigUint::from(5u8),
                score_match_pct: 65.0,
                message_bits: PackedBits::from_bools(&[false, false, false, true, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 1,
                r: BigUint::from(5u8),
                x: BigUint::from(7u8),
                score_match_pct: 55.0,
                message_bits: PackedBits::from_bools(&[false, false, true, true, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 2,
                message_index: 0,
                r: BigUint::from(7u8),
                x: BigUint::from(9u8),
                score_match_pct: 90.0,
                message_bits: PackedBits::from_bools(&[true, true, true, true, true]),
                detail: None,
            },
        ];

        let expected = apply_scored_avalanche_fitness_pass(inputs.clone(), 4, 2, 1, false, 0.580);
        let mut streaming_pool = StreamingScoredAvalancheFitnessPool::new(4, 2, 1, false, 0.580);
        streaming_pool.extend(inputs[..2].to_vec());
        streaming_pool.extend(inputs[2..].to_vec());
        let retained = streaming_pool.finalize(false);

        let expected_keys = expected
            .iter()
            .map(|input| (input.batch_candidate_index, input.message_index))
            .collect::<Vec<_>>();
        let retained_keys = retained
            .iter()
            .map(|input| (input.input.batch_candidate_index, input.input.message_index))
            .collect::<Vec<_>>();
        assert_eq!(retained_keys, expected_keys);
    }

    #[test]
    fn test_build_avalanche_nodes_from_scored_inputs_mirrors_when_enabled() {
        let mut config = Config::default();
        config.engine.mirror_invert_candidates = true;
        config.engine.use_hamming_distance = true;

        let inputs = vec![ScoredAvalancheInput {
            batch_candidate_index: 0,
            message_index: 0,
            r: BigUint::from(3u8),
            x: BigUint::from(1u8),
            score_match_pct: 75.0,
            message_bits: PackedBits::from_bools(&[true, false]),
            detail: None,
        }];

        let nodes =
            build_avalanche_nodes_from_scored_inputs(&inputs, &config.engine, &[true, true])
                .expect("sampled avalanche nodes should build");

        let bits: Vec<Vec<bool>> = nodes.iter().map(|node| node.message_bits_vec()).collect();
        assert_eq!(bits, vec![vec![true, false], vec![false, true]]);
    }

    #[test]
    fn test_select_scored_inputs_for_mixed_r_candidates_caps_combination_size() {
        let inputs = vec![
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 0,
                r: BigUint::from(3u8),
                x: BigUint::from(1u8),
                score_match_pct: 75.0,
                message_bits: PackedBits::from_bools(&[true, false]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 1,
                r: BigUint::from(3u8),
                x: BigUint::from(3u8),
                score_match_pct: 70.0,
                message_bits: PackedBits::from_bools(&[false, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 0,
                r: BigUint::from(5u8),
                x: BigUint::from(1u8),
                score_match_pct: 65.0,
                message_bits: PackedBits::from_bools(&[true, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 1,
                r: BigUint::from(5u8),
                x: BigUint::from(3u8),
                score_match_pct: 60.0,
                message_bits: PackedBits::from_bools(&[false, false]),
                detail: None,
            },
        ];

        let grouped_inputs = group_scored_inputs_by_r_candidate_with_progress(&inputs, None);
        assert_eq!(grouped_inputs.len(), 2);

        let mut rng = RngChoice::from_seed(RngMode::Standard, 7);
        let selected_single =
            select_scored_inputs_for_mixed_r_candidates(&grouped_inputs, 1, 2, &mut rng);
        let selected_single_candidates = selected_single
            .iter()
            .map(|input| input.batch_candidate_index)
            .collect::<HashSet<_>>();
        assert_eq!(selected_single.len(), 2);
        assert_eq!(selected_single_candidates.len(), 1);

        let mut rng = RngChoice::from_seed(RngMode::Standard, 7);
        let selected_double =
            select_scored_inputs_for_mixed_r_candidates(&grouped_inputs, 2, 3, &mut rng);
        let selected_double_candidates = selected_double
            .iter()
            .map(|input| input.batch_candidate_index)
            .collect::<HashSet<_>>();
        assert_eq!(selected_double.len(), 3);
        assert_eq!(selected_double_candidates.len(), 2);
    }

    #[test]
    fn test_select_random_scored_inputs_caps_sample_size_and_uniqueness() {
        let inputs = vec![
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 0,
                r: BigUint::from(3u8),
                x: BigUint::from(1u8),
                score_match_pct: 75.0,
                message_bits: PackedBits::from_bools(&[true, false]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 1,
                r: BigUint::from(3u8),
                x: BigUint::from(3u8),
                score_match_pct: 70.0,
                message_bits: PackedBits::from_bools(&[false, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 0,
                r: BigUint::from(5u8),
                x: BigUint::from(1u8),
                score_match_pct: 65.0,
                message_bits: PackedBits::from_bools(&[true, true]),
                detail: None,
            },
        ];

        let mut rng = RngChoice::from_seed(RngMode::Crypto, 7);
        let selected = select_random_scored_inputs(&inputs, 10, &mut rng);
        let selected_keys = selected
            .iter()
            .map(|input| (input.batch_candidate_index, input.message_index))
            .collect::<HashSet<_>>();

        assert_eq!(selected.len(), 3);
        assert_eq!(selected_keys.len(), 3);
    }

    #[test]
    fn test_prune_scored_inputs_by_hamming_distance_percentile_keeps_central_band() {
        let reference_bits = PackedBits::from_bools(&[false; 10]);
        let inputs = (0usize..10)
            .map(|distance| {
                let mut bits = vec![false; 10];
                for bit in bits.iter_mut().take(distance) {
                    *bit = true;
                }
                ScoredAvalancheInput {
                    batch_candidate_index: distance,
                    message_index: 0,
                    r: BigUint::from(distance + 3),
                    x: BigUint::from(1u8),
                    score_match_pct: 100.0 - (distance as f64 * 10.0),
                    message_bits: PackedBits::from_bools(&bits),
                    detail: None,
                }
            })
            .collect::<Vec<_>>();

        let pruned = prune_scored_inputs_by_hamming_distance_percentile_with_progress(
            &inputs,
            &reference_bits,
            60.0,
            0.0,
            None,
        );
        let retained = pruned
            .selected_inputs
            .iter()
            .map(|input| input.batch_candidate_index)
            .collect::<Vec<_>>();

        assert_eq!(retained, vec![2, 3, 4, 5, 6, 7]);
        assert_eq!(pruned.retained_inlier_count, 6);
        assert_eq!(pruned.available_outlier_count, 4);
        assert_eq!(pruned.preferred_outlier_count, 0);
    }

    #[test]
    fn test_prune_scored_inputs_by_hamming_distance_percentile_adds_preferred_outliers() {
        let reference_bits = PackedBits::from_bools(&[false; 10]);
        let inputs = (0usize..10)
            .map(|distance| {
                let mut bits = vec![false; 10];
                for bit in bits.iter_mut().take(distance) {
                    *bit = true;
                }
                ScoredAvalancheInput {
                    batch_candidate_index: distance,
                    message_index: 0,
                    r: BigUint::from(distance + 3),
                    x: BigUint::from(1u8),
                    score_match_pct: 100.0 - (distance as f64 * 10.0),
                    message_bits: PackedBits::from_bools(&bits),
                    detail: None,
                }
            })
            .collect::<Vec<_>>();

        let pruned = prune_scored_inputs_by_hamming_distance_percentile_with_progress(
            &inputs,
            &reference_bits,
            60.0,
            50.0,
            None,
        );
        let retained = pruned
            .selected_inputs
            .iter()
            .map(|input| input.batch_candidate_index)
            .collect::<Vec<_>>();

        assert_eq!(retained, vec![0, 1, 2, 3, 4, 5, 6, 7, 9]);
        assert_eq!(pruned.retained_inlier_count, 6);
        assert_eq!(pruned.available_outlier_count, 4);
        assert_eq!(pruned.preferred_outlier_count, 3);
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_rejects_invalid_hamming_prune_percentile() {
        let mut config = Config::default();
        config.engine.avalanche_combination_hamming_distance_prune = true;
        config
            .engine
            .avalanche_combination_hamming_distance_keep_percentile = 0.0;
        config.engine.avalanche_combination_samples = 1;
        config.engine.avalanche_combination_size = 1;
        config.engine.avalanche_combination_mixed_r_candidates = 1;

        let scored_inputs = vec![ScoredAvalancheInput {
            batch_candidate_index: 0,
            message_index: 0,
            r: BigUint::from(3u8),
            x: BigUint::from(1u8),
            score_match_pct: 75.0,
            message_bits: PackedBits::from_bools(&[true, false]),
            detail: None,
        }];

        let mut rng = RngChoice::from_seed(RngMode::Standard, 7);
        let error = run_sampled_avalanche_beam_search(
            &config.engine,
            &BigUint::from(1u8),
            &scored_inputs,
            1,
            false,
            &mut rng,
        )
        .expect_err("invalid percentile should be rejected");

        assert!(
            error
                .to_string()
                .contains("avalanche_combination_hamming_distance_keep_percentile")
        );
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_rejects_invalid_outlier_preference_pct() {
        let mut config = Config::default();
        config.engine.avalanche_combination_hamming_distance_prune = true;
        config
            .engine
            .avalanche_combination_hamming_distance_outlier_preference_pct = 150.0;
        config.engine.avalanche_combination_samples = 1;
        config.engine.avalanche_combination_size = 1;
        config.engine.avalanche_combination_mixed_r_candidates = 1;

        let scored_inputs = vec![ScoredAvalancheInput {
            batch_candidate_index: 0,
            message_index: 0,
            r: BigUint::from(3u8),
            x: BigUint::from(1u8),
            score_match_pct: 75.0,
            message_bits: PackedBits::from_bools(&[true, false]),
            detail: None,
        }];

        let mut rng = RngChoice::from_seed(RngMode::Standard, 7);
        let error = run_sampled_avalanche_beam_search(
            &config.engine,
            &BigUint::from(1u8),
            &scored_inputs,
            1,
            false,
            &mut rng,
        )
        .expect_err("invalid outlier preference percentage should be rejected");

        assert!(
            error
                .to_string()
                .contains("avalanche_combination_hamming_distance_outlier_preference_pct")
        );
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_allows_zero_mixed_r_in_chacha20_mode() {
        let mut config = Config::default();
        config.engine.avalanche_random_chacha20_inputs = true;
        config.engine.avalanche_combination_samples = 1;
        config.engine.avalanche_combination_size = 2;
        config.engine.avalanche_combination_mixed_r_candidates = 0;

        let scored_inputs = vec![
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 0,
                r: BigUint::from(3u8),
                x: BigUint::from(1u8),
                score_match_pct: 75.0,
                message_bits: PackedBits::from_bools(&[true, false]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 0,
                r: BigUint::from(5u8),
                x: BigUint::from(3u8),
                score_match_pct: 65.0,
                message_bits: PackedBits::from_bools(&[false, true]),
                detail: None,
            },
        ];

        let mut rng = RngChoice::from_seed(RngMode::Standard, 7);
        let result = run_sampled_avalanche_beam_search(
            &config.engine,
            &BigUint::from(1u8),
            &scored_inputs,
            1,
            false,
            &mut rng,
        )
        .expect("ChaCha20 direct-input mode should not require mixed-r combinations");

        assert_eq!(result.sample_count, 1);
        assert!(result.selected_sample.is_some());
        assert!(result.retained_samples.is_empty());
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_applies_fitness_pass_before_sampling() {
        let mut config = Config::default();
        config.engine.avalanche_random_chacha20_inputs = true;
        config.engine.avalanche_combination_samples = 1;
        config.engine.avalanche_combination_size = 4;
        config.engine.avalanche_combination_mixed_r_candidates = 0;
        config.engine.avalanche_fitness_scoring_pass = true;
        config.engine.avalanche_fitness_r_candidate_limit = 1;
        config.engine.avalanche_fitness_cx_candidate_limit = 1;
        config.engine.avalanche_fitness_use_threshold = false;
        config.engine.avalanche_fitness_bit_width = 4;
        config.engine.message.bits = 5;

        let scored_inputs = vec![
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 0,
                r: BigUint::from(3u8),
                x: BigUint::from(1u8),
                score_match_pct: 80.0,
                message_bits: PackedBits::from_bools(&[false, false, false, false, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 1,
                r: BigUint::from(3u8),
                x: BigUint::from(3u8),
                score_match_pct: 70.0,
                message_bits: PackedBits::from_bools(&[false, true, true, true, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 0,
                r: BigUint::from(5u8),
                x: BigUint::from(1u8),
                score_match_pct: 95.0,
                message_bits: PackedBits::from_bools(&[false, false, true, true, true]),
                detail: None,
            },
        ];

        let mut rng = RngChoice::from_seed(RngMode::Standard, 7);
        let result = run_sampled_avalanche_beam_search(
            &config.engine,
            &BigUint::zero(),
            &scored_inputs,
            1,
            false,
            &mut rng,
        )
        .expect("fitness-preprocessed sampled avalanche should succeed");

        assert_eq!(result.sample_count, 1);
        let selected = result
            .selected_sample
            .expect("sampled avalanche should retain one selected sample");
        assert_eq!(selected.input_count, 1);
        assert_eq!(
            selected.node.message_bits_vec(),
            vec![false, false, false, false, true]
        );
    }

    #[test]
    fn test_resolve_fitness_top_cohort_count_uses_configured_percentage() {
        assert_eq!(resolve_fitness_top_cohort_count(0, 0.30), 0);
        assert_eq!(resolve_fitness_top_cohort_count(1, 0.30), 1);
        assert_eq!(resolve_fitness_top_cohort_count(10, 0.30), 3);
        assert_eq!(resolve_fitness_top_cohort_count(10, 0.35), 4);
        assert_eq!(resolve_fitness_top_cohort_count(10, 1.0), 10);
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_errors_when_fitness_threshold_removes_all_candidates()
    {
        let mut config = Config::default();
        config.engine.avalanche_random_chacha20_inputs = true;
        config.engine.avalanche_combination_samples = 1;
        config.engine.avalanche_combination_size = 1;
        config.engine.avalanche_combination_mixed_r_candidates = 0;
        config.engine.avalanche_fitness_scoring_pass = true;
        config.engine.avalanche_fitness_use_threshold = true;
        config.engine.avalanche_fitness_threshold = 0.580;
        config.engine.avalanche_fitness_bit_width = 4;
        config.engine.message.bits = 4;

        let scored_inputs = vec![
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 0,
                r: BigUint::from(3u8),
                x: BigUint::from(1u8),
                score_match_pct: 70.0,
                message_bits: PackedBits::from_bools(&[false, true, true, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 0,
                r: BigUint::from(5u8),
                x: BigUint::from(1u8),
                score_match_pct: 65.0,
                message_bits: PackedBits::from_bools(&[true, true, true, true]),
                detail: None,
            },
        ];

        let mut rng = RngChoice::from_seed(RngMode::Standard, 19);
        let error = run_sampled_avalanche_beam_search(
            &config.engine,
            &BigUint::from(1u8),
            &scored_inputs,
            1,
            false,
            &mut rng,
        )
        .expect_err("fitness threshold should reject an empty retained pool");

        assert!(error.to_string().contains("avalanche_fitness_threshold"));
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_rejects_invalid_fitness_log_top_pct() {
        let mut config = Config::default();
        config.engine.avalanche_random_chacha20_inputs = true;
        config.engine.avalanche_combination_samples = 1;
        config.engine.avalanche_combination_size = 1;
        config.engine.avalanche_combination_mixed_r_candidates = 0;
        config.engine.avalanche_fitness_use_threshold = false;
        config.engine.avalanche_fitness_log_top_pct = 0.0;

        let scored_inputs = vec![ScoredAvalancheInput {
            batch_candidate_index: 0,
            message_index: 0,
            r: BigUint::from(3u8),
            x: BigUint::from(1u8),
            score_match_pct: 70.0,
            message_bits: PackedBits::from_bools(&[false, true, true, true]),
            detail: None,
        }];

        let mut rng = RngChoice::from_seed(RngMode::Standard, 19);
        let error = run_sampled_avalanche_beam_search(
            &config.engine,
            &BigUint::from(1u8),
            &scored_inputs,
            1,
            false,
            &mut rng,
        )
        .expect_err("zero top cohort percentage should be rejected");

        assert!(error.to_string().contains("avalanche_fitness_log_top_pct"));
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_scores_only_payload_bits() {
        let mut config = Config::default();
        config.engine.avalanche_random_chacha20_inputs = true;
        config.engine.avalanche_combination_samples = 1;
        config.engine.avalanche_combination_size = 1;
        config.engine.avalanche_combination_mixed_r_candidates = 0;
        config.engine.avalanche_fitness_shift_bytes = 1;
        config.engine.message.bits = 2;
        config.engine.avalanche_beam_top_k = 1;

        let scored_inputs = vec![ScoredAvalancheInput {
            batch_candidate_index: 0,
            message_index: 0,
            r: BigUint::from(3u8),
            x: BigUint::from(1u8),
            score_match_pct: 80.0,
            message_bits: PackedBits::from_bools(&[
                true, true, true, true, true, true, true, true, true, false,
            ]),
            detail: None,
        }];

        let mut rng = RngChoice::from_seed(RngMode::Standard, 11);
        let result = run_sampled_avalanche_beam_search(
            &config.engine,
            &BigUint::from(1u8),
            &scored_inputs,
            1,
            false,
            &mut rng,
        )
        .expect("payload-only sampled avalanche scoring should succeed");

        let selected = result
            .selected_sample
            .expect("sampled avalanche should retain one selected sample");
        assert_eq!(selected.majority_vote_match_pct, 100.0);
        assert_eq!(selected.top_beam_match_pct, Some(100.0));
        assert_eq!(selected.best_match_pct, 100.0);
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_serializes_payload_only_display_bits() {
        let mut config = Config::default();
        config.engine.avalanche_statistics_collection = true;
        config
            .engine
            .avalanche_combination_keep_all_samples_in_memory = true;
        config.engine.avalanche_random_chacha20_inputs = true;
        config.engine.avalanche_combination_samples = 1;
        config.engine.avalanche_combination_size = 1;
        config.engine.avalanche_combination_mixed_r_candidates = 0;
        config.engine.avalanche_fitness_shift_bytes = 1;
        config.engine.message.bits = 2;
        config.engine.avalanche_beam_top_k = 1;

        let scored_inputs = vec![ScoredAvalancheInput {
            batch_candidate_index: 0,
            message_index: 0,
            r: BigUint::from(3u8),
            x: BigUint::from(1u8),
            score_match_pct: 80.0,
            message_bits: PackedBits::from_bools(&[
                true, true, true, true, true, true, true, true, true, false,
            ]),
            detail: Some(ScoredAvalancheInputDetail {
                target_exponent: BigDecimal::from(1u8),
                hbc_ciphertext_r: BigUint::from(1u8),
                candidate_decryption: BigUint::from(1u8),
            }),
        }];

        let mut rng = RngChoice::from_seed(RngMode::Standard, 17);
        let result = run_sampled_avalanche_beam_search(
            &config.engine,
            &BigUint::from(1u8),
            &scored_inputs,
            1,
            false,
            &mut rng,
        )
        .expect("payload-only sampled avalanche serialization should succeed");

        let retained = result
            .retained_samples
            .first()
            .expect("sampled avalanche should retain serialized sample");
        assert_eq!(retained.majority_vote_bits, vec![true, false]);
        assert_eq!(retained.beam_results.len(), 1);
        assert_eq!(retained.beam_results[0].hex, "01");
        assert_eq!(retained.beam_results[0].bit_width, 2);
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_records_recursive_tier_statistics() {
        let mut config = Config::default();
        config.engine.avalanche_random_chacha20_inputs = true;
        config.engine.avalanche_combination_samples = 4;
        config.engine.avalanche_combination_size = 1;
        config.engine.avalanche_combination_mixed_r_candidates = 0;
        config.engine.avalanche_combination_recursion_depth = 2;
        config.engine.avalanche_combination_recursive_group_size = vec![2];

        let scored_inputs = vec![
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 0,
                r: BigUint::from(3u8),
                x: BigUint::from(1u8),
                score_match_pct: 80.0,
                message_bits: PackedBits::from_bools(&[true, false]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 0,
                r: BigUint::from(5u8),
                x: BigUint::from(1u8),
                score_match_pct: 70.0,
                message_bits: PackedBits::from_bools(&[true, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 2,
                message_index: 0,
                r: BigUint::from(7u8),
                x: BigUint::from(1u8),
                score_match_pct: 60.0,
                message_bits: PackedBits::from_bools(&[false, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 3,
                message_index: 0,
                r: BigUint::from(11u8),
                x: BigUint::from(1u8),
                score_match_pct: 55.0,
                message_bits: PackedBits::from_bools(&[false, false]),
                detail: None,
            },
        ];

        let mut rng = RngChoice::from_seed(RngMode::Standard, 19);
        let result = run_sampled_avalanche_beam_search(
            &config.engine,
            &BigUint::from(1u8),
            &scored_inputs,
            1,
            false,
            &mut rng,
        )
        .expect("recursive sampled avalanche should succeed");

        assert_eq!(result.sample_count, 4);
        assert_eq!(result.tier_statistics.len(), 2);
        assert_eq!(result.tier_statistics[0].tier_index, 1);
        assert_eq!(result.tier_statistics[0].sample_count, 4);
        assert_eq!(result.tier_statistics[1].tier_index, 2);
        assert_eq!(result.tier_statistics[1].sample_count, 2);
        assert_eq!(result.final_tier_samples.len(), 2);
        assert!(result.selected_sample.is_some());
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_skips_statistics_when_disabled() {
        let mut config = Config::default();
        config.engine.avalanche_statistics_collection = false;
        config
            .engine
            .avalanche_combination_keep_all_samples_in_memory = true;
        config.engine.avalanche_random_chacha20_inputs = true;
        config.engine.avalanche_combination_samples = 4;
        config.engine.avalanche_combination_size = 1;
        config.engine.avalanche_combination_mixed_r_candidates = 0;
        config.engine.avalanche_combination_recursion_depth = 2;
        config.engine.avalanche_combination_recursive_group_size = vec![2];

        let scored_inputs = vec![
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 0,
                r: BigUint::from(3u8),
                x: BigUint::from(1u8),
                score_match_pct: 80.0,
                message_bits: PackedBits::from_bools(&[true, false]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 0,
                r: BigUint::from(5u8),
                x: BigUint::from(1u8),
                score_match_pct: 70.0,
                message_bits: PackedBits::from_bools(&[true, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 2,
                message_index: 0,
                r: BigUint::from(7u8),
                x: BigUint::from(1u8),
                score_match_pct: 60.0,
                message_bits: PackedBits::from_bools(&[false, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 3,
                message_index: 0,
                r: BigUint::from(11u8),
                x: BigUint::from(1u8),
                score_match_pct: 55.0,
                message_bits: PackedBits::from_bools(&[false, false]),
                detail: None,
            },
        ];

        let mut rng = RngChoice::from_seed(RngMode::Standard, 19);
        let result = run_sampled_avalanche_beam_search(
            &config.engine,
            &BigUint::from(1u8),
            &scored_inputs,
            1,
            false,
            &mut rng,
        )
        .expect("sampled avalanche without statistics should succeed");

        assert_eq!(result.sample_count, 4);
        assert!(result.tier_statistics.is_empty());
        assert!(result.retained_samples.is_empty());
        assert_eq!(result.final_tier_samples.len(), 2);
        assert!(result.selected_sample.is_some());
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_compacts_retained_inputs_before_recursion() {
        let mut config = Config::default();
        config
            .engine
            .avalanche_combination_keep_all_samples_in_memory = true;
        config.engine.avalanche_random_chacha20_inputs = true;
        config.engine.avalanche_combination_samples = 4;
        config.engine.avalanche_combination_size = 1;
        config.engine.avalanche_combination_mixed_r_candidates = 0;
        config.engine.avalanche_combination_recursion_depth = 2;
        config.engine.avalanche_combination_recursive_group_size = vec![2];

        let scored_inputs = (0..4usize)
            .map(|index| ScoredAvalancheInput {
                batch_candidate_index: index,
                message_index: 0,
                r: BigUint::from((index + 3) as u32),
                x: BigUint::from(1u8),
                score_match_pct: 80.0 - (index as f64 * 5.0),
                message_bits: PackedBits::from_bools(&[index % 2 == 0, index < 2]),
                detail: Some(ScoredAvalancheInputDetail {
                    target_exponent: BigDecimal::from(2u8),
                    hbc_ciphertext_r: BigUint::from((index + 10) as u32),
                    candidate_decryption: BigUint::from((index + 20) as u32),
                }),
            })
            .collect::<Vec<_>>();

        let mut rng = RngChoice::from_seed(RngMode::Standard, 41);
        let result = run_sampled_avalanche_beam_search(
            &config.engine,
            &BigUint::from(1u8),
            &scored_inputs,
            1,
            false,
            &mut rng,
        )
        .expect("recursive sampled avalanche with retained samples should succeed");

        assert_eq!(result.retained_samples.len(), 4);
        assert!(
            result
                .retained_samples
                .iter()
                .all(|sample| sample.inputs.is_empty())
        );
        assert_eq!(result.final_tier_samples.len(), 2);
    }

    #[test]
    fn test_recursive_tier_bits_use_top_beam_when_enabled() {
        let config = Config::default();
        let sample = SelectedAvalancheSample {
            sample_index: 1,
            tier_index: 2,
            input_count: 2,
            average_score_pct: 75.0,
            beam_results: Vec::new(),
            majority_vote_bits: vec![true, false, true],
            majority_vote_match_pct: 75.0,
            majority_vote_ones_match_pct: 100.0,
            best_bits: vec![true, false, true],
            top_beam_score: 0.0,
            top_beam_match_pct: None,
            best_match_pct: 75.0,
            center_biases: Vec::new(),
            node: AvalancheNode::new(vec![false, false, false], vec![2.5, -1.0, 4.0]),
        };

        assert_eq!(
            recursive_tier_bits(&sample, &config.engine),
            &[true, false, true]
        );
    }

    #[test]
    fn test_recursive_tier_bits_use_majority_vote_when_top_beam_disabled() {
        let mut config = Config::default();
        config.engine.avalanche_use_top_beam = false;
        let sample = SelectedAvalancheSample {
            sample_index: 1,
            tier_index: 2,
            input_count: 2,
            average_score_pct: 75.0,
            beam_results: Vec::new(),
            majority_vote_bits: vec![false, true, false],
            majority_vote_match_pct: 75.0,
            majority_vote_ones_match_pct: 100.0,
            best_bits: vec![true, false, true],
            top_beam_score: 0.0,
            top_beam_match_pct: None,
            best_match_pct: 75.0,
            center_biases: Vec::new(),
            node: AvalancheNode::new(vec![true, true, true], vec![2.5, -1.0, 4.0]),
        };

        assert_eq!(
            recursive_tier_bits(&sample, &config.engine),
            &[false, true, false]
        );
    }

    #[test]
    fn test_compact_recursive_avalanche_source_samples_use_configured_recursive_bits() {
        let sample = SelectedAvalancheSample {
            sample_index: 1,
            tier_index: 2,
            input_count: 2,
            average_score_pct: 75.0,
            beam_results: Vec::new(),
            majority_vote_bits: vec![false, true, false],
            majority_vote_match_pct: 75.0,
            majority_vote_ones_match_pct: 100.0,
            best_bits: vec![true, false, true],
            top_beam_score: 0.0,
            top_beam_match_pct: None,
            best_match_pct: 81.0,
            center_biases: Vec::new(),
            node: AvalancheNode::new(vec![true, true, true], vec![2.5, -1.0, 4.0]),
        };

        let top_beam_sources = compact_recursive_avalanche_source_samples(
            vec![sample.clone()],
            &Config::default().engine,
        );
        assert_eq!(top_beam_sources[0].best_match_pct, 81.0);
        assert_eq!(
            top_beam_sources[0].message_bits.to_bools(),
            vec![true, false, true]
        );

        let mut config = Config::default();
        config.engine.avalanche_use_top_beam = false;
        let majority_sources =
            compact_recursive_avalanche_source_samples(vec![sample], &config.engine);
        assert_eq!(
            majority_sources[0].message_bits.to_bools(),
            vec![false, true, false]
        );
    }

    #[test]
    fn test_build_center_bias_entries_filters_probabilities_near_half() {
        let entries = build_center_bias_entries(&[0.48, 0.505, 0.63, 0.495], 0.02);

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].bit_index_lsb0, 0);
        assert_eq!(entries[0].probability_one, 0.48);
        assert!((entries[0].signed_distance_from_half + 0.02).abs() < f64::EPSILON);
        assert_eq!(entries[1].bit_index_lsb0, 1);
        assert_eq!(entries[2].bit_index_lsb0, 3);
    }

    #[test]
    fn test_build_final_tier_bias_reports_preserves_sample_metadata() {
        let reports = build_final_tier_bias_reports(&[SelectedAvalancheSample {
            sample_index: 2,
            tier_index: 4,
            input_count: 3,
            average_score_pct: 72.0,
            beam_results: Vec::new(),
            majority_vote_bits: vec![true],
            majority_vote_match_pct: 72.0,
            majority_vote_ones_match_pct: 72.0,
            best_bits: vec![true],
            top_beam_score: 0.0,
            top_beam_match_pct: Some(72.0),
            best_match_pct: 72.0,
            center_biases: vec![AvalancheCenterBiasEntry {
                bit_index_lsb0: 5,
                probability_one: 0.501,
                signed_distance_from_half: 0.001,
            }],
            node: AvalancheNode::new(vec![true], vec![0.0]),
        }]);

        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].tier_index, 4);
        assert_eq!(reports[0].sample_index, 2);
        assert_eq!(reports[0].center_biases.len(), 1);
        assert_eq!(reports[0].center_biases[0].bit_index_lsb0, 5);
    }

    #[test]
    fn test_should_replace_selected_sample_prefers_match_pct_by_default() {
        let current = SelectedAvalancheSample {
            sample_index: 1,
            tier_index: 1,
            input_count: 2,
            average_score_pct: 60.0,
            beam_results: Vec::new(),
            majority_vote_bits: vec![true],
            majority_vote_match_pct: 60.0,
            majority_vote_ones_match_pct: 60.0,
            best_bits: vec![true],
            top_beam_score: 0.80,
            top_beam_match_pct: Some(60.0),
            best_match_pct: 60.0,
            center_biases: Vec::new(),
            node: AvalancheNode::new(vec![true], vec![0.0]),
        };
        let candidate = SelectedAvalancheSample {
            sample_index: 2,
            tier_index: 1,
            input_count: 2,
            average_score_pct: 55.0,
            beam_results: Vec::new(),
            majority_vote_bits: vec![false],
            majority_vote_match_pct: 75.0,
            majority_vote_ones_match_pct: 75.0,
            best_bits: vec![false],
            top_beam_score: 0.20,
            top_beam_match_pct: Some(75.0),
            best_match_pct: 75.0,
            center_biases: Vec::new(),
            node: AvalancheNode::new(vec![false], vec![0.0]),
        };

        assert!(should_replace_selected_sample(
            Some(&current),
            &candidate,
            false,
        ));
        assert!(!should_replace_selected_sample(
            Some(&current),
            &candidate,
            true,
        ));
    }

    #[test]
    fn test_should_replace_selected_sample_prefers_beam_score_in_public_mode() {
        let current = SelectedAvalancheSample {
            sample_index: 1,
            tier_index: 1,
            input_count: 2,
            average_score_pct: 60.0,
            beam_results: Vec::new(),
            majority_vote_bits: vec![true],
            majority_vote_match_pct: 60.0,
            majority_vote_ones_match_pct: 60.0,
            best_bits: vec![true],
            top_beam_score: 0.30,
            top_beam_match_pct: Some(90.0),
            best_match_pct: 90.0,
            center_biases: Vec::new(),
            node: AvalancheNode::new(vec![true], vec![0.0]),
        };
        let candidate = SelectedAvalancheSample {
            sample_index: 2,
            tier_index: 1,
            input_count: 2,
            average_score_pct: 65.0,
            beam_results: Vec::new(),
            majority_vote_bits: vec![false],
            majority_vote_match_pct: 55.0,
            majority_vote_ones_match_pct: 55.0,
            best_bits: vec![false],
            top_beam_score: 0.90,
            top_beam_match_pct: Some(55.0),
            best_match_pct: 55.0,
            center_biases: Vec::new(),
            node: AvalancheNode::new(vec![false], vec![0.0]),
        };

        assert!(should_replace_selected_sample(
            Some(&current),
            &candidate,
            true,
        ));
        assert!(!should_replace_selected_sample(
            Some(&current),
            &candidate,
            false,
        ));
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_discards_stored_sample_biases() {
        let mut config = Config::default();
        config.engine.avalanche_random_chacha20_inputs = true;
        config.engine.avalanche_combination_samples = 1;
        config.engine.avalanche_combination_size = 2;
        config.engine.avalanche_combination_mixed_r_candidates = 0;

        let scored_inputs = vec![
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 0,
                r: BigUint::from(3u8),
                x: BigUint::from(1u8),
                score_match_pct: 75.0,
                message_bits: PackedBits::from_bools(&[true, false]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 0,
                r: BigUint::from(5u8),
                x: BigUint::from(3u8),
                score_match_pct: 65.0,
                message_bits: PackedBits::from_bools(&[false, true]),
                detail: None,
            },
        ];

        let mut rng = RngChoice::from_seed(RngMode::Standard, 7);
        let result = run_sampled_avalanche_beam_search(
            &config.engine,
            &BigUint::from(1u8),
            &scored_inputs,
            1,
            false,
            &mut rng,
        )
        .expect("sampled avalanche should succeed");
        let selected_sample = result
            .selected_sample
            .expect("sampled avalanche should retain a selected sample");

        assert_eq!(
            selected_sample.node.biases,
            vec![0.0; selected_sample.node.bit_len()]
        );
    }

    #[test]
    fn test_build_unique_flat_sample_plans_return_distinct_sets_with_cross_plan_reuse() {
        let inputs = (0..5usize).collect::<Vec<_>>();

        let mut rng = RngChoice::from_seed(RngMode::Crypto, 23);
        let sample_plans = build_unique_flat_sample_plans(&inputs, 2, 4, &mut rng);
        let signatures = sample_plans
            .iter()
            .map(|plan| {
                let mut signature = plan.clone();
                signature.sort_unstable();
                signature
            })
            .collect::<Vec<_>>();
        let unique_signatures = signatures.iter().cloned().collect::<HashSet<_>>();
        let flattened = sample_plans
            .iter()
            .flat_map(|plan| plan.iter().copied())
            .collect::<Vec<_>>();
        let unique_inputs = flattened.iter().copied().collect::<HashSet<_>>();

        assert_eq!(sample_plans.len(), 4);
        assert!(sample_plans.iter().all(|plan| plan.len() == 2));
        assert!(
            sample_plans
                .iter()
                .all(|plan| { plan.iter().copied().collect::<HashSet<_>>().len() == plan.len() })
        );
        assert_eq!(unique_signatures.len(), sample_plans.len());
        assert!(flattened.len() > unique_inputs.len());
    }

    #[test]
    fn test_build_unique_grouped_sample_plans_return_distinct_sets_with_cross_plan_reuse() {
        let grouped_inputs = vec![
            vec![(0usize, 0usize), (0usize, 1usize), (0usize, 2usize)],
            vec![(1usize, 0usize), (1usize, 1usize), (1usize, 2usize)],
        ];

        let mut rng = RngChoice::from_seed(RngMode::Crypto, 29);
        let sample_plans = build_unique_grouped_sample_plans(&grouped_inputs, 1, 2, 4, &mut rng);
        let signatures = sample_plans
            .iter()
            .map(|plan| {
                let mut signature = plan.clone();
                signature.sort_unstable();
                signature
            })
            .collect::<Vec<_>>();
        let unique_signatures = signatures.iter().cloned().collect::<HashSet<_>>();
        let flattened = sample_plans
            .iter()
            .flat_map(|plan| plan.iter().copied())
            .collect::<Vec<_>>();
        let unique_inputs = flattened.iter().copied().collect::<HashSet<_>>();

        assert_eq!(sample_plans.len(), 4);
        assert!(sample_plans.iter().all(|plan| plan.len() == 2));
        assert!(
            sample_plans
                .iter()
                .all(|plan| { plan.iter().copied().collect::<HashSet<_>>().len() == plan.len() })
        );
        assert!(sample_plans.iter().all(|plan| {
            plan.iter()
                .map(|(batch_candidate_index, _)| *batch_candidate_index)
                .collect::<HashSet<_>>()
                .len()
                == 1
        }));
        assert_eq!(unique_signatures.len(), sample_plans.len());
        assert!(flattened.len() > unique_inputs.len());
    }

    #[test]
    fn test_resolve_recursive_avalanche_tier_config_reuses_last_array_entry() {
        let mut config = Config::default();
        config.engine.avalanche_combination_recursive_group_size = vec![5, 3];
        config.engine.avalanche_combination_recursive_resample_count = vec![11];

        let first_recursive_tier = resolve_recursive_avalanche_tier_config(&config.engine, 1);
        let second_recursive_tier = resolve_recursive_avalanche_tier_config(&config.engine, 2);
        let third_recursive_tier = resolve_recursive_avalanche_tier_config(&config.engine, 3);

        assert_eq!(first_recursive_tier.group_size, 5);
        assert_eq!(first_recursive_tier.resample_count, 11);
        assert_eq!(second_recursive_tier.group_size, 3);
        assert_eq!(second_recursive_tier.resample_count, 11);
        assert_eq!(third_recursive_tier.group_size, 3);
        assert_eq!(third_recursive_tier.resample_count, 11);
    }

    #[test]
    fn test_recursive_avalanche_sample_from_indices_uses_prior_tier_best_bits_when_enabled() {
        let mut config = Config::default();
        config.engine.avalanche_combination_majority_vote = true;
        config.engine.avalanche_beam_top_k = 1;

        let source_samples = vec![
            SelectedAvalancheSample {
                sample_index: 1,
                tier_index: 1,
                input_count: 1,
                average_score_pct: 60.0,
                beam_results: Vec::new(),
                majority_vote_bits: vec![false, false, false],
                majority_vote_match_pct: 100.0,
                majority_vote_ones_match_pct: 100.0,
                best_bits: vec![true, true, true],
                top_beam_score: 0.0,
                top_beam_match_pct: None,
                best_match_pct: 100.0,
                center_biases: Vec::new(),
                node: AvalancheNode::new(vec![false, false, false], vec![0.0, 0.0, 0.0]),
            },
            SelectedAvalancheSample {
                sample_index: 2,
                tier_index: 1,
                input_count: 1,
                average_score_pct: 60.0,
                beam_results: Vec::new(),
                majority_vote_bits: vec![false, false, false],
                majority_vote_match_pct: 100.0,
                majority_vote_ones_match_pct: 100.0,
                best_bits: vec![true, true, true],
                top_beam_score: 0.0,
                top_beam_match_pct: None,
                best_match_pct: 100.0,
                center_biases: Vec::new(),
                node: AvalancheNode::new(vec![false, false, false], vec![0.0, 0.0, 0.0]),
            },
        ];

        let compact_sources =
            compact_recursive_avalanche_source_samples(source_samples, &config.engine);
        let recursive = execute_recursive_avalanche_sample_from_indices(
            &config.engine,
            &[true, true, true],
            &[true, true, true],
            &compact_sources,
            &[1, 0],
            2,
            0,
        )
        .expect("recursive avalanche sample from indices should succeed");

        assert_eq!(recursive.majority_vote_bits, vec![true, true, true]);
    }

    #[test]
    fn test_recursive_avalanche_sample_from_indices_uses_prior_tier_majority_bits_when_disabled() {
        let mut config = Config::default();
        config.engine.avalanche_combination_majority_vote = true;
        config.engine.avalanche_beam_top_k = 1;
        config.engine.avalanche_use_top_beam = false;

        let source_samples = vec![
            SelectedAvalancheSample {
                sample_index: 1,
                tier_index: 1,
                input_count: 1,
                average_score_pct: 60.0,
                beam_results: Vec::new(),
                majority_vote_bits: vec![true, true, true],
                majority_vote_match_pct: 100.0,
                majority_vote_ones_match_pct: 100.0,
                best_bits: vec![false, false, false],
                top_beam_score: 0.0,
                top_beam_match_pct: None,
                best_match_pct: 100.0,
                center_biases: Vec::new(),
                node: AvalancheNode::new(vec![false, false, false], vec![0.0, 0.0, 0.0]),
            },
            SelectedAvalancheSample {
                sample_index: 2,
                tier_index: 1,
                input_count: 1,
                average_score_pct: 60.0,
                beam_results: Vec::new(),
                majority_vote_bits: vec![true, true, true],
                majority_vote_match_pct: 100.0,
                majority_vote_ones_match_pct: 100.0,
                best_bits: vec![false, false, false],
                top_beam_score: 0.0,
                top_beam_match_pct: None,
                best_match_pct: 100.0,
                center_biases: Vec::new(),
                node: AvalancheNode::new(vec![false, false, false], vec![0.0, 0.0, 0.0]),
            },
        ];

        let compact_sources =
            compact_recursive_avalanche_source_samples(source_samples, &config.engine);
        let recursive = execute_recursive_avalanche_sample_from_indices(
            &config.engine,
            &[true, true, true],
            &[true, true, true],
            &compact_sources,
            &[1, 0],
            2,
            0,
        )
        .expect("recursive avalanche sample from indices should succeed");

        assert_eq!(recursive.majority_vote_bits, vec![true, true, true]);
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_reuses_prior_tier_outputs_across_unique_groups() {
        let mut config = Config::default();
        config.engine.avalanche_random_chacha20_inputs = true;
        config.engine.avalanche_combination_samples = 4;
        config.engine.avalanche_combination_size = 1;
        config.engine.avalanche_combination_mixed_r_candidates = 0;
        config.engine.avalanche_combination_recursion_depth = 2;
        config.engine.avalanche_combination_recursive_group_size = vec![2];
        config.engine.avalanche_combination_recursive_resample_count = vec![5];

        let scored_inputs = vec![
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 0,
                r: BigUint::from(3u8),
                x: BigUint::from(1u8),
                score_match_pct: 80.0,
                message_bits: PackedBits::from_bools(&[true, false]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 0,
                r: BigUint::from(5u8),
                x: BigUint::from(1u8),
                score_match_pct: 70.0,
                message_bits: PackedBits::from_bools(&[true, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 2,
                message_index: 0,
                r: BigUint::from(7u8),
                x: BigUint::from(1u8),
                score_match_pct: 60.0,
                message_bits: PackedBits::from_bools(&[false, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 3,
                message_index: 0,
                r: BigUint::from(11u8),
                x: BigUint::from(1u8),
                score_match_pct: 55.0,
                message_bits: PackedBits::from_bools(&[false, false]),
                detail: None,
            },
        ];

        let mut rng = RngChoice::from_seed(RngMode::Standard, 37);
        let result = run_sampled_avalanche_beam_search(
            &config.engine,
            &BigUint::from(1u8),
            &scored_inputs,
            1,
            false,
            &mut rng,
        )
        .expect("recursive sampled avalanche unique input-set grouping should succeed");

        assert_eq!(result.tier_statistics.len(), 2);
        assert_eq!(result.tier_statistics[1].sample_count, 5);
        assert_eq!(result.final_tier_samples.len(), 5);
        assert_eq!(
            result.tier_statistics[1].source_kind,
            "recursive-resampled-samples"
        );
    }

    #[test]
    fn test_resolve_avalanche_cache_db_path_uses_seed() {
        assert_eq!(
            resolve_avalanche_cache_db_path(Some(42), "/tmp"),
            PathBuf::from("/tmp/rsa_avalanche_42.db")
        );
        assert_eq!(
            resolve_avalanche_cache_db_path(None, "/tmp/"),
            PathBuf::from("/tmp/rsa_avalanche_0.db")
        );
    }

    #[test]
    fn test_avalanche_cache_creates_intermediate_db_folders() {
        let temp_root = temp_path("cache_db_folder");
        let nested_folder = temp_root
            .join("mnt")
            .join("alternate_highspeed_device_mount")
            .join("tmp");
        let mut engine = EngineConfig::default();
        engine.sqlite_db_folder = format!("{}/", nested_folder.to_string_lossy());

        let cache =
            AvalancheCacheGuard::new(Some(10_003), &engine).expect("cache should initialize");

        assert!(nested_folder.is_dir());
        assert!(cache.path.exists());
        assert_eq!(cache.path, nested_folder.join("rsa_avalanche_10003.db"));

        drop(cache);
        let _ = fs::remove_dir_all(temp_root);
    }

    #[test]
    fn test_avalanche_cache_sets_sqlite_memory_pragmas() {
        let engine = EngineConfig::default();

        #[derive(Debug, QueryableByName)]
        struct SoftHeapLimitRow {
            #[diesel(sql_type = diesel::sql_types::BigInt, column_name = soft_heap_limit)]
            soft_heap_limit: i64,
        }

        #[derive(Debug, QueryableByName)]
        struct HardHeapLimitRow {
            #[diesel(sql_type = diesel::sql_types::BigInt, column_name = hard_heap_limit)]
            hard_heap_limit: i64,
        }

        #[derive(Debug, QueryableByName)]
        struct MmapSizeRow {
            #[diesel(sql_type = diesel::sql_types::BigInt, column_name = mmap_size)]
            mmap_size: i64,
        }

        let cache =
            AvalancheCacheGuard::new(Some(10_001), &engine).expect("cache should initialize");
        let mut connection = cache.pool().expect("pool should exist").get().unwrap();

        let soft_heap_limit = sql_query("PRAGMA soft_heap_limit")
            .get_result::<SoftHeapLimitRow>(&mut connection)
            .expect("soft heap limit pragma should be readable");
        let hard_heap_limit = sql_query("PRAGMA hard_heap_limit")
            .get_result::<HardHeapLimitRow>(&mut connection)
            .expect("hard heap limit pragma should be readable");
        let mmap_size = sql_query("PRAGMA mmap_size")
            .get_result::<MmapSizeRow>(&mut connection)
            .expect("mmap size pragma should be readable");

        assert_eq!(
            soft_heap_limit.soft_heap_limit,
            i64::try_from(engine.sqlite_soft_heap).unwrap()
        );
        assert_eq!(
            hard_heap_limit.hard_heap_limit,
            i64::try_from(engine.sqlite_hard_heap).unwrap()
        );
        assert!(mmap_size.mmap_size > 0);
        assert!(mmap_size.mmap_size <= i64::try_from(engine.sqlite_mmap_size).unwrap());
    }

    #[test]
    fn test_avalanche_cache_supports_shared_in_memory_mode() {
        let mut engine = EngineConfig::default();
        engine.sqlite_in_memory = true;

        #[derive(Debug, QueryableByName)]
        struct CountStarRow {
            #[diesel(sql_type = diesel::sql_types::BigInt, column_name = table_count)]
            count: i64,
        }

        let cache =
            AvalancheCacheGuard::new(Some(10_004), &engine).expect("cache should initialize");
        assert!(
            cache
                .path
                .to_string_lossy()
                .contains("mode=memory&cache=shared")
        );
        assert!(!cache.path.exists());

        let mut connection = cache.pool().expect("pool should exist").get().unwrap();
        let table_count = sql_query(
            "SELECT COUNT(*) AS table_count
             FROM sqlite_master
             WHERE type = 'table'
               AND name IN ('avalanche_cache_inputs', 'avalanche_cache_samples')",
        )
        .get_result::<CountStarRow>(&mut connection)
        .expect("schema should exist in shared in-memory cache");
        assert_eq!(table_count.count, 2);
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_cached_errors_when_fitness_threshold_removes_all_candidates()
     {
        let mut config = Config::default();
        config.engine.avalanche_random_chacha20_inputs = true;
        config.engine.avalanche_combination_samples = 1;
        config.engine.avalanche_combination_size = 1;
        config.engine.avalanche_combination_mixed_r_candidates = 0;
        config.engine.avalanche_fitness_scoring_pass = true;
        config.engine.avalanche_fitness_use_threshold = true;
        config.engine.avalanche_fitness_threshold = 0.580;
        config.engine.avalanche_fitness_bit_width = 4;
        config.engine.message.bits = 4;

        let scored_inputs = vec![
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 0,
                r: BigUint::from(3u8),
                x: BigUint::from(1u8),
                score_match_pct: 70.0,
                message_bits: PackedBits::from_bools(&[false, true, true, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 0,
                r: BigUint::from(5u8),
                x: BigUint::from(1u8),
                score_match_pct: 65.0,
                message_bits: PackedBits::from_bools(&[true, true, true, true]),
                detail: None,
            },
        ];

        let cache = AvalancheCacheGuard::new(Some(10_002), &config.engine)
            .expect("cache should initialize");
        cache.clear_batch(1).expect("cache clear should succeed");
        let ranked_inputs = scored_inputs
            .iter()
            .cloned()
            .map(|input| RankedScoredAvalancheInput {
                fitness: single_message_avalanche_fitness_score(&input.message_bits, 4),
                input,
            })
            .collect::<Vec<_>>();
        insert_cached_scored_inputs(&cache, 1, &ranked_inputs)
            .expect("cache insert should succeed");

        let mut rng = RngChoice::from_seed(RngMode::Standard, 19);
        let error = run_sampled_avalanche_beam_search_cached(
            &config.engine,
            &BigUint::from(1u8),
            &cache,
            1,
            false,
            &mut rng,
        )
        .expect_err("fitness threshold should reject an empty cached retained pool");

        assert!(error.to_string().contains("avalanche_fitness_threshold"));
    }

    #[test]
    fn test_run_sampled_avalanche_beam_search_cached_matches_in_memory_path() {
        let mut config = Config::default();
        config.engine.avalanche_random_chacha20_inputs = true;
        config.engine.avalanche_combination_samples = 1;
        config.engine.avalanche_combination_size = 2;
        config.engine.avalanche_combination_mixed_r_candidates = 0;
        config.engine.avalanche_combination_recursion_depth = 1;
        config.engine.avalanche_beam_top_k = 1;
        config.engine.avalanche_fitness_use_threshold = false;

        let scored_inputs = vec![
            ScoredAvalancheInput {
                batch_candidate_index: 0,
                message_index: 0,
                r: BigUint::from(3u8),
                x: BigUint::from(1u8),
                score_match_pct: 80.0,
                message_bits: PackedBits::from_bools(&[true, false, true]),
                detail: None,
            },
            ScoredAvalancheInput {
                batch_candidate_index: 1,
                message_index: 0,
                r: BigUint::from(5u8),
                x: BigUint::from(1u8),
                score_match_pct: 70.0,
                message_bits: PackedBits::from_bools(&[false, true, true]),
                detail: None,
            },
        ];

        let cache =
            AvalancheCacheGuard::new(Some(9_999), &config.engine).expect("cache should initialize");
        cache.clear_batch(1).expect("cache clear should succeed");
        let ranked_inputs = scored_inputs
            .iter()
            .cloned()
            .map(|input| RankedScoredAvalancheInput {
                fitness: single_message_avalanche_fitness_score(
                    &input.message_bits,
                    resolve_avalanche_fitness_bit_width(&config.engine),
                ),
                input,
            })
            .collect::<Vec<_>>();
        insert_cached_scored_inputs(&cache, 1, &ranked_inputs)
            .expect("cache insert should succeed");

        let mut in_memory_rng = RngChoice::from_seed(RngMode::Standard, 77);
        let in_memory = run_sampled_avalanche_beam_search(
            &config.engine,
            &BigUint::from(5u8),
            &scored_inputs,
            1,
            false,
            &mut in_memory_rng,
        )
        .expect("in-memory sampled avalanche should succeed");

        let mut cached_rng = RngChoice::from_seed(RngMode::Standard, 77);
        let cached = run_sampled_avalanche_beam_search_cached(
            &config.engine,
            &BigUint::from(5u8),
            &cache,
            1,
            false,
            &mut cached_rng,
        )
        .expect("cached sampled avalanche should succeed");

        let in_memory_selected = in_memory
            .selected_sample
            .expect("in-memory sampled avalanche should select one sample");
        let cached_selected = cached
            .selected_sample
            .expect("cached sampled avalanche should select one sample");
        assert_eq!(cached.sample_count, in_memory.sample_count);
        assert_eq!(cached.evaluated_candidates, in_memory.evaluated_candidates);
        assert_eq!(cached_selected.best_bits, in_memory_selected.best_bits);
        assert_eq!(
            cached_selected.majority_vote_bits,
            in_memory_selected.majority_vote_bits
        );
        assert_eq!(
            cached_selected.beam_results,
            in_memory_selected.beam_results
        );
        assert_eq!(
            cached.final_tier_samples.len(),
            in_memory.final_tier_samples.len()
        );
    }
}
