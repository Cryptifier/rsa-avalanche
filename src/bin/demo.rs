/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>

use std::{
    collections::HashSet,
    error::Error,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use clap::Parser;
use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};
use rand::RngCore;
use rayon::prelude::*;

use rsademo::config::{load_config, Config, EngineConfig};
use rsademo::math::{
    compute_totient, mod_inverse, modular_sqrt, random_biguint_bits, to_hex,
};
use rsademo::r_candidates::{generate_r_candidates_batch, RCandidateSettings};
use rsademo::rng::{RngChoice, RngMode};
use rsademo::avalanche::{search_avalanche_tree, AvalancheNode};
use rsademo::search::beam_search_top_k;

const DEFAULT_DEMO_BATCH_SIZE: u64 = 1000;

#[derive(Parser, Debug)]
#[command(name = "demo", about = "Speculative RSA decrypt demo", author, version)]
struct Args {
    /// Path to a JSON/JSON5 config file
    #[arg(short = 'c', long, default_value = "config/rsa_config.json")]
    config: String,

    /// Encrypt a plaintext (hex string) using the config RSA key
    #[arg(long, conflicts_with = "decrypt")]
    encrypt: bool,

    /// Decrypt a ciphertext using speculative oracle selection
    #[arg(long, conflicts_with = "encrypt")]
    decrypt: bool,

    /// Plaintext hex string (required with --encrypt)
    #[arg(long, value_name = "HEX")]
    plaintext_hex: Option<String>,

    /// Ciphertext to decrypt (overrides config.verify.ciphertext)
    #[arg(long, value_name = "VALUE")]
    ciphertext: Option<String>,

    /// Expected bit width for decrypted values
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=8192))]
    bits: Option<u32>,

    /// Multiply ciphertext by encrypted 2 before base conversion
    #[arg(long)]
    shift: bool,

    /// Enable batched speculative decryption with ciphertext exponent variants
    #[arg(long)]
    batch: bool,

    /// Number of ciphertext exponent variants to include in the batch
    #[arg(long = "batch-size", value_parser = clap::value_parser!(u64).range(1..))]
    batch_size: Option<u64>,

    /// Number of decrypt batches to run
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    batches: Option<u64>,
}

#[derive(Clone, Debug)]
struct DemoContext {
    n: BigUint,
    e: BigUint,
}

#[derive(Clone, Debug)]
struct OracleCandidate {
    r: BigUint,
    phi_new: BigUint,
    r_pow_y: BigUint,
}

#[derive(Clone, Debug)]
struct OracleBitSelection {
    oracle_idx: usize,
    invert: bool,
    match_pct: f64,
}

/// Entry point for the demo CLI.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error on failure.
///
/// # Expected Output
/// - Prints ciphertext or speculative decryption results.
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let config = load_config(&args.config)?;

    if !args.encrypt && !args.decrypt {
        return Err("choose --encrypt or --decrypt".into());
    }

    let ctx = build_demo_context(&config)?;

    if args.encrypt {
        let plaintext = args
            .plaintext_hex
            .as_ref()
            .ok_or("--plaintext-hex is required with --encrypt")?;
        let message = parse_biguint_arg(plaintext)?;
        if message.is_zero() {
            return Err("plaintext cannot be zero".into());
        }
        if message >= ctx.n {
            return Err("plaintext must be smaller than modulus n".into());
        }
        let ciphertext = message.modpow(&ctx.e, &ctx.n);
        println!("Ciphertext (hex): {}", to_hex(&ciphertext));
        return Ok(());
    }

    let ciphertext = resolve_ciphertext(&args, &config)?;
    let batch_size = resolve_demo_batch_size(&config.engine, &args)?;
    let batch_runs = args.batches.unwrap_or(1) as usize;
    let mut aggregate_max: Option<(BeamCandidateBits, usize)> = None;
    for batch_idx in 0..batch_runs {
        if batch_runs > 1 {
            println!("\n===== DEMO BATCH {} / {} =====", batch_idx + 1, batch_runs);
        }
        let start = std::time::Instant::now();
    let recovered = match run_speculative_decrypt(
        &ctx,
        &config,
        &ciphertext,
        args.shift,
        batch_size,
        args.bits,
    ) {
            Ok(result) => result,
            Err(err) => {
                eprintln!(
                    "Demo batch {} failed: {}",
                    batch_idx + 1,
                    err
                );
                continue;
            }
        };
        println!("Recovered (best-case) hex: {}", to_hex(&recovered.best_case));
        println!("Recovered (majority) hex: {}", to_hex(&recovered.majority));
        if let Some(candidate) = recovered.beam_top.clone() {
            let replace = match aggregate_max {
                Some((ref current, _)) => candidate.score > current.score,
                None => true,
            };
            if replace {
                aggregate_max = Some((candidate, batch_idx + 1));
            }
        }
        let duration_s = start.elapsed().as_secs_f64();
        println!(
            "Demo batch {} completed in {:.3}s",
            batch_idx + 1,
            duration_s
        );
    }
    if let Some((candidate, batch_number)) = aggregate_max {
        let mut hex = to_hex(&bits_le_to_biguint(&candidate.bits));
        let hex_len = (candidate.bits.len() + 3) / 4;
        if hex.len() < hex_len {
            let padding = "0".repeat(hex_len - hex.len());
            hex = format!("{}{}", padding, hex);
        }
        println!(
            "Avalanche beam aggregate max after {} batches: score {} batch {} hex {}",
            batch_runs,
            format_beam_float(candidate.score, BEAM_SCORE_DECIMALS),
            batch_number,
            hex
        );
        let max_value = bits_le_to_biguint(&candidate.bits);
        println!(
            "Avalanche beam aggregate max bits: total {} biguint {}",
            candidate.bits.len(),
            max_value.bits()
        );
        let msb = candidate.bits.last().copied().unwrap_or(false);
        println!(
            "Avalanche beam aggregate max MSB: {}",
            if msb { 1 } else { 0 }
        );
    } else if batch_runs > 1 {
        println!(
            "Avalanche beam aggregate max after {} batches: N/A",
            batch_runs
        );
    }

    Ok(())
}

#[derive(Debug)]
struct RecoveryResult {
    best_case: BigUint,
    majority: BigUint,
    beam_top: Option<BeamCandidateBits>,
}

#[derive(Debug, Clone)]
struct BeamCandidateBits {
    score: f64,
    bits: Vec<bool>,
}

/// Builds the demo RSA context from configuration.
///
/// # Parameters
/// - `config`: Loaded configuration with RSA key parameters.
///
/// # Returns
/// - `Result<DemoContext, Box<dyn Error>>`: Context with modulus and exponent.
///
/// # Expected Output
/// - Returns an error if the key material is missing or generated.
fn build_demo_context(config: &Config) -> Result<DemoContext, Box<dyn Error>> {
    if config.rsa_keypair.generate {
        return Err("demo requires rsa_keypair.generate = false".into());
    }
    let p = config
        .rsa_keypair
        .p
        .clone()
        .ok_or("config.rsa_keypair.p must be set")?;
    let q = config
        .rsa_keypair
        .q
        .clone()
        .ok_or("config.rsa_keypair.q must be set")?;
    let n = &p * &q;
    let e = BigUint::from(config.rsa_keypair.e);

    Ok(DemoContext { n, e })
}

/// Resolves the ciphertext input from CLI or config.
///
/// # Parameters
/// - `args`: Parsed CLI arguments.
/// - `config`: Loaded configuration with verify defaults.
///
/// # Returns
/// - `Result<BigUint, Box<dyn Error>>`: Ciphertext value.
///
/// # Expected Output
/// - Returns an error if no ciphertext is provided.
fn resolve_ciphertext(args: &Args, config: &Config) -> Result<BigUint, Box<dyn Error>> {
    if let Some(raw) = &args.ciphertext {
        return parse_biguint_arg(raw);
    }
    if let Some(hex) = &config.verify.ciphertext_hex {
        let trimmed = hex.trim();
        let prefixed = if trimmed.starts_with("0x") || trimmed.starts_with("0X") {
            trimmed.to_string()
        } else {
            format!("0x{}", trimmed)
        };
        return parse_biguint_arg(&prefixed);
    }
    if let Some(value) = &config.verify.ciphertext {
        return Ok(value.clone());
    }
    Err("ciphertext not provided; set --ciphertext or config.verify.ciphertext".into())
}

/// Resolves the demo batch size based on CLI arguments and config defaults.
///
/// # Parameters
/// - `engine`: Engine configuration containing batch defaults.
/// - `args`: Parsed CLI arguments.
///
/// # Returns
/// - `Result<usize, Box<dyn Error>>`: Batch size to use for demo decryption.
///
/// # Expected Output
/// - Returns the resolved batch size; no side effects.
fn resolve_demo_batch_size(engine: &EngineConfig, args: &Args) -> Result<usize, Box<dyn Error>> {
    let batch_enabled = engine.analysis_batch_enable || args.batch || args.batch_size.is_some();
    let size = if batch_enabled {
        if let Some(batch_size) = args.batch_size {
            batch_size
        } else if engine.analysis_batch_enable {
            engine.analysis_batch_messages.max(1)
        } else {
            DEFAULT_DEMO_BATCH_SIZE
        }
    } else {
        1
    };

    usize::try_from(size).map_err(|_| "demo batch size exceeds usize range".into())
}

/// Resolves the bit width for demo decryption outputs.
///
/// # Parameters
/// - `engine`: Engine configuration containing defaults.
/// - `n`: RSA modulus used for sizing defaults.
/// - `expected_bits`: Optional bit-width override from the CLI.
///
/// # Returns
/// - `Result<usize, Box<dyn Error>>`: Bit width to use for decoding.
///
/// # Expected Output
/// - Returns the resolved bit width; no stdout/stderr output.
fn resolve_demo_bit_width(
    engine: &EngineConfig,
    n: &BigUint,
    expected_bits: Option<u32>,
) -> Result<usize, Box<dyn Error>> {
    if let Some(bits) = expected_bits {
        return usize::try_from(bits).map_err(|_| "demo bits exceeds usize range".into());
    }
    Ok(analysis_bit_width(engine, n))
}

/// Computes an increasing odd exponent `x` per batch instance so that `e * x` remains odd.
///
/// # Parameters
/// - `e`: RSA public exponent.
/// - `instance_idx`: Zero-based batch instance index.
///
/// # Returns
/// - `Result<BigUint, Box<dyn Error>>`: Odd exponent value for the instance.
///
/// # Expected Output
/// - Returns the computed exponent; no side effects.
fn odd_ciphertext_exponent(
    e: &BigUint,
    instance_idx: usize,
) -> Result<BigUint, Box<dyn Error>> {
    if e.is_even() {
        return Err("demo requires an odd public exponent to keep e*x odd".into());
    }
    let idx =
        u64::try_from(instance_idx).map_err(|_| "demo batch message index exceeds u64 range")?;
    let x_value = idx
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or("demo batch message index exceeds u64 range")?;
    Ok(BigUint::from(x_value))
}

/// Runs the speculative decryption pipeline using r candidates.
///
/// # Parameters
/// - `ctx`: Demo context with modulus and exponent.
/// - `config`: Loaded configuration with engine settings.
/// - `ciphertext`: Ciphertext to decrypt.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
/// - `batch_size`: Number of ciphertext exponent variants to include.
/// - `expected_bits`: Optional expected bit width override.
///
/// # Returns
/// - `Result<RecoveryResult, Box<dyn Error>>`: Best-case and majority results.
///
/// # Expected Output
/// - Returns recovered messages; no stdout/stderr output.
fn run_speculative_decrypt(
    ctx: &DemoContext,
    config: &Config,
    ciphertext: &BigUint,
    shift: bool,
    batch_size: usize,
    expected_bits: Option<u32>,
) -> Result<RecoveryResult, Box<dyn Error>> {
    let engine = &config.engine;
    let mut rng = RngChoice::from_entropy(RngMode::Crypto)?;

    let settings = build_r_candidate_settings(engine);
    let candidate_batch_size = engine.process_count.max(engine.process_min_count).max(1) as usize;
    let candidates = generate_r_candidates_batch(&ctx.n, &settings, &mut rng, candidate_batch_size);
    if candidates.is_empty() {
        return Err("no r candidates generated for demo".into());
    }

    let y = engine.rabin_exponent as u32;
    let mut prepared = Vec::with_capacity(candidates.len());
    for (r, factors) in candidates {
        let phi_new = compute_totient(&factors);
        if mod_inverse(&ctx.e, &phi_new).is_some() {
            let r_pow_y = r.pow(y);
            prepared.push(OracleCandidate {
                r,
                phi_new,
                r_pow_y,
            });
        }
    }

    if prepared.is_empty() {
        return Err("no valid r candidates for demo".into());
    }
    if engine.same_r_batch && prepared.len() > 1 {
        let idx = (rng.next_u64() as usize) % prepared.len();
        let selected = prepared.swap_remove(idx);
        prepared.clear();
        prepared.push(selected);
    }

    let bit_width = resolve_demo_bit_width(engine, &ctx.n, expected_bits)?;
    let screen_iterations = engine.oracle_screen_iterations.max(1) as usize;
    let top_k = engine.combiner_k_oracles.max(1).min(prepared.len());
    let (per_bit_oracles, top_match_pct) = screen_oracles_per_bit(
        ctx,
        engine,
        &prepared,
        screen_iterations,
        top_k,
        &mut rng,
        shift,
        bit_width,
    )?;

    if let Some(stats) = compute_stats(&top_match_pct) {
        println!(
            "Screened per-bit top oracle match %: mean {:.2}, std dev {:.2}, min {:.2}, max {:.2}, n {}",
            stats.mean,
            stats.stddev,
            stats.min,
            stats.max,
            stats.count
        );
    }

    let (best_case_pct, best_case_bits) = compute_per_bit_best_case_match(
        ctx,
        engine,
        &prepared,
        &per_bit_oracles,
        ciphertext,
        shift,
        bit_width,
        batch_size,
    )?;
    if let Some(stats) = compute_stats(&best_case_pct) {
        println!(
            "Best-case per-bit estimated match %: mean {:.2}, std dev {:.2}, min {:.2}, max {:.2}, n {}",
            stats.mean,
            stats.stddev,
            stats.min,
            stats.max,
            stats.count
        );
    }

    let majority_bits = recover_message_bits_majority(
        ctx,
        engine,
        &prepared,
        &per_bit_oracles,
        ciphertext,
        shift,
        bit_width,
        batch_size,
    )?;
    let beam_top = run_avalanche_beam_search(
        ctx,
        engine,
        &prepared,
        ciphertext,
        shift,
        bit_width,
        batch_size,
        Some(&majority_bits),
    )?;

    Ok(RecoveryResult {
        best_case: bits_le_to_biguint(&best_case_bits),
        majority: bits_le_to_biguint(&majority_bits),
        beam_top,
    })
}

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
/// - Returns the distance; no stdout/stderr output.
fn hamming_distance_bits(left: &[bool], right: &[bool]) -> usize {
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
fn normalize_avalanche_biases(biases: &[f64]) -> Vec<f64> {
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

const BEAM_SCORE_DECIMALS: usize = 8;

/// Formats a floating-point value for beam search output.
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
fn format_beam_float(value: f64, precision: usize) -> String {
    format!("{:.precision$}", value, precision = precision)
}

/// Builds avalanche candidates from unique `(e * x)^{-1} mod phi(r)` decryptions.
///
/// # Parameters
/// - `ctx`: Demo context with modulus and exponent.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `candidates`: Prepared r candidates to use as oracles.
/// - `ciphertext`: Ciphertext to decrypt.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
/// - `bit_width`: Bit width used for candidate messages.
/// - `batch_size`: Number of ciphertext exponent variants to include.
/// - `reference_bits`: Optional reference bits for Hamming-distance sorting.
///
/// # Returns
/// - `Result<Vec<AvalancheNode>, Box<dyn Error>>`: Sorted avalanche nodes.
///
/// # Expected Output
/// - Returns candidate nodes; no stdout/stderr output.
fn build_avalanche_nodes_unique_d_demo(
    ctx: &DemoContext,
    engine: &EngineConfig,
    candidates: &[OracleCandidate],
    ciphertext: &BigUint,
    shift: bool,
    bit_width: usize,
    batch_size: usize,
    reference_bits: Option<&[bool]>,
) -> Result<Vec<AvalancheNode>, Box<dyn Error>> {
    if batch_size == 0 {
        return Err("demo avalanche batch size must be >= 1".into());
    }
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);
    let e_big = ctx.e.clone();
    let mut seen: Vec<HashSet<BigUint>> = vec![HashSet::new(); candidates.len()];
    let mut nodes_with_value: Vec<(BigUint, AvalancheNode)> = Vec::new();
    let mut nodes_with_distance: Vec<(usize, BigUint, AvalancheNode)> = Vec::new();

    for instance_idx in 0..batch_size {
        let x_big = odd_ciphertext_exponent(&e_big, instance_idx)?;
        let ciphertext_x = ciphertext.modpow(&x_big, &ctx.n);
        let shifted = maybe_shift_ciphertext(ctx, &ciphertext_x, shift);
        let result_default = get_larger_number(&shifted, &ctx.n, y, true, false);
        let e_x = &e_big * &x_big;

        for (candidate_idx, candidate) in candidates.iter().enumerate() {
            let Some(d_new) = mod_inverse(&e_x, &candidate.phi_new) else {
                continue;
            };
            let seen_set = &mut seen[candidate_idx];
            if !seen_set.insert(d_new.clone()) {
                continue;
            }
            let dm = derive_candidate_message_from_result(
                ctx,
                engine,
                &result_default,
                &candidate.r,
                &d_new,
                &n_pow_y,
                &candidate.r_pow_y,
                y,
                false,
            );
            let message_bits = biguint_to_bits_le(&dm, bit_width);
            let node = AvalancheNode {
                biases: vec![0.0; bit_width],
                message_bits,
            };
            let message_value = bits_le_to_biguint(&node.message_bits);
    if engine.use_hamming_distance {
        if let Some(reference) = reference_bits {
            let distance = hamming_distance_bits(&node.message_bits, reference);
            nodes_with_distance.push((distance, message_value, node));
        } else {
            nodes_with_value.push((message_value, node));
        }
    } else {
        nodes_with_value.push((message_value, node));
    }
        }
    }

    let mut nodes: Vec<AvalancheNode> = if engine.use_hamming_distance {
        if !nodes_with_distance.is_empty() {
            nodes_with_distance.sort_by(|a, b| {
                a.0.cmp(&b.0)
                    .then_with(|| a.1.cmp(&b.1))
            });
            nodes_with_distance
                .into_iter()
                .map(|(_, _, node)| node)
                .collect()
        } else {
            Vec::new()
        }
    } else if !nodes_with_value.is_empty() {
        nodes_with_value.sort_by(|a, b| a.0.cmp(&b.0));
        nodes_with_value
            .into_iter()
            .map(|(_, node)| node)
            .collect()
    } else {
        Vec::new()
    };

    if !nodes.is_empty() {
        let mut sorted_with_value: Vec<(BigUint, AvalancheNode)> = nodes
            .into_iter()
            .map(|node| (bits_le_to_biguint(&node.message_bits), node))
            .collect();
        sorted_with_value.sort_by(|a, b| a.0.cmp(&b.0));
        nodes = sorted_with_value
            .into_iter()
            .map(|(_, node)| node)
            .collect();
    }

    Ok(nodes)
}

/// Runs avalanche tree and beam search for demo candidates.
///
/// # Parameters
/// - `ctx`: Demo context with modulus and exponent.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `candidates`: Prepared r candidates to use as oracles.
/// - `ciphertext`: Ciphertext to decrypt.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
/// - `bit_width`: Bit width used for candidate messages.
/// - `batch_size`: Number of ciphertext exponent variants to include.
/// - `reference_bits`: Optional reference bits for Hamming-distance sorting.
///
/// # Returns
/// - `Result<Option<BeamCandidateBits>, Box<dyn Error>>`: Top candidate or `None` if unavailable.
///
/// # Expected Output
/// - Prints avalanche beam search results; no other side effects.
fn run_avalanche_beam_search(
    ctx: &DemoContext,
    engine: &EngineConfig,
    candidates: &[OracleCandidate],
    ciphertext: &BigUint,
    shift: bool,
    bit_width: usize,
    batch_size: usize,
    reference_bits: Option<&[bool]>,
) -> Result<Option<BeamCandidateBits>, Box<dyn Error>> {
    let avalanche_nodes = build_avalanche_nodes_unique_d_demo(
        ctx,
        engine,
        candidates,
        ciphertext,
        shift,
        bit_width,
        batch_size,
        reference_bits,
    )?;
    if avalanche_nodes.is_empty() {
        println!("Avalanche tree skipped: no unique decryptions");
        return Ok(None);
    }

    println!(
        "Avalanche tree instances: {}",
        avalanche_nodes.len()
    );
    let avalanche_result = search_avalanche_tree(avalanche_nodes)?;
    let normalized_biases = normalize_avalanche_biases(&avalanche_result.biases);
    let beam_result = beam_search_top_k(
        vec![Vec::new()],
        5,
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
                    let bias = normalized_biases
                        .get(idx)
                        .copied()
                        .unwrap_or(0.0);
                    if *bit >= 0.5 { bias } else { 1.0 - bias }
                })
                .sum()
        },
    )?;

    println!(
        "Avalanche beam search top {} candidates (lsb0 order):",
        beam_result.beam.len()
    );
    for (idx, candidate) in beam_result.beam.iter().enumerate() {
        let candidate_bits: Vec<bool> =
            candidate.vector.iter().map(|value| *value >= 0.5).collect();
        let mut candidate_hex = to_hex(&bits_le_to_biguint(&candidate_bits));
        let hex_len = (bit_width + 3) / 4;
        if candidate_hex.len() < hex_len {
            let padding = "0".repeat(hex_len - candidate_hex.len());
            candidate_hex = format!("{}{}", padding, candidate_hex);
        }
        println!(
            "Beam {} score {} hex {}",
            idx + 1,
            format_beam_float(candidate.score, BEAM_SCORE_DECIMALS),
            candidate_hex
        );
        let candidate_value = bits_le_to_biguint(&candidate_bits);
        println!(
            "Beam {} bits: total {} biguint {}",
            idx + 1,
            candidate_bits.len(),
            candidate_value.bits()
        );
    }
    if let Some(top) = beam_result.beam.first() {
        let top_bits: Vec<bool> = top.vector.iter().map(|value| *value >= 0.5).collect();
        let msb = top_bits.last().copied().unwrap_or(false);
        println!("Avalanche beam top MSB: {}", if msb { 1 } else { 0 });
    }

    let top = beam_result.beam.first().cloned().map(|candidate| BeamCandidateBits {
        score: candidate.score,
        bits: candidate.vector.iter().map(|value| *value >= 0.5).collect(),
    });
    Ok(top)
}

#[derive(Debug, Clone, Copy)]
struct StatSummary {
    mean: f64,
    stddev: f64,
    min: f64,
    max: f64,
    count: usize,
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

/// Computes the analysis bit width based on configuration and modulus bounds.
///
/// # Parameters
/// - `engine`: Engine configuration containing message bit-length hints.
/// - `n`: RSA modulus for upper bound sizing.
///
/// # Returns
/// - `usize`: Bit width used for analysis bit vectors.
///
/// # Expected Output
/// - Returns a positive width; no side effects.
fn analysis_bit_width(engine: &EngineConfig, n: &BigUint) -> usize {
    let mut bit_width = engine.message.bits.max(1) as usize;
    if !n.is_zero() {
        bit_width = bit_width.min(n.bits().max(1) as usize);
    }
    bit_width.max(1)
}

/// Samples a random message that is non-zero and less than `n`.
///
/// # Parameters
/// - `engine`: Engine configuration with message bit-length settings.
/// - `n`: Modulus bound; use zero to skip the bound.
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

/// Derives the candidate message given a precomputed first-stage result.
///
/// # Parameters
/// - `ctx`: Demo context containing key material.
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
    ctx: &DemoContext,
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
/// - `ctx`: Demo context containing key material.
/// - `ciphertext`: Ciphertext to optionally shift.
/// - `shift`: Whether to apply the shift.
///
/// # Returns
/// - `BigUint`: Shifted ciphertext when enabled, otherwise the original ciphertext.
///
/// # Expected Output
/// - Returns a ciphertext value; no side effects.
fn maybe_shift_ciphertext(ctx: &DemoContext, ciphertext: &BigUint, shift: bool) -> BigUint {
    if !shift {
        return ciphertext.clone();
    }
    let enc_two = BigUint::from(2u8).modpow(&ctx.e, &ctx.n);
    (ciphertext * enc_two) % &ctx.n
}

/// Screens r candidates to select top oracles per bit based on random-message matches.
///
/// # Parameters
/// - `ctx`: Demo context containing key material.
/// - `engine`: Engine configuration controlling oracle behavior.
/// - `candidates`: Prepared r candidates to evaluate.
/// - `iterations`: Number of random messages to use for screening.
/// - `top_k`: Number of top oracles to select per bit.
/// - `rng`: Random number generator for message sampling.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
/// - `bit_width`: Bit width for output vectors.
///
/// # Returns
/// - `Result<(Vec<Vec<OracleBitSelection>>, Vec<f64>), Box<dyn Error>>`: Per-bit oracle selection and top match %.
///
/// # Expected Output
/// - Prints screening progress; no file output.
fn screen_oracles_per_bit(
    ctx: &DemoContext,
    engine: &EngineConfig,
    candidates: &[OracleCandidate],
    iterations: usize,
    top_k: usize,
    rng: &mut RngChoice,
    shift: bool,
    bit_width: usize,
) -> Result<(Vec<Vec<OracleBitSelection>>, Vec<f64>), Box<dyn Error>> {
    if iterations == 0 {
        return Err("oracle_screen_iterations must be >= 1".into());
    }
    if candidates.is_empty() {
        return Err("no r candidates available for oracle screening".into());
    }
    let top_k = top_k.max(1).min(candidates.len());

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);

    struct ScreeningSample {
        result_default: BigUint,
        message_bits: Vec<bool>,
    }

    let seeds: Vec<u64> = (0..iterations).map(|_| rng.next_u64()).collect();
    let done = Arc::new(AtomicU64::new(0));
    let next_pct = Arc::new(AtomicU64::new(10));
    let iterations_u64 = iterations as u64;
    let samples: Vec<ScreeningSample> = seeds
        .into_par_iter()
        .map(|seed| {
            let mut local_rng = RngChoice::from_seed(rng.mode(), seed);
            let msg = random_message_under_n(engine, &ctx.n, &mut local_rng);
            let ciphertext = msg.modpow(&ctx.e, &ctx.n);
            let message_bits = biguint_to_bits_le(&msg, bit_width);
            let shifted = maybe_shift_ciphertext(ctx, &ciphertext, shift);
            let result_default = get_larger_number(&shifted, &ctx.n, y, true, false);

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

            ScreeningSample { result_default, message_bits }
        })
        .collect();

    let samples = Arc::new(samples);
    let counts: Vec<Vec<u32>> = candidates
        .par_iter()
        .map(|candidate| {
            let Some(d_new) = mod_inverse(&ctx.e, &candidate.phi_new) else {
                return vec![0u32; bit_width];
            };
            samples
                .par_iter()
                .map(|sample| {
                    let mut match_counts = vec![0u32; bit_width];
                    let dm = derive_candidate_message_from_result(
                        ctx,
                        engine,
                        &sample.result_default,
                        &candidate.r,
                        &d_new,
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
                    match_counts
                })
                .reduce(
                    || vec![0u32; bit_width],
                    |mut acc, counts| {
                        for (idx, value) in counts.into_iter().enumerate() {
                            acc[idx] = acc[idx].saturating_add(value);
                        }
                        acc
                    },
                )
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

        for (oracle_idx, pct, invert) in ranked.into_iter().take(top_k) {
            per_bit_oracles[bit_idx].push(OracleBitSelection {
                oracle_idx,
                invert,
                match_pct: pct,
            });
        }
    }

    Ok((per_bit_oracles, top_match_pct))
}

/// Builds oracle bit vectors for each batch instance using ciphertext exponent variants.
///
/// # Parameters
/// - `ctx`: Demo context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `candidates`: Prepared r candidates to use as oracles.
/// - `per_bit_oracles`: Per-bit oracle selection ranked by screening.
/// - `ciphertext`: Base ciphertext to exponentiate.
/// - `batch_size`: Number of ciphertext exponent variants to include.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
/// - `bit_width`: Bit width for output vectors.
///
/// # Returns
/// - `Result<Vec<Vec<Option<Vec<bool>>>>, Box<dyn Error>>`: Oracle bit vectors per batch instance.
///
/// # Expected Output
/// - Returns oracle bit vectors; no side effects.
fn build_oracle_bits_by_instance(
    ctx: &DemoContext,
    engine: &EngineConfig,
    candidates: &[OracleCandidate],
    per_bit_oracles: &[Vec<OracleBitSelection>],
    ciphertext: &BigUint,
    batch_size: usize,
    shift: bool,
    bit_width: usize,
) -> Result<Vec<Vec<Option<Vec<bool>>>>, Box<dyn Error>> {
    if batch_size == 0 {
        return Err("demo batch size must be >= 1".into());
    }

    let y = engine.rabin_exponent as u32;
    let n_pow_y = ctx.n.pow(y);
    let e_big = ctx.e.clone();

    let mut unique_oracle_indices = std::collections::HashSet::new();
    for selections in per_bit_oracles {
        for selection in selections {
            unique_oracle_indices.insert(selection.oracle_idx);
        }
    }
    let mut oracle_index_list: Vec<usize> = unique_oracle_indices.into_iter().collect();
    oracle_index_list.sort_unstable();

    let oracle_bits_by_instance: Vec<Vec<Option<Vec<bool>>>> = (0..batch_size)
        .into_par_iter()
        .map(|instance_idx| {
            let x_big =
                odd_ciphertext_exponent(&ctx.e, instance_idx).map_err(|err| err.to_string())?;
            let ciphertext_x = ciphertext.modpow(&x_big, &ctx.n);
            let shifted = maybe_shift_ciphertext(ctx, &ciphertext_x, shift);
            let result_default = get_larger_number(&shifted, &ctx.n, y, true, false);

            let mut oracle_bits: Vec<Option<Vec<bool>>> = vec![None; candidates.len()];
            for oracle_idx in oracle_index_list.iter().copied() {
                let candidate = &candidates[oracle_idx];
                let e_x = &e_big * &x_big;
                let Some(d_new) = mod_inverse(&e_x, &candidate.phi_new) else {
                    continue;
                };
                let dm = derive_candidate_message_from_result(
                    ctx,
                    engine,
                    &result_default,
                    &candidate.r,
                    &d_new,
                    &n_pow_y,
                    &candidate.r_pow_y,
                    y,
                    false,
                );
                oracle_bits[oracle_idx] = Some(biguint_to_bits_le(&dm, bit_width));
            }
            Ok::<_, String>(oracle_bits)
        })
        .collect::<Result<_, _>>()
        .map_err(|err| -> Box<dyn Error> { err.into() })?;

    Ok(oracle_bits_by_instance)
}

/// Computes weighted best-case match percentages and best-case bits for a ciphertext.
///
/// # Parameters
/// - `ctx`: Demo context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `candidates`: Prepared r candidates to use as oracles.
/// - `per_bit_oracles`: Per-bit oracle selection ranked by screening.
/// - `ciphertext`: Ciphertext to decrypt.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
/// - `bit_width`: Bit width for output vectors.
/// - `batch_size`: Number of ciphertext exponent variants to include.
/// - `batch_size`: Number of ciphertext exponent variants to include.
///
/// # Returns
/// - `Result<(Vec<f64>, Vec<bool>), Box<dyn Error>>`: Weighted match percentages and best-case bits.
///
/// # Expected Output
/// - Returns weighted match percentages and bits; no side effects.
fn compute_per_bit_best_case_match(
    ctx: &DemoContext,
    engine: &EngineConfig,
    candidates: &[OracleCandidate],
    per_bit_oracles: &[Vec<OracleBitSelection>],
    ciphertext: &BigUint,
    shift: bool,
    bit_width: usize,
    batch_size: usize,
) -> Result<(Vec<f64>, Vec<bool>), Box<dyn Error>> {
    if per_bit_oracles.is_empty() {
        return Err("per-bit oracle selection is empty".into());
    }
    let oracle_bits_by_instance = build_oracle_bits_by_instance(
        ctx,
        engine,
        candidates,
        per_bit_oracles,
        ciphertext,
        batch_size,
        shift,
        bit_width,
    )?;

    let mut results = (0..bit_width)
        .into_par_iter()
        .map(|bit_idx| {
            let selections: &[OracleBitSelection] =
                per_bit_oracles.get(bit_idx).map_or(&[], |v| v.as_slice());
            let mut score_one = 0.0;
            let mut score_zero = 0.0;
            for selection in selections {
                for oracle_bits in &oracle_bits_by_instance {
                    if let Some(bits) = oracle_bits
                        .get(selection.oracle_idx)
                        .and_then(|entry| entry.as_ref())
                    {
                        let bit = if selection.invert { !bits[bit_idx] } else { bits[bit_idx] };
                        if bit {
                            score_one += selection.match_pct;
                        } else {
                            score_zero += selection.match_pct;
                        }
                    }
                }
            }
            let total = (score_one + score_zero).max(1.0);
            let best_score = score_one.max(score_zero);
            let best_bit = if (score_one - score_zero).abs() < f64::EPSILON {
                engine.combiner_tie_breaker
            } else {
                score_one > score_zero
            };
            (bit_idx, best_score / total * 100.0, best_bit)
        })
        .collect::<Vec<_>>();

    results.sort_by_key(|(idx, _, _)| *idx);
    let mut per_bit_pct = Vec::with_capacity(bit_width);
    let mut best_case_bits = Vec::with_capacity(bit_width);
    for (_, pct, bit) in results {
        per_bit_pct.push(pct);
        best_case_bits.push(bit);
    }

    Ok((per_bit_pct, best_case_bits))
}

/// Recovers message bits via majority vote across per-bit oracle selections.
///
/// # Parameters
/// - `ctx`: Demo context containing key material.
/// - `engine`: Engine configuration controlling HBC behavior.
/// - `candidates`: Prepared r candidates to use as oracles.
/// - `per_bit_oracles`: Per-bit oracle selection ranked by screening.
/// - `ciphertext`: Ciphertext to decrypt.
/// - `shift`: Whether to shift ciphertext by encrypted 2 before conversion.
/// - `bit_width`: Bit width for output vectors.
///
/// # Returns
/// - `Result<Vec<bool>, Box<dyn Error>>`: Majority vote bit vector.
///
/// # Expected Output
/// - Returns a recovered bit vector; no side effects.
fn recover_message_bits_majority(
    ctx: &DemoContext,
    engine: &EngineConfig,
    candidates: &[OracleCandidate],
    per_bit_oracles: &[Vec<OracleBitSelection>],
    ciphertext: &BigUint,
    shift: bool,
    bit_width: usize,
    batch_size: usize,
) -> Result<Vec<bool>, Box<dyn Error>> {
    if per_bit_oracles.is_empty() {
        return Err("per-bit oracle selection is empty".into());
    }
    let oracle_bits_by_instance = build_oracle_bits_by_instance(
        ctx,
        engine,
        candidates,
        per_bit_oracles,
        ciphertext,
        batch_size,
        shift,
        bit_width,
    )?;

    let mut recovered = (0..bit_width)
        .into_par_iter()
        .map(|bit_idx| {
            let selections: &[OracleBitSelection] =
                per_bit_oracles.get(bit_idx).map_or(&[], |v| v.as_slice());
            let mut ones = 0usize;
            let mut zeros = 0usize;
            for selection in selections {
                for oracle_bits in &oracle_bits_by_instance {
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
            }
            let recovered_bit = if ones == zeros {
                engine.combiner_tie_breaker
            } else {
                ones > zeros
            };
            (bit_idx, recovered_bit)
        })
        .collect::<Vec<_>>();

    recovered.sort_by_key(|(idx, _)| *idx);
    let mut recovered_bits = Vec::with_capacity(bit_width);
    for (_, bit) in recovered {
        recovered_bits.push(bit);
    }

    Ok(recovered_bits)
}

/// Parses a big integer argument in decimal or hex form.
///
/// # Parameters
/// - `raw`: Raw CLI argument string.
///
/// # Returns
/// - `Result<BigUint, Box<dyn Error>>`: Parsed value or an error.
///
/// # Expected Output
/// - Returns an error on invalid input; no side effects.
fn parse_biguint_arg(raw: &str) -> Result<BigUint, Box<dyn Error>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("empty big integer value".into());
    }

    let (digits, radix) = if let Some(hex) = trimmed.strip_prefix("0x") {
        (hex, 16u32)
    } else if let Some(hex) = trimmed.strip_prefix("0X") {
        (hex, 16u32)
    } else {
        (trimmed, 10u32)
    };

    BigUint::parse_bytes(digits.as_bytes(), radix)
        .ok_or_else(|| format!("invalid big integer: {raw}").into())
}
