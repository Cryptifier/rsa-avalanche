/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use std::{
    error::Error,
    sync::{Arc, Mutex},
};

use bigdecimal::BigDecimal;
use clap::Parser;
use rsademo::analytics::{AnalyticsCliArgs, SessionAnalytics};
use rsademo::config::load_config;
use rsademo::logs::write_session_log;
use rsademo::methods::{DemoArgs, run_demo};

#[derive(Parser, Debug)]
#[command(
    name = "analysis",
    about = "Lightweight RSA round-trip demo",
    author,
    version
)]
struct Args {
    /// Bit-length of the primes to generate (kept small for a quick demo)
    #[arg(short, long, default_value_t = 56, value_parser = clap::value_parser!(u32).range(16..=8192))]
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

    /// Use cryptographic RNGs for sampling and candidate generation
    #[arg(long)]
    crypto_rng: bool,

    /// Path to a JSON config matching the original rsa_demo.sage schema
    #[arg(short = 'c', long, default_value = "config/rsa_config.json")]
    config: String,

    /// Run extended analysis tests and sufficiency checks
    #[arg(long)]
    tests: bool,

    /// Export oracle entropy timeline charts
    #[arg(long)]
    export: bool,

    /// Output path for analytics session JSON
    #[arg(long, default_value = "session.json")]
    session_json: String,

    /// Multiply ciphertext by encrypted 2 before base conversion
    #[arg(long)]
    shift: bool,

    /// Report true match percentage without inversion adjustment
    #[arg(long = "true")]
    true_match: bool,

    /// Expected bit width for decrypted values
    #[arg(long = "bits-decrypt", value_parser = clap::value_parser!(u32).range(1..=8192))]
    bits_decrypt: Option<u32>,

    /// Number of r-candidate accuracy batches to run
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    batches: Option<u64>,

    /// Number of messages per accuracy batch
    #[arg(long = "batch-size", value_parser = clap::value_parser!(u64).range(1..))]
    batch_size: Option<u64>,

    /// Number of avalanche combination samples to evaluate per batch
    #[arg(long = "avalanche-combination-samples", value_parser = clap::value_parser!(u64).range(1..))]
    avalanche_combination_samples: Option<u64>,

    /// Number of scored candidates included in each avalanche combination sample
    #[arg(long = "avalanche-combination-size", value_parser = clap::value_parser!(u64).range(1..))]
    avalanche_combination_size: Option<u64>,

    /// Number of top scored candidates retained for avalanche combination sampling
    #[arg(long = "avalanche-combination-pool-size", value_parser = clap::value_parser!(u64).range(1..))]
    avalanche_combination_pool_size: Option<u64>,

    /// Whether sampled avalanche uses per-bit majority-vote probabilities from the combination outputs
    #[arg(long = "avalanche-combination-majority-vote")]
    avalanche_combination_majority_vote: Option<bool>,

    /// Whether sampled avalanche smooths per-bit majority-vote probabilities before beam search
    #[arg(long = "avalanche-combination-sample-smoothing")]
    avalanche_combination_sample_smoothing: Option<bool>,

    /// Whether sampled avalanche prints a separate majority-vote summary for the selected sample
    #[arg(long = "avalanche-combination-majority-vote-print")]
    avalanche_combination_majority_vote_print: Option<bool>,

    /// Raise ciphertext to a monotonically increasing exponent per batch
    #[arg(long)]
    ciphertext_modify: bool,

    /// Reuse a single r candidate across each batch
    #[arg(long)]
    same_r_batch: bool,

    /// Sort avalanche candidates by Hamming distance
    #[arg(long = "use-hamming-distance")]
    use_hamming_distance: bool,

    /// Legacy compatibility flag; inversion is now automatic in Hamming-distance mode
    #[arg(long = "mirror-invert-candidates")]
    mirror_invert_candidates: bool,

    /// Upper bound for the sampled total exponent used when retargeting speculative r candidates
    #[arg(long = "r-candidate-target-exponent")]
    r_candidate_target_exponent: Option<BigDecimal>,

    /// Lower bound for the sampled total exponent used when retargeting speculative r candidates
    #[arg(long = "r-candidate-target-exponent-minimum")]
    r_candidate_target_exponent_minimum: Option<BigDecimal>,
}

/// Entry point for the RSA round-trip demo CLI.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error on failure.
///
/// # Expected Output
/// - Prints key generation, encryption/decryption, and analysis results; may write output files.
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let mut config = load_config(&args.config)?;
    let mut batch_enable = config.engine.analysis_batch_enable;
    if let Some(batch_size) = args.batch_size {
        config.engine.analysis_batch_messages = batch_size;
        batch_enable = true;
    }
    if let Some(batch_count) = args.batches {
        config.engine.analysis_batch_batches = batch_count;
        batch_enable = true;
    }
    if let Some(sample_count) = args.avalanche_combination_samples {
        config.engine.avalanche_combination_samples = sample_count;
    }
    if let Some(sample_size) = args.avalanche_combination_size {
        config.engine.avalanche_combination_size = usize::try_from(sample_size)
            .map_err(|_| "avalanche combination size exceeds usize range")?;
    }
    if let Some(pool_size) = args.avalanche_combination_pool_size {
        config.engine.avalanche_combination_pool_size = usize::try_from(pool_size)
            .map_err(|_| "avalanche combination pool size exceeds usize range")?;
    }
    if let Some(majority_vote) = args.avalanche_combination_majority_vote {
        config.engine.avalanche_combination_majority_vote = majority_vote;
    }
    if let Some(sample_smoothing) = args.avalanche_combination_sample_smoothing {
        config.engine.avalanche_combination_sample_smoothing = sample_smoothing;
    }
    if let Some(majority_vote_print) = args.avalanche_combination_majority_vote_print {
        config.engine.avalanche_combination_majority_vote_print = majority_vote_print;
    }
    config.engine.analysis_batch_enable = batch_enable;
    if args.ciphertext_modify {
        config.engine.ciphertext_modify = true;
    }
    if args.same_r_batch {
        config.engine.same_r_batch = true;
    }
    if args.use_hamming_distance {
        config.engine.use_hamming_distance = true;
    }
    if args.mirror_invert_candidates {
        config.engine.mirror_invert_candidates = true;
    }
    if let Some(target_exponent) = &args.r_candidate_target_exponent {
        config.engine.r_candidate_target_exponent = target_exponent.clone();
    }
    if let Some(target_exponent_minimum) = &args.r_candidate_target_exponent_minimum {
        config.engine.r_candidate_target_exponent_minimum = target_exponent_minimum.clone();
    }
    let analytics = Arc::new(Mutex::new(SessionAnalytics::new(AnalyticsCliArgs {
        bits: args.bits,
        message_override: args.message.clone(),
        public_exponent: args.public_exponent,
        seed: args.seed,
        crypto_rng: args.crypto_rng,
        config_path: args.config.clone(),
        tests: args.tests,
        export: args.export,
        session_json: args.session_json.clone(),
        shift: args.shift,
        ciphertext_modify: args.ciphertext_modify,
        use_hamming_distance: config.engine.use_hamming_distance,
        mirror_invert_candidates: config.engine.mirror_invert_candidates,
        beam_bit_one_threshold: config.engine.beam_bit_one_threshold,
        avalanche_beam_top_k: config.engine.avalanche_beam_top_k,
        avalanche_probability_spread_exponent: config.engine.avalanche_probability_spread_exponent,
        avalanche_combination_samples: config.engine.avalanche_combination_samples,
        avalanche_combination_size: config.engine.avalanche_combination_size,
        avalanche_combination_mixed_r_candidates: config
            .engine
            .avalanche_combination_mixed_r_candidates,
        avalanche_combination_pool_size: config.engine.avalanche_combination_pool_size,
        avalanche_combination_majority_vote: config.engine.avalanche_combination_majority_vote,
        avalanche_combination_sample_smoothing: config
            .engine
            .avalanche_combination_sample_smoothing,
        avalanche_combination_majority_vote_print: config
            .engine
            .avalanche_combination_majority_vote_print,
        bits_decrypt: args.bits_decrypt,
        r_candidate_target_exponent: args
            .r_candidate_target_exponent
            .as_ref()
            .map(|value| value.normalized().to_string()),
        r_candidate_target_exponent_minimum: args
            .r_candidate_target_exponent_minimum
            .as_ref()
            .map(|value| value.normalized().to_string()),
    })));

    let analytics_for_handler = Arc::clone(&analytics);
    ctrlc::set_handler(move || {
        if let Ok(mut guard) = analytics_for_handler.lock() {
            guard.finish(Some("interrupted".to_string()));
            let output_path = guard.session_json_path().to_string();
            if let Err(err) = write_session_log(&output_path, &guard) {
                eprintln!("Failed to write {}: {}", output_path, err);
            }
        }
        std::process::exit(130);
    })?;

    let demo_args = DemoArgs {
        bits: args.bits,
        message: args.message.clone(),
        public_exponent: args.public_exponent,
        seed: args.seed,
        crypto_rng: args.crypto_rng,
        tests: args.tests,
        export: args.export,
        shift: args.shift,
        true_match: args.true_match,
        bits_decrypt: args.bits_decrypt,
    };

    let result = run_demo(demo_args, config, &analytics);
    if let Ok(mut guard) = analytics.lock() {
        guard.finish(result.as_ref().err().map(|err| err.to_string()));
        let output_path = guard.session_json_path().to_string();
        if let Err(err) = write_session_log(&output_path, &guard) {
            eprintln!("Failed to write {}: {}", output_path, err);
        }
    }
    result
}
