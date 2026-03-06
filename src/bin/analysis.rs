/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>

use std::{
    error::Error,
    sync::{Arc, Mutex},
};

use clap::Parser;
use rsademo::analytics::{AnalyticsCliArgs, SessionAnalytics, write_session_json};
use rsademo::config::load_config;
use rsademo::methods::{DemoArgs, run_demo};

#[derive(Parser, Debug)]
#[command(name = "analysis", about = "Lightweight RSA round-trip demo", author, version)]
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

    /// Number of r-candidate accuracy batches to run
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    batches: Option<u64>,

    /// Number of messages per accuracy batch
    #[arg(long = "batch-size", value_parser = clap::value_parser!(u64).range(1..))]
    batch_size: Option<u64>,

    /// Raise ciphertext to a monotonically increasing exponent per batch
    #[arg(long)]
    ciphertext_modify: bool,

    /// Reuse a single r candidate across each batch
    #[arg(long)]
    same_r_batch: bool,

    /// Sort avalanche candidates by Hamming distance
    #[arg(long = "use-hamming-distance")]
    use_hamming_distance: bool,
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
        use_hamming_distance: args.use_hamming_distance,
    })));

    let analytics_for_handler = Arc::clone(&analytics);
    ctrlc::set_handler(move || {
        if let Ok(mut guard) = analytics_for_handler.lock() {
            guard.finish(Some("interrupted".to_string()));
            let output_path = guard.session_json_path().to_string();
            if let Err(err) = write_session_json(&output_path, &guard) {
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
    };

    let result = run_demo(demo_args, config, &analytics);
    if let Ok(mut guard) = analytics.lock() {
        guard.finish(result.as_ref().err().map(|err| err.to_string()));
        let output_path = guard.session_json_path().to_string();
        if let Err(err) = write_session_json(&output_path, &guard) {
            eprintln!("Failed to write {}: {}", output_path, err);
        }
    }
    result
}
