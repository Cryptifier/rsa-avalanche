/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use std::{
    collections::HashSet,
    error::Error,
    fs,
    io::{BufRead, BufReader},
};

use clap::{Parser, ValueEnum};
use num_bigint::BigUint;
use rsademo::config::{Config, EngineConfig, load_config};
use rsademo::math::random_prime_with_bits;
use rsademo::r_candidates::{
    RCandidate, RCandidateMode, RCandidateSettings, generate_r_candidates,
    generate_retargeted_r_candidates_batch, resolve_retargeted_r_candidates_path,
    write_candidates_csv,
};
use rsademo::rng::{RngChoice, RngMode};

#[derive(Parser, Debug)]
#[command(
    name = "rgen",
    about = "Generate r candidates for analysis CSV reuse",
    author,
    version
)]
struct Args {
    /// Path to a JSON/JSON5 config file (defaults to config/rsa_config_small_batch.json)
    #[arg(
        short = 'c',
        long,
        default_value = "config/rsa_config_small_batch.json"
    )]
    config: String,

    /// Output CSV path (defaults to config reuse_r_candidates_path)
    #[arg(short = 'o', long)]
    output: Option<String>,

    /// Append to the output CSV instead of overwriting
    #[arg(short = 'a', long)]
    append: bool,

    /// Deterministic RNG seed for reproducible candidate generation
    #[arg(long)]
    seed: Option<u64>,

    /// Use cryptographic RNGs for candidate generation
    #[arg(long)]
    crypto_rng: bool,

    /// RSA modulus n to target (decimal or 0x-prefixed hex)
    #[arg(long)]
    n: Option<String>,

    /// RSA prime p to compute n (decimal or 0x-prefixed hex)
    #[arg(long)]
    p: Option<String>,

    /// RSA prime q to compute n (decimal or 0x-prefixed hex)
    #[arg(long)]
    q: Option<String>,

    /// Generate random p/q with the given bit size (16..=63)
    #[arg(long, value_parser = clap::value_parser!(u32).range(16..=63))]
    bits: Option<u32>,

    /// Target bit length for r candidates (small-primes mode)
    #[arg(long, alias = "bit-length", value_name = "BITS")]
    r_bits: Option<u64>,

    /// Percent smaller than modulus bit length for r candidates (default 30 when flag is present)
    #[arg(long, value_name = "PCT", num_args = 0..=1, default_missing_value = "30")]
    r_bits_percent: Option<f64>,

    /// In factoring mode, sample candidate bounds from a random N^a window where a is in [0.8, 0.9]
    #[arg(long)]
    random_power_window: bool,

    /// Number of r candidates to generate (overrides config counts)
    #[arg(long)]
    count: Option<u64>,

    /// Minimum number of r candidates to generate
    #[arg(long, value_name = "MIN")]
    min_count: Option<u64>,

    /// Minimum prime factor for r candidates
    #[arg(long, value_name = "MIN")]
    min_factor: Option<u64>,

    /// Process scale for factoring-mode candidates
    #[arg(long)]
    scale: Option<u32>,

    /// Candidate generation mode (factoring or small_primes)
    #[arg(long, value_enum)]
    mode: Option<ModeArg>,

    /// Small primes list for small-primes mode (comma-separated)
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    small_primes: Vec<u64>,

    /// Small prime factor count per candidate
    #[arg(long, value_name = "COUNT")]
    small_prime_factors: Option<usize>,

    /// Maximum total factors per candidate (small-primes mode)
    #[arg(long, value_name = "COUNT")]
    max_factors: Option<usize>,

    /// Override candidate r to factor directly
    #[arg(long, value_name = "R")]
    override_r: Option<String>,

    /// Load the base reuse CSV, retarget up to `--count` samples with ChaCha20, and write a keyed retarget cache CSV
    #[arg(long)]
    retargeted: bool,
}

#[derive(Clone, Debug, ValueEnum)]
enum ModeArg {
    Factoring,
    SmallPrimes,
}

impl From<ModeArg> for RCandidateMode {
    fn from(value: ModeArg) -> Self {
        match value {
            ModeArg::Factoring => RCandidateMode::Factoring,
            ModeArg::SmallPrimes => RCandidateMode::SmallPrimes,
        }
    }
}

/// Entry point for the r candidate generator CLI.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error on failure.
///
/// # Expected Output
/// - Writes a CSV of r candidates and prints a summary to stdout.
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let config = load_config(&args.config)?;
    run_rgen(args, config)
}

/// Runs the r candidate generation flow.
///
/// # Parameters
/// - `args`: Parsed CLI arguments controlling key selection and output.
/// - `config`: Loaded configuration providing defaults.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error on failure.
///
/// # Expected Output
/// - Prints generation details and writes or appends to the output CSV.
fn run_rgen(args: Args, config: Config) -> Result<(), Box<dyn Error>> {
    let rng_mode = if args.retargeted || args.crypto_rng {
        RngMode::Crypto
    } else {
        RngMode::Standard
    };
    let mut rng: RngChoice = match args.seed {
        Some(seed) => RngChoice::from_seed(rng_mode, seed),
        None => RngChoice::from_entropy(rng_mode)?,
    };

    let modulus = resolve_modulus(&args, &config, &mut rng)?;
    let key_bit_width = resolve_key_bit_width(&args, &config, modulus.as_ref())?;
    let output_path = if let Some(path) = args.output.clone() {
        path
    } else if args.retargeted {
        resolve_retargeted_r_candidates_path(
            &config.engine.reuse_retargeted_r_candidates_path_prefix,
            key_bit_width,
        )
    } else {
        config.engine.reuse_r_candidates_path.clone()
    };
    let target_bit_length_override = resolve_target_bit_length_override(&args, modulus.as_ref())?;
    let settings = build_r_candidate_settings(
        &config.engine,
        &args,
        &output_path,
        target_bit_length_override,
        key_bit_width,
    )?;

    if (settings.mode == RCandidateMode::Factoring || args.retargeted) && modulus.is_none() {
        return Err(
            "factoring mode requires --n, --p/--q, --bits, or config/rsa_config_small_batch.json with p and q"
                .into(),
        );
    }

    let n_for_generation = modulus.clone().unwrap_or_else(|| BigUint::from(1u8));
    let candidates = if args.retargeted {
        generate_retargeted_r_candidates_batch(
            &n_for_generation,
            &settings,
            &mut rng,
            settings
                .process_count
                .max(settings.process_min_count)
                .max(1) as usize,
        )?
    } else {
        generate_r_candidates(&n_for_generation, &settings, &mut rng)
    };

    if candidates.is_empty() {
        return Err("no r candidates generated".into());
    }

    let total_generated = candidates.len();
    let (candidates, skipped, existing_count) =
        dedup_candidates(&output_path, candidates, args.append)?;
    if candidates.is_empty() {
        println!(
            "No new r candidates after dedup; {} duplicates skipped.",
            skipped
        );
        return Ok(());
    }

    let header = build_header_lines(
        modulus.as_ref(),
        &settings,
        candidates.len(),
        args.retargeted,
        key_bit_width,
    );
    let written = write_candidates_csv(&output_path, &candidates, args.append, &header)?;

    if args.append {
        println!(
            "Generated {} r candidates, skipped {} duplicates (existing: {}), wrote {} to {} (mode: {:?})",
            total_generated, skipped, existing_count, written, output_path, settings.mode
        );
    } else {
        println!(
            "Generated {} r candidates, skipped {} duplicates, wrote {} to {} (mode: {:?})",
            total_generated, skipped, written, output_path, settings.mode
        );
    }
    if args.retargeted {
        println!(
            "Retargeted cache generation used ChaCha20 and wrote keyed cache {}",
            output_path
        );
    }

    Ok(())
}

/// Resolves the RSA modulus from CLI overrides or configuration.
///
/// # Parameters
/// - `args`: CLI arguments with key overrides.
/// - `config`: Loaded configuration for fallback key material.
/// - `rng`: Random number generator used when generating primes.
///
/// # Returns
/// - `Result<Option<BigUint>, Box<dyn Error>>`: Modulus when available.
///
/// # Expected Output
/// - Returns `None` when no key material is provided; no side effects.
fn resolve_modulus(
    args: &Args,
    config: &Config,
    rng: &mut RngChoice,
) -> Result<Option<BigUint>, Box<dyn Error>> {
    let mut sources = 0u8;
    if args.n.is_some() {
        sources += 1;
    }
    if args.p.is_some() || args.q.is_some() {
        sources += 1;
    }
    if args.bits.is_some() {
        sources += 1;
    }

    if sources > 1 {
        return Err("choose only one key source: --n, --p/--q, or --bits".into());
    }

    if let Some(ref raw_n) = args.n {
        return Ok(Some(parse_biguint_arg(raw_n)?));
    }

    if args.p.is_some() || args.q.is_some() {
        let p_raw = args
            .p
            .as_ref()
            .ok_or("--p and --q must be provided together")?;
        let q_raw = args
            .q
            .as_ref()
            .ok_or("--p and --q must be provided together")?;
        let p = parse_biguint_arg(p_raw)?;
        let q = parse_biguint_arg(q_raw)?;
        return Ok(Some(p * q));
    }

    if let Some(bits) = args.bits {
        let p = random_prime_with_bits(bits, rng);
        let mut q = random_prime_with_bits(bits, rng);
        while q == p {
            q = random_prime_with_bits(bits, rng);
        }
        return Ok(Some(p * q));
    }

    if let (Some(p), Some(q)) = (config.rsa_keypair.p.clone(), config.rsa_keypair.q.clone()) {
        return Ok(Some(p * q));
    }

    Ok(None)
}

/// Builds r candidate settings with CLI overrides applied.
///
/// # Parameters
/// - `engine`: Engine configuration used for defaults.
/// - `args`: CLI arguments that override defaults.
/// - `output_path`: Output file path used for reuse metadata.
/// - `target_bit_length_override`: Optional target bit length override from percent-based settings.
/// - `key_bit_width`: Bit width of the original RSA key used to derive the retargeted cache path.
///
/// # Returns
/// - `Result<RCandidateSettings, Box<dyn Error>>`: Fully populated settings.
///
/// # Expected Output
/// - Returns a settings struct; no side effects.
fn build_r_candidate_settings(
    engine: &EngineConfig,
    args: &Args,
    output_path: &str,
    target_bit_length_override: Option<u64>,
    key_bit_width: u64,
) -> Result<RCandidateSettings, Box<dyn Error>> {
    let override_best_r = if let Some(ref raw) = args.override_r {
        Some(parse_biguint_arg(raw)?)
    } else if let Some(raw) = engine.override_best_r.as_ref() {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(parse_biguint_arg(trimmed)?)
        }
    } else {
        None
    };

    let mut process_count = engine.process_count;
    let mut process_min_count = engine.process_min_count;
    if let Some(count) = args.count {
        process_count = count;
        if args.min_count.is_none() {
            process_min_count = count;
        }
    }
    if let Some(min_count) = args.min_count {
        process_min_count = min_count;
    }

    let mode = args
        .mode
        .clone()
        .map(Into::into)
        .unwrap_or(engine.r_candidate_mode);

    let small_primes = if args.small_primes.is_empty() {
        engine.r_candidate_small_primes.clone()
    } else {
        args.small_primes.clone()
    };

    Ok(RCandidateSettings {
        mode,
        override_best_r,
        process_min_factor: BigUint::from(args.min_factor.unwrap_or(engine.process_min_factor)),
        process_count,
        process_min_count,
        process_scale: args.scale.unwrap_or(engine.process_scale),
        reuse_r_candidates_path: if args.retargeted {
            engine.reuse_r_candidates_path.clone()
        } else {
            output_path.to_string()
        },
        reuse_r_candidates: false,
        reuse_r_candidates_append_only: false,
        reuse_retargeted_r_candidates: false,
        reuse_retargeted_r_candidates_path: resolve_retargeted_r_candidates_path(
            &engine.reuse_retargeted_r_candidates_path_prefix,
            key_bit_width,
        ),
        small_primes: small_primes.into_iter().map(BigUint::from).collect(),
        small_prime_factors_per_candidate: args
            .small_prime_factors
            .unwrap_or(engine.r_candidate_small_prime_factors),
        max_factors_per_candidate: args.max_factors.unwrap_or(engine.r_candidate_max_factors),
        target_bit_length: args
            .r_bits
            .or(target_bit_length_override)
            .or(engine.r_candidate_bit_length),
        random_power_window: args.random_power_window || engine.r_candidate_random_power_window,
        target_exponent_minimum: engine.r_candidate_target_exponent_minimum.clone(),
        target_exponent: engine.r_candidate_target_exponent.clone(),
        retarget_partition_count: engine.r_candidate_retarget_partition_count,
        retarget_minimum_exponent: engine.r_candidate_retarget_minimum_exponent.clone(),
    })
}

/// Resolves the original RSA key bit width used to namespace retargeted cache files.
///
/// # Parameters
/// - `args`: CLI arguments that may override key material.
/// - `config`: Loaded configuration for fallback key material.
/// - `modulus`: Resolved modulus when available.
///
/// # Returns
/// - `Result<u64, Box<dyn Error>>`: Key bit width for the current run.
///
/// # Expected Output
/// - Returns a deterministic bit width or an error when no key material is available.
fn resolve_key_bit_width(
    args: &Args,
    config: &Config,
    modulus: Option<&BigUint>,
) -> Result<u64, Box<dyn Error>> {
    if let (Some(p_raw), Some(q_raw)) = (args.p.as_ref(), args.q.as_ref()) {
        let p = parse_biguint_arg(p_raw)?;
        let q = parse_biguint_arg(q_raw)?;
        return Ok(p.bits().saturating_add(q.bits()));
    }
    if let Some(bits) = args.bits {
        return Ok(u64::from(bits).saturating_mul(2));
    }
    if let (Some(p), Some(q)) = (config.rsa_keypair.p.as_ref(), config.rsa_keypair.q.as_ref()) {
        return Ok(p.bits().saturating_add(q.bits()));
    }
    if let Some(modulus) = modulus {
        return Ok(modulus.bits());
    }
    Err("unable to resolve key bit width for retargeted cache naming".into())
}

/// Resolves a target bit length override from the modulus and percent flag.
///
/// # Parameters
/// - `args`: CLI arguments that may include a percent reduction.
/// - `modulus`: Resolved RSA modulus used to compute the target length.
///
/// # Returns
/// - `Result<Option<u64>, Box<dyn Error>>`: Computed target bit length override when set.
///
/// # Expected Output
/// - Returns `Ok(None)` when no percent is provided; no side effects.
fn resolve_target_bit_length_override(
    args: &Args,
    modulus: Option<&BigUint>,
) -> Result<Option<u64>, Box<dyn Error>> {
    let Some(percent) = args.r_bits_percent else {
        return Ok(None);
    };

    if !(percent > 0.0 && percent < 100.0) || percent.is_nan() {
        return Err("--r-bits-percent must be greater than 0 and less than 100".into());
    }

    let modulus = modulus
        .ok_or("--r-bits-percent requires a modulus from config, --n, --p/--q, or --bits")?;
    let bits = modulus.bits();
    if bits == 0 {
        return Err("modulus bit length must be non-zero".into());
    }

    let target = ((bits as f64) * (1.0 - (percent / 100.0))).floor() as u64;
    if target == 0 {
        return Err("--r-bits-percent produces a zero bit-length target".into());
    }

    Ok(Some(target))
}

/// Deduplicates candidates against an existing CSV file (when appending).
///
/// # Parameters
/// - `path`: Output CSV path used to load existing candidates.
/// - `entries`: Newly generated candidate entries.
/// - `append`: Whether to load existing entries for deduplication.
///
/// # Returns
/// - `Result<(Vec<RCandidate>, usize, usize), Box<dyn Error>>`:
///   Filtered candidates, count of skipped duplicates, and count of existing entries.
///
/// # Expected Output
/// - Reads the existing CSV when `append` is true; no other side effects.
fn dedup_candidates(
    path: &str,
    entries: Vec<RCandidate>,
    append: bool,
) -> Result<(Vec<RCandidate>, usize, usize), Box<dyn Error>> {
    let mut seen = if append {
        load_existing_candidate_keys(path)?
    } else {
        HashSet::new()
    };
    let existing_count = seen.len();

    let mut skipped = 0usize;
    let mut filtered = Vec::with_capacity(entries.len());
    for candidate in entries.into_iter() {
        let key = candidate.r.to_string();
        if seen.insert(key) {
            filtered.push(candidate);
        } else {
            skipped += 1;
        }
    }

    Ok((filtered, skipped, existing_count))
}

/// Loads existing r candidate keys from a CSV file.
///
/// # Parameters
/// - `path`: Output CSV path to read.
///
/// # Returns
/// - `Result<HashSet<String>, Box<dyn Error>>`: Set of canonical r values.
///
/// # Expected Output
/// - Reads the file when it exists; prints parse errors for invalid rows.
fn load_existing_candidate_keys(path: &str) -> Result<HashSet<String>, Box<dyn Error>> {
    let mut keys = HashSet::new();
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(err) => {
            if err.kind() == std::io::ErrorKind::NotFound {
                return Ok(keys);
            }
            return Err(err.into());
        }
    };

    let reader = BufReader::new(file);
    for (idx, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(line) => line,
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
        if r_str.is_empty() {
            println!("Skipping line {} in reuse file: missing r entry", idx + 1);
            continue;
        }
        match r_str.parse::<BigUint>() {
            Ok(r) => {
                keys.insert(r.to_string());
            }
            Err(err) => {
                println!(
                    "Skipping line {} in reuse file: invalid r '{}': {}",
                    idx + 1,
                    r_str,
                    err
                );
            }
        }
    }

    Ok(keys)
}

/// Builds comment header lines for a new CSV file.
///
/// # Parameters
/// - `modulus`: Optional modulus used for factoring mode.
/// - `settings`: Candidate generation settings.
/// - `count`: Number of candidates generated.
/// - `retargeted`: Whether the header describes a keyed retargeted-cache file.
/// - `key_bit_width`: Bit width of the original RSA key associated with the output file.
///
/// # Returns
/// - `Vec<String>`: Header lines starting with `#`.
///
/// # Expected Output
/// - Returns header lines; no side effects.
fn build_header_lines(
    modulus: Option<&BigUint>,
    settings: &RCandidateSettings,
    count: usize,
    retargeted: bool,
    key_bit_width: u64,
) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(if retargeted {
        "# rgen data/retargeted_r_candidates.csv".to_string()
    } else {
        "# rgen data/r_candidates.csv".to_string()
    });
    lines.push(format!("# mode={}", mode_label(settings.mode)));
    lines.push(format!("# count={}", count));
    lines.push(format!("# key_bits={}", key_bit_width));
    lines.push(format!("# retargeted={}", retargeted));

    if let Some(n) = modulus {
        lines.push(format!("# n={}", n));
        lines.push(format!("# n_bits={}", n.bits()));
    }

    if retargeted {
        lines.push(format!(
            "# source_reuse_path={}",
            settings.reuse_r_candidates_path
        ));
    }

    if settings.mode == RCandidateMode::Factoring {
        lines.push(format!("# min_factor={}", settings.process_min_factor));
        lines.push(format!("# scale={}", settings.process_scale));
        lines.push(format!(
            "# random_power_window={}",
            settings.random_power_window
        ));
        if settings.random_power_window {
            lines.push("# random_power_window_exponent_range=0.8..=0.9".to_string());
        }
    }

    if settings.mode == RCandidateMode::SmallPrimes {
        let primes = settings
            .small_primes
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(",");
        lines.push(format!("# small_primes={}", primes));
        lines.push(format!(
            "# target_exponent_minimum={}",
            settings.target_exponent_minimum.normalized()
        ));
        lines.push(format!(
            "# target_exponent={}",
            settings.target_exponent.normalized()
        ));
        lines.push(format!(
            "# retarget_partition_count={}",
            settings.retarget_partition_count
        ));
        lines.push(format!(
            "# retarget_minimum_exponent={}",
            settings.retarget_minimum_exponent.normalized()
        ));
        lines.push(format!(
            "# small_prime_factors={}",
            settings.small_prime_factors_per_candidate
        ));
        lines.push(format!(
            "# max_factors={}",
            settings.max_factors_per_candidate
        ));
        if let Some(bits) = settings.target_bit_length {
            lines.push(format!("# target_bits={}", bits));
        }
    }

    lines
}

/// Parses a BigUint CLI argument (decimal or 0x-prefixed hex).
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

/// Returns a lowercase label for a candidate mode.
///
/// # Parameters
/// - `mode`: Candidate generation mode.
///
/// # Returns
/// - `&'static str`: Lowercase label string.
///
/// # Expected Output
/// - Returns a constant string; no side effects.
fn mode_label(mode: RCandidateMode) -> &'static str {
    match mode {
        RCandidateMode::Factoring => "factoring",
        RCandidateMode::SmallPrimes => "small_primes",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;
    use rsademo::rng::{RngChoice, RngMode};
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let mut rng = RngChoice::from_entropy(RngMode::Crypto).expect("rng entropy");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "rgen_{}_{}_{}.csv",
            name,
            std::process::id(),
            rng.next_u64()
        ));
        path
    }

    fn base_args() -> Args {
        Args {
            config: "config/rsa_config_small_batch.json".to_string(),
            output: None,
            append: false,
            seed: None,
            crypto_rng: false,
            n: None,
            p: None,
            q: None,
            bits: None,
            count: None,
            min_count: None,
            min_factor: None,
            scale: None,
            mode: None,
            small_primes: Vec::new(),
            small_prime_factors: None,
            max_factors: None,
            override_r: None,
            r_bits: None,
            r_bits_percent: None,
            random_power_window: false,
            retargeted: false,
        }
    }

    #[test]
    fn test_parse_biguint_arg_decimal_and_hex() {
        let val = parse_biguint_arg("12345").expect("decimal parse failed");
        assert_eq!(val, BigUint::from(12345u64));

        let hex = parse_biguint_arg("0xFF").expect("hex parse failed");
        assert_eq!(hex, BigUint::from(255u64));
    }

    #[test]
    fn test_build_r_candidate_settings_uses_overrides() {
        let mut engine = EngineConfig::default();
        engine.process_count = 5;
        engine.process_min_count = 4;
        engine.process_min_factor = 7;
        engine.process_scale = 11;
        engine.r_candidate_small_primes = vec![3, 5, 7, 11];
        engine.r_candidate_small_prime_factors = 3;
        engine.r_candidate_max_factors = 9;
        engine.r_candidate_bit_length = Some(64);
        engine.r_candidate_random_power_window = true;

        let mut args = base_args();
        args.count = Some(10);
        args.min_count = Some(8);
        args.min_factor = Some(13);
        args.scale = Some(7);
        args.mode = Some(ModeArg::SmallPrimes);
        args.small_primes = vec![3, 5, 7, 11, 13];
        args.small_prime_factors = Some(4);
        args.max_factors = Some(12);
        args.r_bits = Some(80);

        let settings =
            build_r_candidate_settings(&engine, &args, "data/r_candidates.csv", None, 512)
                .expect("settings failed");
        assert_eq!(settings.process_count, 10);
        assert_eq!(settings.process_min_count, 8);
        assert_eq!(settings.process_min_factor, BigUint::from(13u64));
        assert_eq!(settings.process_scale, 7);
        assert_eq!(settings.mode, RCandidateMode::SmallPrimes);
        assert_eq!(settings.small_primes.len(), 5);
        assert_eq!(settings.small_prime_factors_per_candidate, 4);
        assert_eq!(settings.max_factors_per_candidate, 12);
        assert_eq!(settings.target_bit_length, Some(80));
        assert!(settings.random_power_window);
        assert_eq!(
            settings.target_exponent_minimum,
            engine.r_candidate_target_exponent_minimum
        );
        assert_eq!(settings.target_exponent, engine.r_candidate_target_exponent);
    }

    #[test]
    fn test_resolve_key_bit_width_prefers_configured_primes() {
        let mut config = Config::default();
        config.rsa_keypair.generate = false;
        config.rsa_keypair.p = Some(BigUint::from(1u8) << 255);
        config.rsa_keypair.q = Some(BigUint::from(1u8) << 255);
        let args = base_args();
        let key_bits = resolve_key_bit_width(&args, &config, None).expect("key width failed");
        assert_eq!(key_bits, 512);
    }

    #[test]
    fn test_build_header_lines_small_primes_metadata() {
        let settings = RCandidateSettings {
            mode: RCandidateMode::SmallPrimes,
            override_best_r: None,
            process_min_factor: BigUint::from(3u8),
            process_count: 2,
            process_min_count: 1,
            process_scale: 8,
            reuse_r_candidates_path: "data/r_candidates.csv".to_string(),
            reuse_r_candidates: false,
            reuse_r_candidates_append_only: false,
            reuse_retargeted_r_candidates: false,
            reuse_retargeted_r_candidates_path: "data/rgen_retargeted_512.csv".to_string(),
            small_primes: vec![3u8, 5u8, 7u8].into_iter().map(BigUint::from).collect(),
            small_prime_factors_per_candidate: 3,
            max_factors_per_candidate: 6,
            target_bit_length: Some(64),
            random_power_window: true,
            target_exponent_minimum: bigdecimal::BigDecimal::parse_bytes(b"0.8", 10)
                .expect("valid exponent"),
            target_exponent: bigdecimal::BigDecimal::parse_bytes(b"2.005", 10)
                .expect("valid exponent"),
            retarget_partition_count: 3,
            retarget_minimum_exponent: bigdecimal::BigDecimal::parse_bytes(b"0.45", 10)
                .expect("valid minimum exponent"),
        };

        let lines = build_header_lines(None, &settings, 2, false, 512);
        let joined = lines.join("\n");
        assert!(joined.contains("mode=small_primes"));
        assert!(joined.contains("key_bits=512"));
        assert!(joined.contains("retargeted=false"));
        assert!(joined.contains("small_primes=3,5,7"));
        assert!(joined.contains("target_exponent_minimum=0.8"));
        assert!(joined.contains("target_exponent=2.005"));
        assert!(joined.contains("retarget_partition_count=3"));
        assert!(joined.contains("retarget_minimum_exponent=0.45"));
        assert!(joined.contains("small_prime_factors=3"));
        assert!(joined.contains("max_factors=6"));
        assert!(joined.contains("target_bits=64"));
    }

    #[test]
    fn test_build_header_lines_factoring_power_window_metadata() {
        let settings = RCandidateSettings {
            mode: RCandidateMode::Factoring,
            override_best_r: None,
            process_min_factor: BigUint::from(3u8),
            process_count: 2,
            process_min_count: 1,
            process_scale: 8,
            reuse_r_candidates_path: "data/r_candidates.csv".to_string(),
            reuse_r_candidates: false,
            reuse_r_candidates_append_only: false,
            reuse_retargeted_r_candidates: false,
            reuse_retargeted_r_candidates_path: "data/rgen_retargeted_512.csv".to_string(),
            small_primes: Vec::new(),
            small_prime_factors_per_candidate: 3,
            max_factors_per_candidate: 6,
            target_bit_length: None,
            random_power_window: true,
            target_exponent_minimum: bigdecimal::BigDecimal::parse_bytes(b"0.8", 10)
                .expect("valid exponent"),
            target_exponent: bigdecimal::BigDecimal::parse_bytes(b"2.005", 10)
                .expect("valid exponent"),
            retarget_partition_count: 3,
            retarget_minimum_exponent: bigdecimal::BigDecimal::parse_bytes(b"0.45", 10)
                .expect("valid minimum exponent"),
        };

        let lines = build_header_lines(Some(&BigUint::from(10_000u64)), &settings, 2, false, 512);
        let joined = lines.join("\n");
        assert!(joined.contains("mode=factoring"));
        assert!(joined.contains("random_power_window=true"));
        assert!(joined.contains("random_power_window_exponent_range=0.8..=0.9"));
    }

    #[test]
    fn test_build_header_lines_retargeted_metadata() {
        let settings = RCandidateSettings {
            mode: RCandidateMode::SmallPrimes,
            override_best_r: None,
            process_min_factor: BigUint::from(3u8),
            process_count: 2,
            process_min_count: 1,
            process_scale: 8,
            reuse_r_candidates_path: "data/rgen_output.csv".to_string(),
            reuse_r_candidates: false,
            reuse_r_candidates_append_only: false,
            reuse_retargeted_r_candidates: false,
            reuse_retargeted_r_candidates_path: "data/rgen_retargeted_512.csv".to_string(),
            small_primes: vec![3u8, 5u8, 7u8].into_iter().map(BigUint::from).collect(),
            small_prime_factors_per_candidate: 3,
            max_factors_per_candidate: 6,
            target_bit_length: Some(64),
            random_power_window: false,
            target_exponent_minimum: bigdecimal::BigDecimal::parse_bytes(b"0.8", 10)
                .expect("valid exponent"),
            target_exponent: bigdecimal::BigDecimal::parse_bytes(b"0.9", 10)
                .expect("valid exponent"),
            retarget_partition_count: 3,
            retarget_minimum_exponent: bigdecimal::BigDecimal::parse_bytes(b"0.45", 10)
                .expect("valid minimum exponent"),
        };

        let lines = build_header_lines(Some(&BigUint::from(3233u32)), &settings, 2, true, 512);
        let joined = lines.join("\n");
        assert!(joined.contains("retargeted=true"));
        assert!(joined.contains("source_reuse_path=data/rgen_output.csv"));
        assert!(joined.contains("n=3233"));
    }

    #[test]
    fn test_resolve_target_bit_length_override_percent() {
        let mut args = base_args();
        args.r_bits_percent = Some(30.0);
        let modulus = BigUint::from(1u8) << 99;
        let target = resolve_target_bit_length_override(&args, Some(&modulus))
            .expect("resolve percent failed")
            .expect("missing target bits");
        assert_eq!(target, 70);
    }

    #[test]
    fn test_dedup_candidates_filters_existing() {
        let path = temp_path("dedup");
        let existing = "105,3^1;5^1;7^1\n";
        fs::write(&path, existing).expect("write failed");

        let entries = vec![
            RCandidate::new(
                BigUint::from(105u64),
                vec![
                    (BigUint::from(3u8), 1),
                    (BigUint::from(5u8), 1),
                    (BigUint::from(7u8), 1),
                ],
            ),
            RCandidate::new(
                BigUint::from(1155u64),
                vec![
                    (BigUint::from(3u8), 1),
                    (BigUint::from(5u8), 1),
                    (BigUint::from(7u8), 1),
                    (BigUint::from(11u8), 1),
                ],
            ),
        ];

        let (filtered, skipped, existing_count) =
            dedup_candidates(path.to_str().unwrap(), entries, true).expect("dedup failed");
        assert_eq!(existing_count, 1);
        assert_eq!(skipped, 1);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].r, BigUint::from(1155u64));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_resolve_modulus_from_pq() {
        let mut args = base_args();
        args.p = Some("61".to_string());
        args.q = Some("53".to_string());
        let config = Config::default();
        let mut rng = RngChoice::from_seed(RngMode::Standard, 123);
        let modulus = resolve_modulus(&args, &config, &mut rng)
            .expect("resolve failed")
            .expect("missing modulus");
        assert_eq!(modulus, BigUint::from(61u64) * BigUint::from(53u64));
    }
}
