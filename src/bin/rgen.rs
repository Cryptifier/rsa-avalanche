/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>

use std::{
    collections::HashSet,
    error::Error,
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
};

use clap::{Parser, ValueEnum};
use num_bigint::BigUint;
use rsademo::config::{load_config, Config, EngineConfig};
use rsademo::math::random_prime_with_bits;
use rsademo::rng::{RngChoice, RngMode};
use rsademo::r_candidates::{generate_r_candidates, RCandidateMode, RCandidateSettings};

#[derive(Parser, Debug)]
#[command(
    name = "rgen",
    about = "Generate r candidates for analysis CSV reuse",
    author,
    version
)]
struct Args {
    /// Path to a JSON/JSON5 config file (defaults to rsa_config.json)
    #[arg(short = 'c', long, default_value = "rsa_config.json")]
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
    let rng_mode = if args.crypto_rng {
        RngMode::Crypto
    } else {
        RngMode::Standard
    };
    let mut rng: RngChoice = match args.seed {
        Some(seed) => RngChoice::from_seed(rng_mode, seed),
        None => RngChoice::from_entropy(rng_mode)?,
    };

    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| config.engine.reuse_r_candidates_path.clone());

    let settings = build_r_candidate_settings(&config.engine, &args, &output_path)?;
    let modulus = resolve_modulus(&args, &config, &mut rng)?;

    if settings.mode == RCandidateMode::Factoring && modulus.is_none() {
        return Err(
            "factoring mode requires --n, --p/--q, --bits, or rsa_config.json with p and q".into(),
        );
    }

    let n_for_generation = modulus
        .clone()
        .unwrap_or_else(|| BigUint::from(1u8));
    let candidates = generate_r_candidates(&n_for_generation, &settings, &mut rng);

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

    let header = build_header_lines(modulus.as_ref(), &settings, candidates.len());
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

    if let (Some(p), Some(q)) = (
        config.rsa_keypair.p.clone(),
        config.rsa_keypair.q.clone(),
    ) {
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
        reuse_r_candidates_path: output_path.to_string(),
        reuse_r_candidates: false,
        reuse_r_candidates_append_only: false,
        small_primes: small_primes.into_iter().map(BigUint::from).collect(),
        small_prime_factors_per_candidate: args
            .small_prime_factors
            .unwrap_or(engine.r_candidate_small_prime_factors),
        max_factors_per_candidate: args
            .max_factors
            .unwrap_or(engine.r_candidate_max_factors),
        target_bit_length: args.r_bits.or(engine.r_candidate_bit_length),
    })
}

/// Writes r candidates to a CSV file, optionally appending.
///
/// # Parameters
/// - `path`: Output CSV path.
/// - `entries`: Candidate entries to write.
/// - `append`: Whether to append instead of overwriting.
/// - `header_lines`: Comment header lines to include when creating a new file.
///
/// # Returns
/// - `Result<usize, Box<dyn Error>>`: Number of candidates written.
///
/// # Expected Output
/// - Writes to disk at `path`; may create or append the file.
fn write_candidates_csv(
    path: &str,
    entries: &[(BigUint, Vec<(BigUint, u64)>)],
    append: bool,
    header_lines: &[String],
) -> Result<usize, Box<dyn Error>> {
    let needs_header = if append {
        fs::metadata(path).map(|meta| meta.len() == 0).unwrap_or(true)
    } else {
        true
    };

    let mut file = if append {
        OpenOptions::new().create(true).append(true).open(path)?
    } else {
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?
    };

    if needs_header {
        for line in header_lines {
            writeln!(file, "{line}")?;
        }
    }

    for (r, factors) in entries {
        let factors_str = format_factors_csv(factors);
        writeln!(file, "{r},{factors_str}")?;
    }

    Ok(entries.len())
}

/// Deduplicates candidates against an existing CSV file (when appending).
///
/// # Parameters
/// - `path`: Output CSV path used to load existing candidates.
/// - `entries`: Newly generated candidate entries.
/// - `append`: Whether to load existing entries for deduplication.
///
/// # Returns
/// - `Result<(Vec<(BigUint, Vec<(BigUint, u64)>)>, usize, usize), Box<dyn Error>>`:
///   Filtered candidates, count of skipped duplicates, and count of existing entries.
///
/// # Expected Output
/// - Reads the existing CSV when `append` is true; no other side effects.
fn dedup_candidates(
    path: &str,
    entries: Vec<(BigUint, Vec<(BigUint, u64)>)>,
    append: bool,
) -> Result<(Vec<(BigUint, Vec<(BigUint, u64)>)>, usize, usize), Box<dyn Error>> {
    let mut seen = if append {
        load_existing_candidate_keys(path)?
    } else {
        HashSet::new()
    };
    let existing_count = seen.len();

    let mut skipped = 0usize;
    let mut filtered = Vec::with_capacity(entries.len());
    for (r, factors) in entries.into_iter() {
        let key = r.to_string();
        if seen.insert(key) {
            filtered.push((r, factors));
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
) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push("# rgen r_candidates.csv".to_string());
    lines.push(format!("# mode={}", mode_label(settings.mode)));
    lines.push(format!("# count={}", count));

    if let Some(n) = modulus {
        lines.push(format!("# n={}", n));
        lines.push(format!("# n_bits={}", n.bits()));
    }

    if settings.mode == RCandidateMode::Factoring {
        lines.push(format!("# min_factor={}", settings.process_min_factor));
        lines.push(format!("# scale={}", settings.process_scale));
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
            config: "rsa_config.json".to_string(),
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

        let settings = build_r_candidate_settings(&engine, &args, "r_candidates.csv")
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
            reuse_r_candidates_path: "r_candidates.csv".to_string(),
            reuse_r_candidates: false,
            reuse_r_candidates_append_only: false,
            small_primes: vec![3u8, 5u8, 7u8].into_iter().map(BigUint::from).collect(),
            small_prime_factors_per_candidate: 3,
            max_factors_per_candidate: 6,
            target_bit_length: Some(64),
        };

        let lines = build_header_lines(None, &settings, 2);
        let joined = lines.join("\n");
        assert!(joined.contains("mode=small_primes"));
        assert!(joined.contains("small_primes=3,5,7"));
        assert!(joined.contains("small_prime_factors=3"));
        assert!(joined.contains("max_factors=6"));
        assert!(joined.contains("target_bits=64"));
    }

    #[test]
    fn test_dedup_candidates_filters_existing() {
        let path = temp_path("dedup");
        let existing = "105,3^1;5^1;7^1\n";
        fs::write(&path, existing).expect("write failed");

        let entries = vec![
            (
                BigUint::from(105u64),
                vec![
                    (BigUint::from(3u8), 1),
                    (BigUint::from(5u8), 1),
                    (BigUint::from(7u8), 1),
                ],
            ),
            (
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
        assert_eq!(filtered[0].0, BigUint::from(1155u64));

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
