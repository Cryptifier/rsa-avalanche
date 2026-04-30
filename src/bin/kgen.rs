/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use std::{
    error::Error,
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use clap::{Parser, ValueEnum};
use num_bigint::BigUint;
use num_traits::One;
use rsademo::config::{RsaKeyFileFormat, load_rsa_key_material_from_yaml_path};
use rsademo::math::{choose_exponent, mod_inverse, random_prime_with_bits};
use rsademo::rng::{RngChoice, RngMode};
use serde::Serialize;

const DEFAULT_OUTPUT_PATH: &str = "config/keys/private_key.yaml";
const DEFAULT_PRIME_BITS: u32 = 56;
const DEFAULT_MODULUS_BITS: u32 = 144;
const DEFAULT_PUBLIC_EXPONENT: u64 = 65_537;
const MODULUS_MODE_ATTEMPT_LIMIT: usize = 10_000;

#[derive(Parser, Debug)]
#[command(
    name = "kgen",
    about = "Generate RSA private keys and emit matching public-key YAML",
    author,
    version
)]
struct Args {
    /// Whether sizing is driven by prime bits or exact modulus bits
    #[arg(long, value_enum, default_value_t = SizeMode::Prime)]
    size_mode: SizeMode,

    /// Prime bit length used in prime-sized generation mode
    #[arg(long, default_value_t = DEFAULT_PRIME_BITS, value_parser = clap::value_parser!(u32).range(16..=8192))]
    prime_bits: u32,

    /// Exact modulus bit length targeted in modulus-sized generation mode
    #[arg(long, default_value_t = DEFAULT_MODULUS_BITS, value_parser = clap::value_parser!(u32).range(32..=16384))]
    modulus_bits: u32,

    /// Starting public exponent candidate
    #[arg(short = 'e', long, default_value_t = DEFAULT_PUBLIC_EXPONENT)]
    public_exponent: u64,

    /// YAML output path for the generated private key
    #[arg(short = 'o', long, default_value = DEFAULT_OUTPUT_PATH)]
    output: String,

    /// Optional YAML output path for the public key corresponding to the generated or imported private key
    #[arg(long)]
    public_output: Option<String>,

    /// Existing rsa-private-key-v1 YAML file to convert into rsa-public-key-v1 output
    #[arg(long)]
    input_private_key: Option<String>,

    /// Overwrite the output path if it already exists
    #[arg(long)]
    force: bool,

    /// Deterministic RNG seed for reproducible key generation
    #[arg(long)]
    seed: Option<u64>,

    /// Use cryptographic RNGs for key generation
    #[arg(long)]
    crypto_rng: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum SizeMode {
    Prime,
    Modulus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum KeyRngMode {
    Standard,
    Crypto,
}

#[derive(Debug)]
struct GeneratedPrivateKey {
    p: BigUint,
    q: BigUint,
    n: BigUint,
    phi: BigUint,
    e: BigUint,
    d: BigUint,
}

#[derive(Debug, Serialize)]
struct RsaPrivateKeyFile {
    format: String,
    algorithm: String,
    public_exponent: String,
    private_exponent: String,
    modulus: String,
    totient: String,
    primes: RsaPrimePair,
    bit_lengths: RsaKeyBitLengths,
    generation: RsaKeyGeneration,
}

#[derive(Debug, Serialize)]
struct RsaPublicKeyFile {
    format: String,
    algorithm: String,
    public_exponent: String,
    modulus: String,
    bit_lengths: RsaPublicKeyBitLengths,
}

#[derive(Debug, Serialize)]
struct RsaPrimePair {
    p: String,
    q: String,
}

#[derive(Debug, Serialize)]
struct RsaKeyBitLengths {
    requested_prime_bits: u32,
    requested_modulus_bits: u32,
    prime_p_bits: u64,
    prime_q_bits: u64,
    modulus_bits: u64,
}

#[derive(Debug, Serialize)]
struct RsaPublicKeyBitLengths {
    modulus_bits: u64,
}

#[derive(Debug, Serialize)]
struct RsaKeyGeneration {
    size_mode: SizeMode,
    requested_public_exponent: u64,
    rng_mode: KeyRngMode,
    seed: Option<u64>,
    created_unix_ms: u128,
}

/// Entry point for the YAML RSA key generator.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error on failure.
///
/// # Expected Output
/// - Writes a YAML private-key document and prints a generation summary to stdout.
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    run_kgen(args)
}

/// Runs the RSA private-key generation or public-key conversion flow.
///
/// # Parameters
/// - `args`: Parsed CLI arguments controlling sizing, RNG behavior, outputs, and optional private-key input.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` on success or an error on failure.
///
/// # Expected Output
/// - Writes generated private/public YAML files or converts a private YAML into public-key YAML and prints a summary to stdout.
fn run_kgen(args: Args) -> Result<(), Box<dyn Error>> {
    let rng_mode = if args.crypto_rng {
        KeyRngMode::Crypto
    } else {
        KeyRngMode::Standard
    };
    let (key, generated_new_private_key) =
        if let Some(input_private_key) = args.input_private_key.as_deref() {
            if args.public_output.is_none() {
                return Err("--input-private-key requires --public-output".into());
            }
            (load_private_key_from_yaml(input_private_key)?, false)
        } else {
            let mut rng = build_rng(&args, rng_mode)?;
            (generate_private_key(&args, &mut rng)?, true)
        };
    validate_generated_key(&key)?;

    if generated_new_private_key {
        let document = build_private_key_document(&args, &key, rng_mode)?;
        write_yaml_file(&args.output, &document, args.force)?;
    }
    if let Some(public_output) = args.public_output.as_deref() {
        let public_document = build_public_key_document(&key);
        write_yaml_file(public_output, &public_document, args.force)?;
    }

    if generated_new_private_key {
        println!(
            "Generated RSA private key: mode {} p-bits {} q-bits {} modulus-bits {} e {} output {}",
            size_mode_label(args.size_mode),
            key.p.bits(),
            key.q.bits(),
            key.n.bits(),
            key.e,
            args.output
        );
        if key.e != BigUint::from(args.public_exponent) {
            println!(
                "Adjusted public exponent from {} to {} to satisfy odd coprime RSA requirements",
                args.public_exponent, key.e
            );
        }
    } else {
        println!(
            "Loaded RSA private key: p-bits {} q-bits {} modulus-bits {} e {}",
            key.p.bits(),
            key.q.bits(),
            key.n.bits(),
            key.e,
        );
    }
    if let Some(public_output) = args.public_output.as_deref() {
        println!("Wrote RSA public key output {}", public_output);
    }

    Ok(())
}

/// Builds the configured random number generator.
///
/// # Parameters
/// - `args`: Parsed CLI arguments containing seed and RNG mode settings.
/// - `rng_mode`: Resolved RNG mode label for the generator.
///
/// # Returns
/// - `Result<RngChoice, Box<dyn Error>>`: Ready-to-use RNG instance.
///
/// # Expected Output
/// - Returns a seeded or entropy-backed RNG; no side effects.
fn build_rng(args: &Args, rng_mode: KeyRngMode) -> Result<RngChoice, Box<dyn Error>> {
    let mode = match rng_mode {
        KeyRngMode::Standard => RngMode::Standard,
        KeyRngMode::Crypto => RngMode::Crypto,
    };

    match args.seed {
        Some(seed) => Ok(RngChoice::from_seed(mode, seed)),
        None => Ok(RngChoice::from_entropy(mode)?),
    }
}

/// Generates a complete RSA private key using the selected sizing mode.
///
/// # Parameters
/// - `args`: Parsed CLI arguments describing the requested key size and exponent.
/// - `rng`: Random number generator used for prime sampling.
///
/// # Returns
/// - `Result<GeneratedPrivateKey, Box<dyn Error>>`: Fully derived RSA private key material.
///
/// # Expected Output
/// - Returns generated key material; no stdout/stderr output.
fn generate_private_key(
    args: &Args,
    rng: &mut RngChoice,
) -> Result<GeneratedPrivateKey, Box<dyn Error>> {
    let (p, q) = match args.size_mode {
        SizeMode::Prime => generate_prime_pair(args.prime_bits, rng),
        SizeMode::Modulus => generate_prime_pair_for_modulus_bits(args.modulus_bits, rng)?,
    };

    let one = BigUint::one();
    let n = &p * &q;
    let phi = (&p - &one) * (&q - &one);
    let e = choose_exponent(args.public_exponent, &phi);
    let d = mod_inverse(&e, &phi)
        .ok_or("public exponent is not invertible; try a different size or exponent")?;

    Ok(GeneratedPrivateKey { p, q, n, phi, e, d })
}

/// Generates two distinct random primes with the requested bit length.
///
/// # Parameters
/// - `prime_bits`: Bit length for each prime.
/// - `rng`: Random number generator used for prime sampling.
///
/// # Returns
/// - `(BigUint, BigUint)`: Distinct prime pair `(p, q)`.
///
/// # Expected Output
/// - Returns two distinct primes; no stdout/stderr output.
fn generate_prime_pair(prime_bits: u32, rng: &mut RngChoice) -> (BigUint, BigUint) {
    let p = random_prime_with_bits(prime_bits, rng);
    let mut q = random_prime_with_bits(prime_bits, rng);
    while q == p {
        q = random_prime_with_bits(prime_bits, rng);
    }
    (p, q)
}

/// Generates two primes whose product has the exact requested modulus bit length.
///
/// # Parameters
/// - `modulus_bits`: Exact target bit length for the RSA modulus.
/// - `rng`: Random number generator used for prime sampling.
///
/// # Returns
/// - `Result<(BigUint, BigUint), Box<dyn Error>>`: Distinct prime pair whose product matches the target width.
///
/// # Expected Output
/// - Returns an error if a matching modulus cannot be found within the attempt budget.
fn generate_prime_pair_for_modulus_bits(
    modulus_bits: u32,
    rng: &mut RngChoice,
) -> Result<(BigUint, BigUint), Box<dyn Error>> {
    let prime_bits = modulus_bits.div_ceil(2);
    for _ in 0..MODULUS_MODE_ATTEMPT_LIMIT {
        let (p, q) = generate_prime_pair(prime_bits, rng);
        if (&p * &q).bits() == u64::from(modulus_bits) {
            return Ok((p, q));
        }
    }

    Err(format!(
        "failed to generate an RSA modulus with exactly {} bits after {} attempts",
        modulus_bits, MODULUS_MODE_ATTEMPT_LIMIT
    )
    .into())
}

/// Validates that the generated key material is internally consistent.
///
/// # Parameters
/// - `key`: Generated RSA private key material to verify.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` when the key is valid.
///
/// # Expected Output
/// - Returns an error when the key components do not satisfy RSA invariants; no side effects.
fn validate_generated_key(key: &GeneratedPrivateKey) -> Result<(), Box<dyn Error>> {
    if key.p == key.q {
        return Err("generated identical primes; expected distinct RSA primes".into());
    }

    let one = BigUint::one();
    let expected_phi = (&key.p - &one) * (&key.q - &one);
    if key.n != (&key.p * &key.q) {
        return Err("generated modulus does not match p * q".into());
    }
    if key.phi != expected_phi {
        return Err("generated totient does not match (p - 1) * (q - 1)".into());
    }
    if (&key.e * &key.d) % &key.phi != one {
        return Err("generated exponents do not satisfy e * d ≡ 1 mod phi(n)".into());
    }

    Ok(())
}

/// Loads a private RSA YAML document for public-key conversion or validation.
///
/// # Parameters
/// - `path`: Filesystem path to the rsa-private-key-v1 YAML file.
///
/// # Returns
/// - `Result<GeneratedPrivateKey, Box<dyn Error>>`: Private key material loaded from the YAML document.
///
/// # Expected Output
/// - Reads the YAML file from disk and returns its parsed private key material.
fn load_private_key_from_yaml(path: &str) -> Result<GeneratedPrivateKey, Box<dyn Error>> {
    let material = load_rsa_key_material_from_yaml_path(Path::new(path))?;
    if material.format != RsaKeyFileFormat::PrivateKeyV1 {
        return Err(format!("{path} is not an rsa-private-key-v1 YAML file").into());
    }

    Ok(GeneratedPrivateKey {
        p: material.p.ok_or("private key YAML is missing prime p")?,
        q: material.q.ok_or("private key YAML is missing prime q")?,
        n: material.modulus,
        phi: material
            .totient
            .ok_or("private key YAML is missing totient")?,
        e: BigUint::from(material.public_exponent),
        d: material
            .private_exponent
            .ok_or("private key YAML is missing private_exponent")?,
    })
}

/// Builds the serialized YAML document for a generated private key.
///
/// # Parameters
/// - `args`: Parsed CLI arguments used for generation.
/// - `key`: Generated RSA private key material.
/// - `rng_mode`: RNG mode used during generation.
///
/// # Returns
/// - `Result<RsaPrivateKeyFile, Box<dyn Error>>`: Serializable YAML document.
///
/// # Expected Output
/// - Returns the structured YAML payload; no stdout/stderr output.
fn build_private_key_document(
    args: &Args,
    key: &GeneratedPrivateKey,
    rng_mode: KeyRngMode,
) -> Result<RsaPrivateKeyFile, Box<dyn Error>> {
    Ok(RsaPrivateKeyFile {
        format: "rsa-private-key-v1".to_string(),
        algorithm: "RSA".to_string(),
        public_exponent: key.e.to_string(),
        private_exponent: key.d.to_string(),
        modulus: key.n.to_string(),
        totient: key.phi.to_string(),
        primes: RsaPrimePair {
            p: key.p.to_string(),
            q: key.q.to_string(),
        },
        bit_lengths: RsaKeyBitLengths {
            requested_prime_bits: args.prime_bits,
            requested_modulus_bits: args.modulus_bits,
            prime_p_bits: key.p.bits(),
            prime_q_bits: key.q.bits(),
            modulus_bits: key.n.bits(),
        },
        generation: RsaKeyGeneration {
            size_mode: args.size_mode,
            requested_public_exponent: args.public_exponent,
            rng_mode,
            seed: args.seed,
            created_unix_ms: current_unix_ms()?,
        },
    })
}

/// Builds the serialized YAML document for the public half of an RSA keypair.
///
/// # Parameters
/// - `key`: Generated or imported RSA private key material.
///
/// # Returns
/// - `RsaPublicKeyFile`: Serializable public-key YAML document.
///
/// # Expected Output
/// - Returns the structured public-key YAML payload; no stdout/stderr output.
fn build_public_key_document(key: &GeneratedPrivateKey) -> RsaPublicKeyFile {
    RsaPublicKeyFile {
        format: "rsa-public-key-v1".to_string(),
        algorithm: "RSA".to_string(),
        public_exponent: key.e.to_string(),
        modulus: key.n.to_string(),
        bit_lengths: RsaPublicKeyBitLengths {
            modulus_bits: key.n.bits(),
        },
    }
}

/// Writes a YAML document to disk.
///
/// # Parameters
/// - `path`: Output path for the YAML file.
/// - `document`: Serializable YAML document.
/// - `force`: Whether to overwrite an existing file.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` when the file is written successfully.
///
/// # Expected Output
/// - Creates parent directories as needed and writes the YAML file at `path`.
fn write_yaml_file<T: Serialize>(
    path: &str,
    document: &T,
    force: bool,
) -> Result<(), Box<dyn Error>> {
    let output_path = Path::new(path);
    if output_path.exists() && !force {
        return Err(format!(
            "output file {} already exists; rerun with --force to overwrite",
            output_path.display()
        )
        .into());
    }

    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    let mut yaml = serde_yaml::to_string(document)?;
    if !yaml.ends_with('\n') {
        yaml.push('\n');
    }
    fs::write(output_path, yaml)?;
    Ok(())
}

/// Returns the current Unix time in milliseconds.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `Result<u128, Box<dyn Error>>`: Current Unix timestamp in milliseconds.
///
/// # Expected Output
/// - Returns the current wall-clock timestamp; no side effects.
fn current_unix_ms() -> Result<u128, Box<dyn Error>> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
}

/// Returns a lowercase label for the sizing mode.
///
/// # Parameters
/// - `mode`: Key sizing mode.
///
/// # Returns
/// - `&'static str`: Lowercase mode label.
///
/// # Expected Output
/// - Returns a constant string; no side effects.
fn size_mode_label(mode: SizeMode) -> &'static str {
    match mode {
        SizeMode::Prime => "prime",
        SizeMode::Modulus => "modulus",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "kgen_{}_{}_{}.yaml",
            name,
            std::process::id(),
            current_unix_ms().expect("unix time")
        ));
        path
    }

    fn base_args() -> Args {
        Args {
            size_mode: SizeMode::Prime,
            prime_bits: DEFAULT_PRIME_BITS,
            modulus_bits: DEFAULT_MODULUS_BITS,
            public_exponent: DEFAULT_PUBLIC_EXPONENT,
            output: DEFAULT_OUTPUT_PATH.to_string(),
            public_output: None,
            input_private_key: None,
            force: false,
            seed: None,
            crypto_rng: false,
        }
    }

    #[test]
    fn test_cli_defaults_match_requested_sizes() {
        let args = Args::parse_from(["kgen"]);

        assert_eq!(args.size_mode, SizeMode::Prime);
        assert_eq!(args.prime_bits, DEFAULT_PRIME_BITS);
        assert_eq!(args.modulus_bits, DEFAULT_MODULUS_BITS);
        assert_eq!(args.output, DEFAULT_OUTPUT_PATH);
        assert!(args.public_output.is_none());
        assert!(args.input_private_key.is_none());
        assert_eq!(args.public_exponent, DEFAULT_PUBLIC_EXPONENT);
    }

    #[test]
    fn test_generate_prime_pair_for_modulus_bits_hits_exact_width() {
        let mut rng = RngChoice::from_seed(RngMode::Standard, 7);
        let (p, q) =
            generate_prime_pair_for_modulus_bits(40, &mut rng).expect("modulus-sized keypair");

        assert_eq!((&p * &q).bits(), 40);
        assert_ne!(p, q);
    }

    #[test]
    fn test_build_private_key_document_tracks_requested_and_actual_sizes() {
        let args = base_args();
        let key = GeneratedPrivateKey {
            p: BigUint::from(61u8),
            q: BigUint::from(53u8),
            n: BigUint::from(3233u32),
            phi: BigUint::from(3120u32),
            e: BigUint::from(17u8),
            d: BigUint::from(2753u32),
        };

        let document = build_private_key_document(&args, &key, KeyRngMode::Standard)
            .expect("private key document");

        assert_eq!(document.format, "rsa-private-key-v1");
        assert_eq!(document.algorithm, "RSA");
        assert_eq!(document.public_exponent, "17");
        assert_eq!(document.primes.p, "61");
        assert_eq!(
            document.bit_lengths.requested_prime_bits,
            DEFAULT_PRIME_BITS
        );
        assert_eq!(
            document.bit_lengths.requested_modulus_bits,
            DEFAULT_MODULUS_BITS
        );
        assert_eq!(document.bit_lengths.modulus_bits, key.n.bits());
        assert_eq!(document.generation.size_mode, SizeMode::Prime);
        assert_eq!(document.generation.rng_mode, KeyRngMode::Standard);
    }

    #[test]
    fn test_build_public_key_document_omits_private_fields() {
        let key = GeneratedPrivateKey {
            p: BigUint::from(61u8),
            q: BigUint::from(53u8),
            n: BigUint::from(3233u32),
            phi: BigUint::from(3120u32),
            e: BigUint::from(17u8),
            d: BigUint::from(2753u32),
        };

        let document = build_public_key_document(&key);
        let yaml = serde_yaml::to_string(&document).expect("serialize public key document");

        assert_eq!(document.format, "rsa-public-key-v1");
        assert_eq!(document.public_exponent, "17");
        assert_eq!(document.modulus, "3233");
        assert!(yaml.contains("format: rsa-public-key-v1"));
        assert!(!yaml.contains("private_exponent"));
        assert!(!yaml.contains("totient"));
        assert!(!yaml.contains("primes:"));
    }

    #[test]
    fn test_write_yaml_file_refuses_overwrite_without_force() {
        let path = temp_path("overwrite_guard");
        let document = RsaPrivateKeyFile {
            format: "rsa-private-key-v1".to_string(),
            algorithm: "RSA".to_string(),
            public_exponent: "17".to_string(),
            private_exponent: "2753".to_string(),
            modulus: "3233".to_string(),
            totient: "3120".to_string(),
            primes: RsaPrimePair {
                p: "61".to_string(),
                q: "53".to_string(),
            },
            bit_lengths: RsaKeyBitLengths {
                requested_prime_bits: 8,
                requested_modulus_bits: 16,
                prime_p_bits: 6,
                prime_q_bits: 6,
                modulus_bits: 12,
            },
            generation: RsaKeyGeneration {
                size_mode: SizeMode::Prime,
                requested_public_exponent: 17,
                rng_mode: KeyRngMode::Standard,
                seed: Some(1),
                created_unix_ms: 1,
            },
        };

        write_yaml_file(path.to_str().expect("utf8 path"), &document, false)
            .expect("initial write");
        let err = write_yaml_file(path.to_str().expect("utf8 path"), &document, false)
            .expect_err("overwrite should fail");
        assert!(err.to_string().contains("--force"));

        let _ = fs::remove_file(path);
    }
}
