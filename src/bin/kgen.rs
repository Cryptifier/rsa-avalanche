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
use num_traits::{One, ToPrimitive};
use rand::RngCore;
use rsademo::config::{RsaKeyFileFormat, load_rsa_key_material_from_yaml_path};
use rsademo::math::{
    choose_exponent, mod_inverse, random_biguint_bits, random_prime_with_bits, to_hex,
};
use rsademo::pgp::{import_rsa_public_key_from_pgp_path, parse_openpgp_file_path};
use rsademo::rng::{RngChoice, RngMode};
use serde::Serialize;

const DEFAULT_OUTPUT_PATH: &str = "config/keys/private_key.yaml";
const DEFAULT_PRIME_BITS: u32 = 56;
const DEFAULT_MODULUS_BITS: u32 = 144;
const DEFAULT_PUBLIC_EXPONENT: u64 = 65_537;
const DEFAULT_ENCRYPT_KEY_BITS: u32 = 128;
const MODULUS_MODE_ATTEMPT_LIMIT: usize = 10_000;

#[derive(Parser, Debug)]
#[command(
    name = "kgen",
    about = "Generate RSA key YAML, convert existing YAML keys, and import OpenPGP files",
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

    /// Existing rsa-public-key-v1 YAML file to use for encryption or normalized output
    #[arg(long)]
    input_public_key: Option<String>,

    /// Existing OpenPGP public-key file to convert into rsa-public-key-v1 output
    #[arg(long)]
    input_pgp_public_key: Option<String>,

    /// Existing OpenPGP encrypted or packetized file to unpack into pgp-file-v1 YAML
    #[arg(long)]
    input_pgp_file: Option<String>,

    /// Optional YAML output path for unpacked OpenPGP packet contents
    #[arg(long)]
    pgp_output: Option<String>,

    /// Optional binary output path for an RSA PKCS#1 v1.5 encrypted random key
    #[arg(long)]
    encrypt_output: Option<String>,

    /// Bit width of the random key appended to the PKCS#1 v1.5 encryption block
    #[arg(long, default_value_t = DEFAULT_ENCRYPT_KEY_BITS, value_parser = clap::value_parser!(u32).range(1..=65536))]
    encrypt_key_bits: u32,

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

#[derive(Debug, PartialEq, Eq)]
struct EncryptedRandomKeyOutput {
    key_bytes: Vec<u8>,
    padded_message: Vec<u8>,
    ciphertext_bytes: Vec<u8>,
    ciphertext: BigUint,
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
/// - Writes generated private/public YAML files, optionally emits an RSA-encrypted PKCS#1 v1.5 ciphertext file, and prints a summary to stdout.
fn run_kgen(args: Args) -> Result<(), Box<dyn Error>> {
    let import_mode_count = usize::from(args.input_private_key.is_some())
        + usize::from(args.input_public_key.is_some())
        + usize::from(args.input_pgp_public_key.is_some())
        + usize::from(args.input_pgp_file.is_some());
    if import_mode_count > 1 {
        return Err(
            "--input-private-key, --input-public-key, --input-pgp-public-key, and --input-pgp-file are mutually exclusive"
                .into(),
        );
    }

    if let Some(input_pgp_file) = args.input_pgp_file.as_deref() {
        if args.pgp_output.is_none() {
            return Err("--input-pgp-file requires --pgp-output".into());
        }
        if args.public_output.is_some()
            || args.input_private_key.is_some()
            || args.input_public_key.is_some()
            || args.encrypt_output.is_some()
        {
            return Err("--input-pgp-file does not support RSA key conversion outputs".into());
        }

        let document = parse_openpgp_file_path(Path::new(input_pgp_file))?;
        let pgp_output = args.pgp_output.as_deref().unwrap_or_default();
        write_yaml_file(pgp_output, &document, args.force)?;
        println!(
            "Unpacked OpenPGP file: packets {} source {} output {}",
            document.packet_count, input_pgp_file, pgp_output
        );
        return Ok(());
    }

    if let Some(input_pgp_public_key) = args.input_pgp_public_key.as_deref() {
        let public_output = args
            .public_output
            .as_deref()
            .ok_or("--input-pgp-public-key requires --public-output")?;
        let imported = import_rsa_public_key_from_pgp_path(Path::new(input_pgp_public_key))?;
        let public_document = build_public_key_document_from_components(
            &imported.modulus,
            &imported.public_exponent,
        )?;
        write_yaml_file(public_output, &public_document, args.force)?;
        if let Some(pgp_output) = args.pgp_output.as_deref() {
            write_yaml_file(pgp_output, &imported.parsed_file, args.force)?;
            println!("Wrote unpacked OpenPGP YAML output {}", pgp_output);
        }
        println!(
            "Imported OpenPGP RSA public key: modulus-bits {} e {} output {}",
            imported.modulus.bits(),
            imported.public_exponent,
            public_output
        );
        return Ok(());
    }

    if args.pgp_output.is_some() {
        return Err("--pgp-output requires --input-pgp-public-key or --input-pgp-file".into());
    }

    let rng_mode = if args.crypto_rng {
        KeyRngMode::Crypto
    } else {
        KeyRngMode::Standard
    };
    let needs_rng = args.encrypt_output.is_some() || args.input_private_key.is_none();
    let mut rng = if needs_rng {
        Some(build_rng(&args, rng_mode)?)
    } else {
        None
    };

    let (public_modulus, public_exponent) = if let Some(input_pgp_public_key) =
        args.input_pgp_public_key.as_deref()
    {
        if args.public_output.is_none() && args.encrypt_output.is_none() {
            return Err(
                "--input-pgp-public-key requires --public-output or --encrypt-output".into(),
            );
        }
        let imported = import_rsa_public_key_from_pgp_path(Path::new(input_pgp_public_key))?;
        let public_document = build_public_key_document_from_components(
            &imported.modulus,
            &imported.public_exponent,
        )?;
        if let Some(public_output) = args.public_output.as_deref() {
            write_yaml_file(public_output, &public_document, args.force)?;
            println!("Wrote RSA public key output {}", public_output);
        }
        if let Some(pgp_output) = args.pgp_output.as_deref() {
            write_yaml_file(pgp_output, &imported.parsed_file, args.force)?;
            println!("Wrote unpacked OpenPGP YAML output {}", pgp_output);
        }
        println!(
            "Imported OpenPGP RSA public key: modulus-bits {} e {}",
            imported.modulus.bits(),
            imported.public_exponent,
        );
        (imported.modulus, imported.public_exponent)
    } else if let Some(input_public_key) = args.input_public_key.as_deref() {
        if args.public_output.is_none() && args.encrypt_output.is_none() {
            return Err("--input-public-key requires --public-output or --encrypt-output".into());
        }
        let material = load_rsa_key_material_from_yaml_path(Path::new(input_public_key))?;
        if material.format != RsaKeyFileFormat::PublicKeyV1 {
            return Err(format!("{input_public_key} is not an rsa-public-key-v1 YAML file").into());
        }
        let public_exponent = BigUint::from(material.public_exponent);
        if let Some(public_output) = args.public_output.as_deref() {
            let public_document =
                build_public_key_document_from_components(&material.modulus, &public_exponent)?;
            write_yaml_file(public_output, &public_document, args.force)?;
            println!("Wrote RSA public key output {}", public_output);
        }
        println!(
            "Loaded RSA public key: modulus-bits {} e {} source {}",
            material.modulus.bits(),
            material.public_exponent,
            input_public_key
        );
        (material.modulus, public_exponent)
    } else {
        let (key, generated_new_private_key) =
            if let Some(input_private_key) = args.input_private_key.as_deref() {
                if args.public_output.is_none() && args.encrypt_output.is_none() {
                    return Err(
                        "--input-private-key requires --public-output or --encrypt-output".into(),
                    );
                }
                (load_private_key_from_yaml(input_private_key)?, false)
            } else {
                let rng = rng
                    .as_mut()
                    .ok_or("internal error: missing RNG for RSA key generation")?;
                (generate_private_key(&args, rng)?, true)
            };
        validate_generated_key(&key)?;

        if generated_new_private_key {
            let document = build_private_key_document(&args, &key, rng_mode)?;
            write_yaml_file(&args.output, &document, args.force)?;
        }
        if let Some(public_output) = args.public_output.as_deref() {
            let public_document = build_public_key_document(&key);
            write_yaml_file(public_output, &public_document, args.force)?;
            println!("Wrote RSA public key output {}", public_output);
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

        (key.n, key.e)
    };

    if let Some(encrypt_output) = args.encrypt_output.as_deref() {
        let rng = rng
            .as_mut()
            .ok_or("internal error: missing RNG for PKCS#1 ciphertext generation")?;
        let encrypted = encrypt_pkcs1_v1_5_random_key(
            &public_modulus,
            &public_exponent,
            args.encrypt_key_bits,
            rng,
        )?;
        write_binary_file(encrypt_output, &encrypted.ciphertext_bytes, args.force)?;
        println!(
            "Encrypted PKCS#1 v1.5 random key: key-bits {} modulus-bits {} output {}",
            args.encrypt_key_bits,
            public_modulus.bits(),
            encrypt_output
        );
        println!(
            "Random key (hex): {}",
            to_hex(&BigUint::from_bytes_be(&encrypted.key_bytes))
        );
        println!("Ciphertext (hex): {}", to_hex(&encrypted.ciphertext));
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
    build_public_key_document_from_components(&key.n, &key.e)
        .expect("generated RSA key should always have a supported public exponent")
}

/// Builds the serialized YAML document for an RSA public key from explicit components.
///
/// # Parameters
/// - `modulus`: RSA modulus `n`.
/// - `public_exponent`: RSA public exponent `e`.
///
/// # Returns
/// - `Result<RsaPublicKeyFile, Box<dyn Error>>`: Serializable public-key YAML document.
///
/// # Expected Output
/// - Returns the structured public-key payload; no stdout/stderr output.
fn build_public_key_document_from_components(
    modulus: &BigUint,
    public_exponent: &BigUint,
) -> Result<RsaPublicKeyFile, Box<dyn Error>> {
    let exponent_u64 = public_exponent.to_u64().ok_or(
        "RSA public exponent does not fit into the existing rsa-public-key-v1 compatibility range",
    )?;
    Ok(RsaPublicKeyFile {
        format: "rsa-public-key-v1".to_string(),
        algorithm: "RSA".to_string(),
        public_exponent: exponent_u64.to_string(),
        modulus: modulus.to_string(),
        bit_lengths: RsaPublicKeyBitLengths {
            modulus_bits: modulus.bits(),
        },
    })
}

/// Generates a fixed-width random key payload for PKCS#1 v1.5 encryption.
///
/// # Parameters
/// - `bits`: Exact payload bit width to generate.
/// - `rng`: Random number generator used for payload sampling.
///
/// # Returns
/// - `Vec<u8>`: Big-endian payload bytes sized to `bits.div_ceil(8)`.
///
/// # Expected Output
/// - Returns an in-memory random payload with its top bit set so it uses the requested width.
fn generate_random_key_bytes(bits: u32, rng: &mut RngChoice) -> Vec<u8> {
    let bytes_len = bits.div_ceil(8) as usize;
    let mut bytes = random_biguint_bits(bits, rng).to_bytes_be();
    if bytes.len() < bytes_len {
        let mut padded = vec![0u8; bytes_len - bytes.len()];
        padded.extend_from_slice(&bytes);
        bytes = padded;
    }
    bytes
}

/// Builds a PKCS#1 v1.5 encryption block containing the provided payload bytes.
///
/// # Parameters
/// - `modulus`: RSA modulus used to determine the padded block width.
/// - `payload_bytes`: Random key bytes appended after the PKCS#1 separator.
/// - `rng`: Random number generator used for the non-zero padding string.
///
/// # Returns
/// - `Result<Vec<u8>, Box<dyn Error>>`: Full padded encryption block in big-endian order.
///
/// # Expected Output
/// - Returns a `0x00 0x02 || PS || 0x00 || payload` block or an error when the modulus is too small.
fn build_pkcs1_v1_5_encryption_block(
    modulus: &BigUint,
    payload_bytes: &[u8],
    rng: &mut RngChoice,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let modulus_len = modulus.bits().div_ceil(8).max(1) as usize;
    if payload_bytes.len() + 11 > modulus_len {
        return Err(format!(
            "payload requires {} byte(s) but PKCS#1 v1.5 with modulus width {} byte(s) allows at most {} byte(s)",
            payload_bytes.len(),
            modulus_len,
            modulus_len.saturating_sub(11)
        )
        .into());
    }

    let padding_len = modulus_len - payload_bytes.len() - 3;
    let mut block = Vec::with_capacity(modulus_len);
    block.push(0x00);
    block.push(0x02);
    for _ in 0..padding_len {
        let mut byte = 0u8;
        while byte == 0 {
            let mut sample = [0u8; 1];
            rng.fill_bytes(&mut sample);
            byte = sample[0];
        }
        block.push(byte);
    }
    block.push(0x00);
    block.extend_from_slice(payload_bytes);
    Ok(block)
}

/// Encrypts a random key embedded in a PKCS#1 v1.5 block using the provided RSA public key.
///
/// # Parameters
/// - `modulus`: RSA modulus `n`.
/// - `public_exponent`: RSA public exponent `e`.
/// - `key_bits`: Payload bit width for the embedded random key.
/// - `rng`: Random number generator used for both the payload and the PKCS padding string.
///
/// # Returns
/// - `Result<EncryptedRandomKeyOutput, Box<dyn Error>>`: Random key bytes, padded plaintext block, and ciphertext.
///
/// # Expected Output
/// - Returns an encrypted random key without writing files or printing.
fn encrypt_pkcs1_v1_5_random_key(
    modulus: &BigUint,
    public_exponent: &BigUint,
    key_bits: u32,
    rng: &mut RngChoice,
) -> Result<EncryptedRandomKeyOutput, Box<dyn Error>> {
    let key_bytes = generate_random_key_bytes(key_bits, rng);
    let padded_message = build_pkcs1_v1_5_encryption_block(modulus, &key_bytes, rng)?;
    let padded_value = BigUint::from_bytes_be(&padded_message);
    if padded_value >= *modulus {
        return Err("generated PKCS#1 v1.5 plaintext is not smaller than the RSA modulus".into());
    }
    let ciphertext = padded_value.modpow(public_exponent, modulus);
    let modulus_len = modulus.bits().div_ceil(8).max(1) as usize;
    let mut ciphertext_bytes = ciphertext.to_bytes_be();
    if ciphertext_bytes.len() < modulus_len {
        let mut padded = vec![0u8; modulus_len - ciphertext_bytes.len()];
        padded.extend_from_slice(&ciphertext_bytes);
        ciphertext_bytes = padded;
    }

    Ok(EncryptedRandomKeyOutput {
        key_bytes,
        padded_message,
        ciphertext_bytes,
        ciphertext,
    })
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

/// Writes a binary file to disk.
///
/// # Parameters
/// - `path`: Output path for the binary file.
/// - `bytes`: File contents to write.
/// - `force`: Whether to overwrite an existing file.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` when the file is written successfully.
///
/// # Expected Output
/// - Creates parent directories as needed and writes the bytes at `path`.
fn write_binary_file(path: &str, bytes: &[u8], force: bool) -> Result<(), Box<dyn Error>> {
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

    fs::write(output_path, bytes)?;
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
            input_public_key: None,
            input_pgp_public_key: None,
            input_pgp_file: None,
            pgp_output: None,
            encrypt_output: None,
            encrypt_key_bits: DEFAULT_ENCRYPT_KEY_BITS,
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
        assert!(args.input_public_key.is_none());
        assert!(args.input_pgp_public_key.is_none());
        assert!(args.input_pgp_file.is_none());
        assert!(args.pgp_output.is_none());
        assert!(args.encrypt_output.is_none());
        assert_eq!(args.encrypt_key_bits, DEFAULT_ENCRYPT_KEY_BITS);
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

    #[test]
    fn test_encrypt_pkcs1_v1_5_random_key_round_trips_and_preserves_payload() {
        let mut key_rng = RngChoice::from_seed(RngMode::Standard, 7);
        let mut key_args = base_args();
        key_args.size_mode = SizeMode::Modulus;
        key_args.modulus_bits = 128;
        let key = generate_private_key(&key_args, &mut key_rng).expect("generate key");
        let mut encrypt_rng = RngChoice::from_seed(RngMode::Standard, 11);

        let encrypted = encrypt_pkcs1_v1_5_random_key(&key.n, &key.e, 24, &mut encrypt_rng)
            .expect("encrypt random key");
        let decrypted = encrypted.ciphertext.modpow(&key.d, &key.n);
        let modulus_len = key.n.bits().div_ceil(8) as usize;
        let mut decrypted_bytes = decrypted.to_bytes_be();
        if decrypted_bytes.len() < modulus_len {
            let mut padded = vec![0u8; modulus_len - decrypted_bytes.len()];
            padded.extend_from_slice(&decrypted_bytes);
            decrypted_bytes = padded;
        }

        assert_eq!(decrypted_bytes, encrypted.padded_message);
        assert_eq!(decrypted_bytes[0], 0x00);
        assert_eq!(decrypted_bytes[1], 0x02);
        let separator_index = decrypted_bytes[2..]
            .iter()
            .position(|byte| *byte == 0x00)
            .map(|index| index + 2)
            .expect("separator byte");
        assert!(separator_index >= 10);
        assert_eq!(&decrypted_bytes[separator_index + 1..], encrypted.key_bytes);
        assert_eq!(encrypted.key_bytes.len(), 3);
    }

    #[test]
    fn test_run_kgen_encrypts_random_key_from_public_yaml() {
        let public_key_path = temp_path("public_encrypt_input");
        let ciphertext_path = temp_path("public_encrypt_output").with_extension("bin");
        let mut key_rng = RngChoice::from_seed(RngMode::Standard, 13);
        let mut key_args = base_args();
        key_args.size_mode = SizeMode::Modulus;
        key_args.modulus_bits = 128;
        let key = generate_private_key(&key_args, &mut key_rng).expect("generate key");
        let public_document = build_public_key_document(&key);
        write_yaml_file(
            public_key_path.to_str().expect("utf8 path"),
            &public_document,
            true,
        )
        .expect("write public key");

        let mut args = base_args();
        args.input_public_key = Some(public_key_path.to_string_lossy().to_string());
        args.encrypt_output = Some(ciphertext_path.to_string_lossy().to_string());
        args.encrypt_key_bits = 32;
        args.seed = Some(19);
        args.force = true;

        run_kgen(args).expect("encrypt from public key");
        let ciphertext_bytes = fs::read(&ciphertext_path).expect("read ciphertext");
        let ciphertext = BigUint::from_bytes_be(&ciphertext_bytes);
        let decrypted = ciphertext.modpow(&key.d, &key.n);
        let modulus_len = key.n.bits().div_ceil(8) as usize;
        let mut decrypted_bytes = decrypted.to_bytes_be();
        if decrypted_bytes.len() < modulus_len {
            let mut padded = vec![0u8; modulus_len - decrypted_bytes.len()];
            padded.extend_from_slice(&decrypted_bytes);
            decrypted_bytes = padded;
        }
        let separator_index = decrypted_bytes[2..]
            .iter()
            .position(|byte| *byte == 0x00)
            .map(|index| index + 2)
            .expect("separator byte");
        let payload = &decrypted_bytes[separator_index + 1..];

        assert_eq!(ciphertext_bytes.len(), modulus_len);
        assert_eq!(decrypted_bytes[0], 0x00);
        assert_eq!(decrypted_bytes[1], 0x02);
        assert_eq!(payload.len(), 4);

        let _ = fs::remove_file(public_key_path);
        let _ = fs::remove_file(ciphertext_path);
    }

    #[test]
    fn test_run_kgen_imports_openpgp_public_key_to_rsa_yaml() {
        let public_key_path = temp_path("pgp_public_key_input").with_extension("asc");
        let public_output_path = temp_path("pgp_public_key_output");
        let pgp_public_key = "-----BEGIN PGP PUBLIC KEY BLOCK-----\nVersion: Test Suite\n\nxg0EEjRWeAEADAyhAAUR\n=wIfx\n-----END PGP PUBLIC KEY BLOCK-----\n";
        fs::write(&public_key_path, pgp_public_key).expect("write pgp public key");

        let mut args = base_args();
        args.input_pgp_public_key = Some(public_key_path.to_string_lossy().to_string());
        args.public_output = Some(public_output_path.to_string_lossy().to_string());
        args.force = true;

        run_kgen(args).expect("import pgp public key");
        let yaml = fs::read_to_string(&public_output_path).expect("read rsa public output");
        let material = load_rsa_key_material_from_yaml_path(&public_output_path)
            .expect("parse rsa public yaml");

        assert!(yaml.contains("format: rsa-public-key-v1"));
        assert_eq!(material.public_exponent, 17);
        assert_eq!(material.modulus.to_string(), "3233");

        let _ = fs::remove_file(public_key_path);
        let _ = fs::remove_file(public_output_path);
    }

    #[test]
    fn test_run_kgen_unpacks_openpgp_file_to_yaml() {
        let message_path = temp_path("pgp_message_input").with_extension("asc");
        let yaml_output_path = temp_path("pgp_message_output");
        let pgp_message = "-----BEGIN PGP MESSAGE-----\nVersion: Test Suite\n\nwQ8DAQIDBAUGBwgBABUSNFbSCQHerb7vyv66vg==\n=qExN\n-----END PGP MESSAGE-----\n";
        fs::write(&message_path, pgp_message).expect("write pgp message");

        let mut args = base_args();
        args.input_pgp_file = Some(message_path.to_string_lossy().to_string());
        args.pgp_output = Some(yaml_output_path.to_string_lossy().to_string());
        args.force = true;

        run_kgen(args).expect("unpack pgp file");
        let yaml = fs::read_to_string(&yaml_output_path).expect("read pgp yaml output");

        assert!(yaml.contains("format: pgp-file-v1"));
        assert!(yaml.contains("block_type: PGP MESSAGE"));
        assert!(yaml.contains("packet_tag: 1"));
        assert!(yaml.contains("packet_tag: 18"));

        let _ = fs::remove_file(message_path);
        let _ = fs::remove_file(yaml_output_path);
    }
}
