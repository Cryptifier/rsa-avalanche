//! Configuration schema and loader for config/rsa_config.json.

use std::{error::Error, fs, path::Path};

use num_bigint::BigUint;
use serde::Deserialize;

use crate::r_candidates::RCandidateMode;

/// Top-level configuration matching the config/rsa_config.json schema.
#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    /// RSA keypair configuration.
    #[serde(default, rename = "rsa_keypair", alias = "key", alias = "keys")]
    pub rsa_keypair: KeyConfig,
    /// Engine configuration for r candidate and analysis behavior.
    #[serde(default)]
    pub engine: EngineConfig,
    /// Polynomial field configuration for coordinate generation.
    #[serde(default)]
    pub polynomial_fields: PolynomialFieldsConfig,
    /// Verification configuration for demo inputs.
    #[serde(default)]
    pub verify: VerifyConfig,
}

/// RSA keypair configuration values.
#[derive(Debug, Deserialize, Clone)]
pub struct KeyConfig {
    /// Whether to generate keys instead of using provided values.
    #[serde(default = "default_generate")]
    pub generate: bool,
    /// RSA prime p (required when not generating).
    #[serde(default, deserialize_with = "deserialize_biguint_option")]
    pub p: Option<BigUint>,
    /// RSA prime q (required when not generating).
    #[serde(default, deserialize_with = "deserialize_biguint_option")]
    pub q: Option<BigUint>,
    /// RSA public exponent.
    #[serde(default = "default_e")]
    pub e: u64,
}

/// Message configuration for generating plaintexts.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct MessageConfig {
    /// Fixed message to encrypt when random selection is disabled.
    #[serde(default = "default_fixed_message")]
    pub fixed_message: String,
    /// Whether to use random messages instead of the fixed string.
    #[serde(default = "default_message_random")]
    pub is_random: bool,
    /// Bit length for randomly generated messages.
    #[serde(default = "default_message_bits")]
    pub bits: u32,
}

/// Engine configuration parameters for candidate generation and analysis.
#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct EngineConfig {
    #[serde(default = "default_base_convert")]
    pub base_convert: bool,
    #[serde(default = "default_invert_bits")]
    pub invert_bits: bool,
    #[serde(default = "default_rabin_exponent")]
    pub rabin_exponent: u64,
    #[serde(default = "default_min_message_trials")]
    pub min_message_trials: u64,
    #[serde(default = "default_overlap_report_threshold")]
    pub overlap_report_threshold: f64,
    #[serde(default = "default_entropy_report_threshold")]
    pub entropy_report_threshold: f64,
    #[serde(default = "default_process_min_count")]
    pub process_min_count: u64,
    #[serde(default = "default_process_count")]
    pub process_count: u64,
    #[serde(default = "default_process_scale")]
    pub process_scale: u32,
    #[serde(default = "default_process_max_best_attempts")]
    pub process_max_best_attempts: u64,
    #[serde(default = "default_process_min_factor")]
    pub process_min_factor: u64,
    #[serde(default = "default_use_rs_decrypt")]
    pub use_rs_decrypt: bool,
    #[serde(default = "default_test_iterations")]
    pub test_iterations: u64,
    #[serde(default = "default_alt_iterations")]
    pub alt_iterations: u64,
    #[serde(default = "default_analysis_tests_iterations")]
    pub analysis_tests_iterations: u64,
    #[serde(default = "default_oracle_screen_iterations")]
    pub oracle_screen_iterations: u64,
    #[serde(default = "default_analysis_tests_window")]
    pub analysis_tests_window: usize,
    #[serde(default = "default_analysis_tests_stride")]
    pub analysis_tests_stride: usize,
    /// Number of homomorphic left-shift multiplications by 2 to compare per candidate.
    #[serde(default = "default_analysis_shift_multiplications")]
    pub analysis_shift_multiplications: usize,
    #[serde(default = "default_analysis_batch_enable")]
    pub analysis_batch_enable: bool,
    #[serde(default = "default_analysis_batch_candidates")]
    pub analysis_batch_candidates: u64,
    #[serde(default = "default_analysis_batch_batches")]
    pub analysis_batch_batches: u64,
    #[serde(default = "default_oracle_accuracy_threshold")]
    pub oracle_accuracy_threshold: f64,
    #[serde(default = "default_r_use_list_enable")]
    pub r_use_list_enable: bool,
    #[serde(default)]
    pub r_use_list: Vec<String>,
    #[serde(default = "default_r_stress_test_enable")]
    pub r_stress_test_enable: bool,
    #[serde(default, deserialize_with = "deserialize_biguint_option")]
    pub r_stress_start: Option<BigUint>,
    #[serde(default, deserialize_with = "deserialize_biguint_option")]
    pub r_stress_end: Option<BigUint>,
    #[serde(default)]
    pub override_best_r: Option<String>,
    #[serde(default = "default_reuse_r_candidates_path")]
    pub reuse_r_candidates_path: String,
    #[serde(default = "default_reuse_r_candidates")]
    pub reuse_r_candidates: bool,
    #[serde(default = "default_reuse_r_candidates_append_only")]
    pub reuse_r_candidates_append_only: bool,
    #[serde(default = "default_r_candidate_mode")]
    pub r_candidate_mode: RCandidateMode,
    #[serde(default = "default_r_candidate_small_primes")]
    pub r_candidate_small_primes: Vec<u64>,
    #[serde(default = "default_r_candidate_small_prime_factors")]
    pub r_candidate_small_prime_factors: usize,
    #[serde(default = "default_r_candidate_max_factors")]
    pub r_candidate_max_factors: usize,
    #[serde(default)]
    pub r_candidate_bit_length: Option<u64>,
    #[serde(default = "default_combiner_enable")]
    pub combiner_enable: bool,
    #[serde(default = "default_combiner_k_oracles")]
    pub combiner_k_oracles: usize,
    #[serde(default = "default_combiner_match_probability")]
    pub combiner_match_probability: f64,
    #[serde(default = "default_combiner_tie_breaker")]
    pub combiner_tie_breaker: bool,
    #[serde(default)]
    pub message: MessageConfig,
    #[serde(default = "default_enciphered_export_enable")]
    pub enciphered_export_enable: bool,
    #[serde(default = "default_enciphered_export_iterations")]
    pub enciphered_export_iterations: u64,
    #[serde(default = "default_enciphered_export_bins")]
    pub enciphered_export_bins: usize,
    #[serde(default = "default_enciphered_export_window")]
    pub enciphered_export_window: usize,
    #[serde(default = "default_enciphered_export_stride")]
    pub enciphered_export_stride: usize,
    #[serde(default = "default_enciphered_export_output_csv")]
    pub enciphered_export_output_csv: String,
    #[serde(default = "default_enciphered_export_ramp_length")]
    pub enciphered_export_ramp_length: usize,
    #[serde(default = "default_enciphered_export_ramp_step_pct")]
    pub enciphered_export_ramp_step_pct: f64,
    #[serde(default = "default_enciphered_export_ramp_tolerances")]
    pub enciphered_export_ramp_tolerances: Vec<f64>,
    #[serde(default = "default_enciphered_export_ramp_csv")]
    pub enciphered_export_ramp_csv: String,
}

/// Polynomial field configuration for coordinate generation.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct PolynomialFieldsConfig {
    /// List of polynomial field definitions.
    #[serde(default)]
    pub fields: Vec<PolynomialFieldConfig>,
}

/// Single polynomial field definition.
#[derive(Debug, Deserialize, Clone)]
pub struct PolynomialFieldConfig {
    /// Prime modulus defining the field (8..=64 bits).
    #[serde(deserialize_with = "deserialize_biguint")]
    pub prime: BigUint,
    /// Seed used to derive polynomial coefficients.
    #[serde(default)]
    pub seed: u64,
}

/// Verification inputs for demo workflows.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct VerifyConfig {
    /// Ciphertext to decrypt in demo mode (decimal string or number).
    #[serde(default, deserialize_with = "deserialize_biguint_option")]
    pub ciphertext: Option<BigUint>,
    /// Optional ciphertext hex string (0x prefix optional).
    #[serde(default)]
    pub ciphertext_hex: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            rsa_keypair: KeyConfig::default(),
            engine: EngineConfig::default(),
            polynomial_fields: PolynomialFieldsConfig::default(),
            verify: VerifyConfig::default(),
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
            entropy_report_threshold: default_entropy_report_threshold(),
            process_min_count: default_process_min_count(),
            process_count: default_process_count(),
            process_scale: default_process_scale(),
            process_max_best_attempts: default_process_max_best_attempts(),
            process_min_factor: default_process_min_factor(),
            use_rs_decrypt: default_use_rs_decrypt(),
            test_iterations: default_test_iterations(),
            alt_iterations: default_alt_iterations(),
            analysis_tests_iterations: default_analysis_tests_iterations(),
            oracle_screen_iterations: default_oracle_screen_iterations(),
            analysis_tests_window: default_analysis_tests_window(),
            analysis_tests_stride: default_analysis_tests_stride(),
            analysis_shift_multiplications: default_analysis_shift_multiplications(),
            analysis_batch_enable: default_analysis_batch_enable(),
            analysis_batch_candidates: default_analysis_batch_candidates(),
            analysis_batch_batches: default_analysis_batch_batches(),
            oracle_accuracy_threshold: default_oracle_accuracy_threshold(),
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
            r_candidate_max_factors: default_r_candidate_max_factors(),
            r_candidate_bit_length: None,
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

/// Loads the JSON/JSON5 config from disk, falling back to defaults if missing.
///
/// # Parameters
/// - `path`: Path to the configuration file.
///
/// # Returns
/// - `Result<Config, Box<dyn Error>>`: Parsed config or a default config if not found.
///
/// # Expected Output
/// - Prints a notice when the file is missing; no other side effects on success.
pub fn load_config(path: &str) -> Result<Config, Box<dyn Error>> {
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

/// Deserializes an optional `BigUint` from a string or number JSON value.
///
/// # Parameters
/// - `deserializer`: Serde deserializer provided by the caller.
///
/// # Returns
/// - `Result<Option<BigUint>, D::Error>`: Parsed value or `None` when absent.
///
/// # Expected Output
/// - Returns a parse error if the value is not a string/number or is invalid.
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

/// Deserializes a `BigUint` from a string or number JSON value.
///
/// # Parameters
/// - `deserializer`: Serde deserializer provided by the caller.
///
/// # Returns
/// - `Result<BigUint, D::Error>`: Parsed value or an error when invalid.
///
/// # Expected Output
/// - Returns a parse error if the value is not a string/number or is invalid.
fn deserialize_biguint<'de, D>(deserializer: D) -> Result<BigUint, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as DeError;

    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::String(s) => s.parse::<BigUint>().map_err(DeError::custom),
        serde_json::Value::Number(num) => num
            .to_string()
            .parse::<BigUint>()
            .map_err(DeError::custom),
        other => Err(DeError::custom(format!(
            "expected string or number for big integer, got {other}"
        ))),
    }
}

/// Default flag for RSA key generation.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: `true` to generate keys by default.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_generate() -> bool {
    true
}

/// Default RSA public exponent.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default public exponent value.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_e() -> u64 {
    65_537
}

/// Default fixed plaintext message.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `String`: Default message string.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_fixed_message() -> String {
    "afterstate".to_string()
}

/// Default flag for random message selection.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: `false` to use a fixed message by default.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_message_random() -> bool {
    false
}

/// Default message bit length for random generation.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u32`: Default bit length.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_message_bits() -> u32 {
    56
}

/// Default flag for homomorphic base conversion.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default base conversion setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_base_convert() -> bool {
    true
}

/// Default flag for inverting derived message bits.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default invert-bits setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_invert_bits() -> bool {
    false
}

/// Default Rabin exponent used in HBC-related computations.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default exponent value.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_rabin_exponent() -> u64 {
    2
}

/// Default minimum number of message trials.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default minimum trials.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_min_message_trials() -> u64 {
    1
}

/// Default overlap percentage threshold for reporting.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `f64`: Default threshold in percent.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_overlap_report_threshold() -> f64 {
    51.0
}

/// Default entropy threshold for analysis timelines.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `f64`: Default entropy threshold.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_entropy_report_threshold() -> f64 {
    0.995
}

/// Default minimum number of r candidates to generate.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default minimum count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_process_min_count() -> u64 {
    1
}

/// Default number of r candidates to generate.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_process_count() -> u64 {
    8
}

/// Default scaling factor for candidate sampling.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u32`: Default scale value.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_process_scale() -> u32 {
    8
}

/// Default maximum attempts for selecting a best r candidate.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default attempt count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_process_max_best_attempts() -> u64 {
    4
}

/// Default minimum prime factor for r candidates.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default minimum factor.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_process_min_factor() -> u64 {
    3
}

/// Default flag for using RSA decryption in HBC flow.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default decrypt setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_use_rs_decrypt() -> bool {
    true
}

/// Default number of test iterations.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default test iteration count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_test_iterations() -> u64 {
    1
}

/// Default number of alternate iterations for fixed r testing.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default alternate iteration count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_alt_iterations() -> u64 {
    0
}

/// Default number of timeline iterations for analysis tests.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default timeline iteration count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_analysis_tests_iterations() -> u64 {
    64
}

/// Default number of iterations for oracle screening.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default screening iteration count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_oracle_screen_iterations() -> u64 {
    512
}

/// Default window size for analysis timeline frames.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default timeline window size.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_analysis_tests_window() -> usize {
    16
}

/// Default stride between analysis timeline frames.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default timeline stride.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_analysis_tests_stride() -> usize {
    4
}

/// Default number of homomorphic left-shift multiplications by 2 in analysis.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default shift count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_analysis_shift_multiplications() -> usize {
    32
}

/// Default flag for r-candidate accuracy batching.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default enable setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_analysis_batch_enable() -> bool {
    false
}

/// Default number of r candidates per accuracy batch.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default candidate count per batch.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_analysis_batch_candidates() -> u64 {
    0
}

/// Default number of r-candidate accuracy batches to evaluate.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default batch count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_analysis_batch_batches() -> u64 {
    1
}

/// Default oracle accuracy threshold for sufficiency tests.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `f64`: Default oracle accuracy threshold in percent.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_oracle_accuracy_threshold() -> f64 {
    55.0
}

/// Default flag for r_use_list stress tests.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default enable setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_r_use_list_enable() -> bool {
    false
}

/// Default flag for r_stress range testing.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default enable setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_r_stress_test_enable() -> bool {
    false
}

/// Default flag for reusing previously generated r candidates.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default reuse setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_reuse_r_candidates() -> bool {
    false
}

/// Default path for the r candidates reuse file.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `String`: Default reuse path.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_reuse_r_candidates_path() -> String {
    "data/r_candidates.csv".to_string()
}

/// Default flag for append-only reuse file behavior.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default append-only setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_reuse_r_candidates_append_only() -> bool {
    false
}

/// Default r candidate generation mode.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `RCandidateMode`: Default mode.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_r_candidate_mode() -> RCandidateMode {
    RCandidateMode::Factoring
}

/// Default list of small primes for r candidate generation.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `Vec<u64>`: Default small primes list.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_r_candidate_small_primes() -> Vec<u64> {
    vec![3, 5, 7, 11, 13, 17]
}

/// Default count of small prime factors per r candidate.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default factor count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_r_candidate_small_prime_factors() -> usize {
    3
}

/// Default maximum number of factors per r candidate.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default maximum factor count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_r_candidate_max_factors() -> usize {
    6
}

/// Default flag for enabling the combiner experiment.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default enable setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_combiner_enable() -> bool {
    false
}

/// Default number of oracles for the combiner experiment.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default oracle count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_combiner_k_oracles() -> usize {
    5
}

/// Default match probability for oracle sampling.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `f64`: Default match probability.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_combiner_match_probability() -> f64 {
    0.75
}

/// Default tie-breaker bit for combiner voting.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default tie-breaker value.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_combiner_tie_breaker() -> bool {
    true
}

/// Default flag for enciphered export generation.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default enable setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_enciphered_export_enable() -> bool {
    false
}

/// Default number of enciphered export iterations.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default iteration count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_enciphered_export_iterations() -> u64 {
    10_000
}

/// Default number of bins for enciphered export histograms.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default bin count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_enciphered_export_bins() -> usize {
    128
}

/// Default window size for enciphered export frames.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default window size.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_enciphered_export_window() -> usize {
    512
}

/// Default stride between enciphered export frames.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default stride.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_enciphered_export_stride() -> usize {
    64
}

/// Default output CSV path for enciphered export.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `String`: Default CSV path.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_enciphered_export_output_csv() -> String {
    "enciphered_decryption_bins.csv".to_string()
}

/// Default ramp length for ramp detection in enciphered export.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default ramp length.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_enciphered_export_ramp_length() -> usize {
    3
}

/// Default ramp step percentage for ramp detection.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `f64`: Default ramp step percentage.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_enciphered_export_ramp_step_pct() -> f64 {
    0.05
}

/// Default list of tolerances for ramp detection.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `Vec<f64>`: Default tolerance values.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_enciphered_export_ramp_tolerances() -> Vec<f64> {
    vec![0.005, 0.01, 0.02]
}

/// Default output CSV path for ramp detection results.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `String`: Default CSV path.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_enciphered_export_ramp_csv() -> String {
    "enciphered_ramps.csv".to_string()
}
