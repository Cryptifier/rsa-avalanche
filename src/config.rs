/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
// Configuration schema and loader for config/rsa_config.json.
use std::{error::Error, fs, path::Path};

use bigdecimal::BigDecimal;
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
    #[serde(default = "default_analysis_batch_messages")]
    pub analysis_batch_messages: u64,
    #[serde(default = "default_analysis_batch_candidates")]
    pub analysis_batch_candidates: u64,
    #[serde(default = "default_analysis_batch_batches")]
    pub analysis_batch_batches: u64,
    /// Number of avalanche combination samples to evaluate per batch.
    #[serde(default = "default_avalanche_combination_samples")]
    pub avalanche_combination_samples: u64,
    /// Legacy sampled-width setting retained for compatibility with older configs.
    #[serde(default = "default_avalanche_combination_size")]
    pub avalanche_combination_size: usize,
    /// Number of distinct r candidates mixed into each sampled avalanche input set.
    #[serde(default = "default_avalanche_combination_mixed_r_candidates")]
    pub avalanche_combination_mixed_r_candidates: usize,
    /// Legacy pool-size setting retained for compatibility; sampled pools now use the full batch.
    #[serde(default = "default_avalanche_combination_pool_size")]
    pub avalanche_combination_pool_size: usize,
    /// Number of Avalanche tiers to execute, including the initial sampled-input tier.
    #[serde(default = "default_avalanche_combination_recursion_depth")]
    pub avalanche_combination_recursion_depth: usize,
    /// Number of sample outputs grouped into each subsequent recursive Avalanche call.
    #[serde(default = "default_avalanche_combination_recursive_group_size")]
    pub avalanche_combination_recursive_group_size: usize,
    /// Number of recursive samples to produce per subsequent Avalanche tier; `0` preserves one-pass grouping.
    #[serde(default = "default_avalanche_combination_recursive_resample_count")]
    pub avalanche_combination_recursive_resample_count: usize,
    /// Whether sampled avalanche prunes scored inputs to a central Hamming-distance percentile band before sampling.
    #[serde(default = "default_avalanche_combination_hamming_distance_prune")]
    pub avalanche_combination_hamming_distance_prune: bool,
    /// Central percentile of Hamming distances retained when sampled-avalanche pruning is enabled.
    #[serde(default = "default_avalanche_combination_hamming_distance_keep_percentile")]
    pub avalanche_combination_hamming_distance_keep_percentile: f64,
    /// Percentage of the retained inlier pool size to add back from the Hamming-distance outlier tails.
    #[serde(default = "default_avalanche_combination_hamming_distance_outlier_preference_pct")]
    pub avalanche_combination_hamming_distance_outlier_preference_pct: f64,
    /// Whether sampled avalanche uses per-bit majority-vote probabilities from the combination outputs.
    #[serde(default = "default_avalanche_combination_majority_vote")]
    pub avalanche_combination_majority_vote: bool,
    /// Whether sampled avalanche smooths per-bit majority-vote probabilities before beam search.
    #[serde(default = "default_avalanche_combination_sample_smoothing")]
    pub avalanche_combination_sample_smoothing: bool,
    /// Whether sampled avalanche prints a separate majority-vote summary for the selected sample.
    #[serde(default = "default_avalanche_combination_majority_vote_print")]
    pub avalanche_combination_majority_vote_print: bool,
    /// Whether recursive Avalanche tiers carry forward the top beam-search bits instead of majority-vote bits.
    #[serde(default = "default_avalanche_use_top_beam")]
    pub avalanche_use_top_beam: bool,
    /// Whether sampled avalanche retains every evaluated sample in memory for downstream consumers.
    #[serde(default = "default_avalanche_combination_keep_all_samples_in_memory")]
    pub avalanche_combination_keep_all_samples_in_memory: bool,
    /// Whether avalanche runs collect per-level and per-sample statistics for analytics output.
    #[serde(default = "default_avalanche_statistics_collection")]
    pub avalanche_statistics_collection: bool,
    /// Whether sampled avalanche bypasses mixed-r combinations and samples raw scored inputs with ChaCha20.
    #[serde(default = "default_avalanche_random_chacha20_inputs")]
    pub avalanche_random_chacha20_inputs: bool,
    /// Whether sampled avalanche applies the trailing-zero fitness pass before sampling.
    #[serde(default = "default_avalanche_fitness_scoring_pass")]
    pub avalanche_fitness_scoring_pass: bool,
    /// Number of bytes to left-shift the plaintext before candidate scoring to create the LSB fitness slice.
    #[serde(default = "default_avalanche_fitness_shift_bytes")]
    pub avalanche_fitness_shift_bytes: usize,
    /// Number of least-significant bits inspected when computing trailing-zero fitness.
    #[serde(default = "default_avalanche_fitness_bit_width")]
    pub avalanche_fitness_bit_width: usize,
    /// Maximum number of r-candidate groups retained by the fitness pass; `0` keeps every group.
    #[serde(default = "default_avalanche_fitness_r_candidate_limit")]
    pub avalanche_fitness_r_candidate_limit: usize,
    /// Maximum number of `c^x` inputs retained per r-candidate group by the fitness pass; `0` keeps every input.
    #[serde(default = "default_avalanche_fitness_cx_candidate_limit")]
    pub avalanche_fitness_cx_candidate_limit: usize,
    #[serde(default = "default_same_r_batch")]
    pub same_r_batch: bool,
    #[serde(default = "default_ciphertext_modify")]
    pub ciphertext_modify: bool,
    #[serde(default = "default_oracle_accuracy_threshold")]
    pub oracle_accuracy_threshold: f64,
    /// Minimum stored beam value interpreted as bit `1`.
    #[serde(default = "default_beam_bit_one_threshold")]
    pub beam_bit_one_threshold: f64,
    /// Number of top avalanche beam-search candidates retained per run.
    #[serde(default = "default_avalanche_beam_top_k")]
    pub avalanche_beam_top_k: usize,
    /// Exponent used to spread normalized avalanche beam probabilities.
    #[serde(default = "default_avalanche_probability_spread_exponent")]
    pub avalanche_probability_spread_exponent: f64,
    /// Whether to sort avalanche candidates by Hamming distance.
    #[serde(default = "default_use_hamming_distance")]
    pub use_hamming_distance: bool,
    /// Whether to mirror avalanche candidates with bitwise-inverted copies before ordering.
    #[serde(default = "default_mirror_invert_candidates")]
    pub mirror_invert_candidates: bool,
    #[serde(default)]
    pub override_best_r: Option<String>,
    #[serde(default = "default_reuse_retargeted_r_candidates")]
    pub reuse_retargeted_r_candidates: bool,
    #[serde(default = "default_reuse_retargeted_r_candidates_path_prefix")]
    pub reuse_retargeted_r_candidates_path_prefix: String,
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
    /// Whether factoring-mode r candidates sample bounds from a random `N^a` window with `a ∈ [0.8, 0.9]`.
    #[serde(default = "default_r_candidate_random_power_window")]
    pub r_candidate_random_power_window: bool,
    /// Lower bound for the sampled total exponent used when retargeting speculative r candidates.
    #[serde(
        default = "default_r_candidate_target_exponent_minimum",
        deserialize_with = "deserialize_bigdecimal"
    )]
    pub r_candidate_target_exponent_minimum: BigDecimal,
    /// Upper bound for the sampled total exponent used when retargeting speculative r candidates.
    #[serde(
        default = "default_r_candidate_target_exponent",
        deserialize_with = "deserialize_bigdecimal"
    )]
    pub r_candidate_target_exponent: BigDecimal,
    /// Maximum number of exponent partitions used when retargeting speculative r candidates.
    #[serde(default = "default_r_candidate_retarget_partition_count")]
    pub r_candidate_retarget_partition_count: usize,
    /// Minimum exponent assigned to each retargeted partition when feasible.
    #[serde(
        default = "default_r_candidate_retarget_minimum_exponent",
        deserialize_with = "deserialize_bigdecimal"
    )]
    pub r_candidate_retarget_minimum_exponent: BigDecimal,
    #[serde(default = "default_combiner_k_oracles")]
    pub combiner_k_oracles: usize,
    #[serde(default = "default_combiner_tie_breaker")]
    pub combiner_tie_breaker: bool,
    #[serde(default)]
    pub message: MessageConfig,
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
            analysis_tests_iterations: default_analysis_tests_iterations(),
            oracle_screen_iterations: default_oracle_screen_iterations(),
            analysis_tests_window: default_analysis_tests_window(),
            analysis_tests_stride: default_analysis_tests_stride(),
            analysis_shift_multiplications: default_analysis_shift_multiplications(),
            analysis_batch_enable: default_analysis_batch_enable(),
            analysis_batch_messages: default_analysis_batch_messages(),
            analysis_batch_candidates: default_analysis_batch_candidates(),
            analysis_batch_batches: default_analysis_batch_batches(),
            avalanche_combination_samples: default_avalanche_combination_samples(),
            avalanche_combination_size: default_avalanche_combination_size(),
            avalanche_combination_mixed_r_candidates:
                default_avalanche_combination_mixed_r_candidates(),
            avalanche_combination_pool_size: default_avalanche_combination_pool_size(),
            avalanche_combination_recursion_depth: default_avalanche_combination_recursion_depth(),
            avalanche_combination_recursive_group_size:
                default_avalanche_combination_recursive_group_size(),
            avalanche_combination_recursive_resample_count:
                default_avalanche_combination_recursive_resample_count(),
            avalanche_combination_hamming_distance_prune:
                default_avalanche_combination_hamming_distance_prune(),
            avalanche_combination_hamming_distance_keep_percentile:
                default_avalanche_combination_hamming_distance_keep_percentile(),
            avalanche_combination_hamming_distance_outlier_preference_pct:
                default_avalanche_combination_hamming_distance_outlier_preference_pct(),
            avalanche_combination_majority_vote: default_avalanche_combination_majority_vote(),
            avalanche_combination_sample_smoothing: default_avalanche_combination_sample_smoothing(
            ),
            avalanche_combination_majority_vote_print:
                default_avalanche_combination_majority_vote_print(),
            avalanche_use_top_beam: default_avalanche_use_top_beam(),
            avalanche_combination_keep_all_samples_in_memory:
                default_avalanche_combination_keep_all_samples_in_memory(),
            avalanche_statistics_collection: default_avalanche_statistics_collection(),
            avalanche_random_chacha20_inputs: default_avalanche_random_chacha20_inputs(),
            avalanche_fitness_scoring_pass: default_avalanche_fitness_scoring_pass(),
            avalanche_fitness_shift_bytes: default_avalanche_fitness_shift_bytes(),
            avalanche_fitness_bit_width: default_avalanche_fitness_bit_width(),
            avalanche_fitness_r_candidate_limit: default_avalanche_fitness_r_candidate_limit(),
            avalanche_fitness_cx_candidate_limit: default_avalanche_fitness_cx_candidate_limit(),
            same_r_batch: default_same_r_batch(),
            ciphertext_modify: default_ciphertext_modify(),
            oracle_accuracy_threshold: default_oracle_accuracy_threshold(),
            beam_bit_one_threshold: default_beam_bit_one_threshold(),
            avalanche_beam_top_k: default_avalanche_beam_top_k(),
            avalanche_probability_spread_exponent: default_avalanche_probability_spread_exponent(),
            use_hamming_distance: default_use_hamming_distance(),
            mirror_invert_candidates: default_mirror_invert_candidates(),
            override_best_r: None,
            reuse_retargeted_r_candidates: default_reuse_retargeted_r_candidates(),
            reuse_retargeted_r_candidates_path_prefix:
                default_reuse_retargeted_r_candidates_path_prefix(),
            r_candidate_mode: default_r_candidate_mode(),
            r_candidate_small_primes: default_r_candidate_small_primes(),
            r_candidate_small_prime_factors: default_r_candidate_small_prime_factors(),
            r_candidate_max_factors: default_r_candidate_max_factors(),
            r_candidate_bit_length: None,
            r_candidate_random_power_window: default_r_candidate_random_power_window(),
            r_candidate_target_exponent_minimum: default_r_candidate_target_exponent_minimum(),
            r_candidate_target_exponent: default_r_candidate_target_exponent(),
            r_candidate_retarget_partition_count: default_r_candidate_retarget_partition_count(),
            r_candidate_retarget_minimum_exponent: default_r_candidate_retarget_minimum_exponent(),
            combiner_k_oracles: default_combiner_k_oracles(),
            combiner_tie_breaker: default_combiner_tie_breaker(),
            message: MessageConfig::default(),
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
        Some(serde_json::Value::String(s)) => {
            s.parse::<BigUint>().map(Some).map_err(DeError::custom)
        }
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
        serde_json::Value::Number(num) => {
            num.to_string().parse::<BigUint>().map_err(DeError::custom)
        }
        other => Err(DeError::custom(format!(
            "expected string or number for big integer, got {other}"
        ))),
    }
}

/// Deserializes a `BigDecimal` from a string or number JSON value.
///
/// # Parameters
/// - `deserializer`: Serde deserializer provided by the caller.
///
/// # Returns
/// - `Result<BigDecimal, D::Error>`: Parsed decimal value or an error when invalid.
///
/// # Expected Output
/// - Returns a parse error if the value is not a string/number or is invalid.
fn deserialize_bigdecimal<'de, D>(deserializer: D) -> Result<BigDecimal, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as DeError;

    let value = serde_json::Value::deserialize(deserializer)?;
    let raw = match value {
        serde_json::Value::String(s) => s,
        serde_json::Value::Number(num) => num.to_string(),
        other => {
            return Err(DeError::custom(format!(
                "expected string or number for decimal value, got {other}"
            )));
        }
    };

    raw.parse::<BigDecimal>().map_err(DeError::custom)
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

/// Default number of messages per r-candidate accuracy batch.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default message count per batch.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_analysis_batch_messages() -> u64 {
    1
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

/// Default number of avalanche combination samples to evaluate per batch.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default sampled-combination count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_samples() -> u64 {
    100
}

/// Default legacy size of each avalanche combination sample.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default compatibility value for older sampled-combination configs.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_size() -> usize {
    50
}

/// Default number of distinct r candidates mixed into each avalanche sample.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default number of r candidates contributing their `c^x` inputs.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_mixed_r_candidates() -> usize {
    1
}

/// Default number of scored candidates retained for avalanche combination sampling.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default scored-candidate pool size.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_pool_size() -> usize {
    100
}

/// Default recursion depth for sampled Avalanche.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: `1` so sampled Avalanche preserves the existing single-tier behavior by default.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_recursion_depth() -> usize {
    1
}

/// Default group size for recursive sampled Avalanche tiers.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Number of prior-tier samples grouped into each subsequent recursive call.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_recursive_group_size() -> usize {
    8
}

/// Default recursive resample count for sampled Avalanche tiers.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: `0` so recursive tiers preserve legacy one-pass regrouping unless explicitly enabled.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_recursive_resample_count() -> usize {
    0
}

/// Default flag for pruning sampled-avalanche inputs by Hamming-distance percentiles.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: `false` so sampled avalanche keeps the full scored pool unless explicitly enabled.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_hamming_distance_prune() -> bool {
    false
}

/// Default central percentile retained when pruning sampled-avalanche Hamming-distance outliers.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `f64`: Default retained percentile, keeping the middle 95% of Hamming distances.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_hamming_distance_keep_percentile() -> f64 {
    95.0
}

/// Default percentage of retained inliers to add back from Hamming-distance outlier tails.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `f64`: `0.0` so no outliers are reintroduced unless explicitly requested.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_hamming_distance_outlier_preference_pct() -> f64 {
    0.0
}

/// Default flag for random ChaCha20 sampled avalanche inputs.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: `false` to keep mixed-r combination sampling enabled by default.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_random_chacha20_inputs() -> bool {
    false
}

/// Default flag for per-bit majority voting across sampled avalanche combinations.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default majority-vote setting for sampled avalanche runs.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_majority_vote() -> bool {
    true
}

/// Default flag for smoothing per-bit majority-vote probabilities across sampled avalanche combinations.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default sample-smoothing setting for sampled avalanche runs.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_sample_smoothing() -> bool {
    false
}

/// Default flag for printing the sampled-combination majority vote alongside beam output.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default majority-vote console-output setting for sampled avalanche runs.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_majority_vote_print() -> bool {
    true
}

/// Default flag for carrying forward the top beam-search result between recursive avalanche tiers.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: `true` so recursive tiers use the prior tier's top beam-search bits by default.
///
/// # Expected Output
/// - Returns the default configuration value; no side effects.
fn default_avalanche_use_top_beam() -> bool {
    true
}

/// Returns the default in-memory sampled-avalanche retention mode.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: `false` so only the selected sample is retained by default.
///
/// # Expected Output
/// - Returns the default configuration value; no side effects.
fn default_avalanche_combination_keep_all_samples_in_memory() -> bool {
    false
}

/// Returns the default avalanche statistics collection mode.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: `true` so avalanche analytics are collected unless explicitly disabled.
///
/// # Expected Output
/// - Returns the default configuration value; no side effects.
fn default_avalanche_statistics_collection() -> bool {
    true
}

/// Default flag for enabling the trailing-zero fitness preprocessing pass.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default enable state for the fitness preprocessing pass.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_fitness_scoring_pass() -> bool {
    false
}

/// Default byte shift used to create the fitness slice ahead of the original message bits.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default byte shift applied to plaintexts before candidate scoring.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_fitness_shift_bytes() -> usize {
    0
}

/// Default trailing-zero fitness window width.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default number of LSBs scored by the fitness pass.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_fitness_bit_width() -> usize {
    32
}

/// Default cap on retained r-candidate groups for the fitness pass.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default r-group retention limit, where `0` disables truncation.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_fitness_r_candidate_limit() -> usize {
    0
}

/// Default cap on retained `c^x` inputs per r-candidate group for the fitness pass.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default per-group retention limit, where `0` disables truncation.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_fitness_cx_candidate_limit() -> usize {
    0
}

/// Default flag for using the same r candidate across a batch.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default batch r-candidate reuse setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_same_r_batch() -> bool {
    false
}

/// Default flag for ciphertext exponent modification in analysis batches.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default ciphertext modification setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_ciphertext_modify() -> bool {
    false
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

/// Default cutoff for interpreting stored beam values as bit `1`.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `f64`: Default beam bit-one threshold.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_beam_bit_one_threshold() -> f64 {
    0.4
}

/// Default number of top avalanche beam-search candidates retained.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default number of beam-search outputs recorded for avalanche runs.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_beam_top_k() -> usize {
    100
}

/// Default exponent for spreading normalized avalanche beam probabilities.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `f64`: Default avalanche probability spread exponent.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_probability_spread_exponent() -> f64 {
    0.5
}

/// Default toggle for Hamming-distance sorting in avalanche candidate ordering.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default enable setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_use_hamming_distance() -> bool {
    false
}

/// Default toggle for mirroring avalanche candidates with inverted bits.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default enable setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_mirror_invert_candidates() -> bool {
    false
}

/// Default flag for reusing previously generated r candidates.
///
/// # Parameters
/// - None.
///
/// Default flag for reading pre-retargeted r candidates from a keyed cache file.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default retargeted-cache reuse setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_reuse_retargeted_r_candidates() -> bool {
    false
}

/// Default path prefix for keyed retargeted r-candidate cache files.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `String`: Default retargeted-cache path prefix.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_reuse_retargeted_r_candidates_path_prefix() -> String {
    "data/rgen_retargeted".to_string()
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

/// Default flag for factoring-mode random `N^a` r-candidate sampling.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default power-window sampling flag.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_r_candidate_random_power_window() -> bool {
    false
}

/// Default upper bound for sampled retarget exponents.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `BigDecimal`: Default retarget exponent upper bound.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_r_candidate_target_exponent() -> BigDecimal {
    BigDecimal::parse_bytes(b"2.005", 10).expect("valid default r candidate target exponent")
}

/// Default lower bound for sampled retarget exponents.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `BigDecimal`: Default retarget exponent lower bound.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_r_candidate_target_exponent_minimum() -> BigDecimal {
    BigDecimal::parse_bytes(b"0.8", 10).expect("valid default r candidate target exponent minimum")
}

/// Default partition count used for speculative r-candidate retargeting.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default partition count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_r_candidate_retarget_partition_count() -> usize {
    3
}

/// Default minimum exponent per retargeted speculative factor.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `BigDecimal`: Default minimum per-part exponent.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_r_candidate_retarget_minimum_exponent() -> BigDecimal {
    BigDecimal::parse_bytes(b"0.45", 10)
        .expect("valid default r candidate retarget minimum exponent")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_config_defaults_disable_avalanche_hamming_distance_pruning() {
        let engine = EngineConfig::default();

        assert!(engine.avalanche_use_top_beam);
        assert!(engine.avalanche_statistics_collection);
        assert_eq!(engine.avalanche_combination_recursive_resample_count, 0);
        assert!(!engine.avalanche_combination_hamming_distance_prune);
        assert_eq!(
            engine.avalanche_combination_hamming_distance_keep_percentile,
            95.0
        );
        assert_eq!(
            engine.avalanche_combination_hamming_distance_outlier_preference_pct,
            0.0
        );
    }
}
