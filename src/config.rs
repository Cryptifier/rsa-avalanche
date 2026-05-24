/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
// Configuration schema and loader for config/rsa_config.json.
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use bigdecimal::BigDecimal;
use num_bigint::BigUint;
use serde::{Deserialize, Deserializer};

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
    /// Source configuration path recorded by the loader for resolving relative keyfile references.
    #[serde(skip)]
    pub source_path: Option<PathBuf>,
}

/// RSA keypair configuration values.
#[derive(Debug, Deserialize, Clone)]
pub struct KeyConfig {
    /// Whether to generate keys instead of using provided values.
    #[serde(default = "default_generate")]
    pub generate: bool,
    /// Relative or absolute YAML keypair path used when `p`/`q` are not configured.
    #[serde(default = "default_keyfile")]
    pub keyfile: String,
    /// Optional private-key YAML path used only for verification peeks when the main keyfile is public.
    #[serde(default = "default_keyfile")]
    pub private_keyfile: String,
    /// RSA prime p (required when not generating).
    #[serde(default, deserialize_with = "deserialize_biguint_option")]
    pub p: Option<BigUint>,
    /// RSA prime q (required when not generating).
    #[serde(default, deserialize_with = "deserialize_biguint_option")]
    pub q: Option<BigUint>,
    /// RSA public exponent.
    #[serde(default = "default_e")]
    pub e: u64,
    /// RSA modulus hydrated from a YAML keyfile when available.
    #[serde(skip)]
    pub modulus: Option<BigUint>,
}

/// Supported RSA YAML keyfile formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RsaKeyFileFormat {
    /// `rsa-private-key-v1` with modulus, exponent, primes, totient, and private exponent.
    PrivateKeyV1,
    /// `rsa-public-key-v1` with modulus and public exponent only.
    PublicKeyV1,
}

/// Parsed RSA YAML key material loaded from a public or private keyfile.
#[derive(Debug, Clone)]
pub struct RsaKeyMaterial {
    /// Parsed YAML format identifier.
    pub format: RsaKeyFileFormat,
    /// RSA modulus `n`.
    pub modulus: BigUint,
    /// RSA public exponent `e`.
    pub public_exponent: u64,
    /// RSA private exponent `d` when the keyfile is private.
    pub private_exponent: Option<BigUint>,
    /// Euler totient `phi(n)` when the keyfile is private.
    pub totient: Option<BigUint>,
    /// Prime `p` when the keyfile is private.
    pub p: Option<BigUint>,
    /// Prime `q` when the keyfile is private.
    pub q: Option<BigUint>,
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
    /// Whether the final-tier Avalanche solver should compare batch-pair sample products for whole-message recovery.
    #[serde(default = "default_avalanche_solver_enable")]
    pub avalanche_solver_enable: bool,
    /// Whether Avalanche runs should log the global majority vote across every final-tier output.
    #[serde(default = "default_avalanche_solver_global_log_enable")]
    pub avalanche_solver_global_log_enable: bool,
    /// Maximum number of differing sample bits the Avalanche solver may brute-force per batch-pair sample comparison.
    #[serde(default = "default_avalanche_solver_max_bits")]
    pub avalanche_solver_max_bits: usize,
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
    /// Per-recursive-tier group sizes, reusing the last entry when recursion exceeds the configured array.
    #[serde(
        default = "default_avalanche_combination_recursive_group_size",
        deserialize_with = "deserialize_nonempty_usize_array_or_scalar"
    )]
    pub avalanche_combination_recursive_group_size: Vec<usize>,
    /// Per-recursive-tier resample counts, reusing the last entry when recursion exceeds the configured array.
    #[serde(
        default = "default_avalanche_combination_recursive_resample_count",
        deserialize_with = "deserialize_nonempty_usize_array_or_scalar"
    )]
    pub avalanche_combination_recursive_resample_count: Vec<usize>,
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
    /// Whether majority-vote console output should include differing bit locations and bias details.
    #[serde(default = "default_avalanche_statistics_show_majority_vote_biases")]
    pub avalanche_statistics_show_majority_vote_biases: bool,
    /// Whether final-tier sampled Avalanche reports near-center beam probabilities in the session log.
    #[serde(default = "default_avalanche_report_biases")]
    pub avalanche_report_biases: bool,
    /// Maximum absolute distance from `0.5` retained in final-tier sampled Avalanche bias reports.
    #[serde(default = "default_avalanche_center_threshold")]
    pub avalanche_center_threshold: f64,
    /// Whether final-tier sampled Avalanche bias reporting should keep only the best overall Avalanche candidate.
    #[serde(default = "default_avalanche_center_threshold_best")]
    pub avalanche_center_threshold_best: bool,
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
    /// Whether sampled avalanche applies the zero-count fitness pass before sampling.
    #[serde(default = "default_avalanche_fitness_scoring_pass")]
    pub avalanche_fitness_scoring_pass: bool,
    /// Number of bytes to left-shift the plaintext before candidate scoring to create the LSB fitness slice.
    #[serde(default = "default_avalanche_fitness_shift_bytes")]
    pub avalanche_fitness_shift_bytes: usize,
    /// Number of least-significant bits inspected when computing zero-count fitness.
    #[serde(default = "default_avalanche_fitness_bit_width")]
    pub avalanche_fitness_bit_width: usize,
    /// Primary retention dimension used to derive the global retained-input cap for the fitness pass; `0` disables this dimension.
    #[serde(default = "default_avalanche_fitness_r_candidate_limit")]
    pub avalanche_fitness_r_candidate_limit: usize,
    /// Secondary retention dimension used to derive the global retained-input cap for the fitness pass; `0` disables this dimension.
    #[serde(default = "default_avalanche_fitness_cx_candidate_limit")]
    pub avalanche_fitness_cx_candidate_limit: usize,
    /// Whether the fitness pass drops candidates whose normalized fitness falls below the configured threshold.
    #[serde(default = "default_avalanche_fitness_use_threshold")]
    pub avalanche_fitness_use_threshold: bool,
    /// Minimum normalized zero-count fitness retained by the fitness pass when thresholding is enabled.
    #[serde(default = "default_avalanche_fitness_threshold")]
    pub avalanche_fitness_threshold: f64,
    /// Percentage of thresholded fitness-ranked candidates logged for each Avalanche batch.
    #[serde(default = "default_avalanche_fitness_log_top_pct")]
    pub avalanche_fitness_log_top_pct: f64,
    /// Number of additional random messages used to test padding-bit fitness for each retained `c^x/r` candidate.
    #[serde(default = "default_avalanche_fitness_additional_random_messages")]
    pub avalanche_fitness_additional_random_messages: usize,
    /// Whether batch scoring should prune the fitness-ranked Avalanche input pool incrementally while candidates are still being processed.
    #[serde(default = "default_avalanche_fitness_streaming_prune")]
    pub avalanche_fitness_streaming_prune: bool,
    /// Whether sampled Avalanche should keep a globally unique set with no repeated `r` or `x` values.
    #[serde(default = "default_avalanche_unique_r_cx_inputs")]
    pub avalanche_unique_r_cx_inputs: bool,
    /// Whether sampled Avalanche should always include one deterministic input built from the
    /// highest-ranked retained candidates in order before randomized sampling.
    #[serde(default = "default_avalanche_include_max_fitness_candidates_in_order")]
    pub avalanche_include_max_fitness_candidates_in_order: bool,
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
    /// Advisory SQLite soft heap limit in bytes used by the Avalanche cache database.
    #[serde(default = "default_sqlite_soft_heap")]
    pub sqlite_soft_heap: u64,
    /// Hard SQLite heap limit in bytes used by the Avalanche cache database.
    #[serde(default = "default_sqlite_hard_heap")]
    pub sqlite_hard_heap: u64,
    /// SQLite mmap size in bytes used by the Avalanche cache database.
    #[serde(default = "default_sqlite_mmap_size")]
    pub sqlite_mmap_size: u64,
    /// SQLite worker count used by the Avalanche cache connection pool.
    #[serde(default = "default_sqlite_worker_count")]
    pub sqlite_worker_count: u32,
    /// Filesystem folder used for the Avalanche cache SQLite database.
    #[serde(default = "default_sqlite_db_folder")]
    pub sqlite_db_folder: String,
    /// Number of rows per SQLite Avalanche cache page used for batched inserts and reads.
    #[serde(default = "default_sqlite_avalanche_page_size")]
    pub sqlite_avalanche_page_size: usize,
    /// Whether the Avalanche cache should use a shared in-memory SQLite database instead of an on-disk file.
    #[serde(default = "default_sqlite_in_memory")]
    pub sqlite_in_memory: bool,
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
            source_path: None,
        }
    }
}

impl Default for KeyConfig {
    fn default() -> Self {
        Self {
            generate: default_generate(),
            keyfile: default_keyfile(),
            private_keyfile: default_keyfile(),
            p: None,
            q: None,
            e: default_e(),
            modulus: None,
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
            avalanche_solver_enable: default_avalanche_solver_enable(),
            avalanche_solver_global_log_enable: default_avalanche_solver_global_log_enable(),
            avalanche_solver_max_bits: default_avalanche_solver_max_bits(),
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
            avalanche_statistics_show_majority_vote_biases:
                default_avalanche_statistics_show_majority_vote_biases(),
            avalanche_report_biases: default_avalanche_report_biases(),
            avalanche_center_threshold: default_avalanche_center_threshold(),
            avalanche_center_threshold_best: default_avalanche_center_threshold_best(),
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
            avalanche_fitness_use_threshold: default_avalanche_fitness_use_threshold(),
            avalanche_fitness_threshold: default_avalanche_fitness_threshold(),
            avalanche_fitness_log_top_pct: default_avalanche_fitness_log_top_pct(),
            avalanche_fitness_additional_random_messages:
                default_avalanche_fitness_additional_random_messages(),
            avalanche_fitness_streaming_prune: default_avalanche_fitness_streaming_prune(),
            avalanche_unique_r_cx_inputs: default_avalanche_unique_r_cx_inputs(),
            avalanche_include_max_fitness_candidates_in_order:
                default_avalanche_include_max_fitness_candidates_in_order(),
            same_r_batch: default_same_r_batch(),
            ciphertext_modify: default_ciphertext_modify(),
            oracle_accuracy_threshold: default_oracle_accuracy_threshold(),
            beam_bit_one_threshold: default_beam_bit_one_threshold(),
            avalanche_beam_top_k: default_avalanche_beam_top_k(),
            avalanche_probability_spread_exponent: default_avalanche_probability_spread_exponent(),
            sqlite_soft_heap: default_sqlite_soft_heap(),
            sqlite_hard_heap: default_sqlite_hard_heap(),
            sqlite_mmap_size: default_sqlite_mmap_size(),
            sqlite_worker_count: default_sqlite_worker_count(),
            sqlite_db_folder: default_sqlite_db_folder(),
            sqlite_avalanche_page_size: default_sqlite_avalanche_page_size(),
            sqlite_in_memory: default_sqlite_in_memory(),
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
    let mut config: Config = match serde_json::from_str(&raw) {
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

    config.source_path = Some(cfg_path.to_path_buf());
    hydrate_keypair_from_keyfile(cfg_path, &mut config.rsa_keypair)?;
    Ok(config)
}

#[derive(Debug, Deserialize)]
struct RsaKeyYamlEnvelope {
    format: String,
    algorithm: String,
    public_exponent: String,
    modulus: String,
    #[serde(default)]
    private_exponent: Option<String>,
    #[serde(default)]
    totient: Option<String>,
    #[serde(default)]
    primes: Option<RsaPrivateKeyYamlPrimes>,
}

#[derive(Debug, Deserialize)]
struct RsaPrivateKeyYamlPrimes {
    p: String,
    q: String,
}

/// Resolves a keyfile path relative to the configuration file that references it.
///
/// # Parameters
/// - `config_path`: Path to the JSON/JSON5 configuration file.
/// - `keyfile`: Configured keyfile string, which may be relative or absolute.
///
/// # Returns
/// - `PathBuf`: Resolved filesystem path to the requested keyfile.
///
/// # Expected Output
/// - Returns a path value without touching the filesystem.
pub fn resolve_keyfile_path(config_path: &Path, keyfile: &str) -> PathBuf {
    let keyfile_path = Path::new(keyfile);
    if keyfile_path.is_absolute() {
        return keyfile_path.to_path_buf();
    }

    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(keyfile_path)
}

/// Parses one decimal bigint field from an RSA keyfile.
///
/// # Parameters
/// - `field_name`: Human-readable field label for error reporting.
/// - `raw`: Raw decimal field value from the YAML file.
/// - `path`: Resolved keyfile path used for diagnostics.
///
/// # Returns
/// - `Result<BigUint, Box<dyn Error>>`: Parsed bigint value.
///
/// # Expected Output
/// - Returns a parsed bigint or an error; no stdout/stderr output.
fn parse_keyfile_biguint(
    field_name: &str,
    raw: &str,
    path: &Path,
) -> Result<BigUint, Box<dyn Error>> {
    raw.trim().parse::<BigUint>().map_err(|err| {
        format!(
            "failed to parse {field_name} in keyfile {}: {err}",
            path.display()
        )
        .into()
    })
}

/// Loads RSA key material from a resolved YAML keyfile path.
///
/// # Parameters
/// - `resolved_path`: Filesystem path to the YAML keyfile.
///
/// # Returns
/// - `Result<RsaKeyMaterial, Box<dyn Error>>`: Parsed RSA key material from the requested YAML file.
///
/// # Expected Output
/// - Reads and deserializes the YAML file from disk; no stdout/stderr output on success.
pub fn load_rsa_key_material_from_yaml_path(
    resolved_path: &Path,
) -> Result<RsaKeyMaterial, Box<dyn Error>> {
    let raw = fs::read_to_string(&resolved_path)?;
    let parsed: RsaKeyYamlEnvelope = serde_yaml::from_str(&raw).map_err(|err| {
        format!(
            "failed to parse RSA keyfile {}: {err}",
            resolved_path.display()
        )
    })?;

    if parsed.algorithm.trim() != "RSA" {
        return Err(format!(
            "unsupported key algorithm {} in {}",
            parsed.algorithm,
            resolved_path.display()
        )
        .into());
    }

    let modulus = parse_keyfile_biguint("modulus", &parsed.modulus, resolved_path)?;
    let public_exponent = parsed
        .public_exponent
        .trim()
        .parse::<u64>()
        .map_err(|err| {
            format!(
                "failed to parse public_exponent in keyfile {}: {err}",
                resolved_path.display()
            )
        })?;

    match parsed.format.trim() {
        "rsa-private-key-v1" => {
            let private_exponent_raw = parsed.private_exponent.as_deref().ok_or_else(|| {
                format!(
                    "rsa-private-key-v1 file {} is missing private_exponent",
                    resolved_path.display()
                )
            })?;
            let totient_raw = parsed.totient.as_deref().ok_or_else(|| {
                format!(
                    "rsa-private-key-v1 file {} is missing totient",
                    resolved_path.display()
                )
            })?;
            let primes = parsed.primes.ok_or_else(|| {
                format!(
                    "rsa-private-key-v1 file {} is missing primes",
                    resolved_path.display()
                )
            })?;
            let private_exponent =
                parse_keyfile_biguint("private_exponent", private_exponent_raw, resolved_path)?;
            let totient = parse_keyfile_biguint("totient", totient_raw, resolved_path)?;
            let p = parse_keyfile_biguint("primes.p", &primes.p, resolved_path)?;
            let q = parse_keyfile_biguint("primes.q", &primes.q, resolved_path)?;
            Ok(RsaKeyMaterial {
                format: RsaKeyFileFormat::PrivateKeyV1,
                modulus,
                public_exponent,
                private_exponent: Some(private_exponent),
                totient: Some(totient),
                p: Some(p),
                q: Some(q),
            })
        }
        "rsa-public-key-v1" => {
            if parsed.private_exponent.is_some()
                || parsed.totient.is_some()
                || parsed.primes.is_some()
            {
                return Err(format!(
                    "rsa-public-key-v1 file {} must not include private_exponent, totient, or primes",
                    resolved_path.display()
                )
                .into());
            }
            Ok(RsaKeyMaterial {
                format: RsaKeyFileFormat::PublicKeyV1,
                modulus,
                public_exponent,
                private_exponent: None,
                totient: None,
                p: None,
                q: None,
            })
        }
        other => Err(format!(
            "unsupported RSA keyfile format {other} in {}",
            resolved_path.display()
        )
        .into()),
    }
}

/// Loads RSA key material from a configuration-relative YAML keyfile path.
///
/// # Parameters
/// - `config_path`: Path to the JSON/JSON5 configuration file requesting the key.
/// - `keyfile`: Configured YAML keyfile path, relative to `config_path` when not absolute.
///
/// # Returns
/// - `Result<RsaKeyMaterial, Box<dyn Error>>`: Parsed RSA key material from the requested YAML file.
///
/// # Expected Output
/// - Reads and deserializes the YAML file from disk; no stdout/stderr output on success.
pub fn load_rsa_key_material_from_config_keyfile(
    config_path: &Path,
    keyfile: &str,
) -> Result<RsaKeyMaterial, Box<dyn Error>> {
    let resolved_path = resolve_keyfile_path(config_path, keyfile);
    load_rsa_key_material_from_yaml_path(&resolved_path)
}

/// Hydrates missing inline RSA key material from a configured YAML keyfile.
///
/// # Parameters
/// - `config_path`: Path to the JSON/JSON5 configuration file.
/// - `key_config`: Mutable RSA keypair configuration to backfill.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` when hydration succeeds or is unnecessary.
///
/// # Expected Output
/// - Reads the configured YAML keyfile when inline primes are absent; no stdout/stderr output on success.
fn hydrate_keypair_from_keyfile(
    config_path: &Path,
    key_config: &mut KeyConfig,
) -> Result<(), Box<dyn Error>> {
    if key_config.generate {
        return Ok(());
    }

    if let (Some(p), Some(q)) = (&key_config.p, &key_config.q) {
        key_config.modulus = Some(p * q);
        return Ok(());
    }

    let keyfile = key_config.keyfile.trim();
    if keyfile.is_empty() {
        return Ok(());
    }

    let material = load_rsa_key_material_from_config_keyfile(config_path, keyfile)?;
    key_config.modulus = Some(material.modulus.clone());
    key_config.e = material.public_exponent;
    key_config.p = material.p;
    key_config.q = material.q;
    Ok(())
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

/// Deserializes a non-empty `usize` array from either a scalar or an array.
///
/// # Parameters
/// - `deserializer`: Serde deserializer provided by the caller.
///
/// # Returns
/// - `Result<Vec<usize>, D::Error>`: Parsed array, wrapping scalars into one-item vectors.
///
/// # Expected Output
/// - Returns a parse error if the input is not a `usize` or `usize` array.
fn deserialize_nonempty_usize_array_or_scalar<'de, D>(
    deserializer: D,
) -> Result<Vec<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error as DeError;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum UsizeArrayOrScalar {
        Scalar(usize),
        Array(Vec<usize>),
    }

    let values = match UsizeArrayOrScalar::deserialize(deserializer)? {
        UsizeArrayOrScalar::Scalar(value) => vec![value],
        UsizeArrayOrScalar::Array(values) => values,
    };

    if values.is_empty() {
        return Err(DeError::custom(
            "expected at least one recursive tier value, got an empty array",
        ));
    }

    Ok(values)
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

/// Default keyfile path.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `String`: Empty string, indicating no YAML keyfile is configured.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_keyfile() -> String {
    String::new()
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

/// Default enable flag for the cross-batch Avalanche solver.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: `false` so the solver only runs when explicitly enabled in JSON config.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_solver_enable() -> bool {
    false
}

/// Default enable flag for logging the global final-tier Avalanche majority vote.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: `true` so runs log the aggregate final-tier majority vote by default.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_solver_global_log_enable() -> bool {
    true
}

/// Default maximum flip count for each Avalanche solver brute-force search.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default maximum number of differing bit positions the solver may brute-force.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_solver_max_bits() -> usize {
    8
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
/// - `Vec<usize>`: Default per-tier group sizes, starting with one entry for the first recursive tier.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_recursive_group_size() -> Vec<usize> {
    vec![8]
}

/// Default recursive resample count for sampled Avalanche tiers.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `Vec<usize>`: Default per-tier resample counts, starting with one entry for the first recursive tier.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_combination_recursive_resample_count() -> Vec<usize> {
    vec![0]
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

/// Default flag for printing differing-bit majority-vote bias details.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: `true` so majority-vote console output includes differing-bit bias diagnostics by default.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_statistics_show_majority_vote_biases() -> bool {
    true
}

/// Default flag for writing filtered final-tier Avalanche bias reports into the session log.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: `false` so runs avoid extra bias-report payloads unless explicitly enabled.
///
/// # Expected Output
/// - Returns the default reporting flag; no stdout/stderr output.
fn default_avalanche_report_biases() -> bool {
    false
}

/// Default half-width around `0.5` retained in final-tier Avalanche bias reports.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `f64`: `0.01`, meaning reported probabilities must lie in `[0.49, 0.51]`.
///
/// # Expected Output
/// - Returns the default center threshold; no stdout/stderr output.
fn default_avalanche_center_threshold() -> f64 {
    0.01
}

/// Default mode for final-tier Avalanche center-bias session reporting.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: `true` so center-bias reports default to the single best overall Avalanche result.
///
/// # Expected Output
/// - Returns the default best-only reporting mode; no stdout/stderr output.
fn default_avalanche_center_threshold_best() -> bool {
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

/// Default flag for enabling the zero-count fitness preprocessing pass.
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

/// Default zero-count fitness window width.
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

/// Default primary retention dimension for the global Avalanche fitness cap.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default primary cap dimension, where `0` disables this dimension.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_fitness_r_candidate_limit() -> usize {
    0
}

/// Default secondary retention dimension for the global Avalanche fitness cap.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default secondary cap dimension, where `0` disables this dimension.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_fitness_cx_candidate_limit() -> usize {
    0
}

/// Default flag for applying the normalized fitness threshold.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default enable state for normalized fitness thresholding.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_fitness_use_threshold() -> bool {
    true
}

/// Default normalized fitness threshold used by the Avalanche fitness pass.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `f64`: Default minimum normalized zero-count fitness.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_fitness_threshold() -> f64 {
    0.580
}

/// Default top-percentage of thresholded Avalanche fitness entries printed to the batch log.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `f64`: Default fraction of retained fitness-ranked candidates included in batch logging.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_fitness_log_top_pct() -> f64 {
    0.30
}

/// Default number of additional random messages used to test padding fitness per candidate.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default number of extra random-message fitness checks.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_fitness_additional_random_messages() -> usize {
    0
}

/// Default flag for streaming Avalanche fitness pruning during batch scoring.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default enable state for incremental fitness-ranked pruning.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_fitness_streaming_prune() -> bool {
    false
}

/// Default flag for enforcing globally unique `r` and `x` Avalanche inputs.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default uniqueness-enforcement setting.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_unique_r_cx_inputs() -> bool {
    false
}

/// Default flag for seeding sampled Avalanche with the top retained candidates in rank order.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default ordered-seed enable state.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_avalanche_include_max_fitness_candidates_in_order() -> bool {
    true
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

/// Default advisory SQLite soft heap limit for the Avalanche cache.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default SQLite soft heap limit in bytes.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_sqlite_soft_heap() -> u64 {
    10 * 1024 * 1024 * 1024
}

/// Default hard SQLite heap limit for the Avalanche cache.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default SQLite hard heap limit in bytes.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_sqlite_hard_heap() -> u64 {
    10 * 1024 * 1024 * 1024
}

/// Default SQLite mmap size for the Avalanche cache.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u64`: Default SQLite mmap size in bytes.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_sqlite_mmap_size() -> u64 {
    10 * 1024 * 1024 * 1024
}

/// Default SQLite worker count for the Avalanche cache connection pool.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u32`: Default SQLite worker count.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_sqlite_worker_count() -> u32 {
    16
}

/// Default filesystem folder for the Avalanche cache SQLite database.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `String`: Default SQLite database folder.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_sqlite_db_folder() -> String {
    "/tmp".to_string()
}

/// Default SQLite Avalanche cache page size used for batched inserts and reads.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `usize`: Default SQLite Avalanche cache page size in rows.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_sqlite_avalanche_page_size() -> usize {
    4_096
}

/// Default flag for keeping the Avalanche cache SQLite database in memory.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `bool`: Default SQLite cache storage mode.
///
/// # Expected Output
/// - Returns a constant default value; no side effects.
fn default_sqlite_in_memory() -> bool {
    false
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
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Builds a unique temporary path for config-loader tests.
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
        std::env::temp_dir().join(format!("rsademo_config_{label}_{nanos}"))
    }

    #[test]
    fn test_engine_config_defaults_disable_avalanche_hamming_distance_pruning() {
        let engine = EngineConfig::default();

        assert!(engine.avalanche_use_top_beam);
        assert!(engine.avalanche_statistics_collection);
        assert!(!engine.avalanche_report_biases);
        assert_eq!(engine.avalanche_center_threshold, 0.01);
        assert!(engine.avalanche_center_threshold_best);
        assert_eq!(engine.avalanche_combination_recursive_group_size, vec![8]);
        assert_eq!(
            engine.avalanche_combination_recursive_resample_count,
            vec![0]
        );
        assert!(!engine.avalanche_combination_hamming_distance_prune);
        assert_eq!(
            engine.avalanche_combination_hamming_distance_keep_percentile,
            95.0
        );
        assert_eq!(
            engine.avalanche_combination_hamming_distance_outlier_preference_pct,
            0.0
        );
        assert!(!engine.avalanche_solver_enable);
        assert_eq!(engine.avalanche_solver_max_bits, 8);
        assert!(engine.avalanche_fitness_use_threshold);
        assert!((engine.avalanche_fitness_threshold - 0.580).abs() < f64::EPSILON);
        assert!((engine.avalanche_fitness_log_top_pct - 0.30).abs() < f64::EPSILON);
        assert_eq!(engine.avalanche_fitness_additional_random_messages, 0);
        assert!(!engine.avalanche_fitness_streaming_prune);
        assert!(!engine.avalanche_unique_r_cx_inputs);
        assert!(engine.avalanche_include_max_fitness_candidates_in_order);
        assert!(engine.avalanche_statistics_show_majority_vote_biases);
        assert_eq!(engine.sqlite_soft_heap, 10 * 1024 * 1024 * 1024);
        assert_eq!(engine.sqlite_hard_heap, 10 * 1024 * 1024 * 1024);
        assert_eq!(engine.sqlite_mmap_size, 10 * 1024 * 1024 * 1024);
        assert_eq!(engine.sqlite_worker_count, 16);
        assert_eq!(engine.sqlite_db_folder, "/tmp");
        assert_eq!(engine.sqlite_avalanche_page_size, 4_096);
        assert!(!engine.sqlite_in_memory);
    }

    #[test]
    fn test_load_config_hydrates_keypair_from_relative_keyfile() {
        let temp_dir = temp_path("keyfile_relative");
        fs::create_dir_all(temp_dir.join("keys")).expect("create temp config dir");
        fs::write(
            temp_dir.join("keys").join("sample.yaml"),
            concat!(
                "format: rsa-private-key-v1\n",
                "algorithm: RSA\n",
                "public_exponent: \"17\"\n",
                "private_exponent: \"2753\"\n",
                "modulus: \"3233\"\n",
                "totient: \"3120\"\n",
                "primes:\n",
                "  p: \"61\"\n",
                "  q: \"53\"\n",
            ),
        )
        .expect("write keyfile");
        fs::write(
            temp_dir.join("config.json"),
            concat!(
                "{\n",
                "  \"rsa_keypair\": {\n",
                "    \"generate\": false,\n",
                "    \"keyfile\": \"keys/sample.yaml\"\n",
                "  }\n",
                "}\n",
            ),
        )
        .expect("write config");

        let config = load_config(temp_dir.join("config.json").to_str().expect("utf8 path"))
            .expect("load config");
        assert_eq!(config.rsa_keypair.e, 17);
        assert_eq!(config.rsa_keypair.modulus, Some(BigUint::from(3233u32)));
        assert_eq!(config.rsa_keypair.p, Some(BigUint::from(61u8)));
        assert_eq!(config.rsa_keypair.q, Some(BigUint::from(53u8)));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_load_config_prefers_inline_primes_over_keyfile() {
        let temp_dir = temp_path("keyfile_inline_override");
        fs::create_dir_all(temp_dir.join("keys")).expect("create temp config dir");
        fs::write(
            temp_dir.join("keys").join("sample.yaml"),
            concat!(
                "format: rsa-private-key-v1\n",
                "algorithm: RSA\n",
                "public_exponent: \"17\"\n",
                "private_exponent: \"2753\"\n",
                "modulus: \"3233\"\n",
                "totient: \"3120\"\n",
                "primes:\n",
                "  p: \"61\"\n",
                "  q: \"53\"\n",
            ),
        )
        .expect("write keyfile");
        fs::write(
            temp_dir.join("config.json"),
            concat!(
                "{\n",
                "  \"rsa_keypair\": {\n",
                "    \"generate\": false,\n",
                "    \"keyfile\": \"keys/sample.yaml\",\n",
                "    \"e\": 65537,\n",
                "    \"p\": \"71\",\n",
                "    \"q\": \"67\"\n",
                "  }\n",
                "}\n",
            ),
        )
        .expect("write config");

        let config = load_config(temp_dir.join("config.json").to_str().expect("utf8 path"))
            .expect("load config");
        assert_eq!(config.rsa_keypair.e, 65537);
        assert_eq!(config.rsa_keypair.modulus, Some(BigUint::from(4757u32)));
        assert_eq!(config.rsa_keypair.p, Some(BigUint::from(71u8)));
        assert_eq!(config.rsa_keypair.q, Some(BigUint::from(67u8)));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_load_config_hydrates_public_keyfile_without_primes() {
        let temp_dir = temp_path("keyfile_public");
        fs::create_dir_all(temp_dir.join("keys")).expect("create temp config dir");
        fs::write(
            temp_dir.join("keys").join("public.yaml"),
            concat!(
                "format: rsa-public-key-v1\n",
                "algorithm: RSA\n",
                "public_exponent: \"17\"\n",
                "modulus: \"3233\"\n",
                "bit_lengths:\n",
                "  modulus_bits: 12\n",
            ),
        )
        .expect("write public keyfile");
        fs::write(
            temp_dir.join("config.json"),
            concat!(
                "{\n",
                "  \"rsa_keypair\": {\n",
                "    \"generate\": false,\n",
                "    \"keyfile\": \"keys/public.yaml\",\n",
                "    \"private_keyfile\": \"keys/private.yaml\"\n",
                "  }\n",
                "}\n",
            ),
        )
        .expect("write config");

        let config = load_config(temp_dir.join("config.json").to_str().expect("utf8 path"))
            .expect("load config");
        assert_eq!(config.rsa_keypair.e, 17);
        assert_eq!(config.rsa_keypair.modulus, Some(BigUint::from(3233u32)));
        assert_eq!(config.rsa_keypair.p, None);
        assert_eq!(config.rsa_keypair.q, None);
        assert_eq!(config.rsa_keypair.private_keyfile, "keys/private.yaml");

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_load_config_accepts_scalar_recursive_tier_values() {
        let temp_dir = temp_path("recursive_scalar_values");
        fs::create_dir_all(&temp_dir).expect("create temp config dir");
        fs::write(
            temp_dir.join("config.json"),
            concat!(
                "{\n",
                "  \"engine\": {\n",
                "    \"avalanche_combination_recursive_group_size\": 16,\n",
                "    \"avalanche_combination_recursive_resample_count\": 2048\n",
                "  }\n",
                "}\n",
            ),
        )
        .expect("write config");

        let config = load_config(temp_dir.join("config.json").to_str().expect("utf8 path"))
            .expect("load config");

        assert_eq!(
            config.engine.avalanche_combination_recursive_group_size,
            vec![16]
        );
        assert_eq!(
            config.engine.avalanche_combination_recursive_resample_count,
            vec![2048]
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_load_config_accepts_recursive_tier_arrays() {
        let temp_dir = temp_path("recursive_array_values");
        fs::create_dir_all(&temp_dir).expect("create temp config dir");
        fs::write(
            temp_dir.join("config.json"),
            concat!(
                "{\n",
                "  \"engine\": {\n",
                "    \"avalanche_combination_recursive_group_size\": [16, 8],\n",
                "    \"avalanche_combination_recursive_resample_count\": [2048, 0]\n",
                "  }\n",
                "}\n",
            ),
        )
        .expect("write config");

        let config = load_config(temp_dir.join("config.json").to_str().expect("utf8 path"))
            .expect("load config");

        assert_eq!(
            config.engine.avalanche_combination_recursive_group_size,
            vec![16, 8]
        );
        assert_eq!(
            config.engine.avalanche_combination_recursive_resample_count,
            vec![2048, 0]
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_load_config_accepts_fitness_log_top_percentage() {
        let temp_dir = temp_path("fitness_log_top_percentage");
        fs::create_dir_all(&temp_dir).expect("create temp config dir");
        fs::write(
            temp_dir.join("config.json"),
            concat!(
                "{\n",
                "  \"engine\": {\n",
                "    \"avalanche_fitness_log_top_pct\": 0.45\n",
                "  }\n",
                "}\n",
            ),
        )
        .expect("write config");

        let config = load_config(temp_dir.join("config.json").to_str().expect("utf8 path"))
            .expect("load config");

        assert!((config.engine.avalanche_fitness_log_top_pct - 0.45).abs() < f64::EPSILON);

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_load_config_accepts_streaming_prune_unique_inputs_and_in_memory_sqlite() {
        let temp_dir = temp_path("stream_prune_unique_sqlite");
        fs::create_dir_all(&temp_dir).expect("create temp config dir");
        fs::write(
            temp_dir.join("config.json"),
            concat!(
                "{\n",
                "  \"engine\": {\n",
                "    \"avalanche_fitness_additional_random_messages\": 3,\n",
                "    \"avalanche_fitness_streaming_prune\": true,\n",
                "    \"avalanche_unique_r_cx_inputs\": true,\n",
                "    \"avalanche_include_max_fitness_candidates_in_order\": false,\n",
                "    \"sqlite_in_memory\": true\n",
                "  }\n",
                "}\n",
            ),
        )
        .expect("write config");

        let config = load_config(temp_dir.join("config.json").to_str().expect("utf8 path"))
            .expect("load config");

        assert_eq!(
            config.engine.avalanche_fitness_additional_random_messages,
            3
        );
        assert!(config.engine.avalanche_fitness_streaming_prune);
        assert!(config.engine.avalanche_unique_r_cx_inputs);
        assert!(
            !config
                .engine
                .avalanche_include_max_fitness_candidates_in_order
        );
        assert!(config.engine.sqlite_in_memory);

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_load_config_accepts_avalanche_solver_settings() {
        let temp_dir = temp_path("avalanche_solver_settings");
        fs::create_dir_all(&temp_dir).expect("create temp config dir");
        fs::write(
            temp_dir.join("config.json"),
            concat!(
                "{\n",
                "  \"engine\": {\n",
                "    \"avalanche_solver_enable\": true,\n",
                "    \"avalanche_solver_global_log_enable\": false,\n",
                "    \"avalanche_solver_max_bits\": 5\n",
                "  }\n",
                "}\n",
            ),
        )
        .expect("write config");

        let config = load_config(temp_dir.join("config.json").to_str().expect("utf8 path"))
            .expect("load config");

        assert!(config.engine.avalanche_solver_enable);
        assert!(!config.engine.avalanche_solver_global_log_enable);
        assert_eq!(config.engine.avalanche_solver_max_bits, 5);

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_load_public_keyfile_rejects_private_fields() {
        let temp_dir = temp_path("keyfile_public_rejects_private");
        fs::create_dir_all(&temp_dir).expect("create temp config dir");
        let path = temp_dir.join("public.yaml");
        fs::write(
            &path,
            concat!(
                "format: rsa-public-key-v1\n",
                "algorithm: RSA\n",
                "public_exponent: \"17\"\n",
                "modulus: \"3233\"\n",
                "private_exponent: \"2753\"\n",
            ),
        )
        .expect("write malformed public keyfile");

        let error = load_rsa_key_material_from_yaml_path(&path)
            .expect_err("public keyfile with private fields should be rejected");
        assert!(
            error
                .to_string()
                .contains("must not include private_exponent")
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
