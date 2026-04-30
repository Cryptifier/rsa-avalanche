/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use bigdecimal::BigDecimal;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant, SystemTime};

use num_bigint::BigUint;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::logs::{LogError, LogWriter, write_session_start};
use crate::r_candidates::{
    RCandidate, RCandidateMode, RCandidateSettings, prepare_r_candidates_batch,
};
use crate::rng::RngChoice;

/// CLI metadata captured for analytics sessions.
#[derive(Debug, Serialize)]
pub struct AnalyticsCliArgs {
    /// Bit-length used for prime generation.
    pub bits: u32,
    /// Optional message override provided by the CLI.
    pub message_override: Option<String>,
    /// Public exponent selected for RSA.
    pub public_exponent: u64,
    /// Optional RNG seed for deterministic runs.
    pub seed: Option<u64>,
    /// Whether cryptographic RNGs were used.
    pub crypto_rng: bool,
    /// Config path used for this run.
    pub config_path: String,
    /// Whether analysis sufficiency tests were enabled.
    pub tests: bool,
    /// Whether export output was enabled.
    pub export: bool,
    /// Output path for the session JSON file.
    pub session_json: String,
    /// Whether ciphertext shifting is enabled.
    pub shift: bool,
    /// Whether ciphertext exponent modification is enabled.
    pub ciphertext_modify: bool,
    /// Whether avalanche candidates are sorted by Hamming distance.
    pub use_hamming_distance: bool,
    /// Whether bitwise-inverted avalanche candidates are mirrored into the grid.
    pub mirror_invert_candidates: bool,
    /// Minimum stored beam value interpreted as bit `1`.
    pub beam_bit_one_threshold: f64,
    /// Number of top avalanche beam-search candidates retained per run.
    pub avalanche_beam_top_k: usize,
    /// Exponent used to spread normalized avalanche beam probabilities.
    pub avalanche_probability_spread_exponent: f64,
    /// Number of avalanche combination samples evaluated per batch.
    pub avalanche_combination_samples: u64,
    /// Legacy sampled-width setting retained for compatibility with older configs.
    pub avalanche_combination_size: usize,
    /// Number of distinct r candidates mixed into each avalanche combination sample.
    pub avalanche_combination_mixed_r_candidates: usize,
    /// Number of top scored candidates retained for avalanche combination sampling.
    pub avalanche_combination_pool_size: usize,
    /// Number of Avalanche tiers to execute, including the initial sampled-input tier.
    pub avalanche_combination_recursion_depth: usize,
    /// Number of prior-tier samples grouped into each recursive Avalanche call.
    pub avalanche_combination_recursive_group_size: usize,
    /// Number of recursive samples produced per subsequent Avalanche tier; `0` preserves one-pass grouping.
    pub avalanche_combination_recursive_resample_count: usize,
    /// Whether sampled avalanche prunes the scored-input pool by Hamming-distance percentile before sampling.
    pub avalanche_combination_hamming_distance_prune: bool,
    /// Central percentile of Hamming distances retained when sampled-avalanche pruning is enabled.
    pub avalanche_combination_hamming_distance_keep_percentile: f64,
    /// Percentage of the retained inlier pool size to add back from Hamming-distance outlier tails.
    pub avalanche_combination_hamming_distance_outlier_preference_pct: f64,
    /// Whether sampled avalanche uses per-bit majority-vote probabilities from the combination outputs.
    pub avalanche_combination_majority_vote: bool,
    /// Whether sampled avalanche smooths per-bit majority-vote probabilities before beam search.
    pub avalanche_combination_sample_smoothing: bool,
    /// Whether sampled avalanche prints a separate majority-vote summary for the selected sample.
    pub avalanche_combination_majority_vote_print: bool,
    /// Whether recursive Avalanche tiers carry forward the top beam-search bits instead of majority-vote bits.
    pub avalanche_use_top_beam: bool,
    /// Whether all sampled-avalanche combinations were retained in memory during the run.
    pub avalanche_combination_keep_all_samples_in_memory: bool,
    /// Whether avalanche runs collected per-level and per-sample statistics.
    pub avalanche_statistics_collection: bool,
    /// Whether sampled avalanche used direct ChaCha20 input sampling instead of mixed-r combinations.
    pub avalanche_random_chacha20_inputs: bool,
    /// Whether sampled avalanche applies the trailing-zero fitness preprocessing pass.
    pub avalanche_fitness_scoring_pass: bool,
    /// Number of bytes used to shift plaintexts before candidate scoring.
    pub avalanche_fitness_shift_bytes: usize,
    /// Number of least-significant bits inspected by the trailing-zero fitness score.
    pub avalanche_fitness_bit_width: usize,
    /// Maximum number of retained r-candidate groups after fitness preprocessing.
    pub avalanche_fitness_r_candidate_limit: usize,
    /// Maximum number of retained `c^x` inputs per r-candidate group after fitness preprocessing.
    pub avalanche_fitness_cx_candidate_limit: usize,
    /// Expected bit width for decryptions.
    pub bits_decrypt: Option<u32>,
    /// Optional CLI override for speculative r-candidate target exponent.
    pub r_candidate_target_exponent: Option<BigDecimal>,
    /// Optional CLI override for speculative r-candidate target exponent minimum.
    pub r_candidate_target_exponent_minimum: Option<BigDecimal>,
}

#[derive(Debug, Serialize)]
pub(crate) struct AnalyticsCliInfo {
    bits: u32,
    message_override: Option<String>,
    public_exponent: u64,
    seed: Option<u64>,
    crypto_rng: bool,
    config_path: String,
    tests: bool,
    export: bool,
    session_json: String,
    shift: bool,
    ciphertext_modify: bool,
    use_hamming_distance: bool,
    mirror_invert_candidates: bool,
    beam_bit_one_threshold: f64,
    avalanche_beam_top_k: usize,
    avalanche_probability_spread_exponent: f64,
    avalanche_combination_samples: u64,
    avalanche_combination_size: usize,
    avalanche_combination_mixed_r_candidates: usize,
    avalanche_combination_pool_size: usize,
    avalanche_combination_recursion_depth: usize,
    avalanche_combination_recursive_group_size: usize,
    avalanche_combination_recursive_resample_count: usize,
    avalanche_combination_hamming_distance_prune: bool,
    avalanche_combination_hamming_distance_keep_percentile: f64,
    avalanche_combination_hamming_distance_outlier_preference_pct: f64,
    avalanche_combination_majority_vote: bool,
    avalanche_combination_sample_smoothing: bool,
    avalanche_combination_majority_vote_print: bool,
    avalanche_use_top_beam: bool,
    avalanche_combination_keep_all_samples_in_memory: bool,
    avalanche_statistics_collection: bool,
    avalanche_random_chacha20_inputs: bool,
    avalanche_fitness_scoring_pass: bool,
    avalanche_fitness_shift_bytes: usize,
    avalanche_fitness_bit_width: usize,
    avalanche_fitness_r_candidate_limit: usize,
    avalanche_fitness_cx_candidate_limit: usize,
    bits_decrypt: Option<u32>,
    r_candidate_target_exponent: Option<BigDecimal>,
    r_candidate_target_exponent_minimum: Option<BigDecimal>,
}

/// Timing entry for a named step.
#[derive(Debug, Serialize)]
pub struct StepTiming {
    name: String,
    duration_ms: u128,
}

/// Aggregate timing summary for repeated steps.
#[derive(Debug, Serialize)]
pub struct StepSummary {
    name: String,
    count: u64,
    total_ms: u128,
    mean_ms: f64,
}

/// Feature-level analytics including duration and structured stats.
#[derive(Debug, Serialize, Default)]
pub struct FeatureAnalytics {
    name: String,
    enabled: bool,
    duration_ms: Option<u128>,
    notes: Vec<String>,
    stats: Map<String, Value>,
}

/// Factor metadata for an r candidate.
#[derive(Debug, Serialize)]
pub struct RCandidateFactor {
    /// Prime factor value.
    pub prime: BigUint,
    /// Exponent for the prime factor.
    pub exponent: u64,
    /// Bit length of the prime factor.
    pub prime_bits: u64,
}

/// Serialized r candidate entry with factors.
#[derive(Debug, Serialize)]
pub struct RCandidateEntry {
    /// Candidate modulus value.
    pub r: BigUint,
    /// Bit length of the candidate modulus.
    pub r_bits: u64,
    /// Decimal target exponent used to retarget the candidate.
    pub target_exponent: BigDecimal,
    /// Prime factorization metadata.
    pub factors: Vec<RCandidateFactor>,
}

/// Step-by-step trace entry for a single r candidate.
#[derive(Debug, Serialize)]
pub struct RCandidateTraceEntry {
    /// Candidate modulus value.
    pub r: BigUint,
    /// Bit length of the candidate modulus.
    pub r_bits: u64,
    /// Decimal target exponent used to retarget the candidate.
    pub target_exponent: BigDecimal,
    /// Ciphertext after homomorphic base conversion into `r`.
    pub hbc_ciphertext_r: BigUint,
    /// Candidate-derived plaintext.
    pub candidate_decryption: BigUint,
}

/// Analytics payload for a batch of r candidates.
#[derive(Debug, Serialize)]
pub struct RCandidateBatchAnalytics {
    /// Label describing the candidate batch usage.
    pub context: String,
    /// Candidate generation mode.
    pub mode: String,
    /// Target count requested for the batch.
    pub target_count: usize,
    /// Actual count generated.
    pub generated_count: usize,
    /// Duration in milliseconds.
    pub duration_ms: u128,
    /// Reuse file path for candidates.
    pub reuse_path: String,
    /// Whether reuse loading is enabled.
    pub reuse_enabled: bool,
    /// Whether reuse append-only mode is enabled.
    pub reuse_append_only: bool,
    /// Minimum factor used for candidate screening.
    pub min_factor: BigUint,
    /// Process scale factor for candidate generation.
    pub process_scale: u32,
    /// Number of small primes per candidate.
    pub small_prime_factors: usize,
    /// Maximum factor count per candidate.
    pub max_factors: usize,
    /// Optional target bit length for candidates.
    pub target_bit_length: Option<u64>,
    /// Number of candidate entries produced for the batch.
    pub candidate_count: usize,
    /// Candidate entries for the batch.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<RCandidateEntry>,
}

/// Per-candidate accuracy entry for a shared message batch.
#[derive(Debug, Serialize)]
pub struct RCandidateAccuracyEntry {
    /// Candidate modulus value.
    pub r: BigUint,
    /// Bit length of the candidate modulus.
    pub r_bits: u64,
    /// Decimal target exponent used to retarget the candidate.
    pub target_exponent: BigDecimal,
    /// Prime factorization metadata.
    pub factors: Vec<RCandidateFactor>,
    /// Mean accuracy percentage across the message batch.
    pub accuracy_pct: f64,
    /// HBC ciphertexts in the candidate modulus (per message).
    pub hbc_ciphertexts_r: Vec<BigUint>,
    /// Candidate-derived plaintexts (per message).
    pub candidate_decryptions: Vec<BigUint>,
}

/// Accuracy batch payload for a shared message set.
#[derive(Debug, Serialize)]
pub struct RCandidateAccuracyBatch {
    /// Label describing the batch usage.
    pub context: String,
    /// Plaintext messages used in the batch.
    pub messages: Vec<BigUint>,
    /// Ciphertexts corresponding to the messages.
    pub ciphertexts: Vec<BigUint>,
    /// Shifted ciphertexts when shift is enabled.
    pub shifted_ciphertexts: Vec<BigUint>,
    /// Rabin exponent used for the batch transforms.
    pub rabin_exponent: u32,
    /// Historical bootstrap-source modulus field (currently the source modulus before HBC).
    pub tonelli_shanks_modulus: BigUint,
    /// Historical bootstrap-source ciphertext field (currently the shifted ciphertexts).
    pub tonelli_shanks_ciphertexts: Vec<BigUint>,
    /// Number of per-candidate accuracy entries evaluated for the batch.
    pub candidate_count: usize,
    /// Per-candidate accuracy entries.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<RCandidateAccuracyEntry>,
    /// Maximum bitwise match percentage among evaluated `c^x` candidates in the batch.
    pub cx_max_match_pct: Option<f64>,
    /// Ciphertext exponent `x` that achieved the batch max bitwise match.
    pub cx_max_x: Option<BigUint>,
    /// Total evaluated `c^x` candidates in the batch.
    pub cx_evaluated_candidates: usize,
    /// Total avalanche candidates evaluated for the batch.
    pub avalanche_evaluated_candidates: usize,
    /// Beam search match percentage for the batch (per-bit accuracy).
    pub beam_match_pct: Option<f64>,
    /// Beam search ones-match percentage for the batch.
    pub beam_ones_match_pct: Option<f64>,
    /// Majority-vote match percentage for the selected or best final-tier sample.
    pub majority_vote_match_pct: Option<f64>,
    /// Majority-vote ones-match percentage for the selected or best final-tier sample.
    pub majority_vote_ones_match_pct: Option<f64>,
    /// Beam search score for the top candidate.
    pub beam_score: Option<f64>,
    /// Bit width of the beam search candidate.
    pub beam_bit_width: Option<usize>,
    /// Index of the highest-scoring avalanche combination sample for the batch.
    pub avalanche_selected_sample_index: Option<usize>,
    /// Mean score percentage of the selected avalanche combination sample.
    pub avalanche_selected_sample_average_score_pct: Option<f64>,
    /// Total sampled avalanche candidates evaluated across all combination samples.
    pub avalanche_sampled_candidates_evaluated: usize,
    /// Number of avalanche combination samples retained for the batch.
    pub avalanche_combination_sample_count: usize,
    /// Per-tier sample-accuracy statistics for recursive Avalanche execution.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub avalanche_tier_statistics: Vec<AvalancheTierStatistics>,
    /// Detailed avalanche combination sample results for the batch.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub avalanche_combination_samples: Vec<AvalancheCombinationSample>,
}

/// Accuracy summary for one sample within an Avalanche tier.
#[derive(Debug, Serialize, Clone)]
pub struct AvalancheTierSampleStat {
    /// One-based sample index within the tier.
    pub sample_index: usize,
    /// Number of source items used to produce the sample.
    pub input_count: usize,
    /// Mean score percentage of the sample inputs.
    pub average_score_pct: f64,
    /// Beam-search match percentage for the top beam candidate when available.
    pub beam_match_pct: Option<f64>,
    /// Majority-vote match percentage for the sample when available.
    pub majority_vote_match_pct: Option<f64>,
    /// Best match percentage across the beam and majority-vote outputs for the sample.
    pub best_match_pct: f64,
}

/// Accuracy distribution captured for an Avalanche tier.
#[derive(Debug, Serialize, Clone)]
pub struct AvalancheTierStatistics {
    /// One-based tier index with `1` representing the initial sampled-input tier.
    pub tier_index: usize,
    /// Number of samples produced in the tier.
    pub sample_count: usize,
    /// Number of inputs grouped into each sample for this tier.
    pub group_size: usize,
    /// Human-readable description of the sample source for this tier.
    pub source_kind: String,
    /// Per-sample accuracy statistics across the full tier.
    pub sample_stats: Vec<AvalancheTierSampleStat>,
}

/// Source candidate included in an avalanche combination sample.
#[derive(Debug, Serialize, Clone)]
pub struct AvalancheCombinationSampleInput {
    /// Zero-based index of the scored batch candidate entry.
    pub batch_candidate_index: usize,
    /// Zero-based index of the message or ciphertext variant inside the batch candidate.
    pub message_index: usize,
    /// Candidate modulus value.
    pub r: BigUint,
    /// Bit length of the candidate modulus.
    pub r_bits: u64,
    /// Decimal target exponent used to retarget the candidate.
    pub target_exponent: BigDecimal,
    /// Ciphertext exponent applied to the source ciphertext.
    pub x: BigUint,
    /// Match percentage used to score the source candidate.
    pub score_match_pct: f64,
    /// HBC ciphertext in the candidate modulus.
    pub hbc_ciphertext_r: BigUint,
    /// Candidate-derived plaintext for this source input.
    pub candidate_decryption: BigUint,
}

/// Beam-search candidate produced from an avalanche combination sample.
#[derive(Debug, Serialize, Clone)]
pub struct AvalancheCombinationBeamResult {
    /// One-based beam rank in descending score order.
    pub rank: usize,
    /// Beam search score for the candidate.
    pub score: f64,
    /// Match percentage against the original message bits.
    pub match_pct: f64,
    /// Percentage of predicted `1` bits that match the message.
    pub ones_match_pct: f64,
    /// Hex encoding of the candidate bits.
    pub hex: String,
    /// Bit width of the candidate.
    pub bit_width: usize,
}

/// Serialized avalanche combination sample including beam-search output.
#[derive(Debug, Serialize, Clone)]
pub struct AvalancheCombinationSample {
    /// One-based sample index within the batch.
    pub sample_index: usize,
    /// Number of scored candidates available for sampling in the batch.
    pub pool_size: usize,
    /// Number of distinct r candidates available for sampling in the batch.
    pub r_candidate_pool_size: usize,
    /// Number of scored candidates selected for this sample.
    pub combination_size: usize,
    /// Number of distinct r candidates selected for this sample.
    pub mixed_r_candidate_count: usize,
    /// Mean match percentage of the sampled scored candidates.
    pub average_score_pct: f64,
    /// Whether this sample used per-bit majority-vote probabilities.
    pub majority_vote_enabled: bool,
    /// Whether this sample smoothed per-bit majority-vote probabilities before beam search.
    pub sample_smoothing_enabled: bool,
    /// Source scored candidates used to build the avalanche sample.
    pub inputs: Vec<AvalancheCombinationSampleInput>,
    /// Majority-vote bit values for the sample when enabled.
    pub majority_vote_bits: Vec<bool>,
    /// Per-bit count of `1` votes across the sampled combination.
    pub majority_vote_ones_count: Vec<usize>,
    /// Per-bit count of `0` votes across the sampled combination.
    pub majority_vote_zeros_count: Vec<usize>,
    /// Per-bit probability of `1` derived from the sampled combination, optionally smoothed.
    pub majority_vote_probability_one: Vec<f64>,
    /// Similarity percentages recorded at each avalanche reduction level.
    pub level_similarity_pct: Vec<f64>,
    /// Pair counts recorded at each avalanche reduction level.
    pub level_pair_counts: Vec<usize>,
    /// Normalized avalanche bias probabilities before spreading.
    pub normalized_bias_probabilities: Vec<f64>,
    /// Beam-search probabilities derived from the normalized avalanche biases.
    pub beam_search_probabilities: Vec<f64>,
    /// Final beam-search candidates produced from the avalanche result.
    pub beam_results: Vec<AvalancheCombinationBeamResult>,
}

/// Trace payload for r candidates evaluated against a specific message.
#[derive(Debug, Serialize)]
pub struct RCandidateTraceBatch {
    /// Label describing the batch usage.
    pub context: String,
    /// Plaintext message used in the trace.
    pub message: BigUint,
    /// Ciphertext corresponding to the message.
    pub ciphertext: BigUint,
    /// Shifted ciphertext when shift is enabled.
    pub shifted_ciphertext: BigUint,
    /// Rabin exponent used for the trace transforms.
    pub rabin_exponent: u32,
    /// Historical bootstrap-source modulus field (currently the source modulus before HBC).
    pub tonelli_shanks_modulus: BigUint,
    /// Historical bootstrap-source ciphertext field (currently the shifted ciphertext).
    pub tonelli_shanks_ciphertext: BigUint,
    /// Number of trace candidate entries evaluated for the batch.
    pub candidate_count: usize,
    /// Per-candidate trace entries.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<RCandidateTraceEntry>,
}

impl RCandidateBatchAnalytics {
    /// Removes per-candidate payloads before session-log persistence.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Clears candidate-detail storage while preserving summary counts.
    fn compact_for_session_log(&mut self) {
        self.candidates.clear();
    }
}

impl RCandidateAccuracyBatch {
    /// Removes per-candidate payloads before session-log persistence.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Clears candidate and sampled-avalanche detail storage while preserving summary counts.
    fn compact_for_session_log(&mut self) {
        self.candidates.clear();
        self.avalanche_combination_samples.clear();
    }
}

impl RCandidateTraceBatch {
    /// Removes per-candidate payloads before session-log persistence.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Clears trace-detail storage while preserving summary counts.
    fn compact_for_session_log(&mut self) {
        self.candidates.clear();
    }
}

/// Top-level analytics session payload.
#[derive(Debug, Serialize)]
pub struct SessionAnalytics {
    pub(crate) started_unix_ms: u128,
    pub(crate) finished_unix_ms: Option<u128>,
    pub(crate) cli: AnalyticsCliInfo,
    pub(crate) steps: Vec<StepTiming>,
    pub(crate) step_summaries: Vec<StepSummary>,
    pub(crate) features: Vec<FeatureAnalytics>,
    pub(crate) r_candidate_batches: Vec<RCandidateBatchAnalytics>,
    pub(crate) r_candidate_accuracy_batches: Vec<RCandidateAccuracyBatch>,
    pub(crate) r_candidate_traces: Vec<RCandidateTraceBatch>,
    pub(crate) errors: Vec<String>,
    #[serde(skip_serializing)]
    pub(crate) stream_writer: Option<LogWriter<BufWriter<File>>>,
    #[serde(skip_serializing)]
    pub(crate) stream_started: bool,
}

impl SessionAnalytics {
    /// Creates a new analytics session seeded from CLI arguments.
    ///
    /// # Parameters
    /// - `args`: CLI metadata to persist in the session.
    ///
    /// # Returns
    /// - `Result<SessionAnalytics, LogError>`: Initialized analytics container or an NDJSON stream error.
    ///
    /// # Expected Output
    /// - Creates or overwrites the configured session NDJSON file and writes `session_start`.
    pub fn new(args: AnalyticsCliArgs) -> Result<Self, LogError> {
        let started_unix_ms = now_unix_ms();
        let cli = AnalyticsCliInfo {
            bits: args.bits,
            message_override: args.message_override,
            public_exponent: args.public_exponent,
            seed: args.seed,
            crypto_rng: args.crypto_rng,
            config_path: args.config_path,
            tests: args.tests,
            export: args.export,
            session_json: args.session_json,
            shift: args.shift,
            ciphertext_modify: args.ciphertext_modify,
            use_hamming_distance: args.use_hamming_distance,
            mirror_invert_candidates: args.mirror_invert_candidates,
            beam_bit_one_threshold: args.beam_bit_one_threshold,
            avalanche_beam_top_k: args.avalanche_beam_top_k,
            avalanche_probability_spread_exponent: args.avalanche_probability_spread_exponent,
            avalanche_combination_samples: args.avalanche_combination_samples,
            avalanche_combination_size: args.avalanche_combination_size,
            avalanche_combination_mixed_r_candidates: args.avalanche_combination_mixed_r_candidates,
            avalanche_combination_pool_size: args.avalanche_combination_pool_size,
            avalanche_combination_recursion_depth: args.avalanche_combination_recursion_depth,
            avalanche_combination_recursive_group_size: args
                .avalanche_combination_recursive_group_size,
            avalanche_combination_recursive_resample_count: args
                .avalanche_combination_recursive_resample_count,
            avalanche_combination_hamming_distance_prune: args
                .avalanche_combination_hamming_distance_prune,
            avalanche_combination_hamming_distance_keep_percentile: args
                .avalanche_combination_hamming_distance_keep_percentile,
            avalanche_combination_hamming_distance_outlier_preference_pct: args
                .avalanche_combination_hamming_distance_outlier_preference_pct,
            avalanche_combination_majority_vote: args.avalanche_combination_majority_vote,
            avalanche_combination_sample_smoothing: args.avalanche_combination_sample_smoothing,
            avalanche_combination_majority_vote_print: args
                .avalanche_combination_majority_vote_print,
            avalanche_use_top_beam: args.avalanche_use_top_beam,
            avalanche_combination_keep_all_samples_in_memory: args
                .avalanche_combination_keep_all_samples_in_memory,
            avalanche_statistics_collection: args.avalanche_statistics_collection,
            avalanche_random_chacha20_inputs: args.avalanche_random_chacha20_inputs,
            avalanche_fitness_scoring_pass: args.avalanche_fitness_scoring_pass,
            avalanche_fitness_shift_bytes: args.avalanche_fitness_shift_bytes,
            avalanche_fitness_bit_width: args.avalanche_fitness_bit_width,
            avalanche_fitness_r_candidate_limit: args.avalanche_fitness_r_candidate_limit,
            avalanche_fitness_cx_candidate_limit: args.avalanche_fitness_cx_candidate_limit,
            bits_decrypt: args.bits_decrypt,
            r_candidate_target_exponent: args.r_candidate_target_exponent,
            r_candidate_target_exponent_minimum: args.r_candidate_target_exponent_minimum,
        };
        let mut stream_writer = LogWriter::create(&cli.session_json)?;
        write_session_start(&mut stream_writer, started_unix_ms, &cli)?;

        Ok(Self {
            started_unix_ms,
            finished_unix_ms: None,
            cli,
            steps: Vec::new(),
            step_summaries: Vec::new(),
            features: Vec::new(),
            r_candidate_batches: Vec::new(),
            r_candidate_accuracy_batches: Vec::new(),
            r_candidate_traces: Vec::new(),
            errors: Vec::new(),
            stream_writer: Some(stream_writer),
            stream_started: true,
        })
    }

    /// Returns the configured output path for the session JSON.
    ///
    /// # Parameters
    /// - None.
    ///
    /// # Returns
    /// - `&str`: Output path for the session JSON.
    ///
    /// # Expected Output
    /// - Returns the stored path; no side effects.
    pub fn session_json_path(&self) -> &str {
        &self.cli.session_json
    }

    /// Finalizes the analytics session and records any terminal error.
    ///
    /// # Parameters
    /// - `error`: Optional error string for a failed run.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Updates timestamps and error list; no stdout/stderr output.
    pub fn finish(&mut self, error: Option<String>) {
        self.finished_unix_ms = Some(now_unix_ms());
        if let Some(err) = error {
            self.errors.push(err);
        }
    }

    /// Records a completed step duration.
    ///
    /// # Parameters
    /// - `name`: Human-readable step name.
    /// - `duration`: Duration for the step.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Appends a step timing entry; no stdout/stderr output.
    pub fn record_step(&mut self, name: &str, duration: Duration) {
        let step = StepTiming {
            name: name.to_string(),
            duration_ms: duration.as_millis(),
        };
        if !self.try_stream_event("step", &step) {
            self.steps.push(step);
        }
    }

    /// Records an aggregate step summary for repeated operations.
    ///
    /// # Parameters
    /// - `name`: Human-readable step name.
    /// - `count`: Number of repetitions.
    /// - `total`: Total duration for all repetitions.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Appends a step summary entry; no stdout/stderr output.
    pub fn record_step_summary(&mut self, name: &str, count: u64, total: Duration) {
        if count == 0 {
            return;
        }
        let total_ms = total.as_millis();
        let mean_ms = total_ms as f64 / count as f64;
        let summary = StepSummary {
            name: name.to_string(),
            count,
            total_ms,
            mean_ms,
        };
        if !self.try_stream_event("step_summary", &summary) {
            self.step_summaries.push(summary);
        }
    }

    /// Marks a feature as enabled or disabled for this session.
    ///
    /// # Parameters
    /// - `name`: Feature name to update.
    /// - `enabled`: Whether the feature is enabled.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Updates the feature record; no stdout/stderr output.
    pub fn mark_feature(&mut self, name: &str, enabled: bool) {
        let feature = self.feature_mut(name);
        feature.enabled = enabled;
    }

    /// Records a feature duration.
    ///
    /// # Parameters
    /// - `name`: Feature name to update.
    /// - `duration`: Duration for the feature execution.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Updates the feature duration entry; no stdout/stderr output.
    pub fn record_feature_duration(&mut self, name: &str, duration: Duration) {
        let feature = self.feature_mut(name);
        feature.duration_ms = Some(duration.as_millis());
    }

    /// Appends a note to a feature record.
    ///
    /// # Parameters
    /// - `name`: Feature name to annotate.
    /// - `note`: Note text to append.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Adds the note to the feature entry; no stdout/stderr output.
    pub fn add_feature_note(&mut self, name: &str, note: &str) {
        let feature = self.feature_mut(name);
        feature.notes.push(note.to_string());
    }

    /// Sets a structured statistic on a feature record.
    ///
    /// # Parameters
    /// - `name`: Feature name to update.
    /// - `key`: Statistic key.
    /// - `value`: Statistic value as a JSON value.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Updates the feature statistics map; no stdout/stderr output.
    pub fn set_feature_stat(&mut self, name: &str, key: &str, value: Value) {
        let feature = self.feature_mut(name);
        feature.stats.insert(key.to_string(), value);
    }

    /// Stores r candidate batch analytics for the session.
    ///
    /// # Parameters
    /// - `batch`: Candidate batch analytics record.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Appends the batch entry; no stdout/stderr output.
    pub fn push_r_candidate_batch(&mut self, batch: RCandidateBatchAnalytics) {
        let mut batch = batch;
        batch.compact_for_session_log();
        if !self.try_stream_event("r_candidate_batch", &batch) {
            self.r_candidate_batches.push(batch);
        }
    }

    /// Stores r candidate accuracy batch data for the session.
    ///
    /// # Parameters
    /// - `batch`: Candidate accuracy batch record.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Appends the accuracy batch entry; no stdout/stderr output.
    pub fn push_r_candidate_accuracy_batch(&mut self, batch: RCandidateAccuracyBatch) {
        let mut batch = batch;
        batch.compact_for_session_log();
        if !self.try_stream_event("r_candidate_accuracy_batch", &batch) {
            self.r_candidate_accuracy_batches.push(batch);
        }
    }

    /// Stores r candidate trace data for a specific message context.
    ///
    /// # Parameters
    /// - `batch`: Candidate trace batch to record.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Appends the trace entry; no stdout/stderr output.
    pub fn push_r_candidate_trace_batch(&mut self, batch: RCandidateTraceBatch) {
        let mut batch = batch;
        batch.compact_for_session_log();
        if !self.try_stream_event("r_candidate_trace_batch", &batch) {
            self.r_candidate_traces.push(batch);
        }
    }

    fn feature_mut(&mut self, name: &str) -> &mut FeatureAnalytics {
        if let Some(idx) = self.features.iter().position(|f| f.name == name) {
            return &mut self.features[idx];
        }
        self.features.push(FeatureAnalytics {
            name: name.to_string(),
            ..FeatureAnalytics::default()
        });
        self.features.last_mut().expect("feature entry missing")
    }

    fn try_stream_event<T: Serialize>(&mut self, event: &str, payload: &T) -> bool {
        let mut stream_error = None;
        let streamed = if let Some(writer) = self.stream_writer.as_mut() {
            match writer.write_event(event, payload) {
                Ok(()) => true,
                Err(err) => {
                    stream_error = Some(err);
                    false
                }
            }
        } else {
            false
        };
        if let Some(err) = stream_error {
            self.stream_writer = None;
            self.errors.push(format!(
                "analytics stream disabled after {event} write failure: {err}"
            ));
        }
        streamed
    }
}

/// Generates r candidates and records analytics for the batch.
///
/// # Parameters
/// - `context`: Human-readable label describing the usage of this candidate batch.
/// - `n`: RSA modulus used to bound/scale candidates.
/// - `settings`: Candidate generation configuration.
/// - `rng`: Random number generator for candidate sampling.
/// - `batch_size`: Target number of candidates to produce.
/// - `analytics`: Session analytics accumulator for r candidate metadata.
///
/// # Returns
/// - `Vec<RCandidate>`: Candidate list with revised modulus metadata.
///
/// # Expected Output
/// - Records candidate metadata in `analytics`; no stdout/stderr output.
pub fn generate_r_candidates_with_analytics(
    context: &str,
    n: &BigUint,
    settings: &RCandidateSettings,
    rng: &mut RngChoice,
    batch_size: usize,
    analytics: &Arc<Mutex<SessionAnalytics>>,
) -> Vec<RCandidate> {
    let start = std::time::Instant::now();
    let candidates = match prepare_r_candidates_batch(n, settings, rng, batch_size) {
        Ok(candidates) => candidates,
        Err(err) => {
            println!("Failed to prepare r candidates: {}", err);
            Vec::new()
        }
    };
    let mode = if settings.reuse_retargeted_r_candidates {
        "retargeted_reuse".to_string()
    } else {
        match settings.mode {
            RCandidateMode::Factoring => "factoring".to_string(),
            RCandidateMode::SmallPrimes => "small_primes".to_string(),
        }
    };
    let reuse_path = if settings.reuse_retargeted_r_candidates {
        settings.reuse_retargeted_r_candidates_path.clone()
    } else {
        settings.reuse_r_candidates_path.clone()
    };
    let duration = start.elapsed();

    let candidate_entry_total = u64::try_from(candidates.len()).unwrap_or(u64::MAX);
    let candidate_entry_started_at = Instant::now();
    let candidate_entry_done = AtomicU64::new(0);
    let candidate_entry_next_log_at_ms =
        AtomicU64::new(Duration::from_secs(5).as_millis().min(u128::from(u64::MAX)) as u64);
    println!(
        "Preparing analytics metadata for {} r candidates",
        candidates.len()
    );
    let candidate_entries = candidates
        .par_iter()
        .map(|candidate| RCandidateEntry {
            r: candidate.r.clone(),
            r_bits: candidate.r.bits(),
            target_exponent: candidate.target_exponent.normalized(),
            factors: candidate
                .factors
                .iter()
                .map(|(p, e)| RCandidateFactor {
                    prime: p.clone(),
                    exponent: *e,
                    prime_bits: p.bits(),
                })
                .collect(),
        })
        .map(|entry| {
            let done = candidate_entry_done.fetch_add(1, Ordering::Relaxed) + 1;
            let elapsed_ms = candidate_entry_started_at
                .elapsed()
                .as_millis()
                .min(u128::from(u64::MAX)) as u64;
            let interval_ms = Duration::from_secs(5)
                .as_millis()
                .min(u128::from(u64::MAX)) as u64;
            loop {
                let scheduled_ms = candidate_entry_next_log_at_ms.load(Ordering::Relaxed);
                if done != candidate_entry_total && elapsed_ms < scheduled_ms {
                    break;
                }

                let next_deadline_ms = if done == candidate_entry_total {
                    u64::MAX
                } else {
                    scheduled_ms.saturating_add(interval_ms)
                };
                if candidate_entry_next_log_at_ms
                    .compare_exchange(
                        scheduled_ms,
                        next_deadline_ms,
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    let percent = (done as f64 / candidate_entry_total as f64) * 100.0;
                    println!(
                        "Analytics metadata progress: {:.5}% ({}/{})",
                        percent, done, candidate_entry_total
                    );
                    break;
                }
            }
            entry
        })
        .collect::<Vec<_>>();

    if let Ok(mut guard) = analytics.lock() {
        guard.push_r_candidate_batch(RCandidateBatchAnalytics {
            context: context.to_string(),
            mode,
            target_count: batch_size.max(1),
            generated_count: candidates.len(),
            duration_ms: duration.as_millis(),
            reuse_path,
            reuse_enabled: settings.reuse_r_candidates || settings.reuse_retargeted_r_candidates,
            reuse_append_only: settings.reuse_r_candidates_append_only,
            min_factor: settings.process_min_factor.clone(),
            process_scale: settings.process_scale,
            small_prime_factors: settings.small_prime_factors_per_candidate,
            max_factors: settings.max_factors_per_candidate,
            target_bit_length: settings.target_bit_length,
            candidate_count: candidate_entries.len(),
            candidates: candidate_entries,
        });
    }

    candidates
}

/// Writes the analytics session JSON to disk.
///
/// # Parameters
/// - `path`: Output path for the JSON file.
/// - `session`: Analytics session to serialize.
///
/// # Returns
/// - `Result<(), Box<dyn std::error::Error>>`: `Ok(())` on success or an error on failure.
///
/// # Expected Output
/// - Writes a JSON file to `path`.
/// Gets the current UNIX time in milliseconds.
///
/// # Parameters
/// - None.
///
/// # Returns
/// - `u128`: Milliseconds since UNIX epoch.
///
/// # Expected Output
/// - Returns a timestamp; no side effects.
fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}
