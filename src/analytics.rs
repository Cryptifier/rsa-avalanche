use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use num_bigint::BigUint;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::r_candidates::{generate_r_candidates_batch, RCandidateMode, RCandidateSettings};
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
    /// Expected bit width for decryptions.
    pub bits_decrypt: Option<u32>,
}

#[derive(Debug, Serialize)]
struct AnalyticsCliInfo {
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
    bits_decrypt: Option<u32>,
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
    pub prime: String,
    /// Exponent for the prime factor.
    pub exponent: u64,
    /// Bit length of the prime factor.
    pub prime_bits: u64,
}

/// Serialized r candidate entry with factors.
#[derive(Debug, Serialize)]
pub struct RCandidateEntry {
    /// Candidate modulus value.
    pub r: String,
    /// Bit length of the candidate modulus.
    pub r_bits: u64,
    /// Prime factorization metadata.
    pub factors: Vec<RCandidateFactor>,
}

/// Step-by-step trace entry for a single r candidate.
#[derive(Debug, Serialize)]
pub struct RCandidateTraceEntry {
    /// Candidate modulus value.
    pub r: String,
    /// Bit length of the candidate modulus.
    pub r_bits: u64,
    /// Ciphertext after homomorphic base conversion into `r`.
    pub hbc_ciphertext_r: String,
    /// Candidate-derived plaintext.
    pub candidate_decryption: String,
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
    pub min_factor: String,
    /// Process scale factor for candidate generation.
    pub process_scale: u32,
    /// Number of small primes per candidate.
    pub small_prime_factors: usize,
    /// Maximum factor count per candidate.
    pub max_factors: usize,
    /// Optional target bit length for candidates.
    pub target_bit_length: Option<u64>,
    /// Candidate entries for the batch.
    pub candidates: Vec<RCandidateEntry>,
}

/// Per-candidate accuracy entry for a shared message batch.
#[derive(Debug, Serialize)]
pub struct RCandidateAccuracyEntry {
    /// Candidate modulus value.
    pub r: String,
    /// Bit length of the candidate modulus.
    pub r_bits: u64,
    /// Prime factorization metadata.
    pub factors: Vec<RCandidateFactor>,
    /// Mean accuracy percentage across the message batch.
    pub accuracy_pct: f64,
    /// HBC ciphertexts in the candidate modulus (per message).
    pub hbc_ciphertexts_r: Vec<String>,
    /// Candidate-derived plaintexts (per message).
    pub candidate_decryptions: Vec<String>,
}

/// Accuracy batch payload for a shared message set.
#[derive(Debug, Serialize)]
pub struct RCandidateAccuracyBatch {
    /// Label describing the batch usage.
    pub context: String,
    /// Plaintext messages used in the batch.
    pub messages: Vec<String>,
    /// Ciphertexts corresponding to the messages.
    pub ciphertexts: Vec<String>,
    /// Shifted ciphertexts when shift is enabled.
    pub shifted_ciphertexts: Vec<String>,
    /// Rabin exponent used for the batch transforms.
    pub rabin_exponent: u32,
    /// Tonelli-Shanks modulus value (`n^k`).
    pub tonelli_shanks_modulus: String,
    /// Tonelli-Shanks ciphertexts (per message).
    pub tonelli_shanks_ciphertexts: Vec<String>,
    /// Per-candidate accuracy entries.
    pub candidates: Vec<RCandidateAccuracyEntry>,
    /// Beam search match percentage for the batch (per-bit accuracy).
    pub beam_match_pct: Option<f64>,
    /// Beam search ones-match percentage for the batch.
    pub beam_ones_match_pct: Option<f64>,
    /// Beam search score for the top candidate.
    pub beam_score: Option<f64>,
    /// Bit width of the beam search candidate.
    pub beam_bit_width: Option<usize>,
}

/// Trace payload for r candidates evaluated against a specific message.
#[derive(Debug, Serialize)]
pub struct RCandidateTraceBatch {
    /// Label describing the batch usage.
    pub context: String,
    /// Plaintext message used in the trace.
    pub message: String,
    /// Ciphertext corresponding to the message.
    pub ciphertext: String,
    /// Shifted ciphertext when shift is enabled.
    pub shifted_ciphertext: String,
    /// Rabin exponent used for the trace transforms.
    pub rabin_exponent: u32,
    /// Tonelli-Shanks modulus value (`n^k`).
    pub tonelli_shanks_modulus: String,
    /// Tonelli-Shanks ciphertext.
    pub tonelli_shanks_ciphertext: String,
    /// Per-candidate trace entries.
    pub candidates: Vec<RCandidateTraceEntry>,
}

/// Top-level analytics session payload.
#[derive(Debug, Serialize)]
pub struct SessionAnalytics {
    started_unix_ms: u128,
    finished_unix_ms: Option<u128>,
    cli: AnalyticsCliInfo,
    steps: Vec<StepTiming>,
    step_summaries: Vec<StepSummary>,
    features: Vec<FeatureAnalytics>,
    r_candidate_batches: Vec<RCandidateBatchAnalytics>,
    r_candidate_accuracy_batches: Vec<RCandidateAccuracyBatch>,
    r_candidate_traces: Vec<RCandidateTraceBatch>,
    errors: Vec<String>,
}

impl SessionAnalytics {
    /// Creates a new analytics session seeded from CLI arguments.
    ///
    /// # Parameters
    /// - `args`: CLI metadata to persist in the session.
    ///
    /// # Returns
    /// - `SessionAnalytics`: Initialized analytics container.
    ///
    /// # Expected Output
    /// - Returns a new session with timestamps and CLI metadata; no side effects.
    pub fn new(args: AnalyticsCliArgs) -> Self {
        Self {
            started_unix_ms: now_unix_ms(),
            finished_unix_ms: None,
            cli: AnalyticsCliInfo {
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
                bits_decrypt: args.bits_decrypt,
            },
            steps: Vec::new(),
            step_summaries: Vec::new(),
            features: Vec::new(),
            r_candidate_batches: Vec::new(),
            r_candidate_accuracy_batches: Vec::new(),
            r_candidate_traces: Vec::new(),
            errors: Vec::new(),
        }
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
        self.steps.push(StepTiming {
            name: name.to_string(),
            duration_ms: duration.as_millis(),
        });
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
        self.step_summaries.push(StepSummary {
            name: name.to_string(),
            count,
            total_ms,
            mean_ms,
        });
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
        self.r_candidate_batches.push(batch);
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
        self.r_candidate_accuracy_batches.push(batch);
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
        self.r_candidate_traces.push(batch);
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
/// - `Vec<(BigUint, Vec<(BigUint, u64)>)>`: Candidate list and factor tuples.
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
) -> Vec<(BigUint, Vec<(BigUint, u64)>)> {
    let start = std::time::Instant::now();
    let candidates = generate_r_candidates_batch(n, settings, rng, batch_size);
    let duration = start.elapsed();

    let candidate_entries = candidates
        .iter()
        .map(|(r, factors)| RCandidateEntry {
            r: r.to_string(),
            r_bits: r.bits(),
            factors: factors
                .iter()
                .map(|(p, e)| RCandidateFactor {
                    prime: p.to_string(),
                    exponent: *e,
                    prime_bits: p.bits(),
                })
                .collect(),
        })
        .collect::<Vec<_>>();

    let mode = match settings.mode {
        RCandidateMode::Factoring => "factoring",
        RCandidateMode::SmallPrimes => "small_primes",
    };

    if let Ok(mut guard) = analytics.lock() {
        guard.push_r_candidate_batch(RCandidateBatchAnalytics {
            context: context.to_string(),
            mode: mode.to_string(),
            target_count: batch_size.max(1),
            generated_count: candidates.len(),
            duration_ms: duration.as_millis(),
            reuse_path: settings.reuse_r_candidates_path.clone(),
            reuse_enabled: settings.reuse_r_candidates,
            reuse_append_only: settings.reuse_r_candidates_append_only,
            min_factor: settings.process_min_factor.to_string(),
            process_scale: settings.process_scale,
            small_prime_factors: settings.small_prime_factors_per_candidate,
            max_factors: settings.max_factors_per_candidate,
            target_bit_length: settings.target_bit_length,
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
pub fn write_session_json(
    path: &str,
    session: &SessionAnalytics,
) -> Result<(), Box<dyn std::error::Error>> {
    let serialized = serde_json::to_string_pretty(session)?;
    std::fs::write(path, serialized)?;
    Ok(())
}

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
