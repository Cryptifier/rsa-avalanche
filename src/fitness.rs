/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use num_bigint::BigUint;
use num_traits::Zero;
use rayon::prelude::*;
use std::{
    cmp::Ordering as CmpOrdering,
    collections::{HashMap, HashSet},
    error::Error,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use crate::config::EngineConfig;
use crate::database::{
    AvalancheCacheGuard, CachedKeysetPage, count_cached_scored_inputs,
    load_cached_scored_input_pages_with_progress, load_cached_scored_input_rows_after_id_page,
    load_cached_scored_input_rows_by_ids,
};
use crate::helpers::{PackedBits, format_beam_float, hamming_distance_packed_bytes};
use crate::methods::{
    BEAM_PCT_DECIMALS, ScoredAvalancheInput, ScoredAvalancheInputGroup, biguint_to_bits_le,
    log_parallel_progress_every_interval, parallel_progress_chunk_size, sample_unique_indices,
};
use crate::rng::RngChoice;

/// Lightweight cached scored-input metadata retained during fitness preprocessing.
#[derive(Debug, Clone)]
pub(crate) struct CachedScoredInputSummary {
    pub(crate) id: i64,
    pub(crate) batch_candidate_index: usize,
    pub(crate) message_index: usize,
    pub(crate) score_match_pct: f64,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) contents_have_been_inverted: bool,
    pub(crate) r: BigUint,
    pub(crate) x: BigUint,
    pub(crate) fitness: AvalancheFitnessScore,
}

/// Stored fitness metadata used to rank retained Avalanche inputs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct AvalancheFitnessScore {
    pub(crate) fitness_score: usize,
    pub(crate) fitness_total_score: usize,
    pub(crate) fitness_message_count: usize,
}

/// Ranked scored-input entry retained by the streaming fitness-pruning path.
#[derive(Debug, Clone)]
pub(crate) struct RankedScoredAvalancheInput {
    pub(crate) input: ScoredAvalancheInput,
    pub(crate) fitness: AvalancheFitnessScore,
}

/// Incremental bounded pool used to prune scored Avalanche inputs while batches are processed.
#[derive(Debug)]
pub(crate) struct StreamingScoredAvalancheFitnessPool {
    fitness_bit_width: usize,
    retained_input_limit: usize,
    use_fitness_threshold: bool,
    fitness_threshold: f64,
    total_inputs: usize,
    total_groups: HashSet<usize>,
    threshold_retained_input_count: usize,
    threshold_retained_groups: HashSet<usize>,
    ranked_inputs: Vec<RankedScoredAvalancheInput>,
}

/// Cached row-id groups keyed by originating `r` candidate.
#[derive(Debug, Clone)]
pub(crate) struct CachedScoredInputGroup {
    pub(crate) input_ids: Vec<i64>,
}

/// Resolves the bit width used by avalanche inputs and reductions.
///
/// # Parameters
/// - `engine`: Engine configuration containing the configured message width.
///
/// # Returns
/// - `usize`: Avalanche bit width derived from `engine.message.bits`.
///
/// # Expected Output
/// - Returns at least `1`; no stdout/stderr output.
pub(crate) fn resolve_avalanche_bit_width(engine: &EngineConfig) -> usize {
    resolve_plaintext_message_bit_width(engine)
        .saturating_add(resolve_avalanche_fitness_shift_bits(engine))
        .max(1)
}

/// Resolves the configured plaintext bit width before any fitness shifting is applied.
///
/// # Parameters
/// - `engine`: Engine configuration containing the configured message width.
///
/// # Returns
/// - `usize`: Plaintext bit width derived directly from `engine.message.bits`.
///
/// # Expected Output
/// - Returns at least `1`; no stdout/stderr output.
pub(crate) fn resolve_plaintext_message_bit_width(engine: &EngineConfig) -> usize {
    engine.message.bits.max(1) as usize
}

/// Resolves the number of LSB fitness bits created by the power-of-two plaintext shift.
///
/// # Parameters
/// - `engine`: Engine configuration containing the configured byte shift.
///
/// # Returns
/// - `usize`: Number of zero-expected LSBs added to the plaintext.
///
/// # Expected Output
/// - Returns a non-negative bit count; no stdout/stderr output.
pub(crate) fn resolve_avalanche_fitness_shift_bits(engine: &EngineConfig) -> usize {
    engine.avalanche_fitness_shift_bytes.saturating_mul(8)
}

/// Resolves the zero-count fitness window width capped to the effective avalanche width.
///
/// # Parameters
/// - `engine`: Engine configuration containing the configured fitness window size.
///
/// # Returns
/// - `usize`: Number of LSBs inspected by the fitness score.
///
/// # Expected Output
/// - Returns at least `1`; no stdout/stderr output.
pub(crate) fn resolve_avalanche_fitness_bit_width(engine: &EngineConfig) -> usize {
    engine
        .avalanche_fitness_bit_width
        .max(1)
        .min(resolve_avalanche_bit_width(engine).max(1))
}

/// Builds the anonymous plaintext transform used to create the fitness scoring slice.
///
/// # Parameters
/// - `engine`: Engine configuration containing the configured byte shift.
///
/// # Returns
/// - `Arc<dyn Fn(&BigUint) -> BigUint + Send + Sync>`: Closure that left-shifts plaintexts by the configured power of two.
///
/// # Expected Output
/// - Returns a reusable in-memory closure; no stdout/stderr output.
pub(crate) fn build_candidate_message_transform(
    engine: &EngineConfig,
) -> Arc<dyn Fn(&BigUint) -> BigUint + Send + Sync> {
    let shift_bits = resolve_avalanche_fitness_shift_bits(engine);
    Arc::new(move |message: &BigUint| {
        if shift_bits == 0 {
            message.clone()
        } else {
            message << shift_bits
        }
    })
}

/// Applies the configured plaintext fitness transform and validates the shifted message against `n`.
///
/// # Parameters
/// - `engine`: Engine configuration containing the configured fitness shift.
/// - `message`: Plaintext message before shifting.
/// - `modulus`: RSA modulus that must still contain the shifted message intact.
/// - `context`: Human-readable error label describing the caller.
///
/// # Returns
/// - `Result<BigUint, Box<dyn Error>>`: Shifted plaintext ready for candidate scoring.
///
/// # Expected Output
/// - Returns the shifted plaintext or an error when the transformed message would wrap modulo `n`.
pub(crate) fn transform_message_for_candidate_scoring(
    engine: &EngineConfig,
    message: &BigUint,
    modulus: &BigUint,
    context: &str,
) -> Result<BigUint, Box<dyn Error>> {
    let transform = build_candidate_message_transform(engine);
    let transformed = transform(message);
    if !modulus.is_zero() && transformed >= *modulus {
        return Err(format!(
            "{} shifted message exceeds modulus: shifted={} modulus={} shift_bytes={}",
            context, transformed, modulus, engine.avalanche_fitness_shift_bytes
        )
        .into());
    }
    Ok(transformed)
}

/// Counts zero bits within the least-significant fitness window up to a fixed width.
///
/// # Parameters
/// - `bits`: Packed candidate bits scored from the least-significant side.
/// - `width`: Maximum number of LSBs to inspect.
///
/// # Returns
/// - `usize`: Zero-bit count within the inspected LSB window.
///
/// # Expected Output
/// - Returns the computed fitness value; no stdout/stderr output.
pub(crate) fn lsb_zero_count_fitness(bits: &PackedBits, width: usize) -> usize {
    let capped_width = width.min(bits.len());
    (0..capped_width)
        .filter(|bit_index| !bits.bit(*bit_index))
        .count()
}

/// Converts the integer zero-count fitness score into a normalized `[0, 1]` ratio.
///
/// # Parameters
/// - `fitness_score`: Raw zero-count fitness retained for one candidate.
/// - `fitness_bit_width`: Number of least-significant bits considered by the fitness pass.
///
/// # Returns
/// - `f64`: Normalized fitness score relative to the configured fitness window width.
///
/// # Expected Output
/// - Returns the normalized fitness value; no stdout/stderr output.
pub(crate) fn normalize_avalanche_fitness_score(
    fitness_score: usize,
    fitness_bit_width: usize,
) -> f64 {
    if fitness_bit_width == 0 {
        return 0.0;
    }

    fitness_score as f64 / fitness_bit_width as f64
}

/// Converts the cumulative padding-bit fitness score into an average normalized `[0, 1]` ratio.
///
/// # Parameters
/// - `fitness_total_score`: Sum of padding-bit fitness scores across all evaluated messages.
/// - `fitness_bit_width`: Number of least-significant bits considered by the fitness pass.
/// - `fitness_message_count`: Number of messages evaluated for the fitness score.
///
/// # Returns
/// - `f64`: Average normalized padding-bit fitness across all evaluated messages.
///
/// # Expected Output
/// - Returns the normalized mean fitness value; no stdout/stderr output.
pub(crate) fn normalize_avalanche_fitness_mean_score(
    fitness_total_score: usize,
    fitness_bit_width: usize,
    fitness_message_count: usize,
) -> f64 {
    if fitness_bit_width == 0 || fitness_message_count == 0 {
        return 0.0;
    }

    fitness_total_score as f64 / (fitness_bit_width * fitness_message_count) as f64
}

/// Builds the default one-message padding-bit fitness score for a scored Avalanche input.
///
/// # Parameters
/// - `message_bits`: Packed candidate bits for the scored input.
/// - `fitness_bit_width`: Number of least-significant bits considered by the fitness pass.
///
/// # Returns
/// - `AvalancheFitnessScore`: Single-message padding-bit fitness metrics.
///
/// # Expected Output
/// - Returns the single-message score with no stdout/stderr output.
pub(crate) fn single_message_avalanche_fitness_score(
    message_bits: &PackedBits,
    fitness_bit_width: usize,
) -> AvalancheFitnessScore {
    let fitness_score = lsb_zero_count_fitness(message_bits, fitness_bit_width);
    AvalancheFitnessScore {
        fitness_score,
        fitness_total_score: fitness_score,
        fitness_message_count: 1,
    }
}

/// Resolves the retained-input cap for the global Avalanche fitness pool.
///
/// # Parameters
/// - `r_candidate_limit`: Configured primary retention dimension for the fitness pass.
/// - `cx_candidate_limit`: Configured secondary retention dimension for the fitness pass.
///
/// # Returns
/// - `usize`: Maximum number of globally ranked inputs to retain, or `0` for no cap.
///
/// # Expected Output
/// - Returns the derived pool cap with no side effects.
pub(crate) fn resolve_avalanche_fitness_retained_input_limit(
    r_candidate_limit: usize,
    cx_candidate_limit: usize,
) -> usize {
    match (r_candidate_limit, cx_candidate_limit) {
        (0, 0) => 0,
        (0, cx_limit) => cx_limit,
        (r_limit, 0) => r_limit,
        (r_limit, cx_limit) => r_limit.saturating_mul(cx_limit),
    }
}

/// Retains and sorts the highest-ranked prefix of a candidate pool.
///
/// # Parameters
/// - `inputs`: Candidate pool to rank in place.
/// - `retained_input_limit`: Maximum number of top-ranked items to keep, or `0` for no cap.
/// - `compare`: Comparator that orders better candidates before worse candidates.
///
/// # Returns
/// - `()`: This function returns nothing.
///
/// # Expected Output
/// - Reorders `inputs` in place, keeping only the best-ranked prefix when a cap is configured.
fn retain_best_ranked_inputs<T, F>(inputs: &mut Vec<T>, retained_input_limit: usize, compare: F)
where
    T: Send,
    F: Fn(&T, &T) -> CmpOrdering + Sync + Send,
{
    if inputs.is_empty() {
        return;
    }

    if retained_input_limit > 0 && inputs.len() > retained_input_limit {
        let last_retained_index = retained_input_limit.saturating_sub(1);
        inputs.select_nth_unstable_by(last_retained_index, |left, right| compare(left, right));
        inputs.truncate(retained_input_limit);
    }

    inputs.par_sort_unstable_by(|left, right| compare(left, right));
}

/// Orders two ranked scored Avalanche inputs from best to worst.
///
/// # Parameters
/// - `left`: Left ranked input participating in the comparison.
/// - `right`: Right ranked input participating in the comparison.
///
/// # Returns
/// - `CmpOrdering`: Ordering that places better candidates before worse candidates.
///
/// # Expected Output
/// - Returns a deterministic ordering with no side effects.
fn compare_ranked_scored_avalanche_inputs(
    left: &RankedScoredAvalancheInput,
    right: &RankedScoredAvalancheInput,
) -> CmpOrdering {
    right
        .fitness
        .fitness_score
        .cmp(&left.fitness.fitness_score)
        .then_with(|| {
            right
                .fitness
                .fitness_total_score
                .cmp(&left.fitness.fitness_total_score)
        })
        .then_with(|| {
            right
                .input
                .score_match_pct
                .total_cmp(&left.input.score_match_pct)
        })
        .then_with(|| {
            left.input
                .batch_candidate_index
                .cmp(&right.input.batch_candidate_index)
        })
        .then_with(|| left.input.message_index.cmp(&right.input.message_index))
        .then_with(|| left.input.x.cmp(&right.input.x))
}

/// Orders two cached scored-input summaries from best to worst.
///
/// # Parameters
/// - `left`: Left cached summary participating in the comparison.
/// - `right`: Right cached summary participating in the comparison.
///
/// # Returns
/// - `CmpOrdering`: Ordering that places better candidates before worse candidates.
///
/// # Expected Output
/// - Returns a deterministic ordering with no side effects.
fn compare_cached_scored_input_summaries(
    left: &CachedScoredInputSummary,
    right: &CachedScoredInputSummary,
) -> CmpOrdering {
    right
        .fitness
        .fitness_score
        .cmp(&left.fitness.fitness_score)
        .then_with(|| {
            right
                .fitness
                .fitness_total_score
                .cmp(&left.fitness.fitness_total_score)
        })
        .then_with(|| right.score_match_pct.total_cmp(&left.score_match_pct))
        .then_with(|| left.batch_candidate_index.cmp(&right.batch_candidate_index))
        .then_with(|| left.message_index.cmp(&right.message_index))
        .then_with(|| left.x.cmp(&right.x))
}

/// Greedily retains a globally unique ranked pool with no repeated `r` or `x` values.
///
/// # Parameters
/// - `ranked_inputs`: Fitness-ranked input pool ordered from best to worst.
///
/// # Returns
/// - `(Vec<RankedScoredAvalancheInput>, usize)`: Unique retained pool plus the number of rejected overlaps.
///
/// # Expected Output
/// - Preserves rank order among retained items and drops later inputs that reuse an earlier `r`
///   or `x`; no stdout/stderr output.
fn retain_globally_unique_ranked_scored_inputs(
    ranked_inputs: Vec<RankedScoredAvalancheInput>,
) -> (Vec<RankedScoredAvalancheInput>, usize) {
    let mut seen_r = HashSet::new();
    let mut seen_x = HashSet::new();
    let mut unique_inputs = Vec::with_capacity(ranked_inputs.len());
    let mut rejected_overlap_count = 0usize;

    for ranked_input in ranked_inputs {
        if seen_r.contains(&ranked_input.input.r) || seen_x.contains(&ranked_input.input.x) {
            rejected_overlap_count += 1;
            continue;
        }

        seen_r.insert(ranked_input.input.r.clone());
        seen_x.insert(ranked_input.input.x.clone());
        unique_inputs.push(ranked_input);
    }

    (unique_inputs, rejected_overlap_count)
}

/// Greedily retains a globally unique cached summary pool with no repeated `r` or `x` values.
///
/// # Parameters
/// - `ranked_inputs`: Fitness-ranked cached summary pool ordered from best to worst.
///
/// # Returns
/// - `(Vec<CachedScoredInputSummary>, usize)`: Unique retained pool plus the number of rejected overlaps.
///
/// # Expected Output
/// - Preserves rank order among retained items and drops later inputs that reuse an earlier `r`
///   or `x`; no stdout/stderr output.
fn retain_globally_unique_cached_scored_inputs(
    ranked_inputs: Vec<CachedScoredInputSummary>,
) -> (Vec<CachedScoredInputSummary>, usize) {
    let mut seen_r = HashSet::new();
    let mut seen_x = HashSet::new();
    let mut unique_inputs = Vec::with_capacity(ranked_inputs.len());
    let mut rejected_overlap_count = 0usize;

    for ranked_input in ranked_inputs {
        if seen_r.contains(&ranked_input.r) || seen_x.contains(&ranked_input.x) {
            rejected_overlap_count += 1;
            continue;
        }

        seen_r.insert(ranked_input.r.clone());
        seen_x.insert(ranked_input.x.clone());
        unique_inputs.push(ranked_input);
    }

    (unique_inputs, rejected_overlap_count)
}

impl StreamingScoredAvalancheFitnessPool {
    /// Creates a streaming bounded pool for incremental scored-input fitness pruning.
    ///
    /// # Parameters
    /// - `fitness_bit_width`: Number of least-significant bits used for the zero-count fitness score.
    /// - `r_candidate_limit`: Primary retention dimension used to derive the global retained-input cap.
    /// - `cx_candidate_limit`: Secondary retention dimension used to derive the global retained-input cap.
    /// - `use_fitness_threshold`: Whether candidates below the normalized threshold should be dropped.
    /// - `fitness_threshold`: Minimum normalized zero-count fitness required when thresholding is enabled.
    ///
    /// # Returns
    /// - `StreamingScoredAvalancheFitnessPool`: Empty incremental retention pool.
    ///
    /// # Expected Output
    /// - Returns an in-memory retention state with no stdout/stderr output.
    pub(crate) fn new(
        fitness_bit_width: usize,
        r_candidate_limit: usize,
        cx_candidate_limit: usize,
        use_fitness_threshold: bool,
        fitness_threshold: f64,
    ) -> Self {
        Self {
            fitness_bit_width,
            retained_input_limit: resolve_avalanche_fitness_retained_input_limit(
                r_candidate_limit,
                cx_candidate_limit,
            ),
            use_fitness_threshold,
            fitness_threshold,
            total_inputs: 0,
            total_groups: HashSet::new(),
            threshold_retained_input_count: 0,
            threshold_retained_groups: HashSet::new(),
            ranked_inputs: Vec::new(),
        }
    }

    /// Extends the streaming fitness pool with one processed chunk of scored inputs.
    ///
    /// # Parameters
    /// - `inputs`: Newly scored Avalanche inputs from the current processing chunk.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Computes per-input fitness scores, applies threshold filtering, and periodically prunes the
    ///   retained pool in place; no stdout/stderr output.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn extend(&mut self, inputs: Vec<ScoredAvalancheInput>) {
        if inputs.is_empty() {
            return;
        }

        let ranked_inputs = inputs
            .into_iter()
            .map(|input| RankedScoredAvalancheInput {
                fitness: single_message_avalanche_fitness_score(
                    &input.message_bits,
                    self.fitness_bit_width,
                ),
                input,
            })
            .collect::<Vec<_>>();
        self.extend_with_scores(ranked_inputs);
    }

    /// Extends the streaming fitness pool with one processed chunk of pre-scored inputs.
    ///
    /// # Parameters
    /// - `ranked_inputs`: Newly scored Avalanche inputs plus their padding-bit fitness metrics.
    ///
    /// # Returns
    /// - `()`: This method returns nothing.
    ///
    /// # Expected Output
    /// - Applies threshold filtering to the provided metrics and periodically prunes the retained
    ///   pool in place; no stdout/stderr output.
    pub(crate) fn extend_with_scores(&mut self, ranked_inputs: Vec<RankedScoredAvalancheInput>) {
        if ranked_inputs.is_empty() {
            return;
        }

        self.total_inputs += ranked_inputs.len();
        self.total_groups.extend(
            ranked_inputs
                .iter()
                .map(|input| input.input.batch_candidate_index),
        );
        for ranked_input in ranked_inputs {
            let fitness_score = ranked_input.fitness.fitness_score;
            if self.use_fitness_threshold
                && normalize_avalanche_fitness_score(fitness_score, self.fitness_bit_width)
                    < self.fitness_threshold
            {
                continue;
            }
            self.threshold_retained_input_count += 1;
            self.threshold_retained_groups
                .insert(ranked_input.input.batch_candidate_index);
            self.ranked_inputs.push(ranked_input);
        }

        if self.retained_input_limit > 0
            && self.ranked_inputs.len() > self.retained_input_limit.saturating_mul(2).max(1)
        {
            retain_best_ranked_inputs(
                &mut self.ranked_inputs,
                self.retained_input_limit,
                compare_ranked_scored_avalanche_inputs,
            );
        }
    }

    /// Finalizes the streaming retention pool into a sorted scored-input vector.
    ///
    /// # Parameters
    /// - `enforce_global_uniqueness`: Whether the finalized retained pool should reject reused `r`
    ///   or `x` values.
    ///
    /// # Returns
    /// - `Vec<ScoredAvalancheInput>`: Final retained scored inputs ordered from best to worst.
    ///
    /// # Expected Output
    /// - Prints summary statistics for the incremental pruning pass and returns the retained pool.
    pub(crate) fn finalize(
        mut self,
        enforce_global_uniqueness: bool,
    ) -> Vec<RankedScoredAvalancheInput> {
        retain_best_ranked_inputs(
            &mut self.ranked_inputs,
            self.retained_input_limit,
            compare_ranked_scored_avalanche_inputs,
        );
        println!(
            "Avalanche fitness streaming prune: processed {} scored inputs spanning {} r-candidate groups and retained {} threshold-passing inputs before final ranking",
            self.total_inputs,
            self.total_groups.len(),
            self.threshold_retained_input_count
        );
        if self.use_fitness_threshold {
            println!(
                "Avalanche fitness threshold: retained {} of {} scored inputs spanning {} of {} r-candidate groups at normalized threshold {} during streaming prune",
                self.threshold_retained_input_count,
                self.total_inputs,
                self.threshold_retained_groups.len(),
                self.total_groups.len(),
                format_beam_float(self.fitness_threshold, 3)
            );
        }
        let mut rejected_overlap_count = 0usize;
        if enforce_global_uniqueness {
            let (unique_inputs, rejected_count) =
                retain_globally_unique_ranked_scored_inputs(self.ranked_inputs);
            self.ranked_inputs = unique_inputs;
            rejected_overlap_count = rejected_count;
        }
        let retained_group_count = self
            .ranked_inputs
            .iter()
            .map(|input| input.input.batch_candidate_index)
            .collect::<HashSet<_>>()
            .len();
        println!(
            "Avalanche fitness streaming prune: retained {} scored inputs spanning {} r-candidate groups after incremental ranking",
            self.ranked_inputs.len(),
            retained_group_count
        );
        if enforce_global_uniqueness {
            println!(
                "Avalanche unique-input filter: retained {} scored inputs after dropping {} overlapping r/x candidates",
                self.ranked_inputs.len(),
                rejected_overlap_count
            );
        }
        if let Some(best_input) = self.ranked_inputs.first() {
            let best_fitness_pct = normalize_avalanche_fitness_score(
                best_input.fitness.fitness_score,
                self.fitness_bit_width,
            ) * 100.0;
            let best_mean_fitness_pct = normalize_avalanche_fitness_mean_score(
                best_input.fitness.fitness_total_score,
                self.fitness_bit_width,
                best_input.fitness.fitness_message_count,
            ) * 100.0;
            println!(
                "Avalanche fitness maxima: best candidate batch-index {} message-index {} x {} inverted {} minimum-padding-fitness {} ({}%) mean-padding-fitness {}% across {} message(s) match {}%",
                best_input.input.batch_candidate_index,
                best_input.input.message_index,
                best_input.input.x,
                best_input.input.contents_have_been_inverted,
                best_input.fitness.fitness_score,
                format_beam_float(best_fitness_pct, BEAM_PCT_DECIMALS),
                format_beam_float(best_mean_fitness_pct, BEAM_PCT_DECIMALS),
                best_input.fitness.fitness_message_count,
                format_beam_float(best_input.input.score_match_pct, BEAM_PCT_DECIMALS),
            );
        }
        self.ranked_inputs
    }
}

/// Validates the configured normalized Avalanche fitness threshold when thresholding is enabled.
///
/// # Parameters
/// - `engine`: Engine configuration that controls the fitness preprocessing pass.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` when the configured threshold is valid.
///
/// # Expected Output
/// - Returns a configuration error when the threshold is non-finite or outside `[0, 1]`.
pub(crate) fn validate_avalanche_fitness_threshold(
    engine: &EngineConfig,
) -> Result<(), Box<dyn Error>> {
    if !engine.avalanche_fitness_scoring_pass || !engine.avalanche_fitness_use_threshold {
        return Ok(());
    }

    if !engine.avalanche_fitness_threshold.is_finite()
        || !(0.0..=1.0).contains(&engine.avalanche_fitness_threshold)
    {
        return Err("avalanche_fitness_threshold must be finite and in [0, 1]".into());
    }

    Ok(())
}

/// Validates the configured percentage of retained Avalanche fitness entries logged per batch.
///
/// # Parameters
/// - `engine`: Engine configuration that controls batch-level fitness logging.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` when the configured logging percentage is valid.
///
/// # Expected Output
/// - Returns a configuration error when the logging percentage is non-finite or outside `(0, 1]`.
pub(crate) fn validate_avalanche_fitness_log_top_pct(
    engine: &EngineConfig,
) -> Result<(), Box<dyn Error>> {
    if !engine.avalanche_fitness_log_top_pct.is_finite()
        || !(0.0..=1.0).contains(&engine.avalanche_fitness_log_top_pct)
        || engine.avalanche_fitness_log_top_pct == 0.0
    {
        return Err("avalanche_fitness_log_top_pct must be finite and in (0, 1]".into());
    }

    Ok(())
}

/// Extracts the original plaintext payload bits from a widened Avalanche bit vector.
///
/// # Parameters
/// - `engine`: Engine configuration containing the fitness-shift and plaintext widths.
/// - `bits`: Full-width widened bit vector containing the fitness slice at the low end.
///
/// # Returns
/// - `Vec<bool>`: Payload-only bits with the leading fitness slice removed.
///
/// # Expected Output
/// - Returns the payload slice used for final accuracy comparisons and display output; no stdout/stderr output.
pub(crate) fn extract_payload_bits_for_accuracy(engine: &EngineConfig, bits: &[bool]) -> Vec<bool> {
    let shift_bits = resolve_avalanche_fitness_shift_bits(engine).min(bits.len());
    let payload_width = resolve_plaintext_message_bit_width(engine);
    let payload_end = shift_bits.saturating_add(payload_width).min(bits.len());
    bits[shift_bits..payload_end].to_vec()
}

/// Converts the configured plaintext message into its payload-width bit vector.
///
/// # Parameters
/// - `engine`: Engine configuration containing the plaintext payload width.
/// - `message`: Plaintext message to convert.
///
/// # Returns
/// - `Vec<bool>`: Plaintext payload bits without the fitness slice.
///
/// # Expected Output
/// - Returns the payload-width message bits; no stdout/stderr output.
pub(crate) fn payload_message_bits(engine: &EngineConfig, message: &BigUint) -> Vec<bool> {
    biguint_to_bits_le(message, resolve_plaintext_message_bit_width(engine))
}

/// Validates that the configured plaintext width and fitness slice can fit under the modulus.
///
/// # Parameters
/// - `engine`: Engine configuration containing payload and fitness-shift widths.
/// - `n`: RSA modulus that must contain the widened message without wrapping.
/// - `context`: Human-readable label for the caller.
///
/// # Returns
/// - `Result<(), Box<dyn Error>>`: `Ok(())` when the configured widened message can fit under `n`.
///
/// # Expected Output
/// - Returns an error when `engine.message.bits + fitness_shift_bits` exceeds the modulus width.
pub(crate) fn validate_message_width_under_modulus(
    engine: &EngineConfig,
    n: &BigUint,
    context: &str,
) -> Result<(), Box<dyn Error>> {
    if n.is_zero() {
        return Ok(());
    }

    let payload_bits = resolve_plaintext_message_bit_width(engine) as u64;
    let fitness_shift_bits =
        u64::try_from(resolve_avalanche_fitness_shift_bits(engine)).unwrap_or(u64::MAX);
    let widened_bits = payload_bits.saturating_add(fitness_shift_bits);
    let modulus_bits = n.bits().max(1);

    if widened_bits > modulus_bits {
        return Err(format!(
            "{context} configured payload width {payload_bits} bits plus fitness shift {fitness_shift_bits} bits exceeds modulus width {modulus_bits} bits"
        )
        .into());
    }

    Ok(())
}

pub(crate) type ScoredAvalanchePreprocessPass =
    Arc<dyn Fn(Vec<ScoredAvalancheInput>) -> Vec<ScoredAvalancheInput> + Send + Sync>;

/// Builds the anonymous scored-input fitness pass used before sampled Avalanche selection.
///
/// # Parameters
/// - `engine`: Engine configuration containing the fitness-pass settings.
///
/// # Returns
/// - `Option<ScoredAvalanchePreprocessPass>`: Closure that downselects `r` and `c^x` inputs, or `None` when disabled.
///
/// # Expected Output
/// - Returns an in-memory closure when the fitness pass is enabled; no stdout/stderr output.
pub(crate) fn build_scored_avalanche_fitness_pass(
    engine: &EngineConfig,
) -> Option<ScoredAvalanchePreprocessPass> {
    if !engine.avalanche_fitness_scoring_pass {
        return None;
    }

    let fitness_bit_width = resolve_avalanche_fitness_bit_width(engine);
    let r_candidate_limit = engine.avalanche_fitness_r_candidate_limit;
    let cx_candidate_limit = engine.avalanche_fitness_cx_candidate_limit;
    let use_fitness_threshold = engine.avalanche_fitness_use_threshold;
    let fitness_threshold = engine.avalanche_fitness_threshold;
    Some(Arc::new(move |inputs| {
        apply_scored_avalanche_fitness_pass(
            inputs,
            fitness_bit_width,
            r_candidate_limit,
            cx_candidate_limit,
            use_fitness_threshold,
            fitness_threshold,
        )
    }))
}

/// Applies the zero-count fitness pass to scored Avalanche inputs.
///
/// # Parameters
/// - `inputs`: Flattened scored inputs to rank and downselect.
/// - `fitness_bit_width`: Number of least-significant bits used for the zero-count fitness score.
/// - `r_candidate_limit`: Primary retention dimension used to derive the global retained-input cap.
/// - `cx_candidate_limit`: Secondary retention dimension used to derive the global retained-input cap.
/// - `use_fitness_threshold`: Whether candidates below the normalized threshold should be dropped.
/// - `fitness_threshold`: Minimum normalized zero-count fitness required when thresholding is enabled.
///
/// # Returns
/// - `Vec<ScoredAvalancheInput>`: Fitness-ranked and truncated scored inputs.
///
/// # Expected Output
/// - Returns the filtered pool in descending fitness order; no stdout/stderr output.
pub(crate) fn apply_scored_avalanche_fitness_pass(
    inputs: Vec<ScoredAvalancheInput>,
    fitness_bit_width: usize,
    r_candidate_limit: usize,
    cx_candidate_limit: usize,
    use_fitness_threshold: bool,
    fitness_threshold: f64,
) -> Vec<ScoredAvalancheInput> {
    if inputs.is_empty() {
        return inputs;
    }

    let ranked_inputs = inputs
        .into_iter()
        .map(|input| RankedScoredAvalancheInput {
            fitness: single_message_avalanche_fitness_score(&input.message_bits, fitness_bit_width),
            input,
        })
        .collect::<Vec<_>>();
    apply_ranked_scored_avalanche_fitness_pass(
        ranked_inputs,
        fitness_bit_width,
        r_candidate_limit,
        cx_candidate_limit,
        use_fitness_threshold,
        fitness_threshold,
    )
    .into_iter()
    .map(|input| input.input)
    .collect()
}

/// Applies the scored-input fitness pass to pre-ranked Avalanche inputs.
///
/// # Parameters
/// - `ranked_inputs`: Flattened scored inputs plus precomputed padding-bit fitness scores.
/// - `fitness_bit_width`: Number of least-significant bits used for the normalized threshold.
/// - `r_candidate_limit`: Primary retention dimension used to derive the global retained-input cap.
/// - `cx_candidate_limit`: Secondary retention dimension used to derive the global retained-input cap.
/// - `use_fitness_threshold`: Whether candidates below the normalized threshold should be dropped.
/// - `fitness_threshold`: Minimum normalized padding-bit fitness required when thresholding is enabled.
///
/// # Returns
/// - `Vec<RankedScoredAvalancheInput>`: Fitness-ranked and truncated scored inputs plus their scores.
///
/// # Expected Output
/// - Returns the filtered pool in descending fitness order; no stdout/stderr output.
pub(crate) fn apply_ranked_scored_avalanche_fitness_pass(
    ranked_inputs: Vec<RankedScoredAvalancheInput>,
    fitness_bit_width: usize,
    r_candidate_limit: usize,
    cx_candidate_limit: usize,
    use_fitness_threshold: bool,
    fitness_threshold: f64,
) -> Vec<RankedScoredAvalancheInput> {
    if ranked_inputs.is_empty() {
        return ranked_inputs;
    }

    let total_inputs = ranked_inputs.len();
    let total_groups = ranked_inputs
        .iter()
        .map(|input| input.input.batch_candidate_index)
        .collect::<HashSet<_>>()
        .len();
    let retained_input_limit =
        resolve_avalanche_fitness_retained_input_limit(r_candidate_limit, cx_candidate_limit);
    println!(
        "Avalanche fitness pass: scoring {} scored inputs in one global pool spanning {} r-candidate groups",
        total_inputs, total_groups
    );

    let mut ranked_inputs = ranked_inputs
        .into_par_iter()
        .filter(|input| {
            !use_fitness_threshold
                || normalize_avalanche_fitness_score(input.fitness.fitness_score, fitness_bit_width)
                    >= fitness_threshold
        })
        .collect::<Vec<_>>();
    let threshold_retained_input_count = ranked_inputs.len();
    let threshold_retained_group_count = ranked_inputs
        .iter()
        .map(|input| input.input.batch_candidate_index)
        .collect::<HashSet<_>>()
        .len();
    if use_fitness_threshold {
        println!(
            "Avalanche fitness threshold: retained {} of {} scored inputs spanning {} of {} r-candidate groups at normalized threshold {}",
            threshold_retained_input_count,
            total_inputs,
            threshold_retained_group_count,
            total_groups,
            format_beam_float(fitness_threshold, 3)
        );
    }
    retain_best_ranked_inputs(
        &mut ranked_inputs,
        retained_input_limit,
        compare_ranked_scored_avalanche_inputs,
    );
    let retained_group_count = ranked_inputs
        .iter()
        .map(|input| input.input.batch_candidate_index)
        .collect::<HashSet<_>>()
        .len();
    println!(
        "Avalanche fitness pass: retained {} scored inputs spanning {} r-candidate groups after global ranking",
        ranked_inputs.len(),
        retained_group_count
    );
    if let Some(best_input) = ranked_inputs.first() {
        let best_fitness_pct =
            normalize_avalanche_fitness_score(best_input.fitness.fitness_score, fitness_bit_width)
                * 100.0;
        let best_mean_fitness_pct = normalize_avalanche_fitness_mean_score(
            best_input.fitness.fitness_total_score,
            fitness_bit_width,
            best_input.fitness.fitness_message_count,
        ) * 100.0;
        println!(
            "Avalanche fitness maxima: best candidate batch-index {} message-index {} x {} inverted {} minimum-padding-fitness {} ({}%) mean-padding-fitness {}% across {} message(s) match {}%",
            best_input.input.batch_candidate_index,
            best_input.input.message_index,
            best_input.input.x,
            best_input.input.contents_have_been_inverted,
            best_input.fitness.fitness_score,
            format_beam_float(best_fitness_pct, BEAM_PCT_DECIMALS),
            format_beam_float(best_mean_fitness_pct, BEAM_PCT_DECIMALS),
            best_input.fitness.fitness_message_count,
            format_beam_float(best_input.input.score_match_pct, BEAM_PCT_DECIMALS),
        );
    }
    ranked_inputs
}

#[derive(Clone, Debug)]
pub(crate) struct HammingDistancePrunedPool {
    pub(crate) selected_inputs: Vec<ScoredAvalancheInput>,
    pub(crate) retained_inlier_count: usize,
    pub(crate) available_outlier_count: usize,
    pub(crate) preferred_outlier_count: usize,
}

/// Prunes scored avalanche inputs to a central Hamming-distance percentile band with optional
/// interval progress logging.
///
/// # Parameters
/// - `inputs`: Flattened scored avalanche inputs available for sampled-avalanche selection.
/// - `reference_message_bits`: Original plaintext bits packed for Hamming-distance scoring.
/// - `keep_percentile`: Central percentile of Hamming distances to retain.
/// - `outlier_preference_pct`: Percentage of the retained inlier count to add back from the
///   Hamming-distance outlier tails.
/// - `progress_label`: Optional human-readable label used for interval progress logging.
///
/// # Returns
/// - `HammingDistancePrunedPool`: Filtered pool plus counts describing the retained inliers and
///   preferred outliers.
///
/// # Expected Output
/// - Optionally prints interval progress updates and returns the filtered inputs in original
///   order; falls back to the unpruned pool when pruning would remove every input or when the
///   requested percentile does not trim any tails.
pub(crate) fn prune_scored_inputs_by_hamming_distance_percentile_with_progress(
    inputs: &[ScoredAvalancheInput],
    reference_message_bits: &PackedBits,
    keep_percentile: f64,
    outlier_preference_pct: f64,
    progress_label: Option<&str>,
) -> HammingDistancePrunedPool {
    let original_pool = HammingDistancePrunedPool {
        selected_inputs: inputs.to_vec(),
        retained_inlier_count: inputs.len(),
        available_outlier_count: 0,
        preferred_outlier_count: 0,
    };
    if inputs.len() < 2 || keep_percentile >= 100.0 {
        return original_pool;
    }

    let tail_fraction = ((100.0 - keep_percentile).max(0.0) / 100.0) / 2.0;
    if tail_fraction <= 0.0 {
        return original_pool;
    }

    let chunk_size = parallel_progress_chunk_size(inputs.len());
    let total_chunks = inputs.len().div_ceil(chunk_size);
    let progress_total = total_chunks.min(u64::MAX as usize) as u64;
    let progress_started_at = Instant::now();
    let progress_done = AtomicU64::new(0);
    let progress_next_log_at_ms =
        AtomicU64::new(Duration::from_secs(5).as_millis().min(u128::from(u64::MAX)) as u64);
    if let Some(label) = progress_label {
        println!("{label}: scoring Hamming distances across {total_chunks} chunk(s)");
    }
    let distances = inputs
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let start_index = chunk_index.saturating_mul(chunk_size);
            let distances = chunk
                .iter()
                .enumerate()
                .map(|(offset, input)| {
                    (
                        start_index + offset,
                        hamming_distance_packed_bytes(
                            input.message_bits.bytes_le(),
                            reference_message_bits.bytes_le(),
                        ),
                    )
                })
                .collect::<Vec<_>>();
            if let Some(label) = progress_label {
                let done = progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                log_parallel_progress_every_interval(
                    done,
                    progress_total,
                    &progress_started_at,
                    &progress_next_log_at_ms,
                    label,
                    Duration::from_secs(5),
                );
            }
            distances
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let mut sorted_distances = distances
        .iter()
        .map(|(_, distance)| *distance)
        .collect::<Vec<_>>();
    sorted_distances.sort_unstable();

    let tail_count = ((inputs.len() as f64) * tail_fraction).round() as usize;
    if tail_count == 0 || tail_count.saturating_mul(2) >= sorted_distances.len() {
        return original_pool;
    }

    let lower_distance = sorted_distances[tail_count];
    let upper_distance = sorted_distances[sorted_distances.len() - tail_count - 1];
    let mut inlier_indices = Vec::new();
    let mut outliers = Vec::new();
    for (index, distance) in distances {
        if distance >= lower_distance && distance <= upper_distance {
            inlier_indices.push(index);
        } else {
            let deviation = if distance < lower_distance {
                lower_distance - distance
            } else {
                distance - upper_distance
            };
            outliers.push((index, deviation));
        }
    }

    if inlier_indices.is_empty() {
        return original_pool;
    }

    let preferred_outlier_count =
        (((inlier_indices.len() as f64) * outlier_preference_pct.max(0.0)) / 100.0).round()
            as usize;
    outliers.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let inlier_index_set = inlier_indices.iter().copied().collect::<HashSet<_>>();
    let preferred_outlier_indices = outliers
        .iter()
        .take(preferred_outlier_count.min(outliers.len()))
        .map(|(index, _)| *index)
        .collect::<HashSet<_>>();

    let selected_inputs = inputs
        .iter()
        .enumerate()
        .filter_map(|(index, input)| {
            (inlier_index_set.contains(&index) || preferred_outlier_indices.contains(&index))
                .then(|| input.clone())
        })
        .collect::<Vec<_>>();

    if selected_inputs.is_empty() {
        original_pool
    } else {
        HammingDistancePrunedPool {
            selected_inputs,
            retained_inlier_count: inlier_indices.len(),
            available_outlier_count: outliers.len(),
            preferred_outlier_count: preferred_outlier_indices.len(),
        }
    }
}

/// Groups scored avalanche inputs by their originating r candidate with optional interval
/// progress logging.
///
/// # Parameters
/// - `inputs`: Scored candidate decryptions produced for the batch.
/// - `progress_label`: Optional human-readable label used for interval progress logging.
///
/// # Returns
/// - `Vec<ScoredAvalancheInputGroup>`: Distinct r-candidate groups preserving every `c^x` input.
///
/// # Expected Output
/// - Optionally prints interval progress updates and returns grouped inputs ordered by
///   batch-candidate index.
pub(crate) fn group_scored_inputs_by_r_candidate_with_progress(
    inputs: &[ScoredAvalancheInput],
    progress_label: Option<&str>,
) -> Vec<ScoredAvalancheInputGroup> {
    if inputs.is_empty() {
        return Vec::new();
    }

    let chunk_size = parallel_progress_chunk_size(inputs.len());
    let total_chunks = inputs.len().div_ceil(chunk_size);
    let progress_total = total_chunks.min(u64::MAX as usize) as u64;
    let progress_started_at = Instant::now();
    let progress_done = AtomicU64::new(0);
    let progress_next_log_at_ms =
        AtomicU64::new(Duration::from_secs(5).as_millis().min(u128::from(u64::MAX)) as u64);
    if let Some(label) = progress_label {
        println!(
            "{label}: grouping {} scored inputs across {total_chunks} chunk(s)",
            inputs.len()
        );
    }

    let mut grouped_inputs = inputs
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut grouped = HashMap::<usize, Vec<ScoredAvalancheInput>>::new();
            for input in chunk {
                grouped
                    .entry(input.batch_candidate_index)
                    .or_default()
                    .push(input.clone());
            }
            if let Some(label) = progress_label {
                let done = progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                log_parallel_progress_every_interval(
                    done,
                    progress_total,
                    &progress_started_at,
                    &progress_next_log_at_ms,
                    label,
                    Duration::from_secs(5),
                );
            }
            grouped
        })
        .reduce(HashMap::new, |mut left, right| {
            for (batch_candidate_index, mut chunk_inputs) in right {
                left.entry(batch_candidate_index)
                    .or_default()
                    .append(&mut chunk_inputs);
            }
            left
        })
        .into_iter()
        .collect::<Vec<_>>();
    grouped_inputs.par_sort_unstable_by_key(|(batch_candidate_index, _)| *batch_candidate_index);

    grouped_inputs
        .into_par_iter()
        .map(|(batch_candidate_index, mut grouped_inputs)| {
            grouped_inputs.sort_by(|left, right| {
                left.message_index
                    .cmp(&right.message_index)
                    .then_with(|| left.x.cmp(&right.x))
                    .then_with(|| right.score_match_pct.total_cmp(&left.score_match_pct))
            });
            ScoredAvalancheInputGroup {
                batch_candidate_index,
                inputs: grouped_inputs,
            }
        })
        .collect()
}

/// Selects a random set of r-candidate groups and caps the flattened sample size.
///
/// # Parameters
/// - `grouped_inputs`: Grouped scored inputs keyed by r candidate.
/// - `mixed_r_candidate_count`: Number of distinct r candidates to include.
/// - `combination_size`: Maximum number of scored inputs to keep after group sampling.
/// - `rng`: Random number generator used for group sampling.
///
/// # Returns
/// - `Vec<ScoredAvalancheInput>`: Sampled scored inputs for the selected r groups.
///
/// # Expected Output
/// - Returns up to `combination_size` sampled `c^x` inputs while preserving selected r-group
///   coverage when possible; no stdout/stderr output.
#[allow(dead_code)]
pub(crate) fn select_scored_inputs_for_mixed_r_candidates(
    grouped_inputs: &[ScoredAvalancheInputGroup],
    mixed_r_candidate_count: usize,
    combination_size: usize,
    rng: &mut RngChoice,
) -> Vec<ScoredAvalancheInput> {
    if combination_size == 0 || grouped_inputs.is_empty() || mixed_r_candidate_count == 0 {
        return Vec::new();
    }

    let sampled_group_indices =
        sample_unique_indices(grouped_inputs.len(), mixed_r_candidate_count, rng);
    let mut sampled_groups = Vec::new();
    for group_idx in sampled_group_indices {
        if let Some(group) = grouped_inputs.get(group_idx) {
            debug_assert_eq!(
                group
                    .inputs
                    .first()
                    .map(|input| input.batch_candidate_index)
                    .unwrap_or(group.batch_candidate_index),
                group.batch_candidate_index
            );
            sampled_groups.push(group);
        }
    }
    if sampled_groups.is_empty() {
        return Vec::new();
    }

    let available_input_count = sampled_groups
        .iter()
        .map(|group| group.inputs.len())
        .sum::<usize>();
    if available_input_count <= combination_size {
        let mut selected_inputs = Vec::with_capacity(available_input_count);
        for group in sampled_groups {
            selected_inputs.extend(group.inputs.iter().cloned());
        }
        return selected_inputs;
    }

    let required_group_slots = sampled_groups.len().min(combination_size);
    let mut selected_inputs = Vec::with_capacity(combination_size);
    let mut leftover_inputs = Vec::with_capacity(available_input_count - required_group_slots);

    for (group_order, group) in sampled_groups.iter().enumerate() {
        let pick_indices = sample_unique_indices(group.inputs.len(), 1, rng);
        if group_order < required_group_slots {
            if let Some(&picked_index) = pick_indices.first() {
                selected_inputs.push(group.inputs[picked_index].clone());
                for (input_idx, input) in group.inputs.iter().enumerate() {
                    if input_idx != picked_index {
                        leftover_inputs.push(input.clone());
                    }
                }
                continue;
            }
        }
        leftover_inputs.extend(group.inputs.iter().cloned());
    }

    let remaining_slots = combination_size.saturating_sub(selected_inputs.len());
    let leftover_indices = sample_unique_indices(leftover_inputs.len(), remaining_slots, rng);
    for leftover_idx in leftover_indices {
        if let Some(input) = leftover_inputs.get(leftover_idx) {
            selected_inputs.push(input.clone());
        }
    }

    selected_inputs
}

/// Loads lightweight cached scored-input summaries for one analysis batch.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// - `batch_number`: One-based analysis batch number.
///
/// # Returns
/// - `Result<Vec<CachedScoredInputSummary>, Box<dyn Error>>`: Lightweight scored-input summaries in row order.
///
/// # Expected Output
/// - Streams cached scored-input pages from SQLite and returns lightweight metadata only.
pub(crate) fn load_cached_scored_input_summaries(
    cache: &AvalancheCacheGuard,
    batch_number: usize,
) -> Result<Vec<CachedScoredInputSummary>, Box<dyn Error>> {
    let total_rows = count_cached_scored_inputs(cache, batch_number)?;
    let progress_label = format!("Accuracy batch {} cached summary loading", batch_number);
    load_cached_scored_input_pages_with_progress(
        total_rows,
        cache.page_rows_usize(),
        &progress_label,
        |after_row_id| {
            let (rows, mut timing) = load_cached_scored_input_rows_after_id_page(
                cache,
                batch_number,
                after_row_id,
                cache.page_rows_i64(),
            )
            .map_err(|err| err.to_string())?;
            let last_row_id = rows.last().map(|row| row.id);
            let decode_start = Instant::now();
            let items = rows
                .into_iter()
                .map(|row| {
                    Ok::<_, String>(CachedScoredInputSummary {
                        id: row.id,
                        batch_candidate_index: usize::try_from(row.batch_candidate_index).map_err(
                            |_| "cached batch candidate index exceeds usize range".to_string(),
                        )?,
                        message_index: usize::try_from(row.message_index)
                            .map_err(|_| "cached message index exceeds usize range".to_string())?,
                        score_match_pct: row.score_match_pct,
                        contents_have_been_inverted: row.contents_have_been_inverted,
                        r: row
                            .r_text
                            .parse::<BigUint>()
                            .map_err(|err| err.to_string())?,
                        x: row
                            .x_text
                            .parse::<BigUint>()
                            .map_err(|err| err.to_string())?,
                        fitness: AvalancheFitnessScore::default(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            timing.row_decode += decode_start.elapsed();
            Ok(CachedKeysetPage {
                items,
                last_row_id,
                timing,
            })
        },
    )
    .map_err(|err| -> Box<dyn Error> { err.into() })
}

/// Applies the zero-count fitness pass to cached scored Avalanche inputs.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// - `batch_number`: One-based analysis batch number.
/// - `fitness_bit_width`: Number of least-significant bits used for the zero-count fitness score.
/// - `r_candidate_limit`: Primary retention dimension used to derive the global retained-input cap.
/// - `cx_candidate_limit`: Secondary retention dimension used to derive the global retained-input cap.
/// - `use_fitness_threshold`: Whether candidates below the normalized threshold should be dropped.
/// - `fitness_threshold`: Minimum normalized zero-count fitness required when thresholding is enabled.
///
/// # Returns
/// - `Result<Vec<CachedScoredInputSummary>, Box<dyn Error>>`: Fitness-ranked and truncated cached summaries.
///
/// # Expected Output
/// - Streams cached rows from SQLite, computes fitness rankings, and returns lightweight retained summaries.
pub(crate) fn apply_cached_scored_avalanche_fitness_pass(
    cache: &AvalancheCacheGuard,
    batch_number: usize,
    fitness_bit_width: usize,
    r_candidate_limit: usize,
    cx_candidate_limit: usize,
    use_fitness_threshold: bool,
    fitness_threshold: f64,
) -> Result<Vec<CachedScoredInputSummary>, Box<dyn Error>> {
    let total_rows = count_cached_scored_inputs(cache, batch_number)?;
    if total_rows == 0 {
        return Ok(Vec::new());
    }

    let page_progress_label = format!(
        "Accuracy batch {} cached fitness page scoring",
        batch_number
    );
    let scored_inputs = load_cached_scored_input_pages_with_progress(
        total_rows,
        cache.page_rows_usize(),
        &page_progress_label,
        |after_row_id| {
            let (rows, mut timing) = load_cached_scored_input_rows_after_id_page(
                cache,
                batch_number,
                after_row_id,
                cache.page_rows_i64(),
            )
            .map_err(|err| err.to_string())?;
            let last_row_id = rows.last().map(|row| row.id);
            let decode_start = Instant::now();
            let items = rows
                .into_iter()
                .map(|row| {
                    let batch_candidate_index = usize::try_from(row.batch_candidate_index)
                        .map_err(|_| {
                            "cached batch candidate index exceeds usize range".to_string()
                        })?;
                    Ok::<_, String>(CachedScoredInputSummary {
                        id: row.id,
                        batch_candidate_index,
                        message_index: usize::try_from(row.message_index)
                            .map_err(|_| "cached message index exceeds usize range".to_string())?,
                        score_match_pct: row.score_match_pct,
                        contents_have_been_inverted: row.contents_have_been_inverted,
                        r: row
                            .r_text
                            .parse::<BigUint>()
                            .map_err(|err| err.to_string())?,
                        x: row
                            .x_text
                            .parse::<BigUint>()
                            .map_err(|err| err.to_string())?,
                        fitness: AvalancheFitnessScore {
                            fitness_score: usize::try_from(row.fitness_score).map_err(|_| {
                                "cached fitness score exceeds usize range".to_string()
                            })?,
                            fitness_total_score: usize::try_from(row.fitness_total_score).map_err(
                                |_| "cached fitness total score exceeds usize range".to_string(),
                            )?,
                            fitness_message_count: usize::try_from(row.fitness_message_count)
                                .map_err(|_| {
                                    "cached fitness message count exceeds usize range".to_string()
                                })?,
                        },
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            timing.row_decode += decode_start.elapsed();
            Ok(CachedKeysetPage {
                items,
                last_row_id,
                timing,
            })
        },
    )
    .map_err(|err| -> Box<dyn Error> { err.into() })?;
    let total_groups = scored_inputs
        .iter()
        .map(|input| input.batch_candidate_index)
        .collect::<HashSet<_>>()
        .len();
    let retained_input_limit =
        resolve_avalanche_fitness_retained_input_limit(r_candidate_limit, cx_candidate_limit);
    println!(
        "Avalanche fitness pass: scoring {} cached scored inputs in one global pool spanning {} r-candidate groups",
        scored_inputs.len(),
        total_groups
    );
    let mut ranked_inputs = scored_inputs;
    if use_fitness_threshold {
        ranked_inputs.retain(|input| {
            normalize_avalanche_fitness_score(input.fitness.fitness_score, fitness_bit_width)
                >= fitness_threshold
        });
    }
    let threshold_retained_input_count = ranked_inputs.len();
    let threshold_retained_group_count = ranked_inputs
        .iter()
        .map(|input| input.batch_candidate_index)
        .collect::<HashSet<_>>()
        .len();
    if use_fitness_threshold {
        println!(
            "Avalanche fitness threshold: retained {} of {} cached scored inputs spanning {} of {} r-candidate groups at normalized threshold {}",
            threshold_retained_input_count,
            total_rows,
            threshold_retained_group_count,
            total_groups,
            format_beam_float(fitness_threshold, 3)
        );
    }
    retain_best_ranked_inputs(
        &mut ranked_inputs,
        retained_input_limit,
        compare_cached_scored_input_summaries,
    );
    let retained_group_count = ranked_inputs
        .iter()
        .map(|input| input.batch_candidate_index)
        .collect::<HashSet<_>>()
        .len();
    println!(
        "Avalanche fitness pass: retained {} cached scored inputs spanning {} r-candidate groups after global ranking",
        ranked_inputs.len(),
        retained_group_count
    );
    if let Some(best_input) = ranked_inputs.first() {
        let best_fitness_pct =
            normalize_avalanche_fitness_score(best_input.fitness.fitness_score, fitness_bit_width)
                * 100.0;
        let best_mean_fitness_pct = normalize_avalanche_fitness_mean_score(
            best_input.fitness.fitness_total_score,
            fitness_bit_width,
            best_input.fitness.fitness_message_count,
        ) * 100.0;
        println!(
            "Avalanche fitness maxima: best cached candidate batch-index {} message-index {} x {} inverted {} minimum-padding-fitness {} ({}%) mean-padding-fitness {}% across {} message(s) match {}%",
            best_input.batch_candidate_index,
            best_input.message_index,
            best_input.x,
            best_input.contents_have_been_inverted,
            best_input.fitness.fitness_score,
            format_beam_float(best_fitness_pct, BEAM_PCT_DECIMALS),
            format_beam_float(best_mean_fitness_pct, BEAM_PCT_DECIMALS),
            best_input.fitness.fitness_message_count,
            format_beam_float(best_input.score_match_pct, BEAM_PCT_DECIMALS),
        );
    }
    Ok(ranked_inputs)
}

/// Enforces a globally unique `r`/`x` set across fitness-ranked scored Avalanche inputs.
///
/// # Parameters
/// - `inputs`: Ranked scored Avalanche inputs ordered from best to worst.
///
/// # Returns
/// - `(Vec<ScoredAvalancheInput>, usize)`: Unique retained pool plus the number of rejected overlaps.
///
/// # Expected Output
/// - Preserves the first-ranked occurrence of each `r` and `x` value and drops later overlaps; no
///   stdout/stderr output.
pub(crate) fn enforce_global_unique_scored_inputs(
    inputs: Vec<ScoredAvalancheInput>,
) -> (Vec<ScoredAvalancheInput>, usize) {
    let ranked_inputs = inputs
        .into_iter()
        .map(|input| RankedScoredAvalancheInput {
            fitness: AvalancheFitnessScore::default(),
            input,
        })
        .collect::<Vec<_>>();
    let (unique_ranked_inputs, rejected_overlap_count) =
        retain_globally_unique_ranked_scored_inputs(ranked_inputs);
    (
        unique_ranked_inputs
            .into_iter()
            .map(|input| input.input)
            .collect::<Vec<_>>(),
        rejected_overlap_count,
    )
}

/// Enforces a globally unique `r`/`x` set across fitness-ranked cached scored-input summaries.
///
/// # Parameters
/// - `inputs`: Ranked cached scored-input summaries ordered from best to worst.
///
/// # Returns
/// - `(Vec<CachedScoredInputSummary>, usize)`: Unique retained pool plus the number of rejected overlaps.
///
/// # Expected Output
/// - Preserves the first-ranked occurrence of each `r` and `x` value and drops later overlaps; no
///   stdout/stderr output.
pub(crate) fn enforce_global_unique_cached_scored_inputs(
    inputs: Vec<CachedScoredInputSummary>,
) -> (Vec<CachedScoredInputSummary>, usize) {
    retain_globally_unique_cached_scored_inputs(inputs)
}

/// Prunes cached scored-input summaries to a central Hamming-distance percentile band with
/// optional interval progress logging.
///
/// # Parameters
/// - `cache`: Shared SQLite cache wrapper.
/// - `summaries`: Lightweight cached input summaries available for pruning.
/// - `reference_message_bits`: Original plaintext bits packed for Hamming-distance scoring.
/// - `keep_percentile`: Central percentile of Hamming distances to retain.
/// - `outlier_preference_pct`: Percentage of the retained inlier count to add back from outlier tails.
/// - `progress_label`: Optional human-readable label used for interval progress logging.
///
/// # Returns
/// - `Result<(Vec<CachedScoredInputSummary>, usize, usize, usize), Box<dyn Error>>`: Retained
///   summaries plus inlier/outlier counts.
///
/// # Expected Output
/// - Optionally prints interval progress updates while loading cached rows needed for
///   Hamming-distance scoring and returns retained summaries in original order.
pub(crate) fn prune_cached_scored_inputs_by_hamming_distance_percentile_with_progress(
    cache: &AvalancheCacheGuard,
    summaries: &[CachedScoredInputSummary],
    reference_message_bits: &PackedBits,
    keep_percentile: f64,
    outlier_preference_pct: f64,
    progress_label: Option<&str>,
) -> Result<(Vec<CachedScoredInputSummary>, usize, usize, usize), Box<dyn Error>> {
    if summaries.len() < 2 || keep_percentile >= 100.0 {
        return Ok((summaries.to_vec(), summaries.len(), 0, 0));
    }

    let tail_fraction = ((100.0 - keep_percentile).max(0.0) / 100.0) / 2.0;
    if tail_fraction <= 0.0 {
        return Ok((summaries.to_vec(), summaries.len(), 0, 0));
    }

    let ids = summaries
        .iter()
        .map(|summary| summary.id)
        .collect::<Vec<_>>();
    let chunk_size = cache.page_rows_usize().max(1);
    let total_chunks = ids.len().div_ceil(chunk_size);
    let progress_total = total_chunks.min(u64::MAX as usize) as u64;
    let progress_started_at = Instant::now();
    let progress_done = AtomicU64::new(0);
    let progress_next_log_at_ms =
        AtomicU64::new(Duration::from_secs(5).as_millis().min(u128::from(u64::MAX)) as u64);
    if let Some(label) = progress_label {
        println!("{label}: scoring cached Hamming distances across {total_chunks} chunk(s)");
    }
    let distance_pairs = ids
        .par_chunks(chunk_size)
        .map(|id_chunk| {
            let rows = load_cached_scored_input_rows_by_ids(cache, id_chunk)
                .map_err(|err| err.to_string())?;
            let distances = rows
                .into_iter()
                .map(|row| {
                    (
                        row.id,
                        hamming_distance_packed_bytes(
                            &row.message_bits,
                            reference_message_bits.bytes_le(),
                        ),
                    )
                })
                .collect::<Vec<_>>();
            if let Some(label) = progress_label {
                let done = progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                log_parallel_progress_every_interval(
                    done,
                    progress_total,
                    &progress_started_at,
                    &progress_next_log_at_ms,
                    label,
                    Duration::from_secs(5),
                );
            }
            Ok::<_, String>(distances)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| -> Box<dyn Error> { err.into() })?;
    let mut distance_by_id = HashMap::with_capacity(ids.len());
    for distance_chunk in distance_pairs {
        for (id, distance) in distance_chunk {
            distance_by_id.insert(id, distance);
        }
    }

    let distances = summaries
        .iter()
        .enumerate()
        .map(|(index, summary)| {
            let distance = *distance_by_id
                .get(&summary.id)
                .ok_or_else(|| format!("missing cached distance row id {}", summary.id))?;
            Ok::<_, Box<dyn Error>>((index, distance))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut sorted_distances = distances
        .iter()
        .map(|(_, distance)| *distance)
        .collect::<Vec<_>>();
    sorted_distances.sort_unstable();

    let tail_count = ((summaries.len() as f64) * tail_fraction).round() as usize;
    if tail_count == 0 || tail_count.saturating_mul(2) >= sorted_distances.len() {
        return Ok((summaries.to_vec(), summaries.len(), 0, 0));
    }

    let lower_distance = sorted_distances[tail_count];
    let upper_distance = sorted_distances[sorted_distances.len() - tail_count - 1];
    let mut inlier_indices = Vec::new();
    let mut outliers = Vec::new();
    for (index, distance) in distances {
        if distance >= lower_distance && distance <= upper_distance {
            inlier_indices.push(index);
        } else {
            let deviation = if distance < lower_distance {
                lower_distance - distance
            } else {
                distance - upper_distance
            };
            outliers.push((index, deviation));
        }
    }
    if inlier_indices.is_empty() {
        return Ok((summaries.to_vec(), summaries.len(), 0, 0));
    }

    let preferred_outlier_count =
        (((inlier_indices.len() as f64) * outlier_preference_pct.max(0.0)) / 100.0).round()
            as usize;
    outliers.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let inlier_index_set = inlier_indices.iter().copied().collect::<HashSet<_>>();
    let preferred_outlier_indices = outliers
        .iter()
        .take(preferred_outlier_count.min(outliers.len()))
        .map(|(index, _)| *index)
        .collect::<HashSet<_>>();

    let selected_inputs = summaries
        .iter()
        .enumerate()
        .filter_map(|(index, summary)| {
            (inlier_index_set.contains(&index) || preferred_outlier_indices.contains(&index))
                .then(|| summary.clone())
        })
        .collect::<Vec<_>>();
    Ok((
        if selected_inputs.is_empty() {
            summaries.to_vec()
        } else {
            selected_inputs
        },
        inlier_indices.len(),
        outliers.len(),
        preferred_outlier_indices.len(),
    ))
}

/// Groups cached scored-input summaries by their originating r candidate with optional interval
/// progress logging.
///
/// # Parameters
/// - `summaries`: Lightweight cached scored-input summaries produced for the batch.
/// - `progress_label`: Optional human-readable label used for interval progress logging.
///
/// # Returns
/// - `Vec<CachedScoredInputGroup>`: Distinct r-candidate groups preserving every cached row id.
///
/// # Expected Output
/// - Optionally prints interval progress updates and returns grouped cached summaries ordered by
///   batch-candidate index.
pub(crate) fn group_cached_scored_inputs_by_r_candidate_with_progress(
    summaries: &[CachedScoredInputSummary],
    progress_label: Option<&str>,
) -> Vec<CachedScoredInputGroup> {
    if summaries.is_empty() {
        return Vec::new();
    }

    let chunk_size = parallel_progress_chunk_size(summaries.len());
    let total_chunks = summaries.len().div_ceil(chunk_size);
    let progress_total = total_chunks.min(u64::MAX as usize) as u64;
    let progress_started_at = Instant::now();
    let progress_done = AtomicU64::new(0);
    let progress_next_log_at_ms =
        AtomicU64::new(Duration::from_secs(5).as_millis().min(u128::from(u64::MAX)) as u64);
    if let Some(label) = progress_label {
        println!(
            "{label}: grouping {} cached scored inputs across {total_chunks} chunk(s)",
            summaries.len()
        );
    }

    let mut grouped_inputs = summaries
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut grouped = HashMap::<usize, Vec<CachedScoredInputSummary>>::new();
            for summary in chunk {
                grouped
                    .entry(summary.batch_candidate_index)
                    .or_default()
                    .push(summary.clone());
            }
            if let Some(label) = progress_label {
                let done = progress_done.fetch_add(1, Ordering::Relaxed) + 1;
                log_parallel_progress_every_interval(
                    done,
                    progress_total,
                    &progress_started_at,
                    &progress_next_log_at_ms,
                    label,
                    Duration::from_secs(5),
                );
            }
            grouped
        })
        .reduce(HashMap::new, |mut left, right| {
            for (batch_candidate_index, mut chunk_inputs) in right {
                left.entry(batch_candidate_index)
                    .or_default()
                    .append(&mut chunk_inputs);
            }
            left
        })
        .into_iter()
        .collect::<Vec<_>>();
    grouped_inputs.par_sort_unstable_by_key(|(batch_candidate_index, _)| *batch_candidate_index);

    grouped_inputs
        .into_par_iter()
        .map(|(_, mut grouped_inputs)| {
            grouped_inputs.sort_by(|left, right| {
                left.message_index
                    .cmp(&right.message_index)
                    .then_with(|| left.x.cmp(&right.x))
                    .then_with(|| right.score_match_pct.total_cmp(&left.score_match_pct))
            });
            CachedScoredInputGroup {
                input_ids: grouped_inputs
                    .into_iter()
                    .map(|summary| summary.id)
                    .collect(),
            }
        })
        .collect()
}

/// Selects random cached scored-input row ids directly from the flattened summary pool.
///
/// # Parameters
/// - `summaries`: Flattened cached scored-input summaries available for sampling.
/// - `sample_size`: Maximum number of row ids to keep.
/// - `rng`: Random number generator used for index sampling.
///
/// # Returns
/// - `Vec<i64>`: Randomly selected cached row ids without replacement.
///
/// # Expected Output
/// - Returns up to `sample_size` unique cached row ids; no stdout/stderr output.
#[allow(dead_code)]
pub(crate) fn select_random_cached_scored_input_ids(
    summaries: &[CachedScoredInputSummary],
    sample_size: usize,
    rng: &mut RngChoice,
) -> Vec<i64> {
    if sample_size == 0 || summaries.is_empty() {
        return Vec::new();
    }

    sample_unique_indices(summaries.len(), sample_size, rng)
        .into_iter()
        .filter_map(|index| summaries.get(index).map(|summary| summary.id))
        .collect()
}

/// Selects cached row ids for a mixed-r sampled Avalanche input set.
///
/// # Parameters
/// - `grouped_inputs`: Cached scored-input groups keyed by r candidate.
/// - `mixed_r_candidate_count`: Number of distinct r candidates to include.
/// - `combination_size`: Maximum number of cached rows to keep after group sampling.
/// - `rng`: Random number generator used for group sampling.
///
/// # Returns
/// - `Vec<i64>`: Sampled cached row ids for the selected r groups.
///
/// # Expected Output
/// - Returns up to `combination_size` cached row ids while preserving selected r-group coverage when possible.
#[allow(dead_code)]
pub(crate) fn select_cached_scored_input_ids_for_mixed_r_candidates(
    grouped_inputs: &[CachedScoredInputGroup],
    mixed_r_candidate_count: usize,
    combination_size: usize,
    rng: &mut RngChoice,
) -> Vec<i64> {
    if combination_size == 0 || grouped_inputs.is_empty() || mixed_r_candidate_count == 0 {
        return Vec::new();
    }

    let sampled_group_indices =
        sample_unique_indices(grouped_inputs.len(), mixed_r_candidate_count, rng);
    let mut sampled_groups = Vec::new();
    for group_idx in sampled_group_indices {
        if let Some(group) = grouped_inputs.get(group_idx) {
            sampled_groups.push(group);
        }
    }
    if sampled_groups.is_empty() {
        return Vec::new();
    }

    let available_input_count = sampled_groups
        .iter()
        .map(|group| group.input_ids.len())
        .sum::<usize>();
    if available_input_count <= combination_size {
        let mut selected_ids = Vec::with_capacity(available_input_count);
        for group in sampled_groups {
            selected_ids.extend(group.input_ids.iter().copied());
        }
        return selected_ids;
    }

    let required_group_slots = sampled_groups.len().min(combination_size);
    let mut selected_ids = Vec::with_capacity(combination_size);
    let mut leftover_ids = Vec::with_capacity(available_input_count - required_group_slots);
    for (group_order, group) in sampled_groups.iter().enumerate() {
        let pick_indices = sample_unique_indices(group.input_ids.len(), 1, rng);
        if group_order < required_group_slots {
            if let Some(&picked_index) = pick_indices.first() {
                selected_ids.push(group.input_ids[picked_index]);
                for (input_idx, input_id) in group.input_ids.iter().enumerate() {
                    if input_idx != picked_index {
                        leftover_ids.push(*input_id);
                    }
                }
                continue;
            }
        }
        leftover_ids.extend(group.input_ids.iter().copied());
    }

    let remaining_slots = combination_size.saturating_sub(selected_ids.len());
    let leftover_indices = sample_unique_indices(leftover_ids.len(), remaining_slots, rng);
    for leftover_idx in leftover_indices {
        if let Some(input_id) = leftover_ids.get(leftover_idx) {
            selected_ids.push(*input_id);
        }
    }

    selected_ids
}
