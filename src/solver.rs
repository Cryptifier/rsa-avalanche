/// Eclipse Public License 2.0
/// SPDX-License-Identifier: EPL-2.0
/// Copyright (c) 2025 Nicholas LaRoche <nlaroche@cryptifier.dev>
use std::{
    error::Error,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use rayon::prelude::*;

use crate::{
    config::EngineConfig,
    methods::{log_parallel_progress_every_interval, parallel_progress_chunk_size},
};

const GREEN_ANSI: &str = "\u{1b}[32m";
const RESET_ANSI: &str = "\u{1b}[0m";

/// Stable output marker used by scripts to detect solver success.
pub(crate) const AVALANCHE_SOLVER_SUCCESS_MARKER: &str = "AVALANCHE SOLVER FOUND MESSAGE";

/// Final-tier sample payload carried into the cross-batch Avalanche solver.
#[derive(Clone, Debug)]
pub(crate) struct AvalancheSolverSample {
    pub(crate) batch_number: usize,
    pub(crate) tier_index: usize,
    pub(crate) sample_index: usize,
    pub(crate) bits: Vec<bool>,
    pub(crate) majority_vote_bits: Vec<bool>,
}

/// Per-batch final-tier data consumed by the cross-batch Avalanche solver.
#[derive(Clone, Debug)]
pub(crate) struct AvalancheSolverBatch {
    pub(crate) batch_number: usize,
    pub(crate) message_bits: Vec<bool>,
    pub(crate) samples: Vec<AvalancheSolverSample>,
}

/// Exact whole-message recovery reported by the cross-batch Avalanche solver.
#[derive(Clone, Debug)]
pub(crate) struct AvalancheSolverFoundCandidate {
    pub(crate) first_batch_number: usize,
    pub(crate) first_tier_index: usize,
    pub(crate) first_sample_index: usize,
    pub(crate) second_batch_number: Option<usize>,
    pub(crate) second_tier_index: Option<usize>,
    pub(crate) second_sample_index: Option<usize>,
    pub(crate) flipped_bit_positions: Vec<usize>,
}

/// Aggregate outcome from one cross-batch Avalanche solver execution.
#[derive(Clone, Debug, Default)]
pub(crate) struct AvalancheSolverResult {
    pub(crate) enabled: bool,
    pub(crate) batch_pair_count: usize,
    pub(crate) sample_pair_count: usize,
    pub(crate) attempted_candidates: u64,
    pub(crate) max_bits: usize,
    pub(crate) found: Option<AvalancheSolverFoundCandidate>,
}

/// Runs the cross-batch Avalanche solver against final-tier sample outputs.
///
/// # Parameters
/// - `engine`: Engine configuration containing the solver enable flag and brute-force bit cap.
/// - `batches`: Final-tier Avalanche samples grouped by analysis batch.
///
/// # Returns
/// - `Result<AvalancheSolverResult, Box<dyn Error>>`: Solver summary including whether an exact whole-message match was found.
///
/// # Expected Output
/// - Prints solver progress every 5 seconds and emits a green success line when the original message is recovered.
pub(crate) fn run_avalanche_solver(
    engine: &EngineConfig,
    batches: &[AvalancheSolverBatch],
) -> Result<AvalancheSolverResult, Box<dyn Error>> {
    if !engine.avalanche_solver_enable {
        return Ok(AvalancheSolverResult::default());
    }
    if batches.len() < 2 {
        return Err("avalanche_solver_enable requires at least 2 analysis batches".into());
    }

    let target_message_bits = validate_solver_batches(batches)?;
    let batch_pairs = build_batch_pairs(batches.len());
    let sample_pair_count = count_sample_pairs(batches, &batch_pairs);
    let mut result = AvalancheSolverResult {
        enabled: true,
        batch_pair_count: batch_pairs.len(),
        sample_pair_count,
        attempted_candidates: 0,
        max_bits: engine.avalanche_solver_max_bits,
        found: None,
    };

    println!(
        "Avalanche solver: checking {} batch pairs across {} final-tier sample pairs with max-bits {}",
        result.batch_pair_count, result.sample_pair_count, result.max_bits
    );

    let direct_attempts = AtomicU64::new(0);
    if let Some(found) = find_direct_sample_match(batches, target_message_bits, &direct_attempts) {
        result.attempted_candidates = direct_attempts.load(Ordering::Relaxed);
        result.found = Some(found.clone());
        log_solver_success(&found);
        return Ok(result);
    }

    let total_sample_pairs =
        u64::try_from(sample_pair_count).map_err(|_| "solver sample pair count exceeds u64")?;
    if total_sample_pairs == 0 {
        result.attempted_candidates = direct_attempts.load(Ordering::Relaxed);
        println!(
            "Avalanche solver: no final-tier sample pairs were available for batch-pair comparison"
        );
        return Ok(result);
    }

    let processed_sample_pairs = AtomicU64::new(0);
    let attempted_candidates = AtomicU64::new(direct_attempts.load(Ordering::Relaxed));
    let started_at = Instant::now();
    let next_log_at_ms =
        AtomicU64::new(Duration::from_secs(5).as_millis().min(u128::from(u64::MAX)) as u64);
    let found_any = Arc::new(AtomicBool::new(false));
    let chunk_size = parallel_progress_chunk_size(batch_pairs.len());

    let found = batch_pairs
        .par_chunks(chunk_size)
        .find_map_any(|batch_pair_chunk| {
            for &(first_idx, second_idx) in batch_pair_chunk {
                if found_any.load(Ordering::Relaxed) {
                    return None;
                }
                let found = search_batch_pair_for_message(
                    &batches[first_idx],
                    &batches[second_idx],
                    target_message_bits,
                    engine.avalanche_solver_max_bits,
                    total_sample_pairs,
                    &processed_sample_pairs,
                    &attempted_candidates,
                    &started_at,
                    &next_log_at_ms,
                    found_any.as_ref(),
                );
                if found.is_some() {
                    return found;
                }
            }
            None
        });

    result.attempted_candidates = attempted_candidates.load(Ordering::Relaxed);
    if let Some(found) = found {
        log_solver_success(&found);
        result.found = Some(found);
    } else {
        println!(
            "Avalanche solver: no whole-message match found after {} batch pairs, {} sample pairs, and {} brute-force attempts",
            result.batch_pair_count, result.sample_pair_count, result.attempted_candidates
        );
    }

    Ok(result)
}

/// Validates that every solver batch targets the same original message bits.
///
/// # Parameters
/// - `batches`: Final-tier Avalanche samples grouped by analysis batch.
///
/// # Returns
/// - `Result<&[bool], Box<dyn Error>>`: Shared target message bits used for exact solver comparisons.
///
/// # Expected Output
/// - Returns an error when cross-batch solver inputs do not all target the same message.
fn validate_solver_batches(batches: &[AvalancheSolverBatch]) -> Result<&[bool], Box<dyn Error>> {
    let target_message_bits = batches
        .first()
        .map(|batch| batch.message_bits.as_slice())
        .ok_or("solver requires at least one batch")?;
    for sample in &batches[0].samples {
        if sample.bits.len() != target_message_bits.len() {
            return Err(
                "avalanche solver sample width must match the original message bit width".into(),
            );
        }
    }
    for batch in batches.iter().skip(1) {
        if batch.message_bits != target_message_bits {
            return Err(
                "avalanche_solver_enable requires every batch to target the same original message"
                    .into(),
            );
        }
        for sample in &batch.samples {
            if sample.bits.len() != target_message_bits.len() {
                return Err(
                    "avalanche solver sample width must match the original message bit width"
                        .into(),
                );
            }
        }
    }
    Ok(target_message_bits)
}

/// Builds the unordered batch-pair index list used by the solver.
///
/// # Parameters
/// - `batch_count`: Number of available analysis batches.
///
/// # Returns
/// - `Vec<(usize, usize)>`: Unordered `(first_batch_idx, second_batch_idx)` pairs.
///
/// # Expected Output
/// - Returns index pairs without mutating solver state.
fn build_batch_pairs(batch_count: usize) -> Vec<(usize, usize)> {
    let mut pairs =
        Vec::with_capacity(batch_count.saturating_mul(batch_count.saturating_sub(1)) / 2);
    for first_idx in 0..batch_count {
        for second_idx in (first_idx + 1)..batch_count {
            pairs.push((first_idx, second_idx));
        }
    }
    pairs
}

/// Counts the total cartesian-product sample pairs across all selected batch pairs.
///
/// # Parameters
/// - `batches`: Final-tier Avalanche samples grouped by analysis batch.
/// - `batch_pairs`: Unordered batch-pair indices to compare.
///
/// # Returns
/// - `usize`: Total number of final-tier sample pairs that will be checked.
///
/// # Expected Output
/// - Returns a computed count without mutating solver state.
fn count_sample_pairs(batches: &[AvalancheSolverBatch], batch_pairs: &[(usize, usize)]) -> usize {
    batch_pairs
        .iter()
        .map(|&(first_idx, second_idx)| {
            batches[first_idx]
                .samples
                .len()
                .saturating_mul(batches[second_idx].samples.len())
        })
        .sum()
}

/// Checks every raw final-tier sample for an exact whole-message match before brute forcing.
///
/// # Parameters
/// - `batches`: Final-tier Avalanche samples grouped by analysis batch.
/// - `target_message_bits`: Original message bits used for exact solver comparisons.
/// - `attempted_candidates`: Counter incremented once per direct exact-match check.
///
/// # Returns
/// - `Option<AvalancheSolverFoundCandidate>`: Direct exact-match result when any final-tier sample already equals the original message.
///
/// # Expected Output
/// - Updates the attempt counter and returns the first exact whole-message match found.
fn find_direct_sample_match(
    batches: &[AvalancheSolverBatch],
    target_message_bits: &[bool],
    attempted_candidates: &AtomicU64,
) -> Option<AvalancheSolverFoundCandidate> {
    for batch in batches {
        for sample in &batch.samples {
            attempted_candidates.fetch_add(1, Ordering::Relaxed);
            if sample.bits == target_message_bits {
                return Some(AvalancheSolverFoundCandidate {
                    first_batch_number: sample.batch_number,
                    first_tier_index: sample.tier_index,
                    first_sample_index: sample.sample_index,
                    second_batch_number: None,
                    second_tier_index: None,
                    second_sample_index: None,
                    flipped_bit_positions: Vec::new(),
                });
            }
        }
    }
    None
}

/// Searches one batch pair for an exact whole-message recovery.
///
/// # Parameters
/// - `first_batch`: First batch contributing the base sample that may be bit-flipped.
/// - `second_batch`: Second batch whose differing bits constrain the brute-force search.
/// - `target_message_bits`: Original message bits used for exact solver comparisons.
/// - `max_bits`: Maximum number of differing bit positions that may be flipped per attempt.
/// - `total_sample_pairs`: Total sample-pair count across the full solver run.
/// - `processed_sample_pairs`: Shared progress counter incremented once per completed sample pair.
/// - `attempted_candidates`: Shared attempt counter incremented once per brute-force candidate tested.
/// - `started_at`: Solver start time used for 5-second progress logs.
/// - `next_log_at_ms`: Next progress-log deadline in elapsed milliseconds.
/// - `found_any`: Shared stop flag used to short-circuit other worker threads after success.
///
/// # Returns
/// - `Option<AvalancheSolverFoundCandidate>`: Exact whole-message recovery when this batch pair yields one.
///
/// # Expected Output
/// - Updates progress and attempt counters; does not print success output directly.
fn search_batch_pair_for_message(
    first_batch: &AvalancheSolverBatch,
    second_batch: &AvalancheSolverBatch,
    target_message_bits: &[bool],
    max_bits: usize,
    total_sample_pairs: u64,
    processed_sample_pairs: &AtomicU64,
    attempted_candidates: &AtomicU64,
    started_at: &Instant,
    next_log_at_ms: &AtomicU64,
    found_any: &AtomicBool,
) -> Option<AvalancheSolverFoundCandidate> {
    for first_sample in &first_batch.samples {
        if found_any.load(Ordering::Relaxed) {
            return None;
        }
        for second_sample in &second_batch.samples {
            if found_any.load(Ordering::Relaxed) {
                return None;
            }

            let diff_positions = differing_bit_positions(&first_sample.bits, &second_sample.bits);
            let search_outcome = brute_force_message_match(
                &first_sample.bits,
                &diff_positions,
                target_message_bits,
                max_bits,
            );
            attempted_candidates.fetch_add(search_outcome.attempted_candidates, Ordering::Relaxed);

            let done = processed_sample_pairs.fetch_add(1, Ordering::Relaxed) + 1;
            log_parallel_progress_every_interval(
                done,
                total_sample_pairs,
                started_at,
                next_log_at_ms,
                "Avalanche solver cartesian product",
                Duration::from_secs(5),
            );

            if let Some(flipped_bit_positions) = search_outcome.flipped_bit_positions {
                found_any.store(true, Ordering::Relaxed);
                return Some(AvalancheSolverFoundCandidate {
                    first_batch_number: first_batch.batch_number,
                    first_tier_index: first_sample.tier_index,
                    first_sample_index: first_sample.sample_index,
                    second_batch_number: Some(second_batch.batch_number),
                    second_tier_index: Some(second_sample.tier_index),
                    second_sample_index: Some(second_sample.sample_index),
                    flipped_bit_positions,
                });
            }
        }
    }

    None
}

#[derive(Debug, Default)]
struct BruteForceSearchOutcome {
    attempted_candidates: u64,
    flipped_bit_positions: Option<Vec<usize>>,
}

/// Collects the differing bit positions between two final-tier samples.
///
/// # Parameters
/// - `first_bits`: Base sample bits that may be bit-flipped by the solver.
/// - `second_bits`: Comparison sample bits used to constrain the flip-position search.
///
/// # Returns
/// - `Vec<usize>`: Bit positions where the two samples differ.
///
/// # Expected Output
/// - Returns the differing bit positions without mutating solver state.
fn differing_bit_positions(first_bits: &[bool], second_bits: &[bool]) -> Vec<usize> {
    first_bits
        .iter()
        .zip(second_bits.iter())
        .enumerate()
        .filter_map(|(bit_idx, (first_bit, second_bit))| {
            (first_bit != second_bit).then_some(bit_idx)
        })
        .collect()
}

/// Brute-forces bit-flip combinations from one sample pair to test for an exact message match.
///
/// # Parameters
/// - `base_bits`: Base sample bits to mutate by flipping differing positions.
/// - `diff_positions`: Bit positions where the paired final-tier samples disagree.
/// - `target_message_bits`: Original message bits used for exact whole-message comparisons.
/// - `max_bits`: Maximum number of differing positions that may be flipped in one candidate.
///
/// # Returns
/// - `BruteForceSearchOutcome`: Attempt count plus the first matching flip set when found.
///
/// # Expected Output
/// - Returns brute-force search results without printing or mutating shared solver state.
fn brute_force_message_match(
    base_bits: &[bool],
    diff_positions: &[usize],
    target_message_bits: &[bool],
    max_bits: usize,
) -> BruteForceSearchOutcome {
    if diff_positions.is_empty() || max_bits == 0 {
        return BruteForceSearchOutcome::default();
    }

    let search_limit = max_bits.min(diff_positions.len());
    let mut candidate_bits = base_bits.to_vec();
    let mut selected_positions = Vec::with_capacity(search_limit);
    let mut attempted_candidates = 0u64;
    let flipped_bit_positions = search_flip_combinations(
        diff_positions,
        target_message_bits,
        &mut candidate_bits,
        &mut selected_positions,
        0,
        search_limit,
        &mut attempted_candidates,
    );

    BruteForceSearchOutcome {
        attempted_candidates,
        flipped_bit_positions,
    }
}

/// Recursively enumerates bit-flip combinations until an exact whole-message match is found.
///
/// # Parameters
/// - `diff_positions`: Bit positions where the paired final-tier samples disagree.
/// - `target_message_bits`: Original message bits used for exact whole-message comparisons.
/// - `candidate_bits`: Mutable working buffer derived from the first batch sample.
/// - `selected_positions`: Flip positions currently enabled in `candidate_bits`.
/// - `start_idx`: Starting offset into `diff_positions` for this recursion branch.
/// - `max_bits`: Maximum number of differing positions that may be flipped in one candidate.
/// - `attempted_candidates`: Counter incremented once per tested bit-flip combination.
///
/// # Returns
/// - `Option<Vec<usize>>`: Exact flip positions that recover the original message, if found.
///
/// # Expected Output
/// - Mutates and restores `candidate_bits` while exploring combinations; does not print output.
fn search_flip_combinations(
    diff_positions: &[usize],
    target_message_bits: &[bool],
    candidate_bits: &mut [bool],
    selected_positions: &mut Vec<usize>,
    start_idx: usize,
    max_bits: usize,
    attempted_candidates: &mut u64,
) -> Option<Vec<usize>> {
    if !selected_positions.is_empty() {
        *attempted_candidates += 1;
        if candidate_bits == target_message_bits {
            return Some(selected_positions.clone());
        }
    }
    if selected_positions.len() == max_bits {
        return None;
    }

    for diff_idx in start_idx..diff_positions.len() {
        let bit_position = diff_positions[diff_idx];
        candidate_bits[bit_position] = !candidate_bits[bit_position];
        selected_positions.push(bit_position);

        if let Some(found_positions) = search_flip_combinations(
            diff_positions,
            target_message_bits,
            candidate_bits,
            selected_positions,
            diff_idx + 1,
            max_bits,
            attempted_candidates,
        ) {
            return Some(found_positions);
        }

        selected_positions.pop();
        candidate_bits[bit_position] = !candidate_bits[bit_position];
    }

    None
}

/// Prints the stable green success line used by scripts to detect solver recovery.
///
/// # Parameters
/// - `found`: Exact whole-message recovery reported by the solver.
///
/// # Returns
/// - `()`
///
/// # Expected Output
/// - Prints one green solver success line to stdout.
fn log_solver_success(found: &AvalancheSolverFoundCandidate) {
    match (
        found.second_batch_number,
        found.second_tier_index,
        found.second_sample_index,
    ) {
        (Some(second_batch_number), Some(second_tier_index), Some(second_sample_index)) => {
            println!(
                "{GREEN_ANSI}{AVALANCHE_SOLVER_SUCCESS_MARKER}: batch {} tier {} sample {} + batch {} tier {} sample {} flips {:?}{RESET_ANSI}",
                found.first_batch_number,
                found.first_tier_index,
                found.first_sample_index,
                second_batch_number,
                second_tier_index,
                second_sample_index,
                found.flipped_bit_positions
            );
        }
        _ => {
            println!(
                "{GREEN_ANSI}{AVALANCHE_SOLVER_SUCCESS_MARKER}: batch {} tier {} sample {} flips {:?}{RESET_ANSI}",
                found.first_batch_number,
                found.first_tier_index,
                found.first_sample_index,
                found.flipped_bit_positions
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;

    fn sample(batch_number: usize, sample_index: usize, bits: &[bool]) -> AvalancheSolverSample {
        AvalancheSolverSample {
            batch_number,
            tier_index: 3,
            sample_index,
            bits: bits.to_vec(),
            majority_vote_bits: bits.to_vec(),
        }
    }

    fn batch(
        batch_number: usize,
        message_bits: &[bool],
        samples: Vec<AvalancheSolverSample>,
    ) -> AvalancheSolverBatch {
        AvalancheSolverBatch {
            batch_number,
            message_bits: message_bits.to_vec(),
            samples,
        }
    }

    #[test]
    fn test_run_avalanche_solver_finds_direct_match_without_flips() {
        let mut engine = EngineConfig::default();
        engine.avalanche_solver_enable = true;
        engine.avalanche_solver_max_bits = 2;
        let message_bits = vec![true, false, true, false];
        let batches = vec![
            batch(
                1,
                &message_bits,
                vec![sample(1, 0, &[true, false, false, false])],
            ),
            batch(
                2,
                &message_bits,
                vec![sample(2, 0, &[true, false, true, false])],
            ),
        ];

        let result = run_avalanche_solver(&engine, &batches).expect("solver result");

        assert!(result.enabled);
        assert_eq!(result.batch_pair_count, 1);
        assert_eq!(result.sample_pair_count, 1);
        let found = result.found.expect("direct match should be found");
        assert_eq!(found.first_batch_number, 2);
        assert!(found.flipped_bit_positions.is_empty());
    }

    #[test]
    fn test_run_avalanche_solver_bruteforces_pair_difference_match() {
        let mut engine = EngineConfig::default();
        engine.avalanche_solver_enable = true;
        engine.avalanche_solver_max_bits = 2;
        let message_bits = vec![true, false, true, false];
        let batches = vec![
            batch(
                1,
                &message_bits,
                vec![sample(1, 0, &[true, true, false, false])],
            ),
            batch(
                2,
                &message_bits,
                vec![sample(2, 0, &[false, false, true, false])],
            ),
        ];

        let result = run_avalanche_solver(&engine, &batches).expect("solver result");

        let found = result
            .found
            .expect("pairwise brute force should find a match");
        assert_eq!(found.first_batch_number, 1);
        assert_eq!(found.second_batch_number, Some(2));
        assert_eq!(found.flipped_bit_positions, vec![1, 2]);
        assert!(result.attempted_candidates >= 3);
    }

    #[test]
    fn test_run_avalanche_solver_respects_max_bits_limit() {
        let mut engine = EngineConfig::default();
        engine.avalanche_solver_enable = true;
        engine.avalanche_solver_max_bits = 1;
        let message_bits = vec![true, false, true, false];
        let batches = vec![
            batch(
                1,
                &message_bits,
                vec![sample(1, 0, &[true, true, false, false])],
            ),
            batch(
                2,
                &message_bits,
                vec![sample(2, 0, &[false, false, true, false])],
            ),
        ];

        let result = run_avalanche_solver(&engine, &batches).expect("solver result");

        assert!(result.found.is_none());
    }

    #[test]
    fn test_run_avalanche_solver_rejects_mismatched_batch_messages() {
        let mut engine = EngineConfig::default();
        engine.avalanche_solver_enable = true;
        engine.avalanche_solver_max_bits = 2;
        let error = run_avalanche_solver(
            &engine,
            &[
                batch(
                    1,
                    &[true, false, true, false],
                    vec![sample(1, 0, &[true, false, true, false])],
                ),
                batch(
                    2,
                    &[true, true, true, false],
                    vec![sample(2, 0, &[true, false, true, false])],
                ),
            ],
        )
        .expect_err("mismatched batch messages should fail");

        assert!(
            error
                .to_string()
                .contains("requires every batch to target the same original message")
        );
    }
}
