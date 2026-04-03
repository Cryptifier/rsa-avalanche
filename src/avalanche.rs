use rayon::prelude::*;

use crate::helpers::hamming_distance_bits;

/// Errors returned by avalanche tree search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AvalancheError {
    EmptyCandidates,
    InconsistentBitWidth,
}

impl std::fmt::Display for AvalancheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AvalancheError::EmptyCandidates => write!(f, "no avalanche candidates provided"),
            AvalancheError::InconsistentBitWidth => {
                write!(f, "avalanche bit widths are inconsistent")
            }
        }
    }
}

impl std::error::Error for AvalancheError {}

/// Container for avalanche tree state.
#[derive(Debug, Clone)]
pub struct AvalancheNode {
    pub biases: Vec<f64>,
    pub message_bits: Vec<bool>,
}

/// Result of an avalanche tree search with per-level similarity scores.
#[derive(Debug, Clone)]
pub struct AvalancheSearchResult {
    pub node: AvalancheNode,
    pub level_similarity_pct: Vec<f64>,
    pub level_pair_counts: Vec<usize>,
}

/// Mirrors candidates with their bitwise inversions and sorts by Hamming distance.
///
/// # Parameters
/// - `candidates`: Candidate nodes to duplicate and sort.
/// - `reference_bits`: Reference bit vector used for distance ordering.
///
/// # Returns
/// - `Result<Vec<AvalancheNode>, AvalancheError>`: Original and inverted candidates sorted by distance.
///
/// # Expected Output
/// - Returns a sorted candidate list; no stdout/stderr output.
pub fn sort_candidates_by_hamming_distance(
    candidates: Vec<AvalancheNode>,
    reference_bits: &[bool],
) -> Result<Vec<AvalancheNode>, AvalancheError> {
    let bit_width = validate_candidates(&candidates)?;
    if reference_bits.len() != bit_width {
        return Err(AvalancheError::InconsistentBitWidth);
    }

    let mut mirrored = Vec::with_capacity(candidates.len() * 2);
    for candidate in candidates {
        mirrored.push(candidate.clone());
        mirrored.push(invert_candidate(&candidate));
    }

    mirrored.sort_by(|left, right| {
        hamming_distance_bits(&left.message_bits, reference_bits)
            .cmp(&hamming_distance_bits(&right.message_bits, reference_bits))
            .then_with(|| compare_message_bits_le(&left.message_bits, &right.message_bits))
    });
    Ok(mirrored)
}

/// Recursively reduces candidates by bitwise AND with bias accumulation.
///
/// # Parameters
/// - `candidates`: Candidate nodes with message bits and bias vectors.
///
/// # Returns
/// - `Result<AvalancheNode, AvalancheError>`: Final node after recursive reduction.
///
/// # Expected Output
/// - Returns the reduced node; no stdout/stderr output.
pub fn search_avalanche_tree(
    candidates: Vec<AvalancheNode>,
) -> Result<AvalancheNode, AvalancheError> {
    let bit_width = validate_candidates(&candidates)?;
    if candidates.len() == 1 {
        return Ok(candidates
            .into_iter()
            .next()
            .ok_or(AvalancheError::EmptyCandidates)?);
    }

    let next_level = build_next_level(&candidates, bit_width)?;
    search_avalanche_tree(next_level)
}

/// Recursively reduces candidates while computing per-level similarity scores.
///
/// # Parameters
/// - `candidates`: Candidate nodes with message bits and bias vectors.
///
/// # Returns
/// - `Result<AvalancheSearchResult, AvalancheError>`: Final node and per-level scores.
///
/// # Expected Output
/// - Returns the reduced node and similarity data; no stdout/stderr output.
pub fn search_avalanche_tree_with_scores(
    candidates: Vec<AvalancheNode>,
) -> Result<AvalancheSearchResult, AvalancheError> {
    let bit_width = validate_candidates(&candidates)?;
    if candidates.len() == 1 {
        return Ok(AvalancheSearchResult {
            node: candidates
                .into_iter()
                .next()
                .ok_or(AvalancheError::EmptyCandidates)?,
            level_similarity_pct: Vec::new(),
            level_pair_counts: Vec::new(),
        });
    }

    let mut current = candidates;
    let mut level_similarity_pct = Vec::new();
    let mut level_pair_counts = Vec::new();

    while current.len() > 1 {
        let (next, level_match_weight, level_weight, pair_count) =
            build_next_level_with_similarity(&current, bit_width)?;
        if level_weight > 0.0 {
            level_similarity_pct.push(level_match_weight / level_weight * 100.0);
            level_pair_counts.push(pair_count);
        }
        current = next;
    }

    let node = current
        .into_iter()
        .next()
        .ok_or(AvalancheError::EmptyCandidates)?;
    Ok(AvalancheSearchResult {
        node,
        level_similarity_pct,
        level_pair_counts,
    })
}

/// Validates that candidates are non-empty and consistent in bit width.
///
/// # Parameters
/// - `candidates`: Candidate nodes to validate.
///
/// # Returns
/// - `Result<usize, AvalancheError>`: Shared bit width on success.
///
/// # Expected Output
/// - Returns the bit width when valid; no stdout/stderr output.
fn validate_candidates(candidates: &[AvalancheNode]) -> Result<usize, AvalancheError> {
    if candidates.is_empty() {
        return Err(AvalancheError::EmptyCandidates);
    }
    let bit_width = candidates[0].message_bits.len();
    if bit_width == 0 {
        return Err(AvalancheError::InconsistentBitWidth);
    }
    for candidate in candidates {
        if candidate.message_bits.len() != bit_width || candidate.biases.len() != bit_width {
            return Err(AvalancheError::InconsistentBitWidth);
        }
    }
    Ok(bit_width)
}

/// Builds the bitwise inversion of a candidate node.
///
/// # Parameters
/// - `candidate`: Candidate node to invert.
///
/// # Returns
/// - `AvalancheNode`: Inverted node with the same bias vector.
///
/// # Expected Output
/// - Returns the inverted node; no stdout/stderr output.
fn invert_candidate(candidate: &AvalancheNode) -> AvalancheNode {
    AvalancheNode {
        biases: candidate.biases.clone(),
        message_bits: candidate.message_bits.iter().map(|bit| !*bit).collect(),
    }
}

/// Compares little-endian bit vectors by their numeric value.
///
/// # Parameters
/// - `left`: Left-hand little-endian bit vector.
/// - `right`: Right-hand little-endian bit vector.
///
/// # Returns
/// - `std::cmp::Ordering`: Ordering of the represented integer values.
///
/// # Expected Output
/// - Returns the numeric ordering; no stdout/stderr output.
fn compare_message_bits_le(left: &[bool], right: &[bool]) -> std::cmp::Ordering {
    for idx in (0..left.len()).rev() {
        let ordering = left[idx].cmp(&right[idx]);
        if ordering != std::cmp::Ordering::Equal {
            return ordering;
        }
    }
    std::cmp::Ordering::Equal
}

/// Builds the next avalanche level by pairing most-similar candidates.
///
/// # Parameters
/// - `candidates`: Current level of avalanche nodes.
/// - `bit_width`: Expected bit width for all nodes.
///
/// # Returns
/// - `Result<Vec<AvalancheNode>, AvalancheError>`: Next-level candidates.
///
/// # Expected Output
/// - Returns a reduced candidate list; no stdout/stderr output.
fn build_next_level(
    candidates: &[AvalancheNode],
    bit_width: usize,
) -> Result<Vec<AvalancheNode>, AvalancheError> {
    let mut next = Vec::with_capacity((candidates.len() + 1) / 2);
    let mut used = vec![false; candidates.len()];
    for idx in 0..candidates.len() {
        if used[idx] {
            continue;
        }
        let best_partner = (idx + 1..candidates.len())
            .into_par_iter()
            .filter(|other| !used[*other])
            .map(|other| {
                let distance = hamming_distance_bits(
                    &candidates[idx].message_bits,
                    &candidates[other].message_bits,
                );
                (distance, other)
            })
            .reduce_with(|a, b| {
                if a.0 < b.0 || (a.0 == b.0 && a.1 < b.1) {
                    a
                } else {
                    b
                }
            })
            .map(|(_, other)| other);
        if let Some(other) = best_partner {
            used[idx] = true;
            used[other] = true;
            let combined = combine_candidates(&candidates[idx], &candidates[other], bit_width)?;
            next.push(combined);
        } else {
            used[idx] = true;
            next.push(candidates[idx].clone());
        }
    }
    Ok(next)
}

/// Builds the next avalanche level and computes weighted similarity totals.
///
/// # Parameters
/// - `candidates`: Current level of avalanche nodes.
/// - `bit_width`: Expected bit width for all nodes.
///
/// # Returns
/// - `Result<(Vec<AvalancheNode>, f64, f64, usize), AvalancheError>`: Next-level nodes, match-weight sum, weight sum, and pair count.
///
/// # Expected Output
/// - Returns reduced candidates and similarity totals; no stdout/stderr output.
fn build_next_level_with_similarity(
    candidates: &[AvalancheNode],
    bit_width: usize,
) -> Result<(Vec<AvalancheNode>, f64, f64, usize), AvalancheError> {
    let mut next = Vec::with_capacity((candidates.len() + 1) / 2);
    let mut used = vec![false; candidates.len()];
    let mut match_weight_sum = 0.0f64;
    let mut weight_sum = 0.0f64;
    let mut pair_count = 0usize;

    for idx in 0..candidates.len() {
        if used[idx] {
            continue;
        }
        let best_partner = (idx + 1..candidates.len())
            .into_par_iter()
            .filter(|other| !used[*other])
            .map(|other| {
                let distance = hamming_distance_bits(
                    &candidates[idx].message_bits,
                    &candidates[other].message_bits,
                );
                (distance, other)
            })
            .reduce_with(|a, b| {
                if a.0 < b.0 || (a.0 == b.0 && a.1 < b.1) {
                    a
                } else {
                    b
                }
            })
            .map(|(_, other)| other);
        if let Some(other) = best_partner {
            used[idx] = true;
            used[other] = true;
            let (pair_match_weight, pair_weight) =
                weighted_similarity(&candidates[idx], &candidates[other], bit_width)?;
            match_weight_sum += pair_match_weight;
            weight_sum += pair_weight;
            pair_count += 1;
            let combined = combine_candidates(&candidates[idx], &candidates[other], bit_width)?;
            next.push(combined);
        } else {
            used[idx] = true;
            next.push(candidates[idx].clone());
        }
    }

    Ok((next, match_weight_sum, weight_sum, pair_count))
}

/// Computes weighted similarity between two candidates using bias magnitudes.
///
/// # Parameters
/// - `left`: Left-hand candidate node.
/// - `right`: Right-hand candidate node.
/// - `bit_width`: Expected bit width for the nodes.
///
/// # Returns
/// - `Result<(f64, f64), AvalancheError>`: Match-weight sum and total weight.
///
/// # Expected Output
/// - Returns weighted totals; no stdout/stderr output.
fn weighted_similarity(
    left: &AvalancheNode,
    right: &AvalancheNode,
    bit_width: usize,
) -> Result<(f64, f64), AvalancheError> {
    if left.message_bits.len() != bit_width
        || right.message_bits.len() != bit_width
        || left.biases.len() != bit_width
        || right.biases.len() != bit_width
    {
        return Err(AvalancheError::InconsistentBitWidth);
    }

    let mut match_weight = 0.0f64;
    let mut weight_sum = 0.0f64;
    for idx in 0..bit_width {
        let weight = 1.0 + left.biases[idx].abs() + right.biases[idx].abs();
        weight_sum += weight;
        if left.message_bits[idx] == right.message_bits[idx] {
            match_weight += weight;
        }
    }
    Ok((match_weight, weight_sum))
}

/// Combines two nodes by AND-ing bits and adding bias for `true` positions.
///
/// # Parameters
/// - `left`: Left-hand candidate node.
/// - `right`: Right-hand candidate node.
/// - `bit_width`: Expected bit width for the nodes.
///
/// # Returns
/// - `Result<AvalancheNode, AvalancheError>`: Combined node.
///
/// # Expected Output
/// - Returns a combined node; no stdout/stderr output.
fn combine_candidates(
    left: &AvalancheNode,
    right: &AvalancheNode,
    bit_width: usize,
) -> Result<AvalancheNode, AvalancheError> {
    if left.message_bits.len() != bit_width
        || right.message_bits.len() != bit_width
        || left.biases.len() != bit_width
        || right.biases.len() != bit_width
    {
        return Err(AvalancheError::InconsistentBitWidth);
    }

    let mut message_bits = Vec::with_capacity(bit_width);
    let mut biases = Vec::with_capacity(bit_width);
    for idx in 0..bit_width {
        let and_bit = left.message_bits[idx] & right.message_bits[idx];
        let bias = if and_bit {
            let sum = left.biases[idx] + right.biases[idx];
            if sum == 0.0 { 1.0 } else { sum }
        } else {
            (left.biases[idx] - right.biases[idx]).abs()
        };
        message_bits.push(and_bit);
        biases.push(bias);
    }

    Ok(AvalancheNode {
        biases,
        message_bits,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        AvalancheError, AvalancheNode, search_avalanche_tree, search_avalanche_tree_with_scores,
        sort_candidates_by_hamming_distance,
    };
    use insta::assert_yaml_snapshot;
    use serde_json::json;

    fn node(bits: &[bool], biases: &[f64]) -> AvalancheNode {
        AvalancheNode {
            biases: biases.to_vec(),
            message_bits: bits.to_vec(),
        }
    }

    #[test]
    fn test_avalanche_pair_bias_rule() {
        let candidates = vec![
            node(&[true, false, true], &[0.0, 0.5, 1.0]),
            node(&[true, true, false], &[0.0, 1.5, 2.0]),
        ];
        let result = search_avalanche_tree(candidates).expect("avalanche tree failed");
        let snapshot = json!({
            "message_bits": result.message_bits,
            "biases": result.biases,
        });
        assert_yaml_snapshot!(snapshot);
    }

    #[test]
    fn test_avalanche_recursive_four() {
        let candidates = vec![
            node(&[true, true], &[1.0, 0.0]),
            node(&[true, false], &[2.0, 1.0]),
            node(&[false, true], &[3.0, 2.0]),
            node(&[true, true], &[4.0, 3.0]),
        ];
        let result = search_avalanche_tree(candidates).expect("avalanche tree failed");
        let snapshot = json!({
            "message_bits": result.message_bits,
            "biases": result.biases,
        });
        assert_yaml_snapshot!(snapshot);
    }

    #[test]
    fn test_avalanche_odd_count() {
        let candidates = vec![
            node(&[true, false], &[0.0, 2.0]),
            node(&[true, true], &[0.0, 1.0]),
            node(&[false, true], &[5.0, 5.0]),
        ];
        let result = search_avalanche_tree(candidates).expect("avalanche tree failed");
        let snapshot = json!({
            "message_bits": result.message_bits,
            "biases": result.biases,
        });
        assert_yaml_snapshot!(snapshot);
    }

    #[test]
    fn test_avalanche_inconsistent_width() {
        let candidates = vec![node(&[true, false], &[1.0, 2.0]), node(&[true], &[1.0])];
        let err = search_avalanche_tree(candidates).expect_err("expected error");
        let snapshot = json!({ "error": err.to_string() });
        assert_yaml_snapshot!(snapshot);
    }

    #[test]
    fn test_avalanche_similarity_two_nodes() {
        let candidates = vec![
            node(&[true, false, true], &[0.0, 0.5, 1.0]),
            node(&[true, true, false], &[0.0, 1.5, 2.0]),
        ];
        let result = search_avalanche_tree_with_scores(candidates)
            .expect("avalanche tree with scores failed");
        let snapshot = json!({
            "level_similarity_pct": result.level_similarity_pct,
            "level_pair_counts": result.level_pair_counts,
        });
        assert_yaml_snapshot!(snapshot);
    }

    #[test]
    fn test_avalanche_similarity_recursive_four() {
        let candidates = vec![
            node(&[true, true], &[1.0, 0.0]),
            node(&[true, false], &[2.0, 1.0]),
            node(&[false, true], &[3.0, 2.0]),
            node(&[true, true], &[4.0, 3.0]),
        ];
        let result = search_avalanche_tree_with_scores(candidates)
            .expect("avalanche tree with scores failed");
        let snapshot = json!({
            "level_similarity_pct": result.level_similarity_pct,
            "level_pair_counts": result.level_pair_counts,
        });
        assert_yaml_snapshot!(snapshot);
    }

    #[test]
    fn test_sort_candidates_by_hamming_distance_mirrors_inversions() {
        let candidates = vec![
            node(&[true, false], &[0.5, 1.5]),
            node(&[false, false], &[2.0, 3.0]),
        ];
        let sorted = sort_candidates_by_hamming_distance(candidates, &[true, true])
            .expect("distance sort failed");

        let bits: Vec<Vec<bool>> = sorted
            .iter()
            .map(|node| node.message_bits.clone())
            .collect();
        let biases: Vec<Vec<f64>> = sorted.iter().map(|node| node.biases.clone()).collect();

        assert_eq!(
            bits,
            vec![
                vec![true, true],
                vec![true, false],
                vec![false, true],
                vec![false, false],
            ]
        );
        assert_eq!(
            biases,
            vec![
                vec![2.0, 3.0],
                vec![0.5, 1.5],
                vec![0.5, 1.5],
                vec![2.0, 3.0],
            ]
        );
    }

    #[test]
    fn test_sort_candidates_by_hamming_distance_rejects_reference_width_mismatch() {
        let candidates = vec![node(&[true, false], &[1.0, 2.0])];
        let err = sort_candidates_by_hamming_distance(candidates, &[true])
            .expect_err("expected width mismatch");
        assert_eq!(err, AvalancheError::InconsistentBitWidth);
    }
}
