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
            AvalancheError::InconsistentBitWidth => write!(f, "avalanche bit widths are inconsistent"),
        }
    }
}

impl std::error::Error for AvalancheError {}

use rayon::prelude::*;

/// Container for avalanche tree state.
#[derive(Debug, Clone)]
pub struct AvalancheNode {
    pub biases: Vec<f64>,
    pub message_bits: Vec<bool>,
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

/// Computes the Hamming distance between two bit slices.
///
/// # Parameters
/// - `left`: First bit slice.
/// - `right`: Second bit slice.
///
/// # Returns
/// - `usize`: Count of differing bit positions.
///
/// # Expected Output
/// - Returns the distance; no side effects.
fn hamming_distance_bits(left: &[bool], right: &[bool]) -> usize {
    left.iter()
        .zip(right.iter())
        .filter(|(a, b)| a != b)
        .count()
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

    Ok(AvalancheNode { biases, message_bits })
}

#[cfg(test)]
mod tests {
    use super::{search_avalanche_tree, AvalancheNode};
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
}
