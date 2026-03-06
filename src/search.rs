/// Errors returned by `beam_search` for invalid inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeamSearchError {
    EmptyInitial,
    ZeroBeamWidth,
    ZeroMaxSteps,
}

impl std::fmt::Display for BeamSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BeamSearchError::EmptyInitial => write!(f, "initial beam cannot be empty"),
            BeamSearchError::ZeroBeamWidth => write!(f, "beam width must be >= 1"),
            BeamSearchError::ZeroMaxSteps => write!(f, "max steps must be >= 1"),
        }
    }
}

impl std::error::Error for BeamSearchError {}

/// Summary statistics for a completed beam search.
#[derive(Debug, Clone)]
pub struct BeamSearchResult {
    pub best_vector: Vec<f64>,
    pub best_score: f64,
    pub steps: usize,
    pub evaluated: usize,
}

/// Scored candidate returned by beam search.
#[derive(Debug, Clone)]
pub struct BeamSearchCandidate {
    pub vector: Vec<f64>,
    pub score: f64,
}

/// Summary statistics plus top candidates for a beam search run.
#[derive(Debug, Clone)]
pub struct BeamSearchBeamResult {
    pub beam: Vec<BeamSearchCandidate>,
    pub steps: usize,
    pub evaluated: usize,
}

#[derive(Debug, Clone)]
struct ScoredCandidate {
    vector: Vec<f64>,
    score: f64,
}

/// Performs beam search over `Vec<f64>` candidates using score/expansion callbacks.
///
/// # Parameters
/// - `initial`: Starting candidate vectors for the first beam.
/// - `beam_width`: Maximum number of candidates retained per step.
/// - `max_steps`: Maximum number of expansion steps to run.
/// - `expand`: Function that generates successor vectors for a candidate.
/// - `score`: Function that assigns a score to a candidate (higher is better).
///
/// # Returns
/// - `Result<BeamSearchResult, BeamSearchError>`: Best vector/score plus search statistics.
///
/// # Expected Output
/// - Returns search results; no stdout/stderr output.
pub fn beam_search<FExpand, FScore>(
    initial: Vec<Vec<f64>>,
    beam_width: usize,
    max_steps: usize,
    expand: FExpand,
    score: FScore,
) -> Result<BeamSearchResult, BeamSearchError>
where
    FExpand: Fn(&[f64]) -> Vec<Vec<f64>>,
    FScore: Fn(&[f64]) -> f64,
{
    if initial.is_empty() {
        return Err(BeamSearchError::EmptyInitial);
    }
    if beam_width == 0 {
        return Err(BeamSearchError::ZeroBeamWidth);
    }
    if max_steps == 0 {
        return Err(BeamSearchError::ZeroMaxSteps);
    }

    let mut beam = score_candidates(initial, &score);
    let mut evaluated = beam.len();
    select_top_k(&mut beam, beam_width);

    let mut best = beam[0].clone();
    let mut steps = 0usize;

    for _ in 0..max_steps {
        let mut expanded = Vec::new();
        for candidate in &beam {
            let successors = expand(&candidate.vector);
            for successor in successors {
                let score_value = sanitize_score(score(&successor));
                expanded.push(ScoredCandidate {
                    vector: successor,
                    score: score_value,
                });
            }
        }
        steps += 1;
        evaluated += expanded.len();
        if expanded.is_empty() {
            break;
        }
        select_top_k(&mut expanded, beam_width);
        if expanded[0].score > best.score {
            best = expanded[0].clone();
        }
        beam = expanded;
    }

    Ok(BeamSearchResult {
        best_vector: best.vector,
        best_score: best.score,
        steps,
        evaluated,
    })
}

/// Performs beam search and returns the final beam of top candidates.
///
/// # Parameters
/// - `initial`: Starting candidate vectors for the first beam.
/// - `beam_width`: Maximum number of candidates retained per step.
/// - `max_steps`: Maximum number of expansion steps to run.
/// - `expand`: Function that generates successor vectors for a candidate.
/// - `score`: Function that assigns a score to a candidate (higher is better).
///
/// # Returns
/// - `Result<BeamSearchBeamResult, BeamSearchError>`: Final beam plus search statistics.
///
/// # Expected Output
/// - Returns search results; no stdout/stderr output.
pub fn beam_search_top_k<FExpand, FScore>(
    initial: Vec<Vec<f64>>,
    beam_width: usize,
    max_steps: usize,
    expand: FExpand,
    score: FScore,
) -> Result<BeamSearchBeamResult, BeamSearchError>
where
    FExpand: Fn(&[f64]) -> Vec<Vec<f64>>,
    FScore: Fn(&[f64]) -> f64,
{
    if initial.is_empty() {
        return Err(BeamSearchError::EmptyInitial);
    }
    if beam_width == 0 {
        return Err(BeamSearchError::ZeroBeamWidth);
    }
    if max_steps == 0 {
        return Err(BeamSearchError::ZeroMaxSteps);
    }

    let mut beam = score_candidates(initial, &score);
    let mut evaluated = beam.len();
    select_top_k(&mut beam, beam_width);
    let mut steps = 0usize;

    for _ in 0..max_steps {
        let mut expanded = Vec::new();
        for candidate in &beam {
            let successors = expand(&candidate.vector);
            for successor in successors {
                let score_value = sanitize_score(score(&successor));
                expanded.push(ScoredCandidate {
                    vector: successor,
                    score: score_value,
                });
            }
        }
        steps += 1;
        evaluated += expanded.len();
        if expanded.is_empty() {
            break;
        }
        select_top_k(&mut expanded, beam_width);
        beam = expanded;
    }

    let beam = beam
        .into_iter()
        .map(|candidate| BeamSearchCandidate {
            vector: candidate.vector,
            score: candidate.score,
        })
        .collect();

    Ok(BeamSearchBeamResult {
        beam,
        steps,
        evaluated,
    })
}

/// Scores a list of candidates and normalizes scores for ordering.
///
/// # Parameters
/// - `candidates`: Candidate vectors to score.
/// - `score`: Function that assigns a score to a candidate.
///
/// # Returns
/// - `Vec<ScoredCandidate>`: Scored candidates with NaN scores replaced by `-inf`.
///
/// # Expected Output
/// - Returns scored candidates; no stdout/stderr output.
fn score_candidates<FScore>(candidates: Vec<Vec<f64>>, score: &FScore) -> Vec<ScoredCandidate>
where
    FScore: Fn(&[f64]) -> f64,
{
    candidates
        .into_iter()
        .map(|vector| {
            let score_value = sanitize_score(score(&vector));
            ScoredCandidate {
                vector,
                score: score_value,
            }
        })
        .collect()
}

/// Normalizes scores so NaN values are treated as the lowest possible score.
///
/// # Parameters
/// - `score`: Raw score value.
///
/// # Returns
/// - `f64`: `score` when finite, otherwise negative infinity.
///
/// # Expected Output
/// - Returns a normalized score; no side effects.
fn sanitize_score(score: f64) -> f64 {
    if score.is_nan() {
        f64::NEG_INFINITY
    } else {
        score
    }
}

/// Keeps the top `k` candidates in-place, ordered by descending score.
///
/// # Parameters
/// - `candidates`: Scored candidates to truncate.
/// - `k`: Maximum number of candidates to keep.
///
/// # Returns
/// - `()`: Mutates `candidates` in-place.
///
/// # Expected Output
/// - Sorts and truncates the candidates; no stdout/stderr output.
fn select_top_k(candidates: &mut Vec<ScoredCandidate>, k: usize) {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    if candidates.len() > k {
        candidates.truncate(k);
    }
}

#[cfg(test)]
mod tests {
    use super::{beam_search, BeamSearchError};

    #[test]
    fn beam_search_finds_best_candidate() {
        let initial = vec![vec![0.0], vec![1.0]];
        let expand = |v: &[f64]| vec![vec![v[0] + 1.0], vec![v[0] + 2.0]];
        let score = |v: &[f64]| v[0];

        let result = beam_search(initial, 2, 2, expand, score).expect("beam search failed");

        assert_eq!(result.best_vector, vec![5.0]);
        assert_eq!(result.best_score, 5.0);
        assert_eq!(result.steps, 2);
    }

    #[test]
    fn beam_width_limits_candidates() {
        let initial = vec![vec![0.0]];
        let expand = |v: &[f64]| vec![vec![v[0] + 1.0], vec![v[0] + 2.0], vec![v[0] + 3.0]];
        let score = |v: &[f64]| v[0];

        let result = beam_search(initial, 1, 2, expand, score).expect("beam search failed");

        assert_eq!(result.best_vector, vec![6.0]);
        assert_eq!(result.best_score, 6.0);
        assert_eq!(result.steps, 2);
    }

    #[test]
    fn beam_search_stops_when_no_expansion() {
        let initial = vec![vec![1.0]];
        let expand = |_v: &[f64]| Vec::new();
        let score = |v: &[f64]| v[0];

        let result = beam_search(initial, 2, 5, expand, score).expect("beam search failed");

        assert_eq!(result.best_vector, vec![1.0]);
        assert_eq!(result.best_score, 1.0);
        assert_eq!(result.steps, 1);
    }

    #[test]
    fn beam_search_rejects_invalid_inputs() {
        let expand = |_v: &[f64]| vec![vec![0.0]];
        let score = |_v: &[f64]| 0.0;

        let err = beam_search(Vec::new(), 1, 1, expand, score).expect_err("expected error");
        assert_eq!(err, BeamSearchError::EmptyInitial);

        let err = beam_search(vec![vec![0.0]], 0, 1, expand, score).expect_err("expected error");
        assert_eq!(err, BeamSearchError::ZeroBeamWidth);

        let err = beam_search(vec![vec![0.0]], 1, 0, expand, score).expect_err("expected error");
        assert_eq!(err, BeamSearchError::ZeroMaxSteps);
    }
}
