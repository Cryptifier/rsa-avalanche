# Analysis API

This document describes how the `analysis` binary drives the supported analysis workflows in this repository. It is not a Rust API reference. The focus here is execution flow, feature-specific call chains, and the helper functions available to those flows.

## Scope

The supported analysis surface is currently:

- Base RSA round-trip setup and message selection
- Information sufficiency analysis, enabled with `--tests`
- Optional chart export for sufficiency analysis, enabled with `--export`
- R-candidate batch accuracy analysis, enabled by config or by CLI batch overrides
- Avalanche reduction and beam-search reconstruction inside the supported analysis paths

The older small-batch-disabled experiment branches were removed. This document only covers paths that are still live in the code.

## Entry Points

| Module | Function | Role |
| --- | --- | --- |
| `src/bin/analysis.rs` | `main` | CLI parsing, config loading, CLI override application, analytics session creation |
| `src/methods.rs` | `run_demo` | Shared top-level runtime for key generation, message selection, RSA round-trip, and feature dispatch |
| `src/methods.rs` | `run_information_sufficiency_tests` | Sufficiency workflow behind `--tests` |
| `src/methods.rs` | `run_r_candidate_accuracy_batches` | Batch-scoring workflow behind `analysis_batch_enable` |

## Core Math

The base RSA setup is:

$$
n = pq,\qquad \varphi(n) = (p-1)(q-1)
$$

$$
c = m^e \bmod n,\qquad d \equiv e^{-1} \pmod{\varphi(n)}
$$

The candidate-modulus reconstruction flow used across the analysis code is:

$$
c_r = \operatorname{HBC}(c, r, n)
$$

$$
\tilde{m}_r =
\begin{cases}
c_r^{d_r} \bmod r & \text{if `use_rs_decrypt`} \\
c_r & \text{otherwise}
\end{cases}
$$

$$
\hat{m} = \operatorname{HBC}(\tilde{m}_r, n, r) \bmod n
$$

where \( d_r \equiv e^{-1} \pmod{\varphi(r)} \) or, for modified ciphertext exponents, \( d_r \equiv (ex)^{-1} \pmod{\varphi(r)} \).

Bit-match scoring uses:

$$
\mathrm{match\_pct} = 100 \cdot \frac{\mathrm{matching\_total}}{\mathrm{bit\_width}}
$$

Avalanche beam scoring uses a per-bit probability vector \( p_i \in [0,1] \):

$$
S(b) = \sum_{i=0}^{W-1}
\begin{cases}
p_i & \text{if } b_i = 1 \\
1 - p_i & \text{if } b_i = 0
\end{cases}
$$

## Common Runtime Chain

Every analysis run starts with the same top-level sequence:

```text
src/bin/analysis::main
  -> load_config
  -> apply CLI overrides
  -> SessionAnalytics::new
  -> run_demo
```

Inside `run_demo`, the common chain is:

```text
run_demo
  -> initialize RNG
  -> select or generate RSA primes
  -> derive n, phi(n), e, d
  -> select message
  -> encrypt and decrypt once for round-trip validation
  -> dispatch feature paths:
       args.tests ? run_information_sufficiency_tests
       config.engine.analysis_batch_enable ? run_r_candidate_accuracy_batches
```

This means every feature runs on top of a known-good RSA instance and a validated plaintext/ciphertext pair.

## Feature Call Chains

### 1. Base Round-Trip

This path always runs, even when no analysis features are enabled.

```text
main
  -> run_demo
     -> select_message
     -> message.modpow(e, n)
     -> ciphertext.modpow(d, n)
```

Purpose:

- Validate the chosen RSA parameters
- Materialize the reference `message`, `ciphertext`, and `recovered` values
- Build the `RSAContext` shared by all later analysis routines

Key helpers used here:

- `select_message`
- `random_message_under_n`
- `choose_exponent`
- `mod_inverse`
- `to_hex`

### 2. Information Sufficiency Analysis

This path is enabled with `--tests`.

### 2.1 Top-Level Chain

```text
run_demo
  -> run_information_sufficiency_tests
     -> build_oracle_candidates
     -> record_r_candidate_trace_batch_from_prepared
     -> select_best_candidate
     -> optional timeline generation
     -> oracle screening
     -> per-bit best-case reconstruction
     -> bit-similarity export
     -> speculative bitwise oracle attempt
     -> optional standalone avalanche search
     -> threshold evaluation and PASS/FAIL verdict
```

### 2.2 Candidate Preparation

The first step is to create valid prepared `r` candidates:

```text
run_information_sufficiency_tests
  -> build_oracle_candidates
     -> build_r_candidate_settings
     -> generate_r_candidates_with_analytics
     -> compute_totient
     -> mod_inverse
```

`build_oracle_candidates` converts raw `RCandidate` values into prepared `OracleCandidate` values with:

- `r`
- `phi_new`
- `d_new`
- `x`
- `target_exponent`

If `same_r_batch` is enabled, this stage generates multiple `x` variants over one `r`. Otherwise it prepares one decryptable candidate per `r`.

### 2.3 Best-Candidate Selection

The best prepared candidate is chosen against the original message:

```text
run_information_sufficiency_tests
  -> select_best_candidate
     -> ciphertext_for_candidate
     -> maybe_shift_ciphertext
     -> prepare_candidate_ciphertext
     -> derive_candidate_message_from_result
     -> count_matching_bits
```

This produces the reference `r` candidate used by later sufficiency steps.

### 2.4 Timeline Analysis

When `--export` is set, the sufficiency path emits timeline data and charts.

Oracle entropy timeline chain:

```text
run_information_sufficiency_tests
  -> run_oracle_entropy_timeline
  -> plot_timeline_series
```

Match entropy timeline chain:

```text
run_information_sufficiency_tests
  -> run_match_entropy_timeline
  -> plot_timeline_series
```

These timelines are not required for the verdict. They are observational outputs that feed analytics and optional PNG generation.

### 2.5 Per-Bit Oracle Screening

This stage chooses the strongest oracle candidates for each bit position:

```text
run_information_sufficiency_tests
  -> screen_oracles_per_bit
     -> random_message_under_n
     -> ciphertext_for_candidate
     -> maybe_shift_ciphertext
     -> prepare_candidate_ciphertext
     -> derive_candidate_message_from_result
     -> biguint_to_bits_le
```

The result is:

- `per_bit_oracles`: ranked oracle selections for each bit
- `top_match_pct`: best observed per-bit adjusted match percentages

If a candidate is negatively correlated on a bit, the selection is marked with `invert = true`, so the analysis can intentionally flip that oracle’s bit later.

### 2.6 Best-Case Per-Bit Reconstruction

This is a diagnostic path that answers: if you already knew the best oracle choice per bit, how good could reconstruction be?

```text
run_information_sufficiency_tests
  -> compute_per_bit_best_case_match
     -> ciphertext_for_candidate
     -> maybe_shift_ciphertext
     -> prepare_candidate_ciphertext
     -> derive_candidate_message_from_result
     -> count_matching_bits_le
```

This path does not search. It is a direct oracle-quality upper bound.

### 2.7 Bit Similarity Export

The bit-similarity table is produced by:

```text
run_information_sufficiency_tests
  -> build_bit_similarity_entries
     -> prepare_candidate_ciphertext
     -> derive_candidate_message_from_result
```

This helper explores ciphertext shift levels and records:

- raw match percentage
- adjusted match percentage after masked shift loss
- per-bit match counts

### 2.8 Bitwise Speculative Oracle Attempt

This is the direct bit-recovery path built from the screened per-bit oracle selections:

```text
run_information_sufficiency_tests
  -> run_bitwise_speculative_oracle_attempt
     -> collect_invertible_ciphertext_variants
     -> prepare_candidate_ciphertext
     -> derive_candidate_message_from_result
     -> count_matching_bits_le
     -> bits_le_to_biguint
```

The recovered bit at position \( i \) is the majority vote over the chosen screened oracles for that position. Ties use `engine.combiner_tie_breaker`.

### 2.9 Standalone Avalanche Search

This path only runs inside sufficiency testing when `analysis_batch_enable` is off.

```text
run_information_sufficiency_tests
  -> run_avalanche_search
     -> build_avalanche_nodes_unique_d
        -> collect_invertible_ciphertext_variants
        -> prepare_candidate_ciphertext
        -> derive_candidate_message_from_result
        -> optional mirror_inverted_candidates
        -> optional sort_candidates_by_hamming_distance
     -> search_avalanche_tree_with_scores_progress
     -> normalize_avalanche_biases
     -> spread_normalized_avalanche_biases
     -> beam_search_top_k_with_progress
     -> viterbi_decode
```

If `use_hamming_distance` is enabled, the ordering branch becomes:

```text
build_avalanche_nodes_unique_d
  -> sort_candidates_by_hamming_distance
     -> helpers::hamming_distance_bits
        -> optional packed SIMD backend
```

The SIMD backend is feature-gated:

- `x86-hamming-accel`
- `aarch64-hamming-accel`

### 2.10 Verdict Logic

The sufficiency verdict is a conjunction of threshold checks:

$$
\text{PASS} \iff
\text{oracle\_entropy\_ok}
\land \text{match\_entropy\_ok}
\land \text{match\_pct\_ok}
\land \text{oracle\_accuracy\_ok}
\land \text{speculative\_match\_ok}
$$

The match checks allow either direct or inverted signal:

$$
\text{match\_pct\_ok} \iff \mu_{\text{match}} \ge \tau_{\text{match}}
\;\lor\;
\mu_{\text{match}} \le 100 - \tau_{\text{match}}
$$

The same pattern is used for the speculative match percentage.

### 3. R-Candidate Accuracy Batch Analysis

This path is enabled when `config.engine.analysis_batch_enable` is true, or when CLI batch overrides force it on.

### 3.1 Top-Level Chain

```text
run_demo
  -> run_r_candidate_accuracy_batches
     -> generate_r_candidates_with_analytics
     -> prepare valid batch candidates
     -> per-batch scoring loop
     -> run_sampled_avalanche_beam_search
     -> analytics aggregation
```

### 3.2 Raw Candidate Scoring

Per batch, the code computes direct `c^x` candidate quality before avalanche reduction:

```text
run_r_candidate_accuracy_batches
  -> collect_invertible_ciphertext_variants   (when ciphertext_modify)
  -> prepare_candidate_ciphertext
  -> derive_candidate_message_from_result
  -> count_matching_bits
  -> build ScoredAvalancheInput records
```

This produces:

- batch-level direct `c^x` maximum match
- per-candidate analytics entries
- `ScoredAvalancheInput` values consumed by the sampled avalanche stage

### 3.3 Sampled Avalanche and Beam Search

The sampled avalanche stage is the main batch-analysis feature:

```text
run_r_candidate_accuracy_batches
  -> run_sampled_avalanche_beam_search
     -> group_scored_inputs_by_r_candidate
     -> select_scored_inputs_for_mixed_r_candidates
     -> build_avalanche_nodes_from_scored_inputs
     -> search_avalanche_tree_with_scores
     -> majority_vote_with_distribution
     -> optional smooth_probability_one_jeffreys
     -> spread_normalized_avalanche_biases
     -> beam_search_top_k
     -> compute_bit_match_percentages
```

The batch path samples many mixed subsets of scored candidate decryptions and keeps the best sample by:

1. highest average source score
2. then highest top-beam score

This path is what powers the small-batch beam script.

### 3.4 What the Batch Path Emits

For each batch it records:

- direct `c^x` baseline maxima
- selected avalanche sample metadata
- avalanche beam results
- majority-vote reconstruction statistics
- all sampled-combination analytics payloads

At the end of all batches it prints the best overall:

- beam-search run
- majority-vote run
- raw `c^x` run

## Helper Function Inventory

This section groups the reusable helpers that analysis relies on.

### 1. `src/methods.rs` Orchestration Helpers

| Function | Purpose |
| --- | --- |
| `select_message` | Chooses CLI override, random message, or fixed configured message |
| `random_message_under_n` | Samples a nonzero message below `n` |
| `analysis_bit_width` | Resolves bit width for timeline and oracle workflows |
| `build_r_candidate_settings` | Converts `EngineConfig` into `RCandidateSettings` |
| `build_oracle_candidates` | Builds prepared `OracleCandidate` values from raw `r` candidates |
| `select_best_candidate` | Scores candidates against the true message and returns the best one |
| `screen_oracles_per_bit` | Ranks candidates per bit position across screening samples |
| `compute_per_bit_best_case_match` | Computes optimistic per-bit reconstruction quality |
| `build_bit_similarity_entries` | Builds shift-aware per-bit similarity diagnostics |
| `run_bitwise_speculative_oracle_attempt` | Performs per-bit reconstruction using screened oracle selections |
| `build_avalanche_nodes_unique_d` | Builds avalanche nodes from unique decryptions |
| `run_avalanche_search` | Runs standalone avalanche reduction, beam search, and Viterbi decoding |
| `run_sampled_avalanche_beam_search` | Runs sampled avalanche reduction inside batch accuracy analysis |
| `record_r_candidate_trace_batch_from_prepared` | Writes prepared candidate traces into analytics |

### 2. `src/methods.rs` Ciphertext and Reconstruction Helpers

| Function | Purpose |
| --- | --- |
| `prepare_candidate_ciphertext` | Applies HBC from source modulus to candidate modulus |
| `derive_candidate_message` | Full candidate reconstruction from a ciphertext input |
| `derive_candidate_message_from_result` | Reconstruction when the candidate-modulus ciphertext is already prepared |
| `maybe_shift_ciphertext` | Applies the encrypted-2 homomorphic left shift when requested |
| `count_matching_bits` | Computes LSB run and total bit matches for `BigUint` values |
| `count_matching_bits_le` | Computes LSB run and total bit matches for `Vec<bool>` bit slices |
| `pack_bits_to_bytes_le` | Packs internal little-endian bit slices into bytes |

### 3. `src/helpers.rs` Bit and Beam Helpers

| Function | Purpose |
| --- | --- |
| `hamming_distance_bits` | Computes Hamming distance on boolean bit slices |
| `matching_bit_counts_bytes_le` | Computes total matches and contiguous LSB matches on packed byte slices |
| `pack_bits_to_bytes` | Packs booleans into byte storage for accelerated paths |
| `hamming_distance_packed_bytes` | Packed-byte Hamming distance with scalar or SIMD backend |
| `normalize_avalanche_biases` | Converts raw avalanche bias magnitudes into `[0,1]` |
| `spread_normalized_avalanche_biases` | Sharpens or softens decision confidence around `0.5` |
| `stored_beam_value_is_one` | Interprets a stored beam value as bit `1` or `0` |
| `format_beam_float` | Stable float formatting for logs and analytics |

### 4. `src/avalanche.rs` Reduction Helpers

| Function | Purpose |
| --- | --- |
| `mirror_inverted_candidates` | Duplicates candidates with bitwise-inverted copies |
| `sort_candidates_by_hamming_distance` | Orders candidates against a reference bit string |
| `search_avalanche_tree_with_scores` | Reduces candidates while recording per-level similarity |
| `search_avalanche_tree_with_scores_progress` | Same as above, with progress output |

### 5. `src/combiner.rs` Voting Helpers

| Function | Purpose |
| --- | --- |
| `majority_vote_per_bit` | Hard majority vote across oracle bit vectors |
| `majority_vote_with_distribution` | Majority vote plus per-bit one/zero counts and probabilities |
| `generate_oracle_samples` | Synthetic noisy-oracle generator used by combiner tests and experiments |
| `optimal_combiner_test` | End-to-end simulated combiner experiment |

The current analysis runtime uses `majority_vote_with_distribution` directly in avalanche and speculative-oracle workflows.

### 6. `src/search.rs` Search Helpers

| Function | Purpose |
| --- | --- |
| `beam_search_top_k` | Beam search without progress logging |
| `beam_search_top_k_with_progress` | Beam search with progress logging |
| `viterbi_decode` | Hidden Markov Model decoding over per-bit log probabilities |

### 7. `src/analytics.rs` Candidate and Session Helpers

| Function | Purpose |
| --- | --- |
| `generate_r_candidates_with_analytics` | Generates raw `r` candidates, optionally retargets them, and records the batch into session analytics |

## Configuration and CLI Controls That Matter to Analysis

The most important runtime switches are:

| Control | Effect |
| --- | --- |
| `--tests` | Enables information sufficiency analysis |
| `--export` | Enables timeline chart export inside sufficiency analysis |
| `--batches` / `--batch-size` | Forces batch accuracy analysis on and overrides batch sizing |
| `analysis_batch_enable` | Enables the batch accuracy workflow from config |
| `same_r_batch` | Reuses one `r` candidate per batch and varies `x` instead |
| `ciphertext_modify` | Uses invertible ciphertext exponent variants \( c^x \) in batch analysis |
| `use_hamming_distance` | Enables Hamming-distance ordering for avalanche candidates |
| `mirror_invert_candidates` | Duplicates avalanche candidates with inverted bit strings |
| `beam_bit_one_threshold` | Threshold used to interpret beam candidate floats as bits |
| `avalanche_probability_spread_exponent` | Controls confidence sharpening/softening before beam search |
| `avalanche_combination_samples` | Number of sampled avalanche subsets per batch |
| `avalanche_combination_size` | Size of each sampled avalanche subset |
| `avalanche_combination_mixed_r_candidates` | Number of distinct `r` candidates allowed in each sample |

## Reading the Code by Call Chain

If you want to follow the runtime from the outside in, read the code in this order:

1. `src/bin/analysis.rs::main`
2. `src/methods.rs::run_demo`
3. `src/methods.rs::run_information_sufficiency_tests`
4. `src/methods.rs::run_r_candidate_accuracy_batches`
5. `src/methods.rs::run_sampled_avalanche_beam_search`
6. `src/avalanche.rs`, `src/combiner.rs`, `src/search.rs`, and `src/helpers.rs`

That order follows the actual execution flow rather than the module graph.
