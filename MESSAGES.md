# Message Formats

This project emits analytics data as streaming JSON (NDJSON). Each line is a single JSON object with two top-level keys:
- `event`: string event name
- `payload`: event-specific object

Consumers should treat missing fields as empty string, null, or 0 (per field type), and ignore unknown `event` values.

## Session Events

`session_start` payload:
- `started_unix_ms`: number (milliseconds since epoch)
- `cli`: object

`cli` fields:
- `bits`: number
- `message_override`: string or null
- `public_exponent`: number
- `seed`: number or null
- `crypto_rng`: boolean
- `config_path`: string
- `tests`: boolean
- `export`: boolean
- `session_json`: string (output path)
- `shift`: boolean
- `ciphertext_modify`: boolean
- `use_hamming_distance`: boolean
- `mirror_invert_candidates`: boolean
- `beam_bit_one_threshold`: number
- `avalanche_probability_spread_exponent`: number
- `avalanche_combination_samples`: number
- `avalanche_solver_global_log_enable`: boolean
- `avalanche_combination_size`: number
- `avalanche_combination_mixed_r_candidates`: number
- `avalanche_combination_pool_size`: number
- `avalanche_combination_majority_vote`: boolean
- `avalanche_combination_sample_smoothing`: boolean
- `avalanche_combination_majority_vote_print`: boolean
- `bits_decrypt`: number or null

`session_finish` payload:
- `finished_unix_ms`: number or null
- `errors`: array of strings

`step` payload:
- `name`: string
- `duration_ms`: number

`step_summary` payload:
- `name`: string
- `count`: number
- `total_ms`: number
- `mean_ms`: number

`feature` payload:
- `name`: string
- `enabled`: boolean
- `duration_ms`: number or null
- `notes`: array of strings
- `stats`: object (string keys to JSON values)

## r Candidate Events

`r_candidate_batch` payload:
- `context`: string
- `mode`: string
- `target_count`: number
- `generated_count`: number
- `duration_ms`: number
- `reuse_path`: string
- `reuse_enabled`: boolean
- `reuse_append_only`: boolean
- `min_factor`: string
- `process_scale`: number
- `small_prime_factors`: number
- `max_factors`: number
- `target_bit_length`: number or null
- `candidates`: array

`r_candidate_batch.candidates[]` fields:
- `r`: string
- `r_bits`: number
- `factors`: array

`r_candidate_batch.candidates[].factors[]` fields:
- `prime`: string
- `exponent`: number
- `prime_bits`: number

`r_candidate_accuracy_batch` payload:
- `context`: string
- `messages`: array of strings
- `ciphertexts`: array of strings
- `shifted_ciphertexts`: array of strings
- `rabin_exponent`: number
- `tonelli_shanks_modulus`: string
- `tonelli_shanks_ciphertexts`: array of strings
- `candidates`: array
- `beam_match_pct`: number or null
- `beam_ones_match_pct`: number or null
- `beam_score`: number or null
- `beam_bit_width`: number or null

`r_candidate_accuracy_batch.candidates[]` fields:
- `r`: string
- `r_bits`: number
- `factors`: array
- `accuracy_pct`: number
- `hbc_ciphertexts_r`: array of strings
- `candidate_decryptions`: array of strings

`r_candidate_trace_batch` payload:
- `context`: string
- `message`: string
- `ciphertext`: string
- `shifted_ciphertext`: string
- `rabin_exponent`: number
- `tonelli_shanks_modulus`: string
- `tonelli_shanks_ciphertext`: string
- `candidates`: array

`r_candidate_trace_batch.candidates[]` fields:
- `r`: string
- `r_bits`: number
- `hbc_ciphertext_r`: string
- `candidate_decryption`: string

## Bitflow Events

`bitflow_run` payload:
- `run_id`: string
- `bit_width`: number
- `min_partition_size`: number
- `max_partition_size`: number
- `progression`: string
- `max_iterations`: number
- `max_partitions_to_flip`: number
- `per_candidate_trials`: number
- `seed`: number
- `pow_mod_base`: number
- `pow_mod_modulus`: number
- `message_bits`: array of `0`/`1` values (little-endian bit order)

`bitflow_candidate` payload:
- `run_id`: string
- `iteration`: number
- `trial`: number
- `partition_size`: number
- `inverted_partitions`: array of numbers
- `bits`: array of `0`/`1` values (little-endian bit order)
