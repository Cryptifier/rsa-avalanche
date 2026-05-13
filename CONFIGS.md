# Configuration Reference

This file collects the configuration material that used to live in `README.md`.

# Primary configs
- `config/rsa_config_small_batch.json`: Main small-batch Avalanche config used by `make demo` and the current `analysis.rs` proof of concept.
- `config/rsa_config_small_public_key_test.json`: Small public-key regression config that exercises public-key analysis plus a private-key verification peek.
- `config/rsa_config.json`: Baseline general-purpose config with the broader engine schema documented below.

# Key YAML
- `kgen` writes RSA private keys as YAML under `config/keys`.
- `kgen --public-output ...` writes matching RSA public keys as `rsa-public-key-v1` YAML.
- `kgen --input-pgp-public-key ... --public-output ...` imports RSA public keys from OpenPGP files into the same `rsa-public-key-v1` YAML format.
- `kgen --input-pgp-file ... --pgp-output ...` writes unpacked OpenPGP packet contents as `pgp-file-v1` YAML.
- Non-secret example schemas live at `config/keys/private_key.example.yaml` and `config/keys/public_key.example.yaml`.
- A non-secret `pgp-file-v1` schema example lives at `config/keys/pgp_file.example.yaml`.
- The tracked repository ignores the default generated key path `config/keys/private_key.yaml`.
- Tracked examples include both private and public YAMLs for the small-batch keys under `config/keys/`.

# Notes for `config/rsa_config_small_batch.json`
- This is the main replay configuration for the current POC.
- It uses `rsa_keypair.generate = false` and resolves the RSA key from `rsa_keypair.keyfile`.
- It enables batched Avalanche sampling, majority-vote beam search, and Hamming-distance pruning for the small-batch workflow.
- It is the default config for both `analysis.rs` and `demo.rs`.

# Notes for `config/rsa_config_small_public_key_test.json`
- This config exercises the public-key YAML path without changing the default private-key examples.
- `rsa_keypair.keyfile` points at an `rsa-public-key-v1` file.
- `rsa_keypair.private_keyfile` points at the matching private YAML so `analysis` can perform a verification peek while still operating in public-key mode.

# Configuration (`config/rsa_config.json`)
Notes:
- Missing config files fall back to built-in defaults; when present, values below are read.
- Unknown keys are ignored by `analysis.rs`. The `padding` and `engine.max_overlap_min` fields are currently not used.
- When `rsa_keypair.generate` is `false`, supply either inline `rsa_keypair.p`/`rsa_keypair.q` or `rsa_keypair.keyfile`.
- `rsa_keypair.keyfile` may point to either `rsa-private-key-v1` or `rsa-public-key-v1`.
- When the main keyfile is public, `analysis` skips round-trip RSA. Set `rsa_keypair.private_keyfile` if you want a matching private-key verification peek; otherwise the public-key run is selected by top beam score.
- Public-key YAMLs are used the same way as private-key YAMLs in config: point `rsa_keypair.keyfile` at the file and leave `rsa_keypair.p` and `rsa_keypair.q` unset.

| Key | Type | Default in `config/rsa_config.json` | Notes |
| --- | --- | --- | --- |
| `rsa_keypair.generate` | bool | `false` | Generate primes when `true`. |
| `rsa_keypair.e` | u64 | `65537` | Public exponent seed. |
| `rsa_keypair.keyfile` | string | `""` | Relative or absolute RSA YAML path used when inline primes are absent; accepts either private or public key YAML. |
| `rsa_keypair.private_keyfile` | string | `""` | Optional relative or absolute private-key YAML used only for verification peeks when `rsa_keypair.keyfile` is public. |
| `rsa_keypair.p` | string (bigint) | `3030152311446024058741` | Prime `p` when not generating. |
| `rsa_keypair.q` | string (bigint) | `4262327550688715209573` | Prime `q` when not generating. |
| `padding` | string | `PKCS1v15` | Present for compatibility; currently unused. |
| `engine.r_stress_test_enable` | bool | `false` | Enable stress test over `r` range. |
| `engine.r_stress_start` | string (bigint) | `12915501679859480667750241440843466838812811` | Stress test range start. |
| `engine.r_stress_end` | string (bigint) | `12915501679859480667750241440843466838812814` | Stress test range end. |
| `engine.r_use_list_enable` | bool | `false` | Use explicit `r` list. |
| `engine.r_use_list` | array(string) | `[]` | Explicit `r` candidates. |
| `engine.max_overlap_min` | number | `0.005` | Present for compatibility; currently unused. |
| `engine.override_best_r` | string | `""` | Overrides best `r` if non-empty. |
| `engine.test_iterations` | u64 | `4` | Number of primary test iterations. |
| `engine.alt_iterations` | u64 | `4` | Number of alternate test iterations. |
| `engine.analysis_tests_iterations` | u64 | `1000` | Timeline iterations for analysis tests. |
| `engine.oracle_screen_iterations` | u64 | `500` | Iterations for per-bit oracle screening. |
| `engine.analysis_tests_window` | usize | `32` | Window size for analysis timelines. |
| `engine.analysis_tests_stride` | usize | `8` | Window stride for analysis timelines. |
| `engine.analysis_batch_enable` | bool | `false` | Enable batched `r`-candidate scoring plus Avalanche sampling. |
| `engine.analysis_batch_messages` | u64 | `1` | Number of ciphertext/message variants scored per batch before Avalanche sampling. |
| `engine.analysis_batch_candidates` | u64 | `0` | Number of `r` candidates scored in each batch. |
| `engine.analysis_batch_batches` | u64 | `1` | Number of batch-analysis runs. |
| `engine.avalanche_combination_samples` | u64 | `100` | Number of sampled combinations evaluated by Avalanche per batch. |
| `engine.avalanche_combination_size` | usize | `50` | Legacy compatibility field retained from the older scored-item sampler. |
| `engine.avalanche_combination_mixed_r_candidates` | usize | `1` | Number of distinct `r` candidates mixed into each Avalanche sample; each selected `r` contributes all of its scored `c^x` inputs. |
| `engine.avalanche_combination_pool_size` | usize | `100` | Legacy compatibility field; runtime sampling now uses the full batch-sized pool. |
| `engine.avalanche_combination_recursion_depth` | usize | `1` | Number of Avalanche tiers to execute, including the initial sampled-input tier. |
| `engine.avalanche_combination_recursive_group_size` | array(usize) | `[8]` | Per-recursive-tier group sizes; when recursion exceeds the array length, the last entry is reused. |
| `engine.avalanche_combination_recursive_resample_count` | array(usize) | `[0]` | Per-recursive-tier resample counts; `0` keeps one-pass regrouping, and the last entry is reused for deeper tiers. |
| `engine.avalanche_combination_majority_vote` | bool | `true` | Use per-bit majority-vote probabilities from each sampled combination. |
| `engine.avalanche_combination_sample_smoothing` | bool | `false` | Apply Jeffreys smoothing to sampled majority-vote probabilities before beam search. |
| `engine.avalanche_combination_majority_vote_print` | bool | `true` | Print a separate sampled-combination majority-vote summary for the selected sample. |
| `engine.avalanche_use_top_beam` | bool | `true` | Carry forward the prior tier's top beam-search bits between recursive Avalanche tiers instead of the prior tier's majority-vote bits. |
| `engine.avalanche_combination_keep_all_samples_in_memory` | bool | `false` | Retain every sampled Avalanche combination in memory for downstream consumers instead of keeping only the selected best sample. |
| `engine.avalanche_statistics_collection` | bool | `true` | Collect recursive Avalanche tier statistics and other heavy per-sample analytics payloads. Set `false` to keep the Avalanche result while skipping those statistics. |
| `engine.process_min_count` | u64 | `25` | Minimum `r` candidates to process. |
| `engine.process_count` | u64 | `25` | Target `r` candidates per batch. |
| `engine.process_scale` | u32 | `12` | Scaling factor for candidate generation. |
| `engine.process_max_best_attempts` | u64 | `500` | Max attempts to improve best `r`. |
| `engine.process_min_factor` | u64 | `117` | Minimum factor threshold. |
| `engine.rabin_exponent` | u64 | `3` | Rabin exponent used in candidate math. |
| `engine.min_message_trials` | u64 | `100` | Minimum message trials per `r` candidate. |
| `engine.overlap_report_threshold` | number | `51` | Overlap % threshold for reporting. |
| `engine.entropy_report_threshold` | number | `0.99` | Entropy threshold for sufficiency checks. |
| `engine.oracle_accuracy_threshold` | number | `51.0` | Oracle accuracy threshold for sufficiency checks. |
| `engine.beam_bit_one_threshold` | number | `0.4` | Minimum stored beam value interpreted as bit `1`. |
| `engine.avalanche_probability_spread_exponent` | number | `0.5` | Power exponent applied to confidence around `0.5`; values below `1.0` sharpen confidence while preserving the original side of `0.5`, and values above `1.0` soften it. |
| `engine.sqlite_soft_heap` | u64 | `10737418240` | Advisory SQLite soft heap limit for the Avalanche cache database in bytes. |
| `engine.sqlite_hard_heap` | u64 | `10737418240` | Hard SQLite heap limit for the Avalanche cache database in bytes. |
| `engine.sqlite_mmap_size` | u64 | `10737418240` | SQLite mmap size for the Avalanche cache database in bytes. |
| `engine.sqlite_worker_count` | u32 | `16` | SQLite connection-pool worker count for the Avalanche cache database. |
| `engine.sqlite_db_folder` | string | `"/tmp"` | Filesystem folder used for the Avalanche cache database; intermediate directories are created automatically. |
| `engine.sqlite_avalanche_page_size` | usize | `4096` | Number of rows per SQLite Avalanche cache page used for batched inserts and paged reads. |
| `engine.r_candidate_mode` | string | `small_primes` | Candidate generation mode. |
| `engine.r_candidate_small_primes` | array(u64) | `[3, 5, 7, 11, 13, 17]` | Small primes for candidate generation. |
| `engine.r_candidate_small_prime_factors` | usize | `3` | Number of small prime factors. |
| `engine.r_candidate_max_factors` | usize | `6` | Maximum total factors per `r` candidate. |
| `engine.r_candidate_bit_length` | u64 | `null` | Optional target bit length for `r` candidates. |
| `engine.r_candidate_random_power_window` | bool | `false` | In factoring mode, sample candidate bounds from a random `N^a` window with `a` chosen in `[0.8, 0.9]` before uniqueness filtering. |
| `engine.r_candidate_target_exponent_minimum` | number | `0.8` | Lower bound for the sampled total exponent used when retargeting speculative `r` candidates. |
| `engine.r_candidate_target_exponent` | number | `2.005` | Upper bound for the sampled total exponent used when retargeting speculative `r` candidates. |
| `engine.r_candidate_retarget_partition_count` | usize | `3` | Number of exponent partitions required for speculative retargeting. |
| `engine.r_candidate_retarget_minimum_exponent` | number | `0.45` | Minimum exponent assigned to each retargeted partition when feasible. |
| `engine.combiner_enable` | bool | `true` | Enable speculative combiner. |
| `engine.combiner_k_oracles` | usize | `5` | Number of oracles to request. |
| `engine.combiner_match_probability` | number | `0.75` | Target oracle match probability. |
| `engine.combiner_tie_breaker` | bool | `true` | Tie-breaking strategy. |
| `engine.base_convert` | bool | `true` | Enable base conversion in analysis. |
| `engine.invert_bits` | bool | `false` | Invert bits during analysis. |
| `engine.use_rs_decrypt` | bool | `true` | Use Rust decrypt path for `r` analysis. |
| `engine.enciphered_export_enable` | bool | `false` | Export enciphered bins/ramp data. |
| `engine.enciphered_export_iterations` | u64 | `10000` | Export iterations. |
| `engine.enciphered_export_bins` | usize | `128` | Histogram bins. |
| `engine.enciphered_export_window` | usize | `128` | Window size. |
| `engine.enciphered_export_stride` | usize | `1` | Window stride. |
| `engine.enciphered_export_output_csv` | string | `enciphered_decryption_bins.csv` | Export CSV path. |
| `engine.enciphered_export_ramp_length` | usize | `3` | Ramp length. |
| `engine.enciphered_export_ramp_step_pct` | number | `0.05` | Ramp step percent. |
| `engine.enciphered_export_ramp_tolerances` | array(number) | `[0.005, 0.01, 0.02]` | Ramp tolerances. |
| `engine.enciphered_export_ramp_csv` | string | `enciphered_ramps.csv` | Ramp CSV path. |
| `engine.message.is_random` | bool | `true` | Use random message. |
| `engine.message.bits` | u32 | `128` | Random message bit length. |
| `engine.message.fixed_message` | string | `HeloWrld1234` | Fixed message when `is_random` is `false`. |

For `config/rsa_config_small_batch.json`, sampled Avalanche now draws every combination from the full scored batch-sized pool rather than truncating to a separate top-pool limit. `engine.avalanche_combination_mixed_r_candidates` controls how many distinct `r` candidates are allowed into a sampled Avalanche input set, and each chosen `r` contributes all of its configured `c^x` message variants. `engine.avalanche_combination_recursion_depth`, `engine.avalanche_combination_recursive_group_size`, and `engine.avalanche_combination_recursive_resample_count` enable tiered Avalanche runs where prior-tier Avalanche outputs are either grouped once or resampled to a target count for subsequent recursive calls. The two `avalanche_combination_recursive_*` settings are arrays keyed by recursive level, and deeper recursion tiers reuse the last configured array entry. When `engine.avalanche_combination_majority_vote` is enabled, which is the default, the beam probabilities come from per-bit majority-vote frequencies across each sampled combination. `engine.avalanche_use_top_beam` controls which finalized prior-tier bit vector is carried into the next recursive tier: `true` reuses the top beam-search candidate and `false` reuses the prior tier's majority-vote bits. Enable `engine.avalanche_combination_sample_smoothing` or `--avalanche-combination-sample-smoothing true` to apply Jeffreys smoothing to those frequencies before beam search. `engine.avalanche_combination_majority_vote_print` controls the separate console summary for the sampled-combination majority vote and defaults to `true`. Set `engine.avalanche_statistics_collection` to `false` to skip recursive tier statistics and other heavy Avalanche analytics payloads while keeping the selected Avalanche result and summary metrics.

# Configuration (`demo` verification inputs)
These optional keys are used by `demo` when `--ciphertext` is not provided.

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `verify.ciphertext` | string (bigint) | `null` | Ciphertext in decimal string form. |
| `verify.ciphertext_hex` | string | `null` | Ciphertext hex string (`0x` prefix optional). |
