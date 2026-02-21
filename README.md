# RSA Analysis Demo
This demo shows homomorphic key switching as a viable method to retrieve up to 52% of the bits of an RSA message on average given a public modulus with private factorization and several homomorphically related keys with easier factorizations.

Proof of concept by Nicholas LaRoche <nlaroche@nicholas.louis.laroche@outlook.com>.

# Theory
- Use regular RSA encryption using a large modulus N = pq where p and q are large private primes.
- Use phi of phi(N) = (p-1)(q-1) to generate a public/private key pair (e, d) such that ed ≡ 1 (mod phi(N)).
- Use the Tonelli-Shanks approach to increase the relation of the ciphertext mod N to the new ciphertext mod N^k for a small k = 3.
- Use homomorphic base switching to go from mod N^k to mod R where R is an easily factored modulus with at least three factors (more than regular RSA).
- Use the easy factorization of R to retrieve partial information about the original message by calculating a new d' such that ed' ≡ 1 (mod phi(R)).
- For analysis, decryption is performed with the candidate modulus R and its d', not with N or the private exponent derived from N (those are only used for the baseline RSA round-trip check).
- Compute the difference between each independent trial with random message and ciphertext using modulus R in a random oracle model to retrieve percentage of bits matching the original message via the ciphertext using modulus N.
- A bit-wise speculative decryption oracle is used to recover message bits.

# Setup
- Use Linux and install Rust.
```bash
cargo build --bin analysis
cargo run --bin analysis | tee output.txt
```

# Command Line (analysis)
```bash
cargo run --bin analysis -- --config rsa_config.json
```

- `-b, --bits <u32>`: Bit-length for generated primes when `rsa_keypair.generate` is `true`. Default `56` (range `16..=63`).
- `-m, --message <STRING>`: Plaintext override; supersedes `engine.message.*`.
- `-e, --public-exponent <u64>`: Public exponent seed. Default `65537`. If left at default, `rsa_keypair.e` from the config is used.
- `--seed <u64>`: Optional RNG seed for deterministic key generation.
- `-c, --config <PATH>`: JSON/JSON5 config path. Default `rsa_config.json`.

# Configuration (rsa_config.json)
Notes:
- Missing config files fall back to built-in defaults; when present, values below are read.
- Unknown keys are ignored by `analysis.rs`. The `padding` and `engine.max_overlap_min` fields are currently not used.
- `rsa_keypair.p` and `rsa_keypair.q` must be set when `rsa_keypair.generate` is `false`.

| Key | Type | Default in `rsa_config.json` | Notes |
| --- | --- | --- | --- |
| `rsa_keypair.generate` | bool | `false` | Generate primes when `true`. |
| `rsa_keypair.e` | u64 | `65537` | Public exponent seed. |
| `rsa_keypair.p` | string (bigint) | `3030152311446024058741` | Prime p when not generating. |
| `rsa_keypair.q` | string (bigint) | `4262327550688715209573` | Prime q when not generating. |
| `padding` | string | `PKCS1v15` | Present for compatibility; currently unused. |
| `engine.r_stress_test_enable` | bool | `false` | Enable stress test over r range. |
| `engine.r_stress_start` | string (bigint) | `12915501679859480667750241440843466838812811` | Stress test range start. |
| `engine.r_stress_end` | string (bigint) | `12915501679859480667750241440843466838812814` | Stress test range end. |
| `engine.r_use_list_enable` | bool | `false` | Use explicit r list. |
| `engine.r_use_list` | array(string) | `[]` | Explicit r candidates. |
| `engine.max_overlap_min` | number | `0.005` | Present for compatibility; currently unused. |
| `engine.override_best_r` | string | `""` | Overrides best r if non-empty. |
| `engine.test_iterations` | u64 | `0` | Number of primary test iterations. |
| `engine.alt_iterations` | u64 | `0` | Number of alternate test iterations. |
| `engine.process_min_count` | u64 | `25` | Minimum r candidates to process. |
| `engine.process_count` | u64 | `25` | Target r candidates per batch. |
| `engine.process_scale` | u32 | `12` | Scaling factor for candidate generation. |
| `engine.process_max_best_attempts` | u64 | `500` | Max attempts to improve best r. |
| `engine.process_min_factor` | u64 | `117` | Minimum factor threshold. |
| `engine.rabin_exponent` | u64 | `3` | Rabin exponent used in candidate math. |
| `engine.min_message_trials` | u64 | `100` | Minimum message trials per r candidate. |
| `engine.overlap_report_threshold` | number | `51` | Overlap % threshold for reporting. |
| `engine.reuse_r_candidates` | bool | `true` | Reuse cached r candidates. |
| `engine.reuse_r_candidates_path` | string | `r_candidates.csv` | Cache file path. |
| `engine.reuse_r_candidates_append_only` | bool | `false` | Append-only reuse file behavior. |
| `engine.r_candidate_mode` | string | `small_primes` | Candidate generation mode. |
| `engine.r_candidate_small_primes` | array(u64) | `[3, 5, 7, 11, 13, 17]` | Small primes for candidate generation. |
| `engine.r_candidate_small_prime_factors` | usize | `3` | Number of small prime factors. |
| `engine.combiner_enable` | bool | `false` | Enable speculative combiner. |
| `engine.combiner_k_oracles` | usize | `5` | Number of oracles to request. |
| `engine.combiner_match_probability` | number | `0.75` | Target oracle match probability. |
| `engine.combiner_tie_breaker` | bool | `true` | Tie-breaking strategy. |
| `engine.base_convert` | bool | `true` | Enable base conversion in analysis. |
| `engine.invert_bits` | bool | `false` | Invert bits during analysis. |
| `engine.use_rs_decrypt` | bool | `true` | Use Rust decrypt path for r analysis. |
| `engine.enciphered_export_enable` | bool | `true` | Export enciphered bins/ramp data. |
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

