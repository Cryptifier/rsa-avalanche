# RSA Demo of Statistical Advantage
This demo shows homomorphic key switching as a viable method to retrieve up to 51% of the bits of an RSA message given a modulus with private factorization and several homomorphically related keys with easier factorizations. This is statistically significant with 51% thresholding over 1,000,000 trials for messages of length k=144 bits.

by Nicholas LaRoche <nlaroche@cryptifier.dev> during 2025.

# Theory
- Use regular RSA encryption using a large modulus N = pq where p and q are large private primes.
- Use phi of phi(N) = (p-1)(q-1) to generate a public/private key pair (e, d) such that ed ≡ 1 (mod phi(N)).
- Use the Tonelli-Shanks approach to increase the relation of the ciphertext mod N to the new ciphertext mod N^k for a small k = 3.
- Use homomorphic base switching to go from mod N^k to mod R where R is an easily factored modulus with at least three factors (more than regular RSA).
- Use the easy factorization of R to retrieve partial information about the original message by calculating a new d' such that ed' ≡ 1 (mod phi(R)).
- Compute the difference between each independent trial with random message and ciphertext using modulus R in a random oracle model to retrieve percentage of bits matching the original message via the ciphertext using modulus N.

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

# Experimental Notes
- I suggest that 51% is a reasonable threshold for success.
- All "max matching bits" logs are regarding the LSB's only. The overlap % is regarding all bits.
- With k=144 bits with at least 51% of the bits matched, observe there is about a 1.5% advantage over random guessing at 1M iterations.
- Canonically RSA should be resiliant against this type of attack.
- Included in this repository are some example R candidates in r_candidates.csv to speed up testing.
- Change the "alt_iterations" parameter in src/bin/analysis.rs to adjust the number of trials (default 1,000,000). About 100,000 trials is more reasonable for quicker testing.

# Results
### Parameters
- Bits per trial: $$k = 144 \textbf{bits}$$
- Threshold: **51%** → success iff $$X \ge 74$$ (since $$0.51 \cdot 144 = 73.44$$)  
- Null model: $$X \sim \mathrm{Binomial}(144,\frac{1}{2})$$
- Trials: $$n = 1000000$$
- Observed successes: $$y = 416159$$

---

### Expected success rate under randomness (null)
Let
$$q = \Pr[X \ge 74], \quad X \sim \mathrm{Binomial}(144,\frac{1}{2}).$$
Using the normal approximation with continuity correction,
$$q \approx 0.401.$$

So the null expectation is
$$\mathbb{E}[Y] = nq \approx 1{,}000{,}000 \cdot 0.401 = 401{,}000.$$

---

### Compare observation to expectation
Observed excess: $$\Delta = y - \mathbb{E}[Y] \approx 416159 - 401000 = 15159.$$

Null standard deviation: $$\sigma_Y = \sqrt{nq(1-q)} \approx \sqrt{1000000 \cdot 0.401 \cdot 0.599} \approx \sqrt{240199} \approx 490.$$

z-score: $$z \approx \frac{\Delta}{\sigma_Y} \approx \frac{15159}{490} \approx 30.9.$$

---

### Interpretation
- Observed success rate: $$\hat q = \frac{416159}{1000000} = 0.416159.$$
- Null expected rate: $$q \approx 0.401$$.
- The deviation is about **31σ**, i.e. astronomically inconsistent with the pure-random null (p-value effectively $$10^{-100}$$).

**Conclusion:**
At a 51% threshold with $$k=144$$, seeing **416,159 successes out of 1,000,000** is overwhelming evidence of a real upward bias, assuming independence and no multiple-comparisons cherry-picking.

# Output
```
Prime p (72 bits): 3030152311446024058741
Prime q (72 bits): 4262327550688715209573
Modulus n (144 bits): 12915501679859480667750241440843407877527593
phi(n): 12915501679859480667742948960981273138259280
Public exponent e: 65537
Private exponent d: 1780347623111380569028026930031963952134433
Plaintext (hex): 7588c4352bf0387a1b1ddc1d04cfe4768a9a
Ciphertext (hex): 575304cf32e511155fe445b3117d2ca8dc2a
Recovered (hex): 7588c4352bf0387a1b1ddc1d04cfe4768a9a
Reuse enabled; loading r candidates from r_candidates.csv
Loaded 5 r candidates from reuse file r_candidates.csv
Generated 5 r candidates for testing
Test iterations progress: 50% (1/2)
Reuse enabled; loading r candidates from r_candidates.csv
Loaded 5 r candidates from reuse file r_candidates.csv
Generated 5 r candidates for testing
Test iterations progress: 100% (2/2)
Best r candidate: 11790428346265840583865602058950085154805827
Factors: [(2801, 1), (95957, 1), (949381, 1), (46206096079782974891103462331, 1)]
Matching bits: LSB run 0 / overlap 88 of 143 bits
Worst r candidate: 11179149394623257219587568505693593936803481
Factors: [(16913213, 1), (25854593, 1), (25564949321231822668323575309, 1)]
Matching bits: LSB run 0 / overlap 88 of 143 bits
Matching bits stats: mean 0.0000, std dev 0.0000, min 0.0000, max 0.0000, n 2
Matching overlap stats (%): mean 61.5385, std dev 0.0000, min 61.5385, max 61.5385, n 2
Overlaps >= 51.00%: count 2
Max matching bits over all test cases: 0
Alt iterations progress: 10% (100000/1000000)
Alt iterations progress: 20% (200000/1000000)
Alt iterations progress: 30% (300000/1000000)
Alt iterations progress: 40% (400000/1000000)
Alt iterations progress: 50% (500000/1000000)
Alt iterations progress: 60% (600000/1000000)
Alt iterations progress: 70% (700000/1000000)
Alt iterations progress: 80% (800000/1000000)
Alt iterations progress: 90% (900000/1000000)
Alt iterations progress: 100% (1000000/1000000)
Alt iterations stats: bits mean 0.9988, std dev 1.4144, min 0.0000, max 26.0000; overlap % mean 49.8013, std dev 4.2044, min 30.0699, max 68.5315; overlaps >= 51.00% count 416159; max bits 26
...
```
