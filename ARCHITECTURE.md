AI was used to build this proof-of-concept.

This project is a demo and analysis tool that aims to show a statistical advantage in recovering RSA message bits when related, easier-to-factor moduli are available. The README frames this as homomorphic key switching and base conversion (with references to Tonelli-Shanks–style lifting and switching from `N^k` to a more factorable modulus), while the code implements a compact RSA round-trip to validate correctness and then drives the statistical experiment. In practice, the tool generates or loads primes, computes `n = p*q` and `phi(n)`, chooses a public exponent, derives the private exponent by modular inverse, and performs modular exponentiation to encrypt/decrypt. Message handling selects fixed or random plaintext, enforces `m < n`, and prints hex outputs to make runs reproducible and auditable. See the overview and experiment framing in `README.md`.

The statistical-advantage analysis adds a secondary modulus search layer that operationalizes the README’s “easier factorization” story. It builds candidate `r` moduli from configurable small-prime products, optionally reuses cached candidates, and factors them to derive alternate decryption exponents. Decryption during the analysis uses the candidate modulus `r` and its derived exponent, not `n` or the private exponent tied to `n` (those are only used for the baseline RSA round-trip validation). The process models converting from the RSA ciphertext space into a more factorable modulus and then measures how many bits of the original message can be recovered through these alternate factorizations across repeated trials. The analysis CLI can optionally shift ciphertext by encrypted 2 before base conversion and reports best-case per-bit recovery alongside the speculative oracle match. The implementation details live in `src/bin/analysis.rs`.

The demo CLI (`src/bin/demo.rs`) focuses on single-shot speculative decryption: it can encrypt a provided plaintext with the configured RSA key or recover a provided ciphertext by screening per-bit oracles and producing best-case and majority-vote recovered hex strings. Demo runs require a fixed RSA keypair in the config and reuse the same r-candidate generation settings, oracle screening iterations, and optional ciphertext shift flag as the analysis pipeline.

The results from the analysis using the `config/rsa_config_base_256.json` config are based on the following premise:
  - Predictions for the match percentage can either be a positive or negative result requiring negation. It has been observed that 50.39% to 60% of bits can be recovered consistently and that when probability is below 50% the entire prediction is bit-wise negated to reveal the other half of the bits as true.
  - This script runs the process with repeated trials `scripts/run_medium_batch.sh` and requires a SEED_START=123456 environment variable to alter the reproducible results for subsequent tests. Running without setting this environment variable results in reproducible tests with 100 iterations. The proof of concept results are available in the `poc_script.log` file.
  - All tests using this config and the run script use the ChaCha cryptographic number generator.

The analytics loop runs repeated trials, records overlap statistics and LSB runs, and surfaces best- and worst-case candidates relative to a threshold (the README highlights 51%). It includes an optional speculative combiner that aggregates noisy oracle bits with majority voting, plus signal-processing routines that scan exported enciphered bins for ramp-like patterns and summarize their strength. CSV exports and plotted artifacts are produced to support offline inspection of distributions, thresholds, and signal features. The companion Python script `scripts/enciphered_bins_video.py` turns exported `enciphered_*` CSVs into a 3D scrolling surface video, supports smoothing and z-scale transforms, and can render frames in parallel before stitching with ffmpeg (or fall back to frame output when ffmpeg is unavailable).

**R Candidate Generation**

The r-candidate module (`src/r_candidates.rs`) provides multiple strategies for building candidate moduli `r`. These methods are used to derive alternate exponents and evaluate bit recovery accuracy. The ciphertext-stream variant is implemented for experimentation but is not currently wired into `analysis` or `rgen`.

**Pseudocode: Factoring Mode**
```
inputs: n, settings, rng
if settings.override_best_r:
  r = override_best_r
  if r is prime: return []
  return [(r, factor(r))]

target = max(settings.process_count, settings.process_min_count)
for attempt in 1..target * scale:
  r = random_composite_near(n, rng)
  factors = factor(r)
  if factors meet min_factor and count constraints:
    collect r
  stop when target collected
return candidates
```

**Pseudocode: Small-Primes Mode**
```
inputs: settings, rng
target = max(settings.process_count, settings.process_min_count)
target_bits = settings.target_bit_length
seed candidates from reuse file if enabled
while candidates < target:
  pick K small primes from list (>= min_factor)
  choose a larger prime so r reaches target_bits
  r = product(small_primes) * large_prime
  collect r and its factor list
return candidates
```

**Pseudocode: Ciphertext Stream (c^x mod N)**
```
inputs: ciphertext c, modulus n, count, start_exponent
x = start_exponent
for i in 1..count:
  r = c^x mod n
  x = x + 1
  collect r (factor list empty)
return candidates
```

**Algorithms**
- RSA key generation (probable primes), modular inverse, and modular exponentiation
- Homomorphic base conversion and candidate modulus construction
- Integer factorization of candidate moduli
- Majority-vote combiner over oracle bit distributions
- Ramp detection and signal strength estimation over binned enciphered data

**References**
- `README.md`
- `src/bin/analysis.rs`
- `scripts/enciphered_bins_video.py`

**Python Script Example**
```bash
python3 scripts/enciphered_bins_video.py \
  --input enciphered_decryption_bins.csv \
  --output enciphered_bins.mp4 \
  --metric float \
  --fps 24 \
  --window 60 \
  --parallel
```
