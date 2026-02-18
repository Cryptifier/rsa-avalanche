This project is a demo and analysis tool that aims to show a statistical advantage in recovering RSA message bits when related, easier-to-factor moduli are available. The README frames this as homomorphic key switching and base conversion (with references to Tonelli-Shanks–style lifting and switching from `N^k` to a more factorable modulus), while the code implements a compact RSA round-trip to validate correctness and then drives the statistical experiment. In practice, the tool generates or loads primes, computes `n = p*q` and `phi(n)`, chooses a public exponent, derives the private exponent by modular inverse, and performs modular exponentiation to encrypt/decrypt. Message handling selects fixed or random plaintext, enforces `m < n`, and prints hex outputs to make runs reproducible and auditable. See the overview and experiment framing in `README.md`.

The statistical-advantage analysis adds a secondary modulus search layer that operationalizes the README’s “easier factorization” story. It builds candidate `r` moduli from configurable small-prime products, optionally reuses cached candidates, and factors them to derive alternate decryption exponents. Decryption during the analysis uses the candidate modulus `r` and its derived exponent, not `n` or the private exponent tied to `n` (those are only used for the baseline RSA round-trip validation). The process models converting from the RSA ciphertext space into a more factorable modulus and then measures how many bits of the original message can be recovered through these alternate factorizations across repeated trials. The implementation details live in `src/bin/analysis.rs`.

The analytics loop runs repeated trials, records overlap statistics and LSB runs, and surfaces best- and worst-case candidates relative to a threshold (the README highlights 51%). It includes an optional speculative combiner that aggregates noisy oracle bits with majority voting, plus signal-processing routines that scan exported enciphered bins for ramp-like patterns and summarize their strength. CSV exports and plotted artifacts are produced to support offline inspection of distributions, thresholds, and signal features. The companion Python script `scripts/enciphered_bins_video.py` turns exported `enciphered_*` CSVs into a 3D scrolling surface video, supports smoothing and z-scale transforms, and can render frames in parallel before stitching with ffmpeg (or fall back to frame output when ffmpeg is unavailable).

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
