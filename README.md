# RSA Analysis Demo
This demo shows homomorphic key switching as a viable method to retrieve up to 74% of the bits of an RSA message on average given a public modulus with private factorization and several homomorphically related keys with easier factorizations.

Proof of concept by Nicholas LaRoche <nlaroche@cryptifier.dev>.

![Example output from `analysis`](77pct.png)

# Theory
- Use regular RSA encryption using a large modulus `N = pq` where `p` and `q` are large private primes.
- Use homomorphic base switching to go from mod `N` to mod `N^0.850` where `N^0.850` is an easily factored modulus with at least three factors.
- Use the easy factorization of `R` to retrieve partial information about the original message by calculating a new `d'` such that `ed' ≡ 1 (mod phi(N^0.850))`.
- A bit-wise speculative decryption oracle is used to recover message bits.

# Setup
- Use Linux and install Rust.
```bash
cargo build --bin analysis
cargo build --bin demo
cargo build --bin kgen
```

# Tool Usage
## Main test path
Use `make demo` as the primary way to exercise the current proof of concept.

```bash
make demo
```

This runs:

```bash
RUNS=20 SEED_START=2100000 ./scripts/run_small_batch_beam.sh
```

The runner executes `analysis` in release mode against the small-batch Avalanche configuration, records per-run logs, and summarizes match percentages across the batch sweep. Override `RUNS`, `SEED_START`, `CONFIG`, `ANALYSIS_EXTRA_ARGS`, `AVALANCHE_BATCHES`, or `AVALANCHE_BATCH_SIZE` in the environment when you want a different replay.

## `analysis.rs` (POC)
`src/bin/analysis.rs` is where the current proof of concept lives.

```bash
cargo run --bin analysis -- --config config/rsa_config_small_batch.json
```

- `-b, --bits <u32>`: Bit-length for generated primes when `rsa_keypair.generate` is `true`. Default `56` (range `16..=8192`).
- `-m, --message <STRING>`: Plaintext override; supersedes `engine.message.*`.
- `-e, --public-exponent <u64>`: Public exponent seed. Default `65537`. If left at default, `rsa_keypair.e` from the config is used.
- `--seed <u64>`: Optional RNG seed for deterministic key generation.
- `--crypto-rng`: Use cryptographic RNGs for sampling and candidate generation.
- `-c, --config <PATH>`: JSON/JSON5 config path. Default `config/rsa_config_small_batch.json`.
- `--r-candidate-target-exponent <DECIMAL>`: Override the speculative retarget exponent used to build `r` candidates.
- `--tests`: Run extended analysis tests and sufficiency checks.
- `--export`: Export oracle entropy timeline charts and enciphered CSV artifacts.
- `--session-json <PATH>`: Output analytics session JSON. Default `session.json`.
- `--shift`: Multiply ciphertext by encrypted `2` before base conversion.
- `--batches <u64>`: Number of `r`-candidate accuracy batches to run.
- `--batch-size <u64>`: Number of ciphertext/message variants scored per batch before Avalanche sampling.
- `--avalanche-combination-samples <u64>`: Number of sampled combinations evaluated by Avalanche per batch. Default `100`.
- `--avalanche-combination-size <u64>`: Number of scored items taken in each sampled combination. Default `50`.
- `--avalanche-combination-pool-size <u64>`: Legacy compatibility override recorded in session metadata; runtime sampling now uses the full batch-sized pool.
- `--avalanche-combination-recursion-depth <u64>`: Number of Avalanche tiers to execute, including the initial sampled-input tier. Default `1`.
- `--avalanche-combination-recursive-group-size <u64>`: Number of prior-tier sample outputs grouped into each recursive Avalanche call. Default `8`.
- `--avalanche-combination-majority-vote <bool>`: Use per-bit majority-vote probabilities from each sampled combination. Default `true`.
- `--avalanche-combination-sample-smoothing <bool>`: Apply Jeffreys smoothing to sampled majority-vote probabilities before beam search. Default `false`.
- `--avalanche-combination-majority-vote-print <bool>`: Print a separate sampled-combination majority-vote summary for the selected sample. Default `true`.
- `--avalanche-use-top-beam <bool>`: Carry forward the prior tier's top beam-search bits between recursive Avalanche tiers instead of the prior tier's majority-vote bits. Default `true`.

## `demo.rs` (WORK-IN-PROGRESS)
`src/bin/demo.rs` is separate from the proof of concept in `analysis.rs`. Treat it as a work-in-progress utility for direct encrypt/decrypt experiments rather than the main validation path.

```bash
cargo run --bin demo -- --encrypt --plaintext-hex 74657374
cargo run --bin demo -- --decrypt --ciphertext 0x1234
```

- `-c, --config <PATH>`: JSON/JSON5 config path. Default `config/rsa_config_small_batch.json`.
- `--encrypt`: Encrypt a plaintext hex string with the configured RSA key.
- `--decrypt`: Run speculative decryption with per-bit oracle screening.
- `--plaintext-hex <HEX>`: Plaintext hex string (required with `--encrypt`).
- `--ciphertext <VALUE>`: Ciphertext override (decimal or hex). Falls back to `verify.ciphertext_hex` or `verify.ciphertext` in the config.
- `--shift`: Multiply ciphertext by encrypted `2` before base conversion.
- Demo runs require `rsa_keypair.generate = false` with `rsa_keypair.p` and `rsa_keypair.q` supplied.

## `kgen`
```bash
cargo run --bin kgen
cargo run --bin kgen -- --size-mode modulus --modulus-bits 144 --output config/keys/private_key.yaml
```

- `--size-mode <prime|modulus>`: Choose whether generation is driven by prime size or modulus size. Default `prime`.
- `--prime-bits <u32>`: Prime bit length used in `prime` mode. Default `56` (range `16..=8192`).
- `--modulus-bits <u32>`: Exact modulus bit length targeted in `modulus` mode. Default `144` (range `32..=16384`).
- `-e, --public-exponent <u64>`: Starting public exponent candidate. Default `65537`; the first odd coprime exponent at or above it is used.
- `-o, --output <PATH>`: YAML output path. Default `config/keys/private_key.yaml`.
- `--force`: Overwrite an existing output file.
- `--seed <u64>`: Optional deterministic RNG seed for reproducible key generation.
- `--crypto-rng`: Use cryptographic RNGs instead of the standard seeded generator.

# Configuration
Configuration reference material has been moved to [CONFIGS.md](CONFIGS.md).

# Why `p` and `q` are required today
This repository's proof of concept currently covers the Avalanche method implemented in `analysis.rs`. That path still depends on the private factorization inputs `p` and `q` for setup, evaluation, and comparison of recovered output. Proving the approach directly on public `N` alone is not what the current POC implements yet, which is why the documented configs still require `p` and `q`.
