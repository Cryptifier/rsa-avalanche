# RSA Analysis Demo
This demo shows homomorphic key switching as a viable method to retrieve up to 95% of the bits of an RSA message on average given a public modulus with private factorization and several homomorphically related keys with easier factorizations.

Proof of concept by Nicholas LaRoche <nlaroche@cryptifier.dev>.

![Example output from `analysis`](95pct.png)

![Example accuracy histogram from `analysis`](90histogram.png)

# Resource Requirements
- Use a ```c8a.12xlarge``` AWS instance with 48 AMD EPYC cores, 16,000 provisioned IOPS and 1,000 MB/s bandwidth for optimal performance.
- Choose a disk size of at least 100 GB to accommodate the caching database and session artifacts.
- Keep statistics logging disabled unless you want to track per-run scoring details in the database, which can significantly increase runtime and disk usage.

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

- Disk I/O and CPU performance should be maximized by choosing an appropriate AWS Instance Type. For example, a ```c8a.12xlarge``` instance with provisioned IOPS and bandwidth is ideal for running each batch. Without this, most of the time spent running the ```analysis``` code will be spent writing to the caching database.

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
- `--avalanche-combination-recursive-group-size <u64>`: Override the per-tier recursive group-size array with one value applied to every recursive tier. Config default `[8]`.
- `--avalanche-combination-recursive-resample-count <u64>`: Override the per-tier recursive resample-count array with one value applied to every recursive tier. Config default `[0]`.
- `--avalanche-combination-majority-vote <bool>`: Use per-bit majority-vote probabilities from each sampled combination. Default `true`.
- `--avalanche-combination-sample-smoothing <bool>`: Apply Jeffreys smoothing to sampled majority-vote probabilities before beam search. Default `false`.
- `--avalanche-combination-majority-vote-print <bool>`: Print a separate sampled-combination majority-vote summary for the selected sample. Default `true`.
- `--avalanche-solver-global-log-enable <bool>`: Print one batch-global majority vote per batch, then a final majority across those batch-global results when the batches target one message. Default `true`.
- `--avalanche-use-top-beam <bool>`: Carry forward the prior tier's top beam-search bits between recursive Avalanche tiers instead of the prior tier's majority-vote bits. Default `true`.
- `analysis` accepts either `rsa-private-key-v1` or `rsa-public-key-v1` in `rsa_keypair.keyfile`.
- When the configured keyfile is public, `analysis` skips the normal RSA round trip. Set `rsa_keypair.private_keyfile` to a matching private YAML if you want a verification peek; otherwise the public-key run is selected by top beam score.

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
- Demo runs require `rsa_keypair.generate = false` with either inline `rsa_keypair.p`/`rsa_keypair.q` or `rsa_keypair.keyfile` supplied.
- `demo` only needs the public modulus and exponent, so `rsa_keypair.keyfile` may point to either a private or public YAML key file.

## `kgen`
```bash
cargo run --bin kgen
cargo run --bin kgen -- --size-mode modulus --modulus-bits 144 --output config/keys/private_key.yaml --public-output config/keys/public_key.yaml
cargo run --bin kgen -- --input-private-key config/keys/private_key.yaml --public-output config/keys/public_key.yaml --force
cargo run --bin kgen -- --input-pgp-public-key public.asc --public-output config/keys/public_key.yaml --force
cargo run --bin kgen -- --input-pgp-file message.asc --pgp-output config/keys/pgp_message.yaml --force
```

- `--size-mode <prime|modulus>`: Choose whether generation is driven by prime size or modulus size. Default `prime`.
- `--prime-bits <u32>`: Prime bit length used in `prime` mode. Default `56` (range `16..=8192`).
- `--modulus-bits <u32>`: Exact modulus bit length targeted in `modulus` mode. Default `144` (range `32..=16384`).
- `-e, --public-exponent <u64>`: Starting public exponent candidate. Default `65537`; the first odd coprime exponent at or above it is used.
- `-o, --output <PATH>`: Private-key YAML output path for generated keys. Default `config/keys/private_key.yaml`.
- `--public-output <PATH>`: Optional public-key YAML output path for the generated or imported private key.
- `--input-private-key <PATH>`: Existing `rsa-private-key-v1` YAML file to convert into `rsa-public-key-v1`, similar to extracting a public key from a private PEM with `openssl`.
- `--input-pgp-public-key <PATH>`: Existing OpenPGP public-key file to convert into `rsa-public-key-v1`. Optionally combine with `--pgp-output` to also save the unpacked packet structure.
- `--input-pgp-file <PATH>`: Existing OpenPGP encrypted or packetized file to unpack into `pgp-file-v1` YAML.
- `--pgp-output <PATH>`: YAML output path for the unpacked `pgp-file-v1` OpenPGP packet representation.
- `--force`: Overwrite an existing output file.
- `--seed <u64>`: Optional deterministic RNG seed for reproducible key generation.
- `--crypto-rng`: Use cryptographic RNGs instead of the standard seeded generator.

# Configuration
Configuration reference material has been moved to [CONFIGS.md](CONFIGS.md).

# Public-Key Workflows
Public-key YAML files are supported for both `analysis` and `demo`. Point `rsa_keypair.keyfile` at an `rsa-public-key-v1` file when you want the run to operate without inline `p` or `q`.

For `analysis`, the plaintext used for scoring still comes from `engine.message.*`, so speculative outputs can still be compared against the chosen/generated message without learning the factorization. If you also set `rsa_keypair.private_keyfile` to a matching `rsa-private-key-v1` file, `analysis` performs a private-key verification peek; otherwise it skips round-trip RSA and ranks the public-key run by beam score.
