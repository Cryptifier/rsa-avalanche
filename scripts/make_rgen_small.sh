#!/usr/bin/env bash
cargo run --bin rgen -- -c rsa_config_small.json -o rgen_output_smaller.csv --min-count 1000000 --mode small-primes --small-primes 117,1103,1009,1913 --max-factors 6 --r-bits-percent=30
