#!/usr/bin/env bash
cargo run --bin rgen -- -c rsa_config.json -o rgen_output_base.csv --min-count 1000000 --mode small-primes --small-primes 117,1103,1009,1913 --max-factors 6
