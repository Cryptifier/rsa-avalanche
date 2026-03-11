#!/usr/bin/env bash
cargo run --bin rgen -- -c config/rsa_config_demo.json -o data/rgen_output_demo_2.csv --min-count 20000 --mode small-primes --small-primes 117,1103,1009,1913 --max-factors 6
