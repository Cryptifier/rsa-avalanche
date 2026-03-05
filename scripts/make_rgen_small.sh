#!/usr/bin/env bash
set -euo pipefail

RGEN_SMALL_CONFIG=${RGEN_SMALL_CONFIG:-"config/rsa_config_small.json"}
RGEN_SMALL_OUTPUT=${RGEN_SMALL_OUTPUT:-"data/rgen_output_smaller.csv"}
RGEN_SMALL_MIN_COUNT=${RGEN_SMALL_MIN_COUNT:-1000000}
RGEN_SMALL_MODE=${RGEN_SMALL_MODE:-"small-primes"}
RGEN_SMALL_PRIMES=${RGEN_SMALL_PRIMES:-"117,1103,1009,1913"}
RGEN_SMALL_MAX_FACTORS=${RGEN_SMALL_MAX_FACTORS:-6}
RGEN_SMALL_R_BITS_PERCENT=${RGEN_SMALL_R_BITS_PERCENT:-30}

RGEN_SMALL_ARGS=(
  --min-count "${RGEN_SMALL_MIN_COUNT}"
  --mode "${RGEN_SMALL_MODE}"
  --small-primes "${RGEN_SMALL_PRIMES}"
  --max-factors "${RGEN_SMALL_MAX_FACTORS}"
  --r-bits-percent "${RGEN_SMALL_R_BITS_PERCENT}"
)

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  cargo run --bin rgen -- -c "${RGEN_SMALL_CONFIG}" -o "${RGEN_SMALL_OUTPUT}" "${RGEN_SMALL_ARGS[@]}"
fi
