#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=make_rgen_small.sh
source "${SCRIPT_DIR}/make_rgen_small.sh"

usage() {
  cat <<'USAGE'
Usage: make_rgen_grid.sh [--config PATH] [--out-dir DIR] [--pcts "5,10,20,30,40,50"] [--size small|medium]

Options:
  --config PATH   Config JSON/JSON5 file (default: rsa_config_small.json)
  --out-dir DIR   Output directory (default: data/rgen_grid)
  --pcts LIST     Comma-separated percentages (default: 5,10,20,30,40,50)
  --size VALUE    Override size label (small or medium). Otherwise inferred from config name.
  -h, --help      Show this help message
USAGE
}

CONFIG=${RGEN_SMALL_CONFIG}
OUT_DIR="data/rgen_grid"
PCTS_RAW="5,10,20,30,40,50"
SIZE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --pcts)
      PCTS_RAW="${2:-}"
      shift 2
      ;;
    --size)
      SIZE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  echo "Missing --config value." >&2
  usage >&2
  exit 1
fi

if [[ -z "${OUT_DIR}" ]]; then
  echo "Missing --out-dir value." >&2
  usage >&2
  exit 1
fi

if [[ -z "${PCTS_RAW}" ]]; then
  echo "Missing --pcts value." >&2
  usage >&2
  exit 1
fi

IFS=',' read -r -a PCTS <<< "${PCTS_RAW}"

if [[ -z "${SIZE}" ]]; then
  config_base=$(basename "${CONFIG}")
  config_base=${config_base,,}
  if [[ "${config_base}" == *small* ]]; then
    SIZE="small"
  elif [[ "${config_base}" == *medium* ]]; then
    SIZE="medium"
  fi
fi

if [[ -n "${SIZE}" && "${SIZE}" != "small" && "${SIZE}" != "medium" ]]; then
  echo "Invalid --size value: ${SIZE} (expected small or medium)." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

for pct in "${PCTS[@]}"; do
  if [[ ! "${pct}" =~ ^[0-9]+$ ]]; then
    echo "Invalid percentage entry: ${pct}" >&2
    exit 1
  fi
  pct_label=$(printf "%02d" "${pct}")
  if [[ -n "${SIZE}" ]]; then
    output="${OUT_DIR}/rgen_grid_${SIZE}_pct_${pct_label}.csv"
  else
    output="${OUT_DIR}/rgen_grid_pct_${pct_label}.csv"
  fi
  cargo run --bin rgen -- -c "${CONFIG}" -o "${output}" \
    --min-count "${RGEN_SMALL_MIN_COUNT}" \
    --mode "${RGEN_SMALL_MODE}" \
    --small-primes "${RGEN_SMALL_PRIMES}" \
    --max-factors "${RGEN_SMALL_MAX_FACTORS}" \
    --r-bits-percent "${pct}"
done
