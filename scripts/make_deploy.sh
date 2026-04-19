#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  cat <<'USAGE'
Usage: make_deploy.sh [--output PATH]

Options:
  --output PATH  Output zip path (default: ./rsa-poc-YYYYmmdd_HHMMSS.zip)
  -h, --help     Show this help message
USAGE
}

OUT_ZIP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUT_ZIP="${2:-}"
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

if [[ -z "${OUT_ZIP}" ]]; then
  timestamp=$(date +"%Y%m%d_%H%M%S")
  OUT_ZIP="${ROOT_DIR}/rsa-poc-${timestamp}.zip"
fi
bundle_name="rsa-poc-$(date +"%Y%m%d_%H%M%S")"

TMP_DIR=$(mktemp -d)
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

rsync -a --prune-empty-dirs \
  --include '/src/***' \
  --include '/Cargo.toml' \
  --include '/Cargo.lock' \
  --include '/Makefile' \
  --include '/config/' \
  --include '/config/rsa_config_demo.json' \
  --include '/config/rsa_config_small_batch.json' \
  --include '/data/rgen_output_256.csv' \
  --include '/*.md' \
  --include '/LICENSE' \
  --include '/scripts/***' \
  --exclude '/historical/***' \
  --exclude '/images/***' \
  --exclude '/logs/***' \
  --exclude '/data/***' \
  --exclude '/target/***' \
  --exclude '/*.json' \
  --exclude '*' \
  "${ROOT_DIR}/" "${TMP_DIR}/${bundle_name}/"

(
  cd "${TMP_DIR}"
  zip -qr "${OUT_ZIP}" .
)

echo "Deploy bundle written to ${OUT_ZIP}"
