#!/usr/bin/env bash
set -euo pipefail

RUNS=${RUNS:-10}
CONFIG=${CONFIG:-"rsa_config_base_256.json"}
SCRIPT_LOG=${SCRIPT_LOG:-"logs_demo_script.log"}
RESUME=${RESUME:-0}
PLAINTEXT_HEX=${PLAINTEXT_HEX:-""}
DIFF_SCRIPT=${DIFF_SCRIPT:-"scripts/hex_bit_diff.py"}

BLUE=$'\033[0;34m'
RESET=$'\033[0m'

if [[ "${RESUME}" != "1" ]]; then
  : > "${SCRIPT_LOG}"
fi

exec > >(tee -a "${SCRIPT_LOG}") 2>&1

progress_bar() {
  local current=$1
  local total=$2
  local width=30
  local percent=$((current * 100 / total))
  local filled=$((percent * width / 100))
  local empty=$((width - filled))
  local bar
  local pad
  bar=$(printf "%${filled}s" "" | tr ' ' '#')
  pad=$(printf "%${empty}s" "")
  printf "\r${BLUE}[%-${width}s]${RESET} %3d%% (%d/%d)" "${bar}${pad}" "${percent}" "${current}" "${total}"
}

if [[ -z "${PLAINTEXT_HEX}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PLAINTEXT_HEX=$(python3 - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
)
  else
    echo "python3 is required to generate a 256-bit message; set PLAINTEXT_HEX instead." >&2
    exit 1
  fi
fi

if [[ ${#PLAINTEXT_HEX} -ne 64 ]]; then
  echo "PLAINTEXT_HEX must be 64 hex characters (256 bits)." >&2
  exit 1
fi

echo "Running ${RUNS} demo encrypt iterations with config ${CONFIG}"
echo "Plaintext (hex): 0x${PLAINTEXT_HEX}"

for i in $(seq 1 "${RUNS}"); do
  echo ""
  echo "===== RUN ${i} ====="
  encrypt_output="$(mktemp)"
  decrypt_output="$(mktemp)"

  cargo run --bin demo -- --config "${CONFIG}" --encrypt --plaintext-hex "0x${PLAINTEXT_HEX}" | tee "${encrypt_output}"
  ciphertext_hex=$(grep -m1 "Ciphertext (hex):" "${encrypt_output}" | awk '{print $3}')
  if [[ -z "${ciphertext_hex}" ]]; then
    echo "Failed to capture ciphertext from demo output." >&2
    rm -f "${encrypt_output}" "${decrypt_output}"
    exit 1
  fi

  cargo run --bin demo -- --config "${CONFIG}" --decrypt --ciphertext "${ciphertext_hex}" | tee "${decrypt_output}"
  best_case_hex=$(grep -m1 "Recovered (best-case) hex:" "${decrypt_output}" | awk '{print $4}')
  majority_hex=$(grep -m1 "Recovered (majority) hex:" "${decrypt_output}" | awk '{print $4}')

  if [[ -x "${DIFF_SCRIPT}" ]]; then
    if [[ -n "${best_case_hex}" ]]; then
      echo "Best-case vs plaintext bit diff:"
      "${DIFF_SCRIPT}" "0x${PLAINTEXT_HEX}" "${best_case_hex}"
    fi
    if [[ -n "${majority_hex}" ]]; then
      echo "Majority vs plaintext bit diff:"
      "${DIFF_SCRIPT}" "0x${PLAINTEXT_HEX}" "${majority_hex}"
    fi
  else
    echo "Diff script not found or not executable: ${DIFF_SCRIPT}" >&2
  fi

  rm -f "${encrypt_output}" "${decrypt_output}"
  progress_bar "${i}" "${RUNS}"
done

printf "\n"
