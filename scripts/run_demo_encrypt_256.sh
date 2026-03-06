#!/usr/bin/env bash
set -euo pipefail

RUNS=${RUNS:-1}
CONFIG=${CONFIG:-"config/rsa_config_demo.json"}
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

best_sum=0
best_count=0
major_sum=0
major_count=0

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

  cargo run --bin demo -- --batch-size 2000 --batches 1 --config "${CONFIG}" --encrypt --plaintext-hex "0x${PLAINTEXT_HEX}" | tee "${encrypt_output}"
  ciphertext_hex=$(grep -m1 "Ciphertext (hex):" "${encrypt_output}" | awk '{print $3}')
  if [[ -z "${ciphertext_hex}" ]]; then
    echo "Failed to capture ciphertext from demo output." >&2
    rm -f "${encrypt_output}" "${decrypt_output}"
    exit 1
  fi

  cargo run --bin demo -- --config "${CONFIG}" --batches 1 --batch-size 1000 --decrypt --ciphertext "0x${ciphertext_hex}" | tee "${decrypt_output}"
  best_case_hex=$(grep -m1 "Recovered (best-case) hex:" "${decrypt_output}" | awk '{print $4}')
  majority_hex=$(grep -m1 "Recovered (majority) hex:" "${decrypt_output}" | awk '{print $4}')

  if [[ -x "${DIFF_SCRIPT}" ]]; then
    if [[ -n "${best_case_hex}" ]]; then
      echo "Best-case vs plaintext bit diff:"
      diff_out="$("${DIFF_SCRIPT}" "0x${PLAINTEXT_HEX}" "${best_case_hex}")"
      echo "${diff_out}"
      match_line=$(echo "${diff_out}" | grep -m1 "^Match:")
      match_pct=$(echo "${match_line}" | awk '{print $2}' | tr -d '%')
      if [[ -n "${match_pct}" ]]; then
        best_sum=$(awk -v s="${best_sum}" -v v="${match_pct}" 'BEGIN { printf "%.6f", s + v }')
        best_count=$((best_count + 1))
      fi
    fi
    if [[ -n "${majority_hex}" ]]; then
      echo "Majority vs plaintext bit diff:"
      diff_out="$("${DIFF_SCRIPT}" "0x${PLAINTEXT_HEX}" "${majority_hex}")"
      echo "${diff_out}"
      match_line=$(echo "${diff_out}" | grep -m1 "^Match:")
      match_pct=$(echo "${match_line}" | awk '{print $2}' | tr -d '%')
      if [[ -n "${match_pct}" ]]; then
        major_sum=$(awk -v s="${major_sum}" -v v="${match_pct}" 'BEGIN { printf "%.6f", s + v }')
        major_count=$((major_count + 1))
      fi
    fi
  else
    echo "Diff script not found or not executable: ${DIFF_SCRIPT}" >&2
  fi

  if [[ "${best_count}" -gt 0 ]]; then
    best_avg=$(awk -v s="${best_sum}" -v n="${best_count}" 'BEGIN { printf "%.2f", s / n }')
  else
    best_avg="N/A"
  fi
  if [[ "${major_count}" -gt 0 ]]; then
    major_avg=$(awk -v s="${major_sum}" -v n="${major_count}" 'BEGIN { printf "%.2f", s / n }')
  else
    major_avg="N/A"
  fi
  echo "Running averages: best-case ${best_avg}%, majority ${major_avg}%"

  rm -f "${encrypt_output}" "${decrypt_output}"
  progress_bar "${i}" "${RUNS}"
done

printf "\n"
