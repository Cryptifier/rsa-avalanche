#!/usr/bin/env bash
set -euo pipefail

RUNS=${RUNS:-10}
CONFIG=${CONFIG:-"rsa_config_base_256.json"}
SCRIPT_LOG=${SCRIPT_LOG:-"logs_demo_script.log"}
RESUME=${RESUME:-0}
PLAINTEXT_HEX=${PLAINTEXT_HEX:-""}

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
  cargo run --bin demo -- --config "${CONFIG}" --encrypt --plaintext-hex "0x${PLAINTEXT_HEX}"
  progress_bar "${i}" "${RUNS}"
done

printf "\n"
