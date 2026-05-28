#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

MODULUS_BITS=${MODULUS_BITS:-512}
PREP_KEY_BASENAME=${PREP_KEY_BASENAME:-"prep_pgp_${MODULUS_BITS}"}
PREP_OUTPUT_SUBDIR=${PREP_OUTPUT_SUBDIR:-"config/pgp/${MODULUS_BITS}"}
PREP_PGP_CIPHER_ALGO=${PREP_PGP_CIPHER_ALGO:-AES128}
ANALYSIS_BATCHES=${ANALYSIS_BATCHES:-2}

if [[ "${PREP_OUTPUT_SUBDIR}" = /* ]]; then
  PREP_OUTPUT_DIR="${PREP_OUTPUT_SUBDIR}"
else
  PREP_OUTPUT_DIR="${REPO_ROOT}/${PREP_OUTPUT_SUBDIR}"
fi

PRIVATE_KEY="${REPO_ROOT}/config/keys/${PREP_KEY_BASENAME}_private_key.yaml"
PUBLIC_KEY="${REPO_ROOT}/config/keys/${PREP_KEY_BASENAME}_public_key.yaml"
FIRST_ASC="${PREP_OUTPUT_DIR}/prep_blob_1.asc"
EXPECTED_PAYLOAD_PATH="${PREP_OUTPUT_DIR}/prep_blob_1.pkcs1_v1_5_payload.hex"

export CONFIG=${CONFIG:-"${REPO_ROOT}/config/rsa_config_small_public_key_demo_${MODULUS_BITS}.json"}
export ANALYSIS_LOG=${ANALYSIS_LOG:-"logs_small_public_key_demo_${MODULUS_BITS}.log"}
export SCRIPT_LOG=${SCRIPT_LOG:-"logs_small_public_key_demo_${MODULUS_BITS}_script.log"}

PREP_PGP_MODULUS_BITS="${MODULUS_BITS}" \
PREP_PGP_KEY_BASENAME="${PREP_KEY_BASENAME}" \
PREP_PGP_OUTPUT_DIR="${PREP_OUTPUT_DIR}" \
PREP_PGP_CIPHER_ALGO="${PREP_PGP_CIPHER_ALGO}" \
  "${SCRIPT_DIR}/do_prep_pgp.sh"

python3 "${SCRIPT_DIR}/build_public_key_demo.py" \
  --input-asc "${FIRST_ASC}" \
  --private-key "${PRIVATE_KEY}" \
  --public-key "${PUBLIC_KEY}" \
  --template-config "${REPO_ROOT}/config/rsa_config_small_batch.json" \
  --output-config "${CONFIG}" \
  --output-dir "${PREP_OUTPUT_DIR}" \
  --analysis-batches "${ANALYSIS_BATCHES}"

if [[ ! -f "${EXPECTED_PAYLOAD_PATH}" ]]; then
  echo "Missing expected PKCS#1 payload artifact ${EXPECTED_PAYLOAD_PATH}" >&2
  exit 1
fi

EXPECTED_PAYLOAD_HEX=$(tr -d '[:space:]' < "${EXPECTED_PAYLOAD_PATH}" | tr 'A-F' 'a-f')
if [[ -z "${EXPECTED_PAYLOAD_HEX}" ]]; then
  echo "Expected PKCS#1 payload artifact ${EXPECTED_PAYLOAD_PATH} is empty" >&2
  exit 1
fi
PAYLOAD_BITS=$(( ${#EXPECTED_PAYLOAD_HEX} * 4 ))
export ANALYSIS_BITS_DECRYPT="${ANALYSIS_BITS_DECRYPT:-${PAYLOAD_BITS}}"

set +e
"${SCRIPT_DIR}/run_small_batch_beam.sh"
runner_status=$?
set -e

GLOBAL_MAJORITY_HEX=$(
  sed -n 's/.*Avalanche global majority vote:.* hex \([0-9a-fA-F][0-9a-fA-F]*\)$/\1/p' "${ANALYSIS_LOG}" | tail -n 1 | tr 'A-F' 'a-f'
)

if [[ -z "${GLOBAL_MAJORITY_HEX}" ]]; then
  echo "Missing final Avalanche global majority vote line in ${ANALYSIS_LOG}" >&2
  if [[ "${runner_status}" -ne 0 ]]; then
    exit "${runner_status}"
  fi
  exit 1
fi

echo "Expected PKCS#1 v1.5 payload hex: ${EXPECTED_PAYLOAD_HEX}"
echo "Observed global-majority hex:    ${GLOBAL_MAJORITY_HEX}"

if [[ "${GLOBAL_MAJORITY_HEX}" != "${EXPECTED_PAYLOAD_HEX}" ]]; then
  echo "Global majority vote did not match the stored PKCS#1 v1.5 payload." >&2
  exit 1
fi

if [[ "${runner_status}" -ne 0 ]]; then
  exit "${runner_status}"
fi

echo "Global majority vote matched the stored PKCS#1 v1.5 payload."
