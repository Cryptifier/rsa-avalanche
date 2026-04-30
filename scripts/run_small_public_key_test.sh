#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export CONFIG=${CONFIG:-"config/rsa_config_small_public_key_test.json"}
export ANALYSIS_LOG=${ANALYSIS_LOG:-"logs_small_public_key_test.log"}
export SCRIPT_LOG=${SCRIPT_LOG:-"logs_small_public_key_test_script.log"}

exec "${SCRIPT_DIR}/run_small_batch_beam.sh"
