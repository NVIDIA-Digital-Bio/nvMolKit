#!/usr/bin/env bash

set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <local_conda_endpoint> <pytest_directory>" >&2
  exit 1
fi

LOCAL_CONDA_ENDPOINT=$1
PYTEST_DIR=$2

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERIFY_SCRIPT="$SCRIPT_DIR/verify_conda_distributable.sh"

if [[ ! -x $VERIFY_SCRIPT ]]; then
  if [[ -f $VERIFY_SCRIPT ]]; then
    chmod +x "$VERIFY_SCRIPT"
  else
    echo "verify_conda_distributable.sh not found at '$VERIFY_SCRIPT'" >&2
    exit 1
  fi
fi

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

SUMMARY_LOG="$LOG_DIR/summary.log"
printf 'Verification Summary - %s\n' "$(date)" >"$SUMMARY_LOG"

PYTHON_VERSIONS=("3.11" "3.12" "3.13")
RDKIT_VERSIONS=("2025.03.1" "2024.09.6")

for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do
  for RDKIT_VERSION in "${RDKIT_VERSIONS[@]}"; do
    PY_LABEL=${PYTHON_VERSION//./}
    RD_LABEL=${RDKIT_VERSION//./}
    LOG_FILE="$LOG_DIR/verify_py${PY_LABEL}_rdkit${RD_LABEL}.log"

    if "$VERIFY_SCRIPT" "$LOCAL_CONDA_ENDPOINT" "$PYTEST_DIR" "$RDKIT_VERSION" "$PYTHON_VERSION" >"$LOG_FILE" 2>&1; then
      printf 'PASS: Python %s, RDKit %s\n' "$PYTHON_VERSION" "$RDKIT_VERSION" | tee -a "$SUMMARY_LOG"
    else
      printf 'FAIL: Python %s, RDKit %s\n' "$PYTHON_VERSION" "$RDKIT_VERSION" | tee -a "$SUMMARY_LOG"
    fi
  done
done

echo "Verification completed. Summary available at $SUMMARY_LOG"

