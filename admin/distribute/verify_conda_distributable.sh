#!/usr/bin/env bash

set -exuo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <local_conda_endpoint> <pytest_directory> <rdkit_version> <python_version>" >&2
  exit 1
fi

LOCAL_CONDA_ENDPOINT=$1
PYTEST_DIR=$2
RDKIT_VERSION=$3
PYTHON_VERSION=$4

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH" >&2
  exit 1
fi

if [[ ! -d $PYTEST_DIR ]]; then
  echo "pytest directory '$PYTEST_DIR' does not exist" >&2
  exit 1
fi

if [[ $RDKIT_VERSION != 20*.*.* ]]; then
  echo "rdkit version must match 20xx.x.x (received '$RDKIT_VERSION')" >&2
  exit 1
fi

if [[ $PYTHON_VERSION != 3.* ]]; then
  echo "python version must match 3.xx (received '$PYTHON_VERSION')" >&2
  exit 1
fi

LOCAL_CHANNEL_SPEC=$LOCAL_CONDA_ENDPOINT

if [[ $LOCAL_CHANNEL_SPEC != file://* ]]; then
  if [[ ! -d $LOCAL_CHANNEL_SPEC ]]; then
    echo "local conda endpoint '$LOCAL_CHANNEL_SPEC' does not exist" >&2
    exit 1
  fi
  LOCAL_CHANNEL_SPEC="file://$LOCAL_CHANNEL_SPEC"
fi

ENV_NAME="nvmolkit_verify_$(date +%s)"

cleanup() {
  local exit_code=$1
  conda deactivate >/dev/null 2>&1 || true
  conda env remove --name "$ENV_NAME" --yes >/dev/null 2>&1 || true
  exit "$exit_code"
}

trap 'cleanup $?' EXIT

eval "$(conda shell.bash hook)"

conda create --name "$ENV_NAME" "python=$PYTHON_VERSION" "rdkit=$RDKIT_VERSION" pytest pandas psutil --yes

conda activate "$ENV_NAME"

conda install --name "$ENV_NAME" --yes -c "$LOCAL_CHANNEL_SPEC" nvmolkit

pytest "$PYTEST_DIR"

