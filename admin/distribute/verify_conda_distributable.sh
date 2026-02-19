#!/usr/bin/env bash

set -exuo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <local_conda_endpoint> <pytest_directory> <rdkit_version> <python_version>" >&2
  echo "Optional: set NVMOLKIT_CUDA_VER=12.9 (default, V100) or 13.0 (H100) to match artifacts in the channel." >&2
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

# Optional: use conda default locations unless you export CONDA_PKGS_DIRS / CONDA_ENVS_PATH (e.g. to scratch)
if [[ -n "${CONDA_PKGS_DIRS:-}" ]]; then
  mkdir -p "$CONDA_PKGS_DIRS"
fi
if [[ -n "${CONDA_ENVS_PATH:-}" ]]; then
  mkdir -p "$CONDA_ENVS_PATH"
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

# CUDA version for run deps (must match nvmolkit artifacts in the channel). Default 12.9 (V100); use 13.0 for H100.
NVMOLKIT_CUDA_VER=${NVMOLKIT_CUDA_VER:-12.9}
if [[ "$NVMOLKIT_CUDA_VER" == "12.9" ]]; then
  CUDA_CUDART_SPEC="cuda-cudart>=12.9,<13"
  CUDA_NVTX_SPEC="cuda-nvtx>=12.9,<13"
  LIBCUBLAS_SPEC="libcublas>=12.9,<13"
elif [[ "$NVMOLKIT_CUDA_VER" == "13.0" ]]; then
  CUDA_CUDART_SPEC="cuda-cudart>=13,<14"
  CUDA_NVTX_SPEC="cuda-nvtx>=13,<14"
  LIBCUBLAS_SPEC="libcublas>=13,<14"
else
  echo "NVMOLKIT_CUDA_VER must be 12.9 or 13.0 (got '$NVMOLKIT_CUDA_VER')" >&2
  exit 1
fi

# Create env with python, rdkit, nvmolkit run deps (numpy, pytorch, cuda-cudart, etc.) from conda-forge.
# Pre-installing cuda run deps allows installing nvmolkit with --no-deps from local only.
conda create -c conda-forge --name "$ENV_NAME" \
  "python=$PYTHON_VERSION" "rdkit=$RDKIT_VERSION" \
  "$CUDA_CUDART_SPEC" "$CUDA_NVTX_SPEC" "$LIBCUBLAS_SPEC" \
  numpy pytorch pytest pandas psutil --yes

conda activate "$ENV_NAME"

# Install nvmolkit from local channel only (--no-deps so conda-forge nvmolkit is never used)
conda install --name "$ENV_NAME" --yes --no-deps -c "$LOCAL_CHANNEL_SPEC" nvmolkit

pytest "$PYTEST_DIR"

