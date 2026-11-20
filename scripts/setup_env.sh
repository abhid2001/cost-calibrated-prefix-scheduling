#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .cache/pip .cache/huggingface tmp
export PIP_CACHE_DIR="$PWD/.cache/pip" HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache" TMPDIR="$PWD/tmp" PIP_CONFIG_FILE=/dev/null
export TRITON_CACHE_DIR="$PWD/.cache/triton" CUDA_CACHE_PATH="$PWD/.cache/cuda"
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch-extensions" FLASHINFER_WORKSPACE_DIR="$PWD/.cache/flashinfer"
PYTHON=${PYTHON:-python3.11}
# venv refuses a symlinked target, so create it at the resolved path; .venv may link to /projects.
"$PYTHON" -m venv "$(readlink -f .venv)"
.venv/bin/python -m pip install --upgrade 'pip==25.3'
# Constrain every transitive dependency to the recorded environment, not just the two top-level pins.
.venv/bin/python -m pip install -c configs/environment-snapshot.txt 'sglang[all]==0.5.10.post1' 'datasets==3.6.0'
.venv/bin/python -m pip freeze > configs/installed-requirements.txt
.venv/bin/python -m pip check
