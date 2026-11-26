#!/usr/bin/env bash
# Source from the primary H200 Slurm job. Sets up a constrained, patched live engine.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from scheduler directory}"
mkdir -p .cache/pip .cache/huggingface tmp
export PATH="$PWD/.venv/bin:/curc/sw/gcc/10.2.0/bin:$PATH"
export CC=/curc/sw/gcc/10.2.0/bin/gcc CXX=/curc/sw/gcc/10.2.0/bin/g++ NVCC_CCBIN=/curc/sw/gcc/10.2.0/bin/g++
export LD_LIBRARY_PATH="/curc/sw/gcc/10.2.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PIP_CACHE_DIR="$PWD/.cache/pip" HF_HOME="$PWD/.cache/huggingface" XDG_CACHE_HOME="$PWD/.cache" TMPDIR="$PWD/tmp" PIP_CONFIG_FILE=/dev/null
export TRITON_CACHE_DIR="$PWD/.cache/triton" CUDA_CACHE_PATH="$PWD/.cache/cuda" TORCH_EXTENSIONS_DIR="$PWD/.cache/torch-extensions" FLASHINFER_WORKSPACE_DIR="$PWD/.cache/flashinfer" TVM_FFI_CACHE_DIR="$PWD/.cache/tvm-ffi"
export HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY="$PWD/.venv/bin/python"
source "$PWD/scripts/python_headers.sh"
test -x "$PY" || { echo 'Missing project venv' >&2; exit 1; }
test -f data/manifest.json || { echo 'Missing dataset manifest' >&2; exit 1; }

LIVE_MEM_FRACTION="${LIVE_MEM_FRACTION:-0.80}"
LIVE_MAX_TOTAL_TOKENS="${LIVE_MAX_TOTAL_TOKENS:-16384}"
LIVE_CONTEXT_LENGTH="${LIVE_CONTEXT_LENGTH:-4096}"
SERVER_PID=""

start_live_server() {
  local policy="$1" run="$2"
  case "$policy" in fcfs|k-lpm|calibrated|dependency-aware) ;; *) echo "Unknown policy: $policy" >&2; return 2;; esac
  mkdir -p "$run"
  "$PY" scripts/install_prefill_instrumentation.py
  "$PY" scripts/install_live_admission.py
  PORT=$("$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
  URL="http://127.0.0.1:$PORT"
  export PREFIX_GPU_PREFILL_TIMING=1 PREFIX_ADMISSION_POLICY="$policy" PREFIX_ADMISSION_LOG="$run/engine-admissions.jsonl"
  if [[ "$policy" == calibrated || "$policy" == dependency-aware ]]; then
    : "${DEVELOPMENT_PREDICTOR:?Set DEVELOPMENT_PREDICTOR to an accepted development-only predictor}"
    export PREFIX_PREDICTOR_PATH="$DEVELOPMENT_PREDICTOR"
  else
    unset PREFIX_PREDICTOR_PATH PREFIX_DEPENDENCY_GRAPH_PATH PREFIX_DEPENDENCY_UNLOCK_BONUS_MS || true
  fi
  if [[ "$policy" == dependency-aware ]]; then
    export PREFIX_DEPENDENCY_GRAPH_PATH="$run/dependency_graph.json"
    export PREFIX_DEPENDENCY_UNLOCK_BONUS_MS="${PREFIX_DEPENDENCY_UNLOCK_BONUS_MS:-5}"
  fi
  "$PY" -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --dtype bfloat16 \
    --host 127.0.0.1 --port "$PORT" --max-running-requests 1 --mem-fraction-static "$LIVE_MEM_FRACTION" \
    --max-total-tokens "$LIVE_MAX_TOTAL_TOKENS" --context-length "$LIVE_CONTEXT_LENGTH" --disable-cuda-graph --disable-piecewise-cuda-graph \
    --disable-overlap-schedule --chunked-prefill-size -1 > "$run/server.log" 2>&1 &
  SERVER_PID=$!
  local ready=0
  for ((i=0;i<180;i++)); do
    if curl -fsS "$URL/health" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -80 "$run/server.log" >&2; return 1; fi
    sleep 10
  done
  [[ "$ready" == 1 ]] || { tail -80 "$run/server.log" >&2; return 1; }
}

stop_live_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
