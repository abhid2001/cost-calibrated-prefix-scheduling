#!/usr/bin/env bash
# Source from an experiment job launched with scheduler/ as its submit directory.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from scheduler directory}"
mkdir -p .cache/pip .cache/huggingface tmp
export PIP_CACHE_DIR="$PWD/.cache/pip" HF_HOME="$PWD/.cache/huggingface" XDG_CACHE_HOME="$PWD/.cache" TMPDIR="$PWD/tmp" PIP_CONFIG_FILE=/dev/null
export TRITON_CACHE_DIR="$PWD/.cache/triton" CUDA_CACHE_PATH="$PWD/.cache/cuda" TORCH_EXTENSIONS_DIR="$PWD/.cache/torch-extensions" FLASHINFER_WORKSPACE_DIR="$PWD/.cache/flashinfer"
PY="$PWD/.venv/bin/python"
test -x "$PY" || { echo 'Missing project venv' >&2; exit 1; }
test -f data/manifest.json || { echo 'Missing dataset manifest' >&2; exit 1; }
OUT="${OUT:-$PWD/results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}}"; mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$OUT/gpu.csv"
"$PY" -c 'import csv,sys; r=list(csv.reader(open(sys.argv[1]))); assert len(r)==1 and "A100" in r[0][0] and int(r[0][1].split()[0])>=39000, r' "$OUT/gpu.csv"
scontrol show job "$SLURM_JOB_ID" > "$OUT/slurm-job.txt"
"$PY" -m pip freeze > "$OUT/installed-requirements.txt"
cp configs/foundation.json data/manifest.json "$OUT/"
PORT=$("$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
URL="http://127.0.0.1:$PORT"
"$PY" -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --dtype bfloat16 --host 127.0.0.1 --port "$PORT" --max-running-requests 1 --schedule-policy fcfs > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
ready=0
for ((i=0;i<180;i++)); do
  if curl -fsS "$URL/health" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo 'Server failed; inspect server.log' >&2; exit 1; fi
  sleep 10
done
[[ "$ready" == 1 ]] || { echo 'Server health timeout' >&2; exit 1; }
