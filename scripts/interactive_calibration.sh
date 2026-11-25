#!/usr/bin/env bash
# Run only inside an existing interactive A100 20 GB allocation.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .cache/pip .cache/huggingface tmp
export PATH="$PWD/.venv/bin:/curc/sw/gcc/10.2.0/bin:$PATH"
export CC=/curc/sw/gcc/10.2.0/bin/gcc CXX=/curc/sw/gcc/10.2.0/bin/g++ NVCC_CCBIN=/curc/sw/gcc/10.2.0/bin/g++
export TVM_FFI_CACHE_DIR="$PWD/.cache/tvm-ffi"
export LD_LIBRARY_PATH="/curc/sw/gcc/10.2.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PIP_CACHE_DIR="$PWD/.cache/pip" HF_HOME="$PWD/.cache/huggingface" XDG_CACHE_HOME="$PWD/.cache" TMPDIR="$PWD/tmp" PIP_CONFIG_FILE=/dev/null
export TRITON_CACHE_DIR="$PWD/.cache/triton" CUDA_CACHE_PATH="$PWD/.cache/cuda" TORCH_EXTENSIONS_DIR="$PWD/.cache/torch-extensions" FLASHINFER_WORKSPACE_DIR="$PWD/.cache/flashinfer"
export HF_HUB_OFFLINE=1 PREFIX_GPU_PREFILL_TIMING=1
PY="$PWD/.venv/bin/python"
source scripts/python_headers.sh
OUT="$PWD/results/interactive-calibration-${SLURM_JOB_ID:?Must run inside the requested interactive Slurm allocation}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
nvidia-smi > "$OUT/nvidia-smi.txt"
nvidia-smi -L > "$OUT/gpu-devices.txt"
"$PY" -c 'import torch; assert torch.cuda.is_available(); n=torch.cuda.device_count(); p=torch.cuda.get_device_properties(0); print("name=%s memory_bytes=%d device_count=%d"%(p.name,p.total_memory,n)); assert n==1 and "A100" in p.name and 18*1024**3 <= p.total_memory < 23*1024**3' | tee "$OUT/cuda-device.txt"
"$PY" -m pip freeze > "$OUT/installed-requirements.txt"
cp configs/foundation.json "$OUT/config.json"
PORT=$("$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
URL="http://127.0.0.1:$PORT"
printf '%s\n' 'GPU: A100 3g.20gb MIG' 'dtype: bfloat16' 'max running requests: 1' 'max total tokens: 4096' 'context length: 4096' 'CUDA graphs: disabled (including piecewise)' 'host compiler: GCC 10.2.0' 'static memory fraction: 0.80' > "$OUT/server-settings.txt"
"$PY" -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --dtype bfloat16 --host 127.0.0.1 --port "$PORT" --max-running-requests 1 --mem-fraction-static 0.80 --max-total-tokens 4096 --context-length 4096 --disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule --chunked-prefill-size -1 > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
ready=0
for ((i=0;i<180;i++)); do
  if curl -fsS "$URL/health" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "Server exited; see $OUT/server.log" >&2; exit 1; fi
  sleep 10
done
[[ "$ready" == 1 ]] || { echo "Server health timeout; see $OUT/server.log" >&2; exit 1; }
"$PY" scripts/validate_prefill.py --url "$URL" --out "$OUT/prefill-validation.json"
printf 'Prefill validation passed. Results: %s\n' "$OUT"
"$PY" scripts/prepare_calibration_prompts.py --questions data/development_questions.jsonl --start 0 --limit 4 --demo-counts 0,2,4 --out "$OUT/train-prompts.jsonl"
"$PY" scripts/prepare_calibration_prompts.py --questions data/development_questions.jsonl --start 4 --limit 2 --demo-counts 0,2 --out "$OUT/heldout-prompts.jsonl"
"$PY" scripts/collect_calibration.py --url "$URL" --prompts "$OUT/train-prompts.jsonl" --fractions 0.25,0.6 --out "$OUT/train.jsonl" --repetitions 2
"$PY" scripts/collect_calibration.py --url "$URL" --prompts "$OUT/heldout-prompts.jsonl" --fractions 0.25,0.6 --out "$OUT/heldout.jsonl" --repetitions 2
"$PY" scripts/fit_calibration.py --train "$OUT/train.jsonl" --heldout "$OUT/heldout.jsonl" --out "$OUT/model" --min-heldout-coverage 0.8 --max-heldout-mae-ms 40 --max-ranking-error-rate 0.5
printf 'Calibration completed. Results: %s\n' "$OUT"
