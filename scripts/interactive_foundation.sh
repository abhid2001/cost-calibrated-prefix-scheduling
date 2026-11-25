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
export HF_HUB_OFFLINE=1
PY="$PWD/.venv/bin/python"
source scripts/python_headers.sh
OUT="$PWD/results/interactive-foundation-${SLURM_JOB_ID:?Must run inside the requested interactive Slurm allocation}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
nvidia-smi > "$OUT/nvidia-smi.txt"
nvidia-smi -L > "$OUT/gpu-devices.txt"
"$PY" -c 'import torch; assert torch.cuda.is_available(); n=torch.cuda.device_count(); p=torch.cuda.get_device_properties(0); print("name=%s memory_bytes=%d device_count=%d"%(p.name,p.total_memory,n)); assert n==1 and "A100" in p.name and 18*1024**3 <= p.total_memory < 23*1024**3' | tee "$OUT/cuda-device.txt"
"$PY" -m pip freeze > "$OUT/installed-requirements.txt"
cp configs/foundation.json "$OUT/config.json"
PORT=$("$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
URL="http://127.0.0.1:$PORT"
printf '%s\n' 'GPU: A100 3g.20gb MIG' 'dtype: bfloat16' 'max running requests: 1' 'max total tokens: 4096' 'context length: 4096' 'CUDA graphs: disabled (including piecewise)' 'host compiler: GCC 10.2.0' 'static memory fraction: 0.80' > "$OUT/server-settings.txt"
"$PY" -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --dtype bfloat16 --host 127.0.0.1 --port "$PORT" --max-running-requests 1 --mem-fraction-static 0.80 --max-total-tokens 4096 --context-length 4096 --disable-cuda-graph --disable-piecewise-cuda-graph > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
ready=0
for ((i=0;i<180;i++)); do
  if curl -fsS "$URL/health" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "Server exited; see $OUT/server.log" >&2; exit 1; fi
  sleep 10
done
[[ "$ready" == 1 ]] || { echo "Server health timeout; see $OUT/server.log" >&2; exit 1; }
"$PY" scripts/check_cache.py --url "$URL" --out "$OUT/cache-proof.json"
for mode in direct cot; do
  "$PY" scripts/run_questions.py --url "$URL" --mode "$mode" --limit 1 --out "$OUT/$mode" --labels data/development_labels.jsonl
  "$PY" - "$OUT/$mode/questions.jsonl" <<'PY'
import json,sys
with open(sys.argv[1]) as f: row=json.loads(f.readline())
if row['status']!='completed': raise SystemExit('Model request did not produce a parsed answer: '+str(row.get('error')))
print('Completed',row['mode'],'request; answer:',row['answer'])
PY
done
printf 'Interactive foundation checks passed. Results: %s\n' "$OUT"
