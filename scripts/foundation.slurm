#!/usr/bin/env bash
#SBATCH --job-name=prefix-foundation
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=foundation-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the scheduler directory}"
export HF_HOME="$PWD/.cache/huggingface"
mkdir -p .cache/pip .cache/huggingface tmp
export PIP_CACHE_DIR="$PWD/.cache/pip" HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache" TMPDIR="$PWD/tmp" PIP_CONFIG_FILE=/dev/null
export TRITON_CACHE_DIR="$PWD/.cache/triton" CUDA_CACHE_PATH="$PWD/.cache/cuda"
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch-extensions" FLASHINFER_WORKSPACE_DIR="$PWD/.cache/flashinfer"
export TVM_FFI_CACHE_DIR="$PWD/.cache/tvm-ffi"
PY="$PWD/.venv/bin/python"
export PATH="$PWD/.venv/bin:/curc/sw/gcc/10.2.0/bin:$PATH"
export CC=/curc/sw/gcc/10.2.0/bin/gcc CXX=/curc/sw/gcc/10.2.0/bin/g++ NVCC_CCBIN=/curc/sw/gcc/10.2.0/bin/g++
export LD_LIBRARY_PATH="/curc/sw/gcc/10.2.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$PWD/scripts/python_headers.sh"
test -x "$PY" || { echo 'Run scripts/setup_env.sh first' >&2; exit 1; }
test -f data/development_questions.jsonl || { echo 'Run scripts/prepare_data.py first' >&2; exit 1; }
OUT="results/foundation-${SLURM_JOB_ID}"; mkdir -p "$OUT"
"$PY" scripts/preflight_experiment.py --mode foundation --require-primary-gpu > "$OUT/preflight.json"
PORT=$("$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
URL="http://127.0.0.1:$PORT"
nvidia-smi | tee "$OUT/nvidia-smi.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$OUT/gpu.csv"
"$PY" - "$OUT/hardware-validation.json" configs/platform.json <<'PY'
import json, sys, torch
platform = json.load(open(sys.argv[2]))
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('Expected exactly one Slurm-visible CUDA device; found ' + str(torch.cuda.device_count()))
properties = torch.cuda.get_device_properties(0)
name = properties.name
memory_mib = properties.total_memory // (1024 * 1024)
if platform['gpu_name_contains'] not in name or memory_mib < platform['min_memory_mib']:
    raise SystemExit('Expected primary ' + platform['name'] + ' with one ' + platform['gpu_name_contains'] + ' device of at least ' + str(platform['min_memory_mib']) + ' MiB; found: ' + name + ' with ' + str(memory_mib) + ' MiB')
profile = platform['name']
json.dump({'profile': profile, 'name': name, 'memory_mib': memory_mib, 'cuda_version': torch.version.cuda, 'platform_config': platform}, open(sys.argv[1], 'w'), indent=2)
print('Validated Slurm-visible hardware profile:', profile, name, str(memory_mib) + ' MiB')
PY
scontrol show job "$SLURM_JOB_ID" > "$OUT/slurm-job.txt"
"$PY" -m pip freeze > "$OUT/installed-requirements.txt"
cp configs/foundation.json "$OUT/config.json"
cp configs/platform.json "$OUT/platform.json"
printf '%s\n' 'Primary H200 3g.71gb platform' > "$OUT/run-mode.txt"
"$PY" -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 --dtype bfloat16 \
  --host 127.0.0.1 --port "$PORT" --max-running-requests 1 --mem-fraction-static 0.80 \
  --max-total-tokens 16384 --context-length 4096 --disable-cuda-graph --disable-piecewise-cuda-graph \
  --disable-overlap-schedule --chunked-prefill-size -1 > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'status=$?; kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; if [[ $status -ne 0 ]]; then tail -120 "$OUT/server.log" >&2 || true; fi; exit $status' EXIT
ready=0
for ((i=0; i<180; i++)); do
  if curl -fsS "$URL/health" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo 'Server exited; inspect server.log' >&2; exit 1; fi
  sleep 10
done
if [[ "$ready" != 1 ]]; then echo 'Server did not become healthy' >&2; exit 1; fi
"$PY" scripts/check_cache.py --url "$URL" --out "$OUT/cache-proof.json"
"$PY" scripts/run_questions.py --url "$URL" --mode direct --out "$OUT/direct" --labels data/development_labels.jsonl
"$PY" scripts/run_questions.py --url "$URL" --mode cot --out "$OUT/cot" --labels data/development_labels.jsonl
"$PY" - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
for mode in ('direct', 'cot'):
    records = [json.loads(line) for line in (out / mode / 'questions.jsonl').read_text().splitlines()]
    if len(records) != 10 or not any(r['status'] == 'completed' for r in records):
        raise SystemExit('%s: expected 10 records and at least one parsed completion' % mode)
    print('%s: %d/10 parsed completions' % (mode, sum(r['status'] == 'completed' for r in records)))
PY
