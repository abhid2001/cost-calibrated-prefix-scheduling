#!/usr/bin/env bash
#SBATCH --job-name=prefix-pilot
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=pilot-%j.out
set -euo pipefail

source "${SLURM_SUBMIT_DIR:?}/scripts/live_experiment_common.sh"
OUT="${PILOT_OUT:-$PWD/results/pilot-${SLURM_JOB_ID}}"
mkdir -p "$OUT"
trap stop_live_server EXIT
"$PY" scripts/preflight_experiment.py --mode pilot --require-primary-gpu > "$OUT/preflight.json"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$OUT/gpu.csv"
scontrol show job "$SLURM_JOB_ID" > "$OUT/slurm-job.txt"
"$PY" -m pip freeze > "$OUT/installed-requirements.txt"
cp configs/foundation.json configs/platform.json data/manifest.json "$OUT/"
MAX_NEW=$("$PY" -c 'import json; print(json.load(open("configs/foundation.json"))["max_new_tokens"])')
for RATE in 0.20 0.50 1.00; do
  RUN="$OUT/rate-$RATE"
  TRACE="$RUN/trace.jsonl"
  mkdir -p "$RUN"
  if [[ -f "$RUN/completed-run.json" ]] && "$PY" scripts/verify_policy_run.py --run "$RUN" --policy fcfs --trace "$TRACE" --max-new-tokens "$MAX_NEW" >/dev/null 2>&1; then
    printf 'Reusing completed pilot rate %s\n' "$RATE"
    continue
  fi
  "$PY" scripts/prepare_trace.py --questions data/development_questions.jsonl --limit 20 --rate "$RATE" --seed 20261114 --modes direct,cot,self_consistency,tot --output "$TRACE"
  if start_live_server fcfs "$RUN" && curl -fsS -X POST "$URL/flush_cache" > "$RUN/cache-reset.json" && \
      "$PY" scripts/load_reasoning_trace.py --trace "$TRACE" --labels data/development_labels.jsonl --url "$URL" --out "$RUN/workload" --workers "${PILOT_WORKERS:-32}" --max-new-tokens "$MAX_NEW" --admission-log "$PREFIX_ADMISSION_LOG" --require-live-admissions --require-stream-telemetry && \
      "$PY" scripts/validate_workload.py --run "$RUN/workload" --policy fcfs --require-four-modes && \
      "$PY" scripts/analyze_run.py --run "$RUN/workload" --out "$RUN/analysis.json"; then
    "$PY" scripts/record_run.py --out "$RUN/completed-run.json" --kind pilot-rate --status completed --config configs/foundation.json --trace "$TRACE" --hardware "$OUT/gpu.csv" --policy fcfs --max-new-tokens "$MAX_NEW"
    stop_live_server
  else
    status=$?
    stop_live_server
    "$PY" scripts/record_run.py --out "$RUN/failed-run.json" --kind pilot-rate --status failed --config configs/foundation.json --trace "$TRACE" --hardware "$OUT/gpu.csv" --policy fcfs --max-new-tokens "$MAX_NEW" --note 'See preserved workload, admission, and server artifacts in this rate directory.'
    exit "$status"
  fi
done
"$PY" scripts/select_pilot_rate.py --analyses "$OUT/rate-*/analysis.json" --out "$OUT/pilot-selection.json" \
  --min-queue-wait-seconds "${PILOT_MIN_QUEUE_WAIT_SECONDS:-0.01}" \
  --max-p95-queue-wait-seconds "${PILOT_MAX_P95_QUEUE_WAIT_SECONDS:-10}"
"$PY" scripts/record_run.py --out "$OUT/completed-run.json" --kind pilot --status completed --config configs/foundation.json --hardware "$OUT/gpu.csv" --note 'Each rate has an independently reusable completion record.'
printf 'Pilot completed: %s\n' "$OUT"
