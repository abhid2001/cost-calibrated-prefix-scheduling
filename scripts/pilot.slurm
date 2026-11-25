#!/usr/bin/env bash
#SBATCH --job-name=prefix-pilot
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=pilot-%j.out
set -euo pipefail

source "${SLURM_SUBMIT_DIR:?}/scripts/live_experiment_common.sh"
OUT="$PWD/results/pilot-${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap stop_live_server EXIT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$OUT/gpu.csv"
"$PY" -c 'import csv,sys; rows=list(csv.reader(open(sys.argv[1]))); assert len(rows)==1 and "A100" in rows[0][0] and int(rows[0][1].split()[0])>=39000, rows' "$OUT/gpu.csv"
MAX_NEW=$("$PY" -c 'import json; print(json.load(open("configs/foundation.json"))["max_new_tokens"])')
for RATE in 0.20 0.50 1.00; do
  RUN="$OUT/rate-$RATE"
  TRACE="$RUN/trace.jsonl"
  mkdir -p "$RUN"
  "$PY" scripts/prepare_trace.py --questions data/development_questions.jsonl --limit 20 --rate "$RATE" --seed 20261114 --modes direct,cot,self_consistency,tot --output "$TRACE"
  start_live_server fcfs "$RUN"
  curl -fsS -X POST "$URL/flush_cache" > "$RUN/cache-reset.json"
  "$PY" scripts/load_reasoning_trace.py --trace "$TRACE" --labels data/development_labels.jsonl --url "$URL" --out "$RUN/workload" --workers "${PILOT_WORKERS:-32}" --max-new-tokens "$MAX_NEW" --admission-log "$PREFIX_ADMISSION_LOG" --require-live-admissions
  "$PY" scripts/analyze_run.py --run "$RUN/workload" --out "$RUN/analysis.json"
  stop_live_server
done
"$PY" scripts/select_pilot_rate.py --analyses "$OUT/rate-*/analysis.json" --out "$OUT/pilot-selection.json"
printf 'Pilot completed: %s\n' "$OUT"
