#!/usr/bin/env bash
#SBATCH --job-name=prefix-calibration
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=calibration-%j.out
set -euo pipefail

source "${SLURM_SUBMIT_DIR:?}/scripts/live_experiment_common.sh"
OUT="${CALIBRATION_OUT:-$PWD/results/calibration-${SLURM_JOB_ID}}"
mkdir -p "$OUT"
trap stop_live_server EXIT
"$PY" scripts/preflight_experiment.py --mode pilot --require-full-a100 > "$OUT/preflight.json"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$OUT/gpu.csv"
scontrol show job "$SLURM_JOB_ID" > "$OUT/slurm-job.txt"
"$PY" -m pip freeze > "$OUT/installed-requirements.txt"
cp configs/foundation.json data/manifest.json "$OUT/"

start_live_server fcfs "$OUT"
curl -fsS -X POST "$URL/flush_cache" > "$OUT/cache-reset.json"
"$PY" scripts/validate_prefill.py --url "$URL" --out "$OUT/prefill-validation.json"

TRAIN_LIMIT="${CALIBRATION_TRAIN_LIMIT:-8}"
HELDOUT_LIMIT="${CALIBRATION_HELDOUT_LIMIT:-4}"
REPETITIONS="${CALIBRATION_REPETITIONS:-3}"
DEMO_COUNTS="${CALIBRATION_DEMO_COUNTS:-0,2,4}"
"$PY" scripts/prepare_calibration_prompts.py --questions data/development_questions.jsonl --start 0 --limit "$TRAIN_LIMIT" --demo-counts "$DEMO_COUNTS" --out "$OUT/train-prompts.jsonl"
"$PY" scripts/prepare_calibration_prompts.py --questions data/development_questions.jsonl --start "$TRAIN_LIMIT" --limit "$HELDOUT_LIMIT" --demo-counts "$DEMO_COUNTS" --out "$OUT/heldout-prompts.jsonl"
"$PY" scripts/collect_calibration.py --url "$URL" --prompts "$OUT/train-prompts.jsonl" --fractions 0.25,0.60 --out "$OUT/train.jsonl" --repetitions "$REPETITIONS"
"$PY" scripts/collect_calibration.py --url "$URL" --prompts "$OUT/heldout-prompts.jsonl" --fractions 0.25,0.60 --out "$OUT/heldout.jsonl" --repetitions "$REPETITIONS"
"$PY" scripts/fit_calibration.py --train "$OUT/train.jsonl" --heldout "$OUT/heldout.jsonl" --out "$OUT/model" --min-heldout-coverage 0.8 --max-heldout-mae-ms 40 --max-ranking-error-rate 0.5
"$PY" scripts/record_run.py --out "$OUT/completed-run.json" --kind full-a100-calibration --status completed --config configs/foundation.json --hardware "$OUT/gpu.csv" --input "train_prompts=$OUT/train-prompts.jsonl" --input "heldout_prompts=$OUT/heldout-prompts.jsonl" --input "predictor_validation=$OUT/model/validation.json" --note 'Development-only prompts; held-out acceptance is in model/validation.json.'
printf 'Full-A100 calibration completed: %s\n' "$OUT"
