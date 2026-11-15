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
source "${SLURM_SUBMIT_DIR:?}/scripts/experiment_common.sh"
"$PY" scripts/check_cache.py --url "$URL" --out "$OUT/cache-proof.json"
# Ten questions per condition; rates are pilot candidates, not selected experiment rates.
head -n 10 data/development_questions.jsonl > "$OUT/pilot_questions.jsonl"
for mode in self_consistency tot; do
  for rate in 0.02 0.05 0.1; do
    trace="$OUT/${mode}-${rate}-trace.jsonl"
    "$PY" scripts/prepare_trace.py --questions "$OUT/pilot_questions.jsonl" --rate "$rate" --seed 20261114 --mode "$mode" --output "$trace"
    curl -fsS -X POST "$URL/flush_cache" > "$OUT/${mode}-${rate}-cache-reset.txt"
    "$PY" scripts/load_reasoning_trace.py --trace "$trace" --labels data/development_labels.jsonl --url "$URL" --out "$OUT/${mode}-${rate}"
    "$PY" scripts/summarize.py --input "$OUT/${mode}-${rate}/questions.jsonl" --out "$OUT/${mode}-${rate}/summary.json"
    "$PY" scripts/analyze_run.py --run "$OUT/${mode}-${rate}" --out "$OUT/${mode}-${rate}/analysis.json"
  done
done
