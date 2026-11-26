#!/usr/bin/env bash
#SBATCH --job-name=prefix-reasoning
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=reasoning-%j.out
set -euo pipefail
# Provide frozen traces only after selecting a rate from the pilot; override time after estimating runtime.
: "${TRACE_DIR:?Set TRACE_DIR to the frozen evaluation traces}"
source "${SLURM_SUBMIT_DIR:?}/scripts/experiment_common.sh"
for mode in direct cot self_consistency tot; do
  trace="$TRACE_DIR/${mode}.jsonl"
  test -f "$trace" || { echo "Missing frozen trace $trace" >&2; exit 1; }
  cp "$trace" "$OUT/${mode}-trace.jsonl"
  curl -fsS -X POST "$URL/flush_cache" > "$OUT/${mode}-cache-reset.txt"
  "$PY" scripts/load_reasoning_trace.py --trace "$trace" --labels data/evaluation_labels.jsonl --url "$URL" --out "$OUT/$mode"
  "$PY" scripts/summarize.py --input "$OUT/$mode/questions.jsonl" --out "$OUT/$mode/summary.json"
  "$PY" scripts/analyze_run.py --run "$OUT/$mode" --out "$OUT/$mode/analysis.json"
done
