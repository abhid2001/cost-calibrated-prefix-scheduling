#!/usr/bin/env bash
#SBATCH --job-name=prefix-matrix
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=matrix-%j.out
set -euo pipefail

: "${PILOT_SELECTION:?Set PILOT_SELECTION to pilot-selection.json}"
: "${DEVELOPMENT_PREDICTOR:?Set DEVELOPMENT_PREDICTOR to accepted development-only predictor.json}"
source "${SLURM_SUBMIT_DIR:?}/scripts/live_experiment_common.sh"
OUT="$PWD/results/matrix-${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap stop_live_server EXIT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$OUT/gpu.csv"
"$PY" -c 'import csv,sys; rows=list(csv.reader(open(sys.argv[1]))); assert len(rows)==1 and "A100" in rows[0][0] and int(rows[0][1].split()[0])>=39000, rows' "$OUT/gpu.csv"
scontrol show job "$SLURM_JOB_ID" > "$OUT/slurm-job.txt"
"$PY" -m pip freeze > "$OUT/installed-requirements.txt"
cp configs/foundation.json data/manifest.json "$OUT/"
cp "$DEVELOPMENT_PREDICTOR" "$OUT/development-predictor.json"
VALIDATION="$(dirname "$DEVELOPMENT_PREDICTOR")/validation.json"
test -f "$VALIDATION" || { echo 'Missing predictor validation.json' >&2; exit 1; }
"$PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("acceptance",{}).get("accepted") is True, x.get("acceptance")' "$VALIDATION"
cp "$VALIDATION" "$OUT/development-predictor-validation.json"

RATE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["chosen"]["rate"])' "$PILOT_SELECTION")
TRACE="$OUT/frozen-evaluation-trace.jsonl"
"$PY" scripts/prepare_trace.py --questions data/evaluation_questions.jsonl --rate "$RATE" --seed "${TRACE_SEED:-20261114}" --modes direct,cot,self_consistency,tot --output "$TRACE"
TRACE_HASH=$(sha256sum "$TRACE" | awk '{print $1}')
MAX_NEW=$("$PY" -c 'import json; print(json.load(open("configs/foundation.json"))["max_new_tokens"])')

for POLICY in fcfs k-lpm calibrated dependency-aware; do
  RUN="$OUT/$POLICY"
  mkdir -p "$RUN"
  start_live_server "$POLICY" "$RUN"
  curl -fsS -X POST "$URL/flush_cache" > "$RUN/cache-reset.json"
  ARGS=(--trace "$TRACE" --labels data/evaluation_labels.jsonl --url "$URL" --out "$RUN/workload" --workers "${MATRIX_WORKERS:-32}" --max-new-tokens "$MAX_NEW" --admission-log "$PREFIX_ADMISSION_LOG" --require-live-admissions)
  if [[ "$POLICY" == dependency-aware ]]; then ARGS+=(--dependency-graph "$PREFIX_DEPENDENCY_GRAPH_PATH"); fi
  "$PY" scripts/load_reasoning_trace.py "${ARGS[@]}"
  "$PY" scripts/analyze_run.py --run "$RUN/workload" --out "$RUN/workload/analysis.json"
  "$PY" -c 'import json,sys; open(sys.argv[1],"w").write(json.dumps({"policy":sys.argv[2],"trace_sha256":sys.argv[3],"max_new_tokens":int(sys.argv[4]),"predictor":sys.argv[5] if sys.argv[5] else None,"unlock_bonus_ms":sys.argv[6] if sys.argv[6] else None},indent=2)+"\n")' "$RUN/policy.json" "$POLICY" "$TRACE_HASH" "$MAX_NEW" "${PREFIX_PREDICTOR_PATH:-}" "${PREFIX_DEPENDENCY_UNLOCK_BONUS_MS:-}"
  stop_live_server
done
"$PY" scripts/analyze_matrix.py --matrix "$OUT" --out "$OUT/matrix-summary.json"
printf 'Matrix completed: %s\n' "$OUT"
