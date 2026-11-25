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
OUT="${MATRIX_OUT:-$PWD/results/matrix-${SLURM_JOB_ID}}"
mkdir -p "$OUT"
trap stop_live_server EXIT
"$PY" scripts/preflight_experiment.py --mode matrix --predictor "$DEVELOPMENT_PREDICTOR" --pilot-selection "$PILOT_SELECTION" --require-full-a100 > "$OUT/preflight.json"
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
if [[ ! -f "$TRACE" ]]; then
  "$PY" scripts/prepare_trace.py --questions data/evaluation_questions.jsonl --rate "$RATE" --seed "${TRACE_SEED:-20261114}" --modes direct,cot,self_consistency,tot --output "$TRACE"
fi
TRACE_HASH=$(sha256sum "$TRACE" | awk '{print $1}')
MAX_NEW=$("$PY" -c 'import json; print(json.load(open("configs/foundation.json"))["max_new_tokens"])')
"$PY" scripts/record_run.py --out "$OUT/planned-run.json" --kind scheduler-matrix --status planned --config configs/foundation.json --trace "$TRACE" --predictor "$DEVELOPMENT_PREDICTOR" --pilot-selection "$PILOT_SELECTION" --hardware "$OUT/gpu.csv" --max-new-tokens "$MAX_NEW"

for POLICY in fcfs k-lpm calibrated dependency-aware; do
  RUN="$OUT/$POLICY"
  mkdir -p "$RUN"
  VERIFY_ARGS=(--run "$RUN" --policy "$POLICY" --trace "$TRACE" --max-new-tokens "$MAX_NEW")
  if [[ "$POLICY" == calibrated || "$POLICY" == dependency-aware ]]; then VERIFY_ARGS+=(--predictor "$DEVELOPMENT_PREDICTOR"); fi
  if [[ "${MATRIX_RESUME:-1}" == 1 ]] && [[ -f "$RUN/completed-run.json" ]] && "$PY" scripts/verify_policy_run.py "${VERIFY_ARGS[@]}"; then
    printf 'Reusing completed matrix policy %s\n' "$POLICY"
    continue
  fi
  if start_live_server "$POLICY" "$RUN"; then :; else status=$?; "$PY" scripts/record_run.py --out "$RUN/failed-run.json" --kind matrix-policy --status failed --config configs/foundation.json --trace "$TRACE" --hardware "$OUT/gpu.csv" --policy "$POLICY" --max-new-tokens "$MAX_NEW" --note 'Server startup failed; previous completed policies remain reusable.'; exit "$status"; fi
  if curl -fsS -X POST "$URL/flush_cache" > "$RUN/cache-reset.json"; then :; else status=$?; stop_live_server; "$PY" scripts/record_run.py --out "$RUN/failed-run.json" --kind matrix-policy --status failed --config configs/foundation.json --trace "$TRACE" --hardware "$OUT/gpu.csv" --policy "$POLICY" --max-new-tokens "$MAX_NEW" --note 'Cache reset failed; previous completed policies remain reusable.'; exit "$status"; fi
  ARGS=(--trace "$TRACE" --labels data/evaluation_labels.jsonl --url "$URL" --out "$RUN/workload" --workers "${MATRIX_WORKERS:-32}" --max-new-tokens "$MAX_NEW" --admission-log "$PREFIX_ADMISSION_LOG" --require-live-admissions --require-stream-telemetry)
  if [[ "$POLICY" == dependency-aware ]]; then ARGS+=(--dependency-graph "$PREFIX_DEPENDENCY_GRAPH_PATH"); fi
  if "$PY" scripts/load_reasoning_trace.py "${ARGS[@]}" && "$PY" scripts/validate_workload.py --run "$RUN/workload" --policy "$POLICY" --require-four-modes && "$PY" scripts/analyze_run.py --run "$RUN/workload" --out "$RUN/workload/analysis.json"; then :; else
    status=$?
    stop_live_server
    "$PY" scripts/record_run.py --out "$RUN/failed-run.json" --kind matrix-policy --status failed --config configs/foundation.json --trace "$TRACE" --hardware "$OUT/gpu.csv" --policy "$POLICY" --max-new-tokens "$MAX_NEW" --note 'Workload failed; previous completed policies remain reusable.'
    exit "$status"
  fi
  POLICY_PREDICTOR=""
  if [[ -n "${PREFIX_PREDICTOR_PATH:-}" ]]; then POLICY_PREDICTOR="$(readlink -f "$PREFIX_PREDICTOR_PATH")"; fi
  "$PY" -c 'import json,sys; open(sys.argv[1],"w").write(json.dumps({"policy":sys.argv[2],"trace_sha256":sys.argv[3],"max_new_tokens":int(sys.argv[4]),"predictor":sys.argv[5] if sys.argv[5] else None,"unlock_bonus_ms":sys.argv[6] if sys.argv[6] else None},indent=2)+"\n")' "$RUN/policy.json" "$POLICY" "$TRACE_HASH" "$MAX_NEW" "$POLICY_PREDICTOR" "${PREFIX_DEPENDENCY_UNLOCK_BONUS_MS:-}"
  COMPLETE_ARGS=(--out "$RUN/completed-run.json" --kind matrix-policy --status completed --config configs/foundation.json --trace "$TRACE" --hardware "$OUT/gpu.csv" --policy "$POLICY" --max-new-tokens "$MAX_NEW")
  if [[ -n "${PREFIX_PREDICTOR_PATH:-}" ]]; then COMPLETE_ARGS+=(--predictor "$PREFIX_PREDICTOR_PATH"); fi
  "$PY" scripts/record_run.py "${COMPLETE_ARGS[@]}"
  stop_live_server
done
"$PY" scripts/analyze_matrix.py --matrix "$OUT" --out "$OUT/matrix-summary.json"
"$PY" scripts/record_run.py --out "$OUT/completed-run.json" --kind scheduler-matrix --status completed --config configs/foundation.json --trace "$TRACE" --predictor "$DEVELOPMENT_PREDICTOR" --pilot-selection "$PILOT_SELECTION" --hardware "$OUT/gpu.csv" --max-new-tokens "$MAX_NEW"
printf 'Matrix completed: %s\n' "$OUT"
