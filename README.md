# Cost-Calibrated Prefix Scheduling

An experimental LLM inference project that studies whether estimating the GPU prefill time saved by KV-cache reuse can improve scheduling for reasoning workloads. The project uses Qwen2.5-7B-Instruct, GSM8K, SGLang, and a single NVIDIA A100.

The code includes direct and chain-of-thought (CoT) answers, three-sample self-consistency, a bounded Tree of Thoughts (ToT) controller, and standalone FCFS, k-LPM, and calibrated ranking policies. The experiment compares answer quality and runtime under fixed reasoning budgets.

**Project status:** The reasoning runners and standalone policies have local tests. An interactive 20 GB A100 MIG smoke test completed direct and CoT requests and verified prefix-cache reuse. The full 40 GB foundation job has not yet passed. Opt-in GPU prefill telemetry, real-prompt held-out calibration, and the constrained live FCFS, k-LPM, calibrated, and dependency-aware paths have passed interactive smoke tests. Scheduler experiments are still in progress. No scheduler performance results have been established.

---

## Repository Structure

```text
scheduler/
├── README.md
├── configs/                         # Model, dataset, and calibration settings
├── data/
│   └── manifest.json                # Pinned GSM8K splits and record IDs
├── scripts/
│   ├── setup_env.sh                 # Create the project environment
│   ├── prepare_data.py              # Prepare GSM8K questions and labels
│   ├── foundation.slurm             # A100 model and cache-reuse check
│   ├── run_questions.py             # Direct and CoT inference
│   ├── run_reasoning.py             # Self-consistency and bounded ToT
│   ├── prepare_trace.py             # Generate reproducible arrivals
│   ├── load_reasoning_trace.py      # Concurrent materialized-DAG live workload driver
│   ├── collect_calibration.py       # Collect instrumented prefill timings
│   ├── fit_calibration.py           # Fit and validate the predictor
│   ├── calibration.slurm             # Full-A100 development-only calibration
│   ├── pilot.slurm                  # Pilot workload on an A100
│   ├── reasoning_comparison.slurm   # Four-mode comparison
│   └── scheduler_matrix.slurm       # Reserved scheduler-matrix entrypoint
├── src/prefix_scheduler/            # Request records, controllers, and policies
├── tests/                           # Unit and mock-server tests
└── results/                         # Generated results (Git ignored)
```

Model weights, caches, the virtual environment, generated datasets, and results stay local and are excluded from Git.

---

## Quickstart

### 1. Set Up the Environment

Run from the repository root on Linux with Python 3.11. The installation requires network access and significant storage for GPU dependencies.

```bash
PYTHON=python3.11 scripts/setup_env.sh
source .venv/bin/activate
```

The setup creates `.venv/` and places caches and temporary files under the project folder. For subsequent commands in a new shell, set the same paths:

```bash
mkdir -p .cache/pip .cache/huggingface tmp
export PIP_CACHE_DIR="$PWD/.cache/pip" HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache" TMPDIR="$PWD/tmp" PIP_CONFIG_FILE=/dev/null
export TRITON_CACHE_DIR="$PWD/.cache/triton" CUDA_CACHE_PATH="$PWD/.cache/cuda"
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch-extensions" FLASHINFER_WORKSPACE_DIR="$PWD/.cache/flashinfer"
export TVM_FFI_CACHE_DIR="$PWD/.cache/tvm-ffi"
```

The setup pins SGLang `0.5.10.post1` and `datasets` `3.6.0`. Model and dataset revisions are recorded in `configs/foundation.json`. `configs/environment-snapshot.txt` records one resolved environment; it is not a GPU-validated lockfile. On systems without Python development headers, GPU jobs also require matching headers: `scripts/python_headers.sh` checks the system include directory and can use the project-local `.cache/python-devel/usr/include/python3.11` fallback. For the current Alpine Python 3.11.13 environment, that fallback was extracted from the matching Rocky Linux `python3.11-devel-3.11.13-2.el8_10.x86_64` RPM and includes the system `pyconfig-64.h`. The cache is not tracked; provision matching headers on a fresh setup.

### 2. Prepare GSM8K

```bash
python scripts/prepare_data.py
```

The script saves 20 development questions, eight disjoint training demonstrations, and 100 evaluation questions. It writes questions and labels separately and records the dataset revision, seed, splits, and row IDs in `data/manifest.json`. Generated question and label files are Git ignored.

### 3. Run Local Tests

```bash
python -m unittest discover -s tests -v
```

These tests exercise parsing, request states, reasoning dependencies, scheduling choices, and calibration logic without a GPU.

### 4. Check the Model and Prefix Cache on an A100

On CURC Alpine, submit the foundation job from the repository root with an eligible allocation account:

```bash
module load slurm/alpine
sbatch --account=YOUR_ACCOUNT scripts/foundation.slurm
```

The job requests one full 40 GB A100, launches SGLang, sends the same tokenized prompt twice, and checks the engine's cached-token telemetry. It then runs development questions in direct and CoT modes. Results and logs are saved in `results/foundation-<job-id>/`. The pinned model must be available in the project Hugging Face cache or downloadable from the compute node.

For a 20 GB A100 MIG interactive smoke test, run `scripts/interactive_foundation.sh` from an existing allocation. This uses a smaller KV cache and disables CUDA graphs; it is not equivalent to the full 40 GB experiment configuration. On Alpine, the script selects GCC 10 and its C++ runtime for CUDA JIT compilation and keeps TVM's build cache under the project directory.

An H200 `3g.71gb` interactive allocation can run the same foundation pipeline only as a compatibility check. Set `FOUNDATION_ALLOW_H200=1`; its result is labeled `h200-3g.71gb-compatibility` and cannot replace the required full-A100 foundation result or be combined with the A100 experiment matrix.

### 5. Run Reasoning Workloads

With an SGLang server running at `http://127.0.0.1:30000`:

```bash
# Direct and CoT
python scripts/run_questions.py --mode direct --labels data/development_labels.jsonl --out results/direct
python scripts/run_questions.py --mode cot --labels data/development_labels.jsonl --out results/cot

# Self-consistency and bounded ToT
python scripts/run_reasoning.py --mode self_consistency --labels data/development_labels.jsonl --out results/self-consistency
python scripts/run_reasoning.py --mode tot --labels data/development_labels.jsonl --out results/tot
```

Self-consistency votes over three complete solutions. The ToT controller generates and evaluates two candidates in each of two rounds, then synthesizes an answer: seven model calls when all stages succeed. Labels are used for scoring after generation and are not included in prompts.

The runners stream responses from SGLang's native `/generate` endpoint and record `first_token_at` when the first generated token arrives, giving client-observed time-to-first-token (TTFT) from `ready_at`. These client timestamps include HTTP overhead: they support queue-wait and TTFT metrics, but are not GPU prefill measurements.

### 6. Generate and Run an Arrival Trace

```bash
python scripts/prepare_trace.py --questions data/development_questions.jsonl \
  --rate 0.2 --seed 20261114 --modes direct,cot,self_consistency,tot \
  --output data/mixed_trace.jsonl
python scripts/load_reasoning_trace.py --trace data/mixed_trace.jsonl \
  --labels data/development_labels.jsonl --out results/mixed-load
python scripts/analyze_run.py --run results/mixed-load \
  --out results/mixed-load/analysis.json
```

The runner materializes each complete graph before its question arrives: one Direct node, one CoT node, three self-consistency samples, or all seven bounded-ToT nodes. It dispatches every ready node through a bounded concurrent HTTP executor; a child becomes ready only after its recorded parents complete, while parent failure recursively marks descendants failed without sending them to the engine. Request, question, and `events.jsonl` logs preserve IDs, dependencies, state transitions, and client timestamps.

### 7. Run the Full-A100 Foundation Check

The foundation script now uses the same project-local compiler, C++ runtime, caches, conservative memory fraction, and no-overlap/no-chunk constraints as the validated live path. It must run on a full 40 GB A100; the 20 GB `gpu-testing` MIG QoS cannot validate it.

```bash
sbatch --account=YOUR_ACCOUNT scripts/foundation.slurm
```

### 8. Run the Pilot

After the foundation job succeeds, collect an accepted full-A100 development predictor:

```bash
sbatch --account=YOUR_ACCOUNT scripts/calibration.slurm
```

The calibration job measures repeated cold/warm pairs using disjoint development and held-out GSM8K prompt IDs, including zero-, two-, and four-demonstration chat contexts. It writes `model/predictor.json` only with a sibling held-out validation report; the matrix refuses a predictor that the report does not accept.

Then submit the pilot:

```bash
sbatch --account=YOUR_ACCOUNT scripts/pilot.slurm
```

The pilot runs a frozen mixed Direct/CoT/self-consistency/ToT development trace at 0.20, 0.50, and 1.00 arrivals per second against the same constrained live FCFS engine used for the matrix. It records admission logs and selects the highest rate with complete engine telemetry, no request failures, a nonzero queue wait, and bounded p95 engine wait. Its `pilot-selection.json` is the required input to the primary matrix.

### 9. Run the Primary Scheduler Matrix

Keep calibration development-only. The matrix accepts only a predictor whose sibling `validation.json` has `acceptance.accepted: true`, copies both immutable artifacts into the output directory, freezes one 100-question mixed evaluation trace from the pilot-selected rate, and runs FCFS, k-LPM, calibrated, and dependency-aware sequentially from a cold cache.

```bash
PILOT_SELECTION="$PWD/results/pilot-<job-id>/pilot-selection.json" \
DEVELOPMENT_PREDICTOR="$PWD/results/calibration-<job-id>/model/predictor.json" \
sbatch --account=YOUR_ACCOUNT scripts/scheduler_matrix.slurm
```

Each policy receives the same trace hash, seed, token budget, server constraints, cache reset, and accepted predictor. `matrix-summary.json` reports question latency, engine admission wait, throughput, failures, cache telemetry, GPU prefill timing, calibration fallback/coverage, accuracy, dependency decisions, unlock counts, and fairness-cycle errors. The matrix is an experiment, not a claim that dependency-aware scheduling improves performance. The full, resume-safe procedure and final reporting template are in [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) and [`docs/REPORT_TEMPLATE.md`](docs/REPORT_TEMPLATE.md).

---

## Scheduling and Calibration

The standalone policies use an oldest-first admission followed by `k - 1` cache-ranked admissions, with `k = 4` by default. The k-LPM implementation offers cached-token ratio and matched-token count scores; the ratio is a provisional interpretation of the paper's cache-hit signal. Ties are resolved by ready time and request ID. The pinned SGLang LPM implementation instead sorts by matched-token count and preserves existing queue order for equal keys.

The calibrated policy uses a lookup predictor fitted from paired cold and warm prefill timings. If a prediction is unavailable, it falls back to the baseline score. `interactive_calibration.sh` builds varied tokenized GSM8K chat prompts (zero, two, and four real demonstration questions), repeats cold/warm measurements, and accepts a predictor only after disjoint held-out prompt IDs satisfy the configured coverage, MAE, and ranking-error gate. This validates a constrained MIG predictor; repeat calibration on the final workload and full A100 before reporting scheduler performance.

An optional `dependency-aware` policy adds a configurable per-unlocked-child bonus (in milliseconds) to the predicted saving on cache-ranked admissions. It keeps the same oldest-first cycle and falls back to the baseline for the whole choice when prediction or dependency information is unavailable. The workload driver atomically publishes its complete materialized graph (IDs, parent IDs, and states) to the engine hook. The hook counts a blocked child only if every other parent completed, logs the count and dependency score, and falls back uniformly if its predictor or graph snapshot is unavailable. It is an optional fourth policy, not a replacement for the three required policies.


`scripts/install_prefill_instrumentation.py` applies an opt-in patch to the pinned SGLang 0.5.10.post1 environment and saves the original engine files in the ignored `patches/sglang-original/` directory. Enable it with `PREFIX_GPU_PREFILL_TIMING=1` and run with one GPU, `--max-running-requests 1`, `--chunked-prefill-size -1`, and `--disable-overlap-schedule`; incompatible settings fail. The patch records synchronized CUDA events around the model forward on its current stream, before token sampling, and exposes `meta_info.prefill_gpu_ms` in responses. Streaming telemetry has been verified on the 20 GB MIG configuration. Event timing includes GPU work and stream idle gaps within that forward interval; it is not client TTFT. CUDA graph and piecewise graph settings must remain fixed for paired measurements.

`scripts/interactive_prefill.sh` launches a dedicated 20 GB MIG server with those constraints and validates cold and warm telemetry. One interactive smoke test passed with 700 cached tokens; these individual timings are not a performance comparison. `collect_calibration.py` flushes the server cache between paired measurements, so use only a dedicated server. A preliminary 20 GB MIG calibration and held-out validation have been run; more varied prompts and repeated measurements are needed before relying on the predictor.

### Live admission validation

The pinned engine can use the FCFS, k-LPM, or calibrated policy in its actual prefill admission loop. This is an opt-in experimental patch for the constrained, one-GPU/one-running-request/no-overlap/unchunked configuration above; it is deliberately not enabled for general SGLang serving. Install the prefill instrumentation first, then the admission patch:

```bash
.venv/bin/python scripts/install_prefill_instrumentation.py
.venv/bin/python scripts/install_live_admission.py
PREFIX_ADMISSION_POLICY=k-lpm bash scripts/interactive_admission.sh
# Or, with a fitted predictor:
PREFIX_ADMISSION_POLICY=calibrated PREFIX_PREDICTOR_PATH=/absolute/path/to/predictor.json \
  bash scripts/interactive_admission.sh
```

Run the script only inside the documented interactive A100 allocation. It primes a real prefix and submits contending prompts with different matched-prefix lengths. The JSONL log records every candidate's live radix-cache match, score, fairness slot, ranked order, and the request actually accepted by SGLang. The validator independently recomputes each ranking, verifies the oldest-first cycle advances only after an actual admission, and compares each selected request's log entry with SGLang's `cached_tokens` response telemetry. The saved `responses.json` and `admissions.jsonl` are the run evidence. This establishes scheduling correctness under the stated constraints; it is not yet a performance result or a general multi-request admission controller.

To run a materialized four-mode workload through that same patched server, create a mixed trace and set `WORKLOAD_TRACE` when launching the interactive script:

```bash
.venv/bin/python scripts/prepare_trace.py --questions data/development_questions.jsonl \
  --limit 4 --rate 1000 --modes direct,cot,self_consistency,tot \
  --output results/live-smoke-trace.jsonl
WORKLOAD_TRACE="$PWD/results/live-smoke-trace.jsonl" WORKLOAD_WORKERS=7 \
WORKLOAD_MAX_NEW_TOKENS=256 WORKLOAD_REQUIRE_FOUR_MODES=1 PREFIX_ADMISSION_POLICY=k-lpm \
  bash scripts/interactive_admission.sh
```

This mode requires one live admission entry for every dispatched request, streaming prefill/cache/first-token telemetry for every completed request, and agreement between the response and live-cache match. `workload/validation.json` independently checks rankings, the fairness cycle, dependency release order, competition, cache reuse, calibration fallback, and dependency-score behavior where applicable. It also writes `requests.jsonl`, `questions.jsonl`, `events.jsonl`, and filtered `admissions.jsonl`; `admitted_at` is populated only from engine telemetry, while `dispatched_at` remains the client timestamp. The 256-token value is a constrained 20 GB MIG smoke-test budget, not the evaluation budget.

To enable the optional fourth policy, use an accepted predictor and a nonnegative unlock bonus. The script publishes the graph sidecar before dispatch and requires every request to have a live admission record:

```bash
WORKLOAD_TRACE="$PWD/results/live-smoke-trace.jsonl" WORKLOAD_WORKERS=7 \
  WORKLOAD_MAX_NEW_TOKENS=64 PREFIX_ADMISSION_POLICY=dependency-aware \
  PREFIX_PREDICTOR_PATH="$PWD/results/interactive-calibration-<job-id>-<time>/model/predictor.json" \
  PREFIX_DEPENDENCY_UNLOCK_BONUS_MS=5 \
  bash scripts/interactive_admission.sh
```

---

## References

- [LLM Query Scheduling with Prefix Reuse and Latency Constraints](https://arxiv.org/html/2502.04677v3)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [SGLang native API](https://github.com/sgl-project/sglang/blob/v0.5.10.post1/docs/docs/basic_usage/native_api.mdx)
- [CURC Alpine hardware and GPU resources](https://curc.readthedocs.io/en/latest/clusters/alpine/alpine-hardware.html)
