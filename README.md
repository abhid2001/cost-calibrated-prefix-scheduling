# Cost-Calibrated Prefix Scheduling

An experimental LLM inference project that studies whether estimating the GPU prefill time saved by KV-cache reuse can improve scheduling for reasoning workloads. The project uses Qwen2.5-7B-Instruct, GSM8K, SGLang, and a single NVIDIA A100.

The code includes direct and chain-of-thought (CoT) answers, three-sample self-consistency, a bounded Tree of Thoughts (ToT) controller, and standalone FCFS, k-LPM, and calibrated ranking policies. The experiment compares answer quality and runtime under fixed reasoning budgets.

**Project status:** The reasoning runners and standalone policies have local tests. The foundation job reached an A100, but its model server was killed before completing a request. Live scheduler integration, GPU prefill instrumentation, calibration, and the scheduler experiments are still in progress. No scheduler performance results have been established.

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
│   ├── load_reasoning_trace.py      # Run reasoning workloads
│   ├── collect_calibration.py       # Collect instrumented prefill timings
│   ├── fit_calibration.py           # Fit and validate the predictor
│   ├── pilot.slurm                  # Pilot workload on an A100
│   ├── reasoning_comparison.slurm   # Four-mode comparison
│   └── scheduler_matrix.slurm       # Blocked scheduler matrix placeholder
├── src/prefix_scheduler/            # Request records, controllers, and policies
├── tests/                           # Unit and mock-server tests
└── results/                         # Generated results (Git ignored)
```

Model weights, caches, the virtual environment, generated datasets, and results stay local and are excluded from Git.

---

## Quickstart

### 1. Set Up the Environment

Run from the repository root on Linux with Python 3.11. The installation requires network access and significant storage for GPU dependencies.

On CURC, `/home` has a 2 GB quota, which is too small for the environment (about 15 GB) and the model weights (about 15 GB). If the repository is under `/home`, first link the large directories to your `/projects` space:

```bash
L=/projects/$USER/ccps-local; mkdir -p $L/venv $L/cache $L/tmp $L/results
ln -s $L/venv .venv; ln -s $L/cache .cache; ln -s $L/tmp tmp; ln -s $L/results results
```

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
```

The setup pins SGLang `0.5.10.post1` and `datasets` `3.6.0`. Model and dataset revisions are recorded in `configs/foundation.json`. `configs/environment-snapshot.txt` records one resolved environment; it is not a GPU-validated lockfile.

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

### 6. Generate and Run an Arrival Trace

```bash
python scripts/prepare_trace.py --questions data/development_questions.jsonl \
  --rate 0.2 --seed 20261114 --mode tot --output data/tot_trace.jsonl
python scripts/load_reasoning_trace.py --trace data/tot_trace.jsonl \
  --labels data/development_labels.jsonl --out results/tot-load
python scripts/summarize.py --input results/tot-load/questions.jsonl \
  --out results/tot-load/summary.json
```

The runner supports all four modes and concurrent question arrivals. Calls within each question are currently sequential. Request and question logs include outputs, failures, dependencies, and client-side timing.

### 7. Run the Pilot

After the foundation job succeeds, submit the pilot:

```bash
sbatch --account=YOUR_ACCOUNT scripts/pilot.slurm
```

The pilot tests self-consistency and ToT on development questions at three candidate arrival rates. Use its results to size later reservations and choose a workload rate. `scripts/reasoning_comparison.slurm` is prepared for a four-mode FCFS comparison using frozen evaluation traces.

---

## Scheduling and Calibration

The standalone policies use an oldest-first admission followed by `k - 1` cache-ranked admissions, with `k = 4` by default. The k-LPM implementation offers cached-token ratio and matched-token count scores; the ratio is a provisional interpretation of the paper's cache-hit signal. Ties are resolved by ready time and request ID. The pinned SGLang LPM implementation instead sorts by matched-token count and preserves existing queue order for equal keys.

The calibrated policy uses a lookup predictor fitted from paired cold and warm prefill timings. If a prediction is unavailable, it falls back to the baseline score. **These policies are not yet connected to SGLang's live admission loop.**

`collect_calibration.py` expects GPU prefill timing in the engine response and flushes the server cache between measurements. The pinned engine has not yet been instrumented to provide that timing, so collection and predictor validation remain blocked. Client request latency must not be used as GPU prefill time. `scripts/scheduler_matrix.slurm` exits until live admission control and timing instrumentation are implemented.

---

## References

- [LLM Query Scheduling with Prefix Reuse and Latency Constraints](https://arxiv.org/html/2502.04677v3)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [SGLang native API](https://github.com/sgl-project/sglang/blob/v0.5.10.post1/docs/docs/basic_usage/native_api.mdx)
- [CURC Alpine hardware and GPU resources](https://curc.readthedocs.io/en/latest/clusters/alpine/alpine-hardware.html)
