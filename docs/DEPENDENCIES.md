# Dependencies and Platform Requirements

This document defines the supported environment for reproducing the primary experiment. All installation, model, compiler, temporary, and cache paths must remain under the repository directory; the setup does not use the home directory for package or model caches.

## Supported primary platform

| Requirement | Supported configuration |
| --- | --- |
| Operating system | Linux |
| Python | 3.11 |
| GPU | One Slurm-visible NVIDIA H200 NVL MIG `3g.71gb` device with at least 70,000 MiB |
| Slurm request | `ah200`, `gpu-normal`, `gpu:h200_3g.71gb:1` |
| Model | `Qwen/Qwen2.5-7B-Instruct` at revision `a09a35458c702b33eeacc393d103063234e8bc28` |
| Dataset | `openai/gsm8k`, `main`, revision `740312add88f781978c0658806c59bc2815b9866` |
| SGLang | `0.5.10.post1` |
| Datasets | `3.6.0` |

`configs/platform.json` is the machine-readable source of truth for the primary Slurm request and visible-device gate. `configs/foundation.json` pins the model, dataset, generation, and SGLang settings.

## Installation

From the repository root:

```bash
PYTHON=python3.11 scripts/setup_env.sh
source .venv/bin/activate
python -m pip check
```

The setup script creates `.venv/`, installs the two top-level requirements, and constrains transitive packages using `configs/environment-snapshot.txt`. It writes the resolved installation to `configs/installed-requirements.txt`. The snapshot records a tested environment; it does not replace validating GPU drivers or compiler compatibility on a new cluster.

## Required external tools

- Slurm commands (`sbatch`, `srun`, and `scontrol`) for GPU experiments.
- NVIDIA driver/CUDA support compatible with the installed PyTorch/SGLang environment.
- `curl` for server health and cache checks.
- GCC 10.2.0 and matching Python 3.11 headers on the CURC Alpine environment. Job scripts select this compiler and `scripts/python_headers.sh` validates headers before launching SGLang.

## Storage and network behavior

The first setup/model/data run may require package-index and Hugging Face access. After weights and datasets are cached under `.cache/`, live experiment jobs set `HF_HUB_OFFLINE=1` and reuse project-local artifacts. The repository intentionally ignores `.venv/`, `.cache/`, `tmp/`, generated datasets, server logs, and raw result directories because they are large or environment-specific.

## Verification

```bash
python -m unittest discover -s tests -v
python tests/smoke_runners.py
```

For GPU validation, run `scripts/foundation.slurm`; it verifies one Slurm-visible primary GPU, model startup, a fresh request, and matching-prefix cache reuse before calibration, pilot, or matrix work.
