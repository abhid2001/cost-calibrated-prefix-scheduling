# Experiment Protocol

This protocol separates implementation checks, development choices, and evaluation. The H200 `3g.71gb` slice is the primary experiment platform. Historical interactive 20 GB A100 MIG runs validate mechanics only and do not select the evaluation arrival rate, accept the H200 predictor, or support scheduler-performance claims.

## 1. Reproducibility preflight

Run from the repository root using the project-local environment and storage paths. Every Slurm launcher runs `scripts/preflight_experiment.py` before launching SGLang. It verifies pinned SGLang, dataset inputs, configuration, accepted calibration inputs when needed, and exactly one Slurm-visible H200 device with at least 70,000 MiB. The platform request and gate live in `configs/platform.json`. Each rate and policy emits a provenance record containing the Git revision and dirty state, configuration, platform configuration, trace/predictor/input hashes, hardware, Slurm context, Python, SGLang, Torch, Transformers, and dependency-lock versions.

## 2. Primary H200 foundation

Run `scripts/foundation.slurm` first. Accept it only when the H200 `3g.71gb` configuration starts the pinned model, completes a fresh request, confirms cache reuse for a matching prefix, and preserves the required engine telemetry. Save the generated result directory unchanged. Foundation run `36426401` satisfies this stage.

## 3. Development-only calibration

Use only development prompts and `scripts/calibration.slurm` on the H200 slice to collect repeated cold/warm pairs. Keep cache settings, CUDA-graph settings, overlap, and chunked-prefill settings fixed. Fit with `scripts/fit_calibration.py` and accept the predictor only if its held-out `validation.json` says `acceptance.accepted: true`.

Never fit, tune, or replace the predictor using evaluation prompts. Copy the accepted predictor and validation report into the matrix output before starting evaluation.

## 4. Development pilot

Submit `scripts/pilot.slurm` on the H200 slice. It uses a mixed Direct, CoT, self-consistency, and ToT development trace at 0.20, 0.50, and 1.00 arrivals/second. The selected rate must have complete engine admission and streamed GPU-prefill telemetry, no request failures, bounded p95 engine wait, and preferably observable queueing. `pilot-selection.json` records the choice and rule.

The pilot launcher writes reusable completion records per rate. To resume a failed job, set `PILOT_OUT` to its existing output directory; completed rates are verified and reused.

## 5. Frozen evaluation matrix

Set `PILOT_SELECTION` and `DEVELOPMENT_PREDICTOR`, then submit `scripts/scheduler_matrix.slurm`. It freezes one evaluation trace using the selected rate and seed, resets the cache before each policy, and runs FCFS, k-LPM, calibrated, and dependency-aware scheduling sequentially with the same trace hash, token budget, server constraints, and accepted predictor.

The matrix validates every workload before marking it complete. If a later policy fails, completed policy directories remain intact. Resume with `MATRIX_OUT=/absolute/path/to/results/matrix-<job-id>`; the launcher verifies and reuses only policy results whose trace, budget, predictor, completion marker, and validation record match the resumed matrix.

## 6. Analysis and reporting

Run `scripts/analyze_matrix.py` only after all four policy validations pass. Report the values in `docs/REPORT_TEMPLATE.md`, including failures and fallback counts. Treat a missing metric as missing; do not replace engine admission wait or GPU prefill time with client latency.

## Limitations

- The live integration is deliberately constrained to one GPU, one running request, unchunked prefill, and disabled overlap/CUDA graphs.
- Client `first_token_at` includes transport overhead. `prefill_gpu_ms` is the engine-side CUDA-event measurement.
- The dependency-aware policy is an optional fourth comparison. It may worsen latency or throughput; report observed outcomes without assuming improvement.
- Historical A100 MIG smoke runs validate control flow and telemetry only. Report primary scheduler findings only from the frozen H200 evaluation matrix.
