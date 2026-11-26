# Results Summary

Primary platform: one Slurm-visible NVIDIA H200 NVL MIG `3g.71gb` device (71,424 MiB), SGLang `0.5.10.post1`, Qwen2.5-7B-Instruct, GSM8K, and a fixed 512-token generation budget.

## Foundation validation

Foundation run `36426401` loaded the pinned model and completed a fresh request. A matching-prefix request reported 1,530 cached tokens, demonstrating live KV-cache reuse. Direct completed 10/10 development questions and CoT completed 8/10 under the earlier 256-token foundation smoke budget; the final pilot and matrix use the corrected 512-token budget and terminal-LaTex answer parsing.

## Calibration

Development-only cold/warm measurements produced an accepted predictor:

| Metric | Result |
| --- | ---: |
| Held-out measurements | 144 |
| Predictor coverage | 144 / 144 (100%) |
| Mean absolute error | 1.059 ms |
| Pairwise ranking error | 18.44% |
| Acceptance | Accepted |

## Pilot

The mixed Direct, CoT, self-consistency, and ToT development workload tested three arrival rates.

| Arrival rate (questions/s) | p95 engine admission wait | Decision |
| --- | ---: | --- |
| 0.20 | 6.937 s | Selected: queued work while remaining below the 10 s cap |
| 0.50 | 17.347 s | Rejected: above the cap |
| 1.00 | 27.869 s | Rejected: above the cap |

All three pilot rates completed 20 questions and 60 live engine admissions without failures or missing telemetry.

## Primary scheduler matrix

The final H200 matrix used the selected 0.20 questions/s rate, one frozen 100-question trace, identical seed, model revision, predictor, server constraints, 512-token budget, and cold-cache reset before each policy. Every policy completed 100 questions and 300 requests with complete admission/cache/GPU-prefill telemetry, 299 cache-hit responses, zero request failures, and zero fairness-cycle errors.

| Policy | Question latency p50 / p95 (s) | Engine wait p50 / p95 (s) | Throughput (questions/min) | Accuracy | Predictor coverage / fallbacks | Dependency decisions / max unlock |
| --- | --- | --- | ---: | ---: | --- | --- |
| FCFS | 6.898 / 32.875 | 2.946 / 10.631 | 12.518 | 0.77 | N/A | N/A |
| k-LPM | 6.783 / 31.674 | 2.293 / 10.924 | 12.527 | 0.78 | N/A | N/A |
| Calibrated | 6.879 / 25.347 | 2.046 / 10.177 | 12.605 | 0.78 | 54.33% / 124 | N/A |
| Dependency-aware | 6.988 / 29.649 | 2.098 / 10.525 | 12.571 | 0.76 | 55.11% / 127 | 98 / 2 |

## Observations and limitations

- Calibrated scheduling had the lowest observed p95 question latency and highest observed throughput in this one run.
- Dependency-aware scheduling was exercised correctly but did not improve p95 latency over calibrated scheduling in this workload.
- The live predictor covered roughly 54–55% of scored admission candidates, so fallback behavior materially contributes to both calibrated-policy results.
- These are single-seed H200 MIG observations, not statistically significant or general claims. Repeated seeds and additional load levels are needed for stronger comparisons.

For per-policy artifacts, hashes, hardware details, and interpretation, see [H200_EXPERIMENT_REPORT.md](H200_EXPERIMENT_REPORT.md).
