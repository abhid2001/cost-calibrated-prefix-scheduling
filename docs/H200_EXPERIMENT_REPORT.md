# H200 `3g.71gb` Experiment Report

This filled report follows [`REPORT_TEMPLATE.md`](REPORT_TEMPLATE.md). It reports the completed primary H200 run only; historical A100 MIG smoke runs are not included.

## Provenance

| Field | Value |
| --- | --- |
| Matrix / node | `36429334` / `c3gpu-e5-u9` |
| Code revision | `cc77badd55577417a6967e5a0a7fb2ea33b0e9a2` (the provenance record is marked dirty only because of the pre-existing unrelated untracked `&1` file) |
| Visible GPU | NVIDIA H200 NVL MIG `3g.71gb`, 71,424 MiB, CUDA 12.8 |
| Driver / host inventory | 590.48.01; host reports four physical H200 NVL devices, while Slurm exposed exactly one MIG device |
| SGLang / Torch / Transformers | `0.5.10.post1` / `2.9.1` / `5.3.0` |
| Dataset manifest SHA-256 | `a103bf5db46e7e3bd40b9567910a0795fa095b06d7386ae3018827493be4dd43` |
| Accepted predictor SHA-256 | `6e4b37a198639acfd8d7492d1895b2a03bc320a7b75bc499be2238bf708277a4` |
| Held-out predictor validation | 144/144 covered; 1.059 ms MAE; 18.44% pairwise ranking error; accepted |
| Pilot | 20 development questions, rates 0.20/0.50/1.00 s⁻¹; selected 0.20 s⁻¹ because it queued work with 6.937 s p95 engine wait, below the recorded 10 s cap |
| Evaluation trace | SHA-256 `46ca17da1331102d8e24d79d8872b9abdc2502dec7e7e14fd1c97aee2b8d65b3`; seed `20261114`; 512 new-token budget |

## Primary matrix

| Policy | Question p50 / p95 latency (s) | Engine-wait p50 / p95 (s) | Throughput (questions/min) | Question / request failures | Cache-hit requests | GPU prefill p50 / p95 (ms) | Predictor coverage / fallbacks | Accuracy | Dependency decisions / max unlock | Fairness-cycle errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FCFS | 6.898 / 32.875 | 2.946 / 10.631 | 12.518 | 0 / 0 | 299 / 300 | 8.963 / 18.430 | N/A | 0.77 | N/A | 0 |
| k-LPM | 6.783 / 31.674 | 2.293 / 10.924 | 12.527 | 0 / 0 | 299 / 300 | 8.890 / 19.312 | N/A | 0.78 | N/A | 0 |
| Calibrated | 6.879 / 25.347 | 2.046 / 10.177 | 12.605 | 0 / 0 | 299 / 300 | 8.884 / 18.311 | 54.33% / 124 | 0.78 | N/A | 0 |
| Dependency-aware | 6.988 / 29.649 | 2.098 / 10.525 | 12.571 | 0 / 0 | 299 / 300 | 8.955 / 18.603 | 55.11% / 127 | 0.76 | 98 / 2 | 0 |

## Interpretation

Every policy used the same frozen trace, 512-token budget, model revision, accepted development-only predictor, constrained one-GPU SGLang configuration, and cold-cache reset. Each completed 100 materialized question graphs and 300 engine admissions with complete admission, cache, and GPU-prefill telemetry. The calibrated policy had the lowest observed p95 question latency in this single run; that observation does not establish a causal or general performance advantage. The dependency-aware policy made 98 logged dependency-score decisions and observed at most two newly unlockable children, but it did not improve p95 latency over calibrated scheduling here.

## Limitations

- This is one seeded H200 MIG run, not a multi-seed confidence interval.
- Predictor coverage during the live matrix was about 54–55%, so 124–127 admission decisions explicitly used baseline fallback.
- The constrained server has one running request, disabled overlap/CUDA graphs, and unchunked prefill; results do not generalize to unrestricted serving.
- Question accuracy reflects deterministic model outputs and the parser; it is not a scheduler-quality metric.
