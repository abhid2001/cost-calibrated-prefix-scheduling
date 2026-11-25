# Cost-Calibrated Prefix Scheduling: Experiment Report

## Provenance

| Field | Value |
| --- | --- |
| Git commit / dirty state | `TBD` |
| Full-A100 hardware and driver | `TBD` |
| Pinned SGLang / Torch / Transformers | `TBD` |
| Dataset manifest hash | `TBD` |
| Accepted development predictor hash | `TBD` |
| Predictor held-out coverage / MAE / ranking error | `TBD` |
| Pilot trace / selected rate / rationale | `TBD` |
| Evaluation trace hash / seed / token budget | `TBD` |

## Primary matrix

| Policy | Question p50 / p95 latency (s) | Engine-wait p50 / p95 (s) | Throughput (questions/min) | Question / request failures | Cache-hit requests | GPU prefill p50 / p95 (ms) | Predictor coverage / fallbacks | Accuracy | Dependency decisions / max unlock | Fairness-cycle errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FCFS | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `N/A` | `TBD` | `N/A` | `TBD` |
| k-LPM | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `N/A` | `TBD` | `N/A` | `TBD` |
| Calibrated | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `N/A` | `TBD` |
| Dependency-aware | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

## Interpretation

State whether every policy used the same frozen trace, budget, model revision, accepted predictor, cache-reset procedure, and server constraints. Describe failures, missing telemetry, calibration fallback, and dependency unlock behavior before comparing latency or throughput. Separate observed results from conclusions about causal benefit.

## Limitations

Record any deviations from the protocol, missing telemetry, model/server changes, queue effects, or hardware differences. Do not generalize 20 GB MIG smoke results into full-A100 scheduler findings.
