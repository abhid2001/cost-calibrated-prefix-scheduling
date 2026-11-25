# Published Validation Evidence

These artifacts are constrained 20 GB A100 MIG validation runs, not the pending full-A100 scheduler matrix. They are included to make the implementation, calibration gate, and live-admission claims auditable.

| Evidence | Scope | Artifact |
| --- | --- | --- |
| Real-prompt calibration | Repeated cold/warm prefill pairs from disjoint GSM8K prompt IDs; held-out acceptance gate passed | `results/interactive-calibration-36277792-20261124-133349/` |
| k-LPM live workload | Four materialized reasoning graphs, live cache/admission correlation | `results/interactive-admission-36276191-20261124-132026/workload/` |
| Dependency-aware live workload | Same graph workload using the accepted calibration predictor and a 5 ms unlock bonus | `results/interactive-admission-36277792-20261124-133757/workload/` |

The accepted calibration validation reports 32/32 held-out predictions covered, 1.766 ms MAE, and a 21.25% pairwise ranking error rate under its pre-registered constrained-MIG thresholds. The dependency-aware workload contains 12 driver requests with 12 live-engine admission records, four dependency-score decisions, and nonzero unlock counts.

Raw server logs, local environments, caches, failed attempts, and all later experiment outputs remain ignored. The full-A100 foundation, rate pilot, and evaluation matrix are intentionally not represented here until their queued full-GPU execution completes.
