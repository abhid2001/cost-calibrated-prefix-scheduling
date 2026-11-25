# Published Validation Evidence

These artifacts are constrained 20 GB A100 MIG validation runs, not the pending full-A100 scheduler matrix. They are included to make the implementation, calibration gate, and live-admission claims auditable.

| Evidence | Scope | Artifact |
| --- | --- | --- |
| Real-prompt calibration | Repeated cold/warm prefill pairs from disjoint GSM8K prompt IDs; held-out acceptance gate passed | `results/interactive-calibration-36277792-20261124-133349/` |
| k-LPM live workload | Four materialized reasoning graphs, live cache/admission correlation | `results/interactive-admission-36276191-20261124-132026/workload/` |
| Dependency-aware live workload | Same graph workload using the accepted calibration predictor and a 5 ms unlock bonus | `results/interactive-admission-36277792-20261124-133757/workload/` |

The accepted calibration validation reports 32/32 held-out predictions covered, 1.766 ms MAE, and a 21.25% pairwise ranking error rate under its pre-registered constrained-MIG thresholds. The dependency-aware workload contains 12 driver requests with 12 live-engine admission records, four dependency-score decisions, and nonzero unlock counts.

Raw server logs, local environments, caches, failed attempts, and all later experiment outputs remain ignored. The full-A100 foundation, rate pilot, and evaluation matrix are intentionally not represented here until their queued full-GPU execution completes.

On November 25, interactive A100 3g.20gb allocation `36355011` ran a four-question concurrent k-LPM workload with 256 generated tokens per request. All four questions and 12 requests completed. Each request had an engine admission, a streamed first-token timestamp, GPU prefill timing, and a cache count matching the admission log; six admissions had competing requests and 11 requests reused cached tokens. The independent validator reported no ranking, fairness-cycle, timestamp, or dependency errors. The local artifacts are under `results/interactive-admission-36355011-20261125-135710/workload/`. An injected HTTP failure in the mock-server suite separately confirmed that ToT descendants are failed without dispatch, and a missing-timing test confirmed the telemetry gate fails clearly. This is a constrained smoke test, not a policy performance comparison.

On November 25, interactive allocation `36355910` ran FCFS, k-LPM, calibrated, and dependency-aware policies sequentially on the same four-question, 12-request mixed trace (`d698c1d09a713552faeb4155ac7884e9793a92b0b7e64c658be0a904e40ca8cd`), with a 256-token request budget and cold-cache server restart per policy. Every policy completed all 12 requests with one admission record per request, 11 cache-hit responses, complete streamed prefill/first-token telemetry, and no validator errors. k-LPM had cache-ranked admissions under contention. Calibrated scheduling used the accepted MIG predictor for scored choices and recorded five whole-queue baseline fallbacks where coverage was unavailable. Dependency-aware scheduling recorded four dependency-score decisions with nonzero unlock effects, alongside the same five documented fallbacks. The local run directories are `results/interactive-admission-36355910-20261125-{140952,141120,141246,141416}/`. These runs validate live policy integration only; they are too small and MIG-constrained for scheduler performance claims.
