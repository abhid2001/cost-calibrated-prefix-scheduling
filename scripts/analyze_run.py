#!/usr/bin/env python3
"""Analyze one materialized workload without substituting client metrics."""
import argparse, datetime, json, math, pathlib, statistics

p = argparse.ArgumentParser()
p.add_argument('--run', required=True, help='directory containing requests.jsonl and questions.jsonl')
p.add_argument('--out', required=True)
p.add_argument('--k', type=int, default=4)
a = p.parse_args()
if a.k < 1:
    p.error('--k must be positive')
d = pathlib.Path(a.run)


def rows(name):
    path = d / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def sec(left, right):
    return (datetime.datetime.fromisoformat(right) - datetime.datetime.fromisoformat(left)).total_seconds()


def quantile(values, q):
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)] if values else None


questions = rows('questions.jsonl')
requests = rows('requests.jsonl')
admissions = rows('admissions.jsonl')
latency = [sec(row['arrival'], row['finished_at']) for row in questions if row.get('arrival') and row.get('finished_at')]
dispatch_wait = [sec(row['ready_at'], row['dispatched_at']) for row in requests if row.get('ready_at') and row.get('dispatched_at')]
engine_wait = [sec(row['ready_at'], row['admitted_at']) for row in requests if row.get('ready_at') and row.get('admitted_at')]
cached = [row['matched_prefix_tokens'] for row in requests if row.get('matched_prefix_tokens') is not None]
prefill = [row.get('engine_metadata', {}).get('prefill_gpu_ms') for row in requests]
prefill = [value for value in prefill if isinstance(value, (int, float)) and value >= 0]
correct = [row['correct'] for row in questions if row.get('correct') is not None]
duration = sec(min(row['arrival'] for row in questions), max(row['finished_at'] for row in questions)) if questions and all(row.get('arrival') and row.get('finished_at') for row in questions) else 0
candidates = [candidate for event in admissions for candidate in event.get('candidates', [])]
prediction_count = sum(candidate.get('predicted_ms') is not None for candidate in candidates)
fallbacks = sum(event.get('reason') == 'baseline_fallback' for event in admissions)
dependency_events = [event for event in admissions if event.get('reason') == 'dependency_score']
unlock_values = [candidate.get('unlock_count') for candidate in candidates if isinstance(candidate.get('unlock_count'), int)]
cycle_errors = sum(event.get('slot') != event.get('admission', 0) % a.k for event in admissions if 'slot' in event)
result = {
    'question_count': len(questions),
    'question_failures': sum(row['status'] != 'completed' for row in questions),
    'accuracy_over_scored_questions': sum(correct) / len(correct) if correct else None,
    'scored_questions': len(correct),
    'latency_p50_seconds': statistics.median(latency) if latency else None,
    'latency_p95_seconds': quantile(latency, .95),
    'completed_questions_per_minute': sum(row['status'] == 'completed' for row in questions) * 60 / duration if duration > 0 else None,
    'request_count': len(requests),
    'request_failures': sum(row['status'] == 'failed' for row in requests),
    'reported_cache_tokens_total': sum(cached),
    'cache_telemetry_requests': len(cached),
    'max_client_ready_to_dispatch_seconds': max(dispatch_wait) if dispatch_wait else None,
    'engine_admission_wait_p50_seconds': statistics.median(engine_wait) if engine_wait else None,
    'engine_admission_wait_p95_seconds': quantile(engine_wait, .95),
    'engine_admission_telemetry_requests': len(engine_wait),
    'prefill_gpu_ms_p50': statistics.median(prefill) if prefill else None,
    'prefill_gpu_ms_p95': quantile(prefill, .95),
    'prefill_gpu_telemetry_requests': len(prefill),
    'admission_events': len(admissions),
    'max_admission_candidates': max((len(event.get('candidates', [])) for event in admissions), default=0),
    'baseline_fallback_admissions': fallbacks,
    'predictor_candidate_coverage': prediction_count / len(candidates) if candidates else None,
    'dependency_score_admissions': len(dependency_events),
    'dependency_unlock_max': max(unlock_values) if unlock_values else None,
    'fairness_cycle_errors': cycle_errors,
    'evaluator_failures': sum(row.get('evaluator_failures', 0) for row in questions),
    'notes': [
        'Failures remain in question latency and accuracy denominators when labels exist.',
        'Engine queue timing uses admission-log timestamps; client dispatch is reported separately.',
        'Predictor error belongs to the frozen held-out calibration report, not this live workload.',
    ],
}
out = pathlib.Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + '\n')
