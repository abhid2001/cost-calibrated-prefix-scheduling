#!/usr/bin/env python3
"""Independently validate a constrained live workload and its admission log."""
import argparse
import datetime
import json
import math
import pathlib


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(run, policy, k=4, require_four_modes=False):
    trace = rows(run / 'trace.jsonl')
    requests = rows(run / 'requests.jsonl')
    questions = rows(run / 'questions.jsonl')
    admissions = rows(run / 'admissions.jsonl')
    events = rows(run / 'events.jsonl')
    errors = []
    by_id = {r['request_id']: r for r in requests}
    by_question = {q['question_id']: q for q in questions}
    if len(by_id) != len(requests): errors.append('duplicate request IDs')
    if len(by_question) != len(questions) or {q['question_id'] for q in trace} != set(by_question):
        errors.append('trace and question IDs do not match uniquely')
    if require_four_modes and {q['mode'] for q in trace} != {'direct', 'cot', 'self_consistency', 'tot'}:
        errors.append('trace does not contain all four modes')
    expected_counts = {'direct': 1, 'cot': 1, 'self_consistency': 3, 'tot': 7}
    for q in trace:
        actual = [r for r in requests if r['question_id'] == q['question_id']]
        if len(actual) != expected_counts[q['mode']]: errors.append(q['question_id'] + ': incorrect graph size')
        if by_question.get(q['question_id'], {}).get('status') != 'completed': errors.append(q['question_id'] + ': question failed')
    grouped = {}
    for admission in admissions:
        grouped.setdefault(admission['actual_id'], []).append(admission)
    dispatched = {r['request_id'] for r in requests if r.get('dispatched_at')}
    if set(grouped) != dispatched or any(len(v) != 1 for v in grouped.values()):
        errors.append('admissions are missing, duplicated, or have unexpected request IDs')
    counts = [e['admission'] for e in admissions]
    if counts and counts != list(range(counts[0], counts[0] + len(counts))):
        errors.append('admission counters are not consecutive')
    queued = 0
    cache_hits = 0
    for r in requests:
        rid = r['request_id']
        if r['status'] != 'completed': errors.append(rid + ': request did not complete')
        for parent_id in r['parent_ids']:
            parent = by_id.get(parent_id)
            if not parent or parent['status'] != 'completed' or not r.get('ready_at') or parent['finished_at'] > r['ready_at']:
                errors.append(rid + ': invalid parent completion or release order')
        if rid not in grouped or len(grouped[rid]) != 1: continue
        e = grouped[rid][0]
        candidates = e.get('candidates', [])
        candidate = next((c for c in candidates if c['id'] == rid), None)
        if e.get('slot') != e['admission'] % k: errors.append(rid + ': fairness slot mismatch')
        expected_reason = 'oldest' if policy == 'fcfs' or e['slot'] == 0 else {
            'k-lpm': 'k-lpm', 'calibrated': 'calibrated', 'dependency-aware': 'dependency_score',
        }[policy]
        if e.get('reason') == 'baseline_fallback' and policy in ('calibrated', 'dependency-aware'):
            expected_reason = 'baseline_fallback'
        if e.get('reason') != expected_reason:
            errors.append(rid + ': unexpected policy reason')
        if e['reason'] == 'oldest': score = lambda c: 0
        elif e['reason'] in ('k-lpm', 'baseline_fallback'): score = lambda c: c['baseline']
        elif e['reason'] == 'calibrated': score = lambda c: c['predicted_ms']
        elif e['reason'] == 'dependency_score': score = lambda c: c['dependency_score']
        else:
            errors.append(rid + ': unrecognized admission reason')
            continue
        if any(score(c) is None for c in candidates):
            errors.append(rid + ': scoring field missing from admission candidates')
        ranking = [c['id'] for c in sorted(candidates, key=lambda c: (-score(c), c['age'], str(c['id'])))]
        if not ranking or ranking != e.get('ranked_ids') or ranking[0] != rid:
            errors.append(rid + ': engine admission does not match recomputed ranking')
        if len(candidates) > 1: queued += 1
        meta = r.get('engine_metadata') or {}
        cached, ms = meta.get('cached_tokens'), meta.get('prefill_gpu_ms')
        if isinstance(cached, bool) or not isinstance(cached, int) or cached < 0 or not candidate or candidate['cached_tokens'] != cached:
            errors.append(rid + ': cache telemetry missing or disagrees with admission')
        elif cached > 0: cache_hits += 1
        if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms < 0:
            errors.append(rid + ': missing or invalid GPU prefill timing')
        if not r.get('first_token_at') or not r.get('admitted_at'):
            errors.append(rid + ': missing admission or first-token timestamp')
        else:
            timestamps = [datetime.datetime.fromisoformat(r[key]) for key in ('dispatched_at', 'admitted_at', 'first_token_at', 'finished_at')]
            if timestamps != sorted(timestamps): errors.append(rid + ': timestamps out of order')
    materialized = [e['sequence'] for e in events if e['event'] == 'materialized']
    arrivals = [e['sequence'] for e in events if e['event'] == 'question_arrived']
    if not materialized or not arrivals or max(materialized) >= min(arrivals): errors.append('graphs were not materialized before arrivals')
    if not queued: errors.append('no competing requests observed at admission')
    if not cache_hits: errors.append('no cache reuse observed')
    reasons = [e.get('reason') for e in admissions]
    if policy == 'k-lpm' and 'k-lpm' not in reasons:
        errors.append('no cache-ranked k-LPM admission observed')
    if policy == 'calibrated' and 'calibrated' not in reasons:
        errors.append('no predictor-ranked calibrated admission observed')
    if policy == 'dependency-aware':
        dependency_candidates = [c for e in admissions if e.get('reason') == 'dependency_score' for c in e.get('candidates', [])]
        if not dependency_candidates:
            errors.append('no dependency-score admission observed')
        elif not any(c.get('unlock_count', 0) > 0 for c in dependency_candidates):
            errors.append('dependency score was observed but no candidate unlocked a child')
    summary = {'policy': policy, 'questions': len(questions), 'requests': len(requests), 'admissions': len(admissions),
               'admissions_with_competition': queued, 'requests_with_cache_hits': cache_hits,
               'baseline_fallback_admissions': sum(e.get('reason') == 'baseline_fallback' for e in admissions),
               'dependency_score_admissions': sum(e.get('reason') == 'dependency_score' for e in admissions), 'errors': errors}
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=pathlib.Path)
    parser.add_argument('--policy', choices=('fcfs', 'k-lpm', 'calibrated', 'dependency-aware'), required=True)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--require-four-modes', action='store_true')
    args = parser.parse_args()
    result = validate(args.run, args.policy, args.k, args.require_four_modes)
    (args.run / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['errors']: raise SystemExit('Workload validation failed: ' + '; '.join(result['errors']))
    print('Validated live workload:', json.dumps(result))
