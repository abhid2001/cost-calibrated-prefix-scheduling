#!/usr/bin/env python3
"""Collect policy run metrics without comparing mismatched traces or predictors."""
import argparse, json, pathlib

p = argparse.ArgumentParser()
p.add_argument('--matrix', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
root = pathlib.Path(a.matrix)
rows = []
for policy in ('fcfs', 'k-lpm', 'calibrated', 'dependency-aware'):
    run = root / policy
    analysis = json.loads((run / 'workload' / 'analysis.json').read_text())
    config = json.loads((run / 'policy.json').read_text())
    rows.append({'policy': policy, 'trace_sha256': config['trace_sha256'], **analysis})
hashes = {row['trace_sha256'] for row in rows}
if len(hashes) != 1:
    raise SystemExit('policy traces differ; refusing comparison')
pathlib.Path(a.out).write_text(json.dumps({'trace_sha256': hashes.pop(), 'policies': rows}, indent=2) + '\n')
