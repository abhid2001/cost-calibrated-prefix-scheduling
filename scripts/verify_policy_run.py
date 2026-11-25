#!/usr/bin/env python3
"""Verify that a saved policy result can safely be reused after a matrix restart."""
import argparse
import hashlib
import json
import pathlib
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True, type=pathlib.Path)
parser.add_argument('--policy', required=True)
parser.add_argument('--trace', required=True, type=pathlib.Path)
parser.add_argument('--max-new-tokens', required=True, type=int)
parser.add_argument('--predictor')
args = parser.parse_args()

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
try:
    completed = json.loads((args.run / 'completed-run.json').read_text())
    policy = json.loads((args.run / 'policy.json').read_text())
    validation = json.loads((args.run / 'workload' / 'validation.json').read_text())
except (OSError, ValueError) as exc:
    raise SystemExit('Incomplete saved policy result: %s' % exc)
if completed.get('status') != 'completed' or completed.get('policy') != args.policy:
    raise SystemExit('Saved completion marker does not match policy')
if policy.get('policy') != args.policy or policy.get('trace_sha256') != sha(args.trace):
    raise SystemExit('Saved policy has a different trace or policy')
if policy.get('max_new_tokens') != args.max_new_tokens:
    raise SystemExit('Saved policy has a different token budget')
expected_predictor = str(pathlib.Path(args.predictor).resolve()) if args.predictor else None
if policy.get('predictor') != expected_predictor:
    raise SystemExit('Saved policy has a different predictor')
if validation.get('errors'):
    raise SystemExit('Saved workload validation failed: ' + '; '.join(validation['errors']))
print('Reusable completed policy:', args.policy)
