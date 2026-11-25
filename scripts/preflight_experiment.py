#!/usr/bin/env python3
"""Fail early when an experiment lacks immutable inputs or a valid full-A100 setup."""
import argparse
import importlib.metadata
import json
import math
import pathlib
import subprocess
import sys


def require(path, description):
    path = pathlib.Path(path)
    if not path.is_file():
        raise SystemExit('Missing %s: %s' % (description, path))
    return path


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('pilot', 'matrix'), required=True)
parser.add_argument('--config', default='configs/foundation.json')
parser.add_argument('--predictor')
parser.add_argument('--pilot-selection')
parser.add_argument('--require-full-a100', action='store_true')
args = parser.parse_args()

config_path = require(args.config, 'configuration')
config = json.loads(config_path.read_text())
for key in ('model', 'model_revision', 'max_new_tokens', 'request_timeout_seconds'):
    if key not in config:
        raise SystemExit('Configuration lacks ' + key)
if not isinstance(config['max_new_tokens'], int) or config['max_new_tokens'] < 1:
    raise SystemExit('Configuration max_new_tokens must be positive')
require('data/manifest.json', 'dataset manifest')
for name in ('development_questions.jsonl', 'development_labels.jsonl'):
    require(pathlib.Path('data') / name, 'development input')
if args.mode == 'matrix':
    for name in ('evaluation_questions.jsonl', 'evaluation_labels.jsonl'):
        require(pathlib.Path('data') / name, 'evaluation input')
    predictor = require(args.predictor, 'development predictor')
    validation = require(predictor.parent / 'validation.json', 'predictor validation')
    if json.loads(validation.read_text()).get('acceptance', {}).get('accepted') is not True:
        raise SystemExit('Development predictor was not accepted on held-out data')
    selection = json.loads(require(args.pilot_selection, 'pilot selection').read_text())
    rate = selection.get('chosen', {}).get('rate')
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise SystemExit('Pilot selection lacks a positive finite chosen rate')
if importlib.metadata.version('sglang') != '0.5.10.post1':
    raise SystemExit('Pinned SGLang 0.5.10.post1 is required')
hardware = None
if args.require_full_a100:
    try:
        hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit('Cannot inspect GPU for full-A100 preflight: %s' % exc)
    if len(hardware) != 1 or 'A100' not in hardware[0] or int(hardware[0].split(',')[1].split()[0]) < 39000:
        raise SystemExit('Full-A100 preflight requires exactly one 40 GB A100: %s' % hardware)
print(json.dumps({'preflight': 'passed', 'mode': args.mode, 'config': str(config_path), 'hardware': hardware}, sort_keys=True))
