#!/usr/bin/env python3
"""Write an atomic, machine-readable provenance record for an experiment run."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile


def digest(path):
    path = pathlib.Path(path)
    if not path.is_file():
        raise ValueError('missing input file: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(command):
    try:
        return subprocess.check_output(['git', *command], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def input_record(path):
    return {'path': str(pathlib.Path(path).resolve()), 'sha256': digest(path)}


parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True, type=pathlib.Path)
parser.add_argument('--kind', required=True)
parser.add_argument('--status', choices=('planned', 'completed', 'failed'), required=True)
parser.add_argument('--config', required=True)
parser.add_argument('--trace')
parser.add_argument('--predictor')
parser.add_argument('--pilot-selection')
parser.add_argument('--hardware')
parser.add_argument('--dependency-lock', default='configs/environment-snapshot.txt')
parser.add_argument('--dataset-manifest', default='data/manifest.json')
parser.add_argument('--input', action='append', default=[], metavar='LABEL=PATH')
parser.add_argument('--policy')
parser.add_argument('--max-new-tokens', type=int)
parser.add_argument('--note')
args = parser.parse_args()

config_path = pathlib.Path(args.config)
record = {
    'schema_version': 1,
    'kind': args.kind,
    'status': args.status,
    'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'git': {'commit': git(['rev-parse', 'HEAD']), 'dirty': bool(git(['status', '--porcelain']))},
    'runtime': {'python': sys.version, 'platform': platform.platform(),
                'sglang': version('sglang'), 'torch': version('torch'), 'transformers': version('transformers')},
    'slurm': {key: os.environ.get(key) for key in ('SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_NODELIST', 'SLURM_SUBMIT_DIR')},
    'config': input_record(config_path),
    'dependency_lock': input_record(args.dependency_lock),
    'dataset_manifest': input_record(args.dataset_manifest),
    'policy': args.policy,
    'max_new_tokens': args.max_new_tokens,
    'note': args.note,
}
for field in ('trace', 'predictor', 'pilot_selection', 'hardware'):
    value = getattr(args, field.replace('_', '-'), None) if '-' in field else getattr(args, field)
    if value:
        record[field] = input_record(value)
record['inputs'] = {}
for item in args.input:
    try:
        label, value = item.split('=', 1)
    except ValueError as exc:
        raise SystemExit('--input requires LABEL=PATH') from exc
    if not label or label in record['inputs']:
        raise SystemExit('invalid or duplicate input label: ' + label)
    record['inputs'][label] = input_record(value)

args.out.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile('w', dir=args.out.parent, prefix=args.out.name + '.', suffix='.tmp', delete=False) as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write('\n')
    temporary = pathlib.Path(handle.name)
temporary.replace(args.out)
print('Recorded', args.status, 'run provenance at', args.out)
