#!/usr/bin/env python3
"""Fail early when an experiment lacks immutable inputs or the primary H200 setup."""
import argparse
import importlib.metadata
import json
import math
import pathlib
import sys


def require(path, description):
    path = pathlib.Path(path)
    if not path.is_file():
        raise SystemExit('Missing %s: %s' % (description, path))
    return path


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('foundation', 'pilot', 'matrix'), required=True)
parser.add_argument('--config', default='configs/foundation.json')
parser.add_argument('--platform-config', default='configs/platform.json')
parser.add_argument('--predictor')
parser.add_argument('--pilot-selection')
parser.add_argument('--require-primary-gpu', action='store_true')
parser.add_argument('--hardware-json', help='Use saved visible-device data for an offline preflight check')
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
platform_config = require(args.platform_config, 'platform configuration')
platform_spec = json.loads(platform_config.read_text())
for key in ('name', 'partition', 'qos', 'gres', 'gpu_name_contains', 'min_memory_mib'):
    if key not in platform_spec:
        raise SystemExit('Platform configuration lacks ' + key)
hardware = None
if args.require_primary_gpu:
    if args.hardware_json:
        hardware = json.loads(require(args.hardware_json, 'hardware record').read_text())
    else:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit('Cannot inspect primary GPU without torch: %s' % exc)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise SystemExit('Primary platform requires exactly one Slurm-visible CUDA device; found %s' % torch.cuda.device_count())
        properties = torch.cuda.get_device_properties(0)
        hardware = {'name': properties.name,
                    'memory_mib': properties.total_memory // (1024 * 1024),
                    'cuda_version': torch.version.cuda}
    if not isinstance(hardware, dict):
        raise SystemExit('Hardware record must be an object')
    name, memory = hardware.get('name'), hardware.get('memory_mib')
    if not isinstance(name, str) or not isinstance(memory, int) or platform_spec['gpu_name_contains'] not in name or memory < platform_spec['min_memory_mib']:
        raise SystemExit('%s preflight requires one %s device with at least %s MiB; found %s' %
                         (platform_spec['name'], platform_spec['gpu_name_contains'], platform_spec['min_memory_mib'], hardware))
print(json.dumps({'preflight': 'passed', 'mode': args.mode, 'config': str(config_path),
                  'platform_config': str(platform_config), 'platform': platform_spec['name'],
                  'hardware': hardware}, sort_keys=True))
