#!/usr/bin/env python3
"""Select the highest development rate that queues work without overload."""
import argparse, glob, json, pathlib

p = argparse.ArgumentParser()
p.add_argument('--analyses', required=True, help='glob of per-rate analysis.json files')
p.add_argument('--out', required=True)
p.add_argument('--min-queue-wait-seconds', type=float, default=.01)
p.add_argument('--max-p95-queue-wait-seconds', type=float, default=5.0)
a = p.parse_args()
paths = sorted(glob.glob(a.analyses))
if not paths:
    p.error('no pilot analyses found')
candidates = []
for name in paths:
    analysis = json.loads(pathlib.Path(name).read_text())
    rate = float(pathlib.Path(name).parent.name.rsplit('rate-', 1)[1])
    p95 = analysis.get('engine_admission_wait_p95_seconds')
    healthy = (analysis.get('request_failures') == 0 and
               analysis.get('engine_admission_telemetry_requests') == analysis.get('request_count') and
               p95 is not None and p95 <= a.max_p95_queue_wait_seconds)
    queued = healthy and p95 >= a.min_queue_wait_seconds
    candidates.append({'rate': rate, 'analysis': name, 'healthy': healthy, 'queued': queued, 'p95_engine_wait_seconds': p95})
eligible = [item for item in candidates if item['queued']]
if not eligible:
    eligible = [item for item in candidates if item['healthy']]
if not eligible:
    raise SystemExit('no pilot rate completed with complete engine telemetry and bounded queueing')
chosen = max(eligible, key=lambda item: item['rate'])
chosen['selection_rule'] = 'highest healthy queued rate' if chosen['queued'] else 'highest healthy rate; no candidate reached queue threshold'
pathlib.Path(a.out).write_text(json.dumps({'candidates': candidates, 'chosen': chosen}, indent=2) + '\n')
print('Selected rate', chosen['rate'], 'from', chosen['analysis'])
