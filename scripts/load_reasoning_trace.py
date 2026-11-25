#!/usr/bin/env python3
"""Run a trace as concurrent, materialized reasoning DAGs against SGLang.

The engine owns admission.  When --admission-log is provided, this client joins
the patched engine's actual admission records back to request IDs; it never
mistakes a client dispatch timestamp for an engine admission timestamp.
"""
import argparse
import concurrent.futures
import dataclasses
import datetime
import json
import math
import os
import pathlib
import sys
import threading
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from prefix_scheduler.core import chat_token_ids, generate, normalize_number, now
from prefix_scheduler.live_workload import materialize, question_result, release
from transformers import AutoTokenizer


p = argparse.ArgumentParser()
p.add_argument('--trace', required=True)
p.add_argument('--out', required=True)
p.add_argument('--labels')
p.add_argument('--url', default='http://127.0.0.1:30000')
p.add_argument('--config', default='configs/foundation.json')
p.add_argument('--workers', type=int, default=32, help='maximum concurrent HTTP requests')
p.add_argument('--question-workers', type=int, dest='workers', help=argparse.SUPPRESS)
p.add_argument('--max-new-tokens', type=int, help='override the configured generation budget (useful for smoke tests)')
p.add_argument('--admission-log', help='JSONL written by PREFIX_ADMISSION_LOG')
p.add_argument('--dependency-graph', help='atomically updated graph snapshot for dependency-aware engine admission')
p.add_argument('--require-live-admissions', action='store_true',
               help='fail unless every dispatched request has one engine admission record')
p.add_argument('--require-stream-telemetry', action='store_true',
               help='fail unless completed requests have GPU prefill, cache and first-token telemetry')
a = p.parse_args()
if a.require_live_admissions and not a.admission_log:
    p.error('--require-live-admissions requires --admission-log')
if a.workers < 1:
    p.error('--workers must be positive')
if a.max_new_tokens is not None and a.max_new_tokens < 1:
    p.error('--max-new-tokens must be positive')

config = json.loads(pathlib.Path(a.config).read_text())
trace = [json.loads(line) for line in pathlib.Path(a.trace).read_text().splitlines() if line.strip()]
if not trace:
    p.error('trace must be nonempty')
if any(row.get('mode') not in ('direct', 'cot', 'self_consistency', 'tot') for row in trace):
    p.error('trace contains an unsupported mode')
if any(not isinstance(row.get('offset_seconds'), (int, float)) or row['offset_seconds'] < 0 for row in trace):
    p.error('trace offsets must be nonnegative numbers')
if any(trace[i]['offset_seconds'] > trace[i + 1]['offset_seconds'] for i in range(len(trace) - 1)):
    p.error('trace offsets must be ordered')
question_ids = [row.get('question_id') for row in trace]
if any(not isinstance(question_id, str) or not question_id for question_id in question_ids) or len(set(question_ids)) != len(question_ids):
    p.error('trace question IDs must be unique and nonempty')

labels = {} if not a.labels else {
    row['question_id']: normalize_number(row['answer'])
    for row in (json.loads(line) for line in pathlib.Path(a.labels).read_text().splitlines() if line.strip())
}
tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['model_revision'])
epoch = datetime.datetime.now(datetime.timezone.utc)
graphs = [materialize(row['question_id'], row['question'], row['mode'],
                      (epoch + datetime.timedelta(seconds=row['offset_seconds'])).isoformat(),
                      a.max_new_tokens or config['max_new_tokens']) for row in trace]
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)

lock = threading.Lock()
events = []
sequence = 0


def write_dependency_graph():
    """Publish only complete snapshots; SGLang may read while this client runs."""
    if not a.dependency_graph:
        return
    destination = pathlib.Path(a.dependency_graph)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {'version': 1, 'requests': [
        {'request_id': request.request_id, 'parent_ids': request.parent_ids, 'status': request.status}
        for graph in graphs for request in graph.records
    ]}
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(payload, sort_keys=True) + '\n')
    os.replace(temporary, destination)


def event(kind, request=None, **extra):
    """Call while lock is held so the event sequence matches state changes."""
    global sequence
    sequence += 1
    row = {'sequence': sequence, 'event': kind, 'timestamp': now()}
    if request is not None:
        row.update({'request_id': request.request_id, 'question_id': request.question_id,
                    'stage': request.stage, 'status': request.status})
    row.update(extra)
    events.append(row)


for graph in graphs:
    for request in graph.records:
        event('materialized', request, parent_ids=request.parent_ids)
write_dependency_graph()


def prepare(request, messages):
    request.prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    request.input_token_ids = chat_token_ids(tokenizer, messages)


def dispatch(graph, request):
    """One HTTP attempt. It changes only this request; graph release is central."""
    start = time.monotonic()
    try:
        messages, temperature = graph.messages_for(request)
        prepare(request, messages)
        with lock:
            request.transition('running')
            # Only the engine admission log can establish this timestamp.
            request.admitted_at = None
            write_dependency_graph()
            event('dispatched', request, input_tokens=len(request.input_token_ids))
        start = time.monotonic()
        text, metadata = generate(
            a.url + '/generate', request.input_token_ids,
            {'temperature': temperature, 'max_new_tokens': request.max_new_tokens,
             'sampling_seed': request.sampling_seed},
            config['request_timeout_seconds'], record=request, rid=request.request_id)
        if not isinstance(text, str) or not text.strip():
            raise ValueError('empty model output')
        with lock:
            request.client_duration_ms = (time.monotonic() - start) * 1000
            request.output = text
            request.engine_metadata = metadata
            request.matched_prefix_tokens = request.engine_metadata.get('cached_tokens')
            request.transition('completed')
            write_dependency_graph()
            event('completed', request)
    except Exception as exc:
        with lock:
            request.client_duration_ms = (time.monotonic() - start) * 1000
            if request.status in ('ready', 'running'):
                request.transition('failed')
            request.error = '%s: %s' % (type(exc).__name__, exc)
            write_dependency_graph()
            event('failed', request, error=request.error)
    return graph


def record_release(graph):
    """Release ready children or record recursive dependency failures."""
    ready, failed = release(graph)
    write_dependency_graph()
    for request in ready:
        event('ready', request, parent_ids=request.parent_ids)
    for request in failed:
        event('dependency_failed', request, error=request.error, parent_ids=request.parent_ids)
    return ready


ready = []
active = {}
next_graph = 0
started = time.monotonic()
by_question = {graph.question_id: graph for graph in graphs}
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
    while next_graph < len(graphs) or active or ready:
        elapsed = time.monotonic() - started
        while next_graph < len(graphs) and trace[next_graph]['offset_seconds'] <= elapsed:
            graph = graphs[next_graph]
            with lock:
                event('question_arrived', question_id=graph.question_id, mode=graph.mode,
                      scheduled_arrival=graph.arrival)
                ready.extend(record_release(graph))
            next_graph += 1
        while ready:
            request = ready.pop(0)
            active[pool.submit(dispatch, by_question[request.question_id], request)] = request
        if not active:
            if next_graph < len(graphs):
                time.sleep(max(0, trace[next_graph]['offset_seconds'] - (time.monotonic() - started)))
            continue
        timeout = None if next_graph == len(graphs) else max(0, trace[next_graph]['offset_seconds'] - (time.monotonic() - started))
        completed, _ = concurrent.futures.wait(active, timeout=timeout,
                                                return_when=concurrent.futures.FIRST_COMPLETED)
        for future in completed:
            request = active.pop(future)
            graph = future.result()
            with lock:
                ready.extend(record_release(graph))


records = [request for graph in graphs for request in graph.records]
by_id = {request.request_id: request for request in records}
live_events = []
missing = []
cache_mismatches = []
if a.admission_log:
    log_path = pathlib.Path(a.admission_log)
    if not log_path.exists():
        missing = list(by_id)
    else:
        for line in log_path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                if item.get('actual_id') in by_id:
                    live_events.append(item)
        grouped = {}
        for item in live_events:
            grouped.setdefault(item['actual_id'], []).append(item)
        for request_id, request in by_id.items():
            if request.dispatched_at is None:
                continue
            items = grouped.get(request_id, [])
            if len(items) != 1:
                missing.append(request_id)
                continue
            item = items[0]
            candidates = [candidate for candidate in item.get('candidates', []) if candidate.get('id') == request_id]
            if request.status == 'completed' and (len(candidates) != 1 or
                    candidates[0].get('cached_tokens') != request.matched_prefix_tokens):
                cache_mismatches.append(request_id)
            request.admitted_at = datetime.datetime.fromtimestamp(item['ts_ns'] / 1_000_000_000,
                                                                   tz=datetime.timezone.utc).isoformat()
            request.engine_metadata = dict(request.engine_metadata or {})
            request.engine_metadata['live_admission'] = {
                key: item.get(key) for key in ('admission', 'slot', 'reason', 'ranked_ids')
            }
            event('engine_admitted', request, admission=item.get('admission'), slot=item.get('slot'),
                  reason=item.get('reason'), engine_timestamp=request.admitted_at)
    (out / 'admissions.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in live_events))

results = []
for graph, row in zip(graphs, trace):
    result = question_result(graph)
    result['arrival'] = graph.arrival
    result['trace_seed'] = row.get('seed')
    result['correct'] = (result['answer'] == labels[graph.question_id]
                         if graph.question_id in labels and labels[graph.question_id] is not None else None)
    results.append(result)

(out / 'trace.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in trace))
(out / 'events.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in events))
(out / 'requests.jsonl').write_text(''.join(json.dumps(dataclasses.asdict(request)) + '\n' for request in records))
(out / 'questions.jsonl').write_text(''.join(json.dumps(result) + '\n' for result in results))
if a.require_live_admissions and missing:
    raise SystemExit('Missing or non-unique engine admissions for: ' + ', '.join(sorted(missing)))
if a.require_live_admissions and cache_mismatches:
    raise SystemExit('Engine admission and response cache telemetry disagree for: ' + ', '.join(sorted(cache_mismatches)))
if a.require_stream_telemetry:
    errors = []
    for request in records:
        if request.status != 'completed':
            continue
        meta = request.engine_metadata or {}
        ms = meta.get('prefill_gpu_ms')
        cached = meta.get('cached_tokens')
        if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms < 0:
            errors.append(request.request_id + ': missing or invalid prefill_gpu_ms')
        if isinstance(cached, bool) or not isinstance(cached, int) or cached < 0:
            errors.append(request.request_id + ': missing or invalid cached_tokens')
        if request.first_token_at is None:
            errors.append(request.request_id + ': missing first_token_at')
    if errors:
        raise SystemExit('Streaming telemetry validation failed: ' + '; '.join(errors))
print('Saved', len(results), 'materialized question graphs and', len(records), 'request records to', out)
