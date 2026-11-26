"""Shared state, answer parsing, prompt construction, and HTTP transport."""
import dataclasses
from collections.abc import Mapping
import datetime
import decimal
import json
import re
import urllib.request
from typing import Optional


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@dataclasses.dataclass
class Request:
    question_id: str
    request_id: str
    stage: str
    question_arrival: str
    created_at: str
    parent_ids: list = dataclasses.field(default_factory=list)
    status: str = 'blocked'
    ready_at: Optional[str] = None
    # Client dispatch and engine admission are distinct.  The latter is only
    # populated when a live SGLang admission log is supplied by the runner.
    dispatched_at: Optional[str] = None
    admitted_at: Optional[str] = None
    first_token_at: Optional[str] = None
    finished_at: Optional[str] = None
    input_token_ids: Optional[list] = None
    prompt: Optional[str] = None
    matched_prefix_tokens: Optional[int] = None
    predicted_saving_ms: Optional[float] = None
    sampling_seed: Optional[int] = None
    max_new_tokens: int = 256
    output: Optional[str] = None
    error: Optional[str] = None
    engine_metadata: Optional[dict] = None
    client_duration_ms: Optional[float] = None

    def transition(self, state, timestamp=None):
        allowed = {'blocked': {'ready', 'failed'}, 'ready': {'running', 'failed'},
                   'running': {'completed', 'failed'}, 'completed': set(), 'failed': set()}
        if state not in allowed[self.status]:
            raise ValueError('invalid transition: %s -> %s' % (self.status, state))
        stamp = timestamp or now()
        self.status = state
        if state == 'ready': self.ready_at = stamp
        if state == 'running':
            self.dispatched_at = stamp
            # Compatibility for serial runners; the live workload replaces
            # this with the engine-side admission timestamp after log join.
            self.admitted_at = stamp
        if state in ('completed', 'failed'): self.finished_at = stamp


NUMBER = re.compile(r'(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])')
FINAL = re.compile(r'(?:final\s+answer\s*[:=]|####)\s*([^\n]+)', re.I)
BOXED = re.compile(r'\\boxed\s*\{\s*([^{}]+?)\s*\}', re.I)

def normalize_number(text):
    """Return a canonical decimal string only for one whole numeric value."""
    value = text.strip().replace(',', '').replace('$', '')
    if not re.fullmatch(r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)', value): return None
    try:
        n = decimal.Decimal(value)
        if not n.is_finite(): return None
        return format(n.normalize(), 'f') if n else '0'
    except decimal.InvalidOperation:
        return None

def parse_answer(text):
    text = text or ''
    matches = FINAL.findall(text)
    boxed = BOXED.findall(text)
    candidate = matches[-1] if matches else (boxed[-1] if boxed else None)
    if candidate is None: return None
    numbers = NUMBER.findall(candidate)
    return normalize_number(numbers[0]) if len(numbers) == 1 else None

def prompt_for(question, mode):
    if mode == 'direct':
        instruction = 'Answer the math question directly without explaining your work.'
    elif mode == 'cot':
        instruction = 'Solve the math question step by step.'
    else: raise ValueError('unsupported reasoning mode: ' + mode)
    return [{'role': 'system', 'content': instruction + ' End with exactly Final answer: <number>.'},
            {'role': 'user', 'content': question}]

def chat_token_ids(tokenizer, messages):
    """Return a flat, JSON-serializable chat-token list across transformers versions."""
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    # `return_dict=False` handles current Transformers releases.  Normalize
    # older/fake tokenizers too, rather than sending a BatchEncoding or batch
    # dimension to SGLang's native input_ids API.
    if isinstance(tokens, Mapping):
        tokens = tokens.get('input_ids')
    if hasattr(tokens, 'tolist'):
        tokens = tokens.tolist()
    if isinstance(tokens, (list, tuple)) and len(tokens) == 1 and isinstance(tokens[0], (list, tuple)):
        tokens = tokens[0]
    if not isinstance(tokens, (list, tuple)) or not all(isinstance(token, int) and not isinstance(token, bool) for token in tokens):
        raise ValueError('chat template returned invalid token IDs')
    return list(tokens)


def post_json(url, data, timeout=300):
    request = urllib.request.Request(url, json.dumps(data).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def generate(url, input_ids, sampling_params, timeout=300, record=None, rid=None):
    """Stream native SGLang output and record client-observed TTFT when requested."""
    body = {'input_ids': input_ids, 'sampling_params': sampling_params, 'stream': True}
    if rid is not None:
        body['rid'] = rid
    request = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    last = None
    metadata = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            data = line[len('data:'):].strip()
            if data == '[DONE]':
                break
            try:
                chunk = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError('malformed streaming event: %s' % exc) from exc
            if not isinstance(chunk, dict):
                raise RuntimeError('malformed streaming event: expected JSON object')
            if 'error' in chunk:
                raise RuntimeError('engine error: %s' % (chunk['error'],))
            meta = chunk.get('meta_info') or {}
            metadata.update(meta)
            generated = meta.get('completion_tokens', 0) >= 1 or bool(chunk.get('text'))
            if record is not None and record.first_token_at is None and generated:
                record.first_token_at = now()
            last = chunk
    if last is None:
        raise RuntimeError('stream ended without any output chunk')
    return last.get('text', ''), metadata
