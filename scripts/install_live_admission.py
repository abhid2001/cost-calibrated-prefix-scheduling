#!/usr/bin/env python3
"""Install opt-in single-request admission hooks into pinned SGLang."""
import importlib.metadata,pathlib,sys
# Purpose: plug LiveAdmission into SGLang by text-patching its scheduler.py; without this, every policy runs as SGLang's own ordering
# Safety: exact SGLang version only, locate scheduler.py (lib64 or lib), skip if already patched
if importlib.metadata.version('sglang')!='0.5.10.post1':raise SystemExit('Wrong SGLang version')
p=pathlib.Path(sys.prefix)/'lib64/python3.11/site-packages/sglang/srt/managers/scheduler.py'
if not p.exists(): p=pathlib.Path(sys.prefix)/'lib/python3.11/site-packages/sglang/srt/managers/scheduler.py'
s=p.read_text()
if 'PREFIX_LIVE_ADMISSION_V1' in s:print('Already installed');sys.exit()
# Hook 1 (before admission): a = original SGLang lines, b = replacement.
# If PREFIX_ADMISSION_POLICY is set: require 1 GPU / 1 request / no overlap / no chunking, then LiveAdmission.order() sorts the queue.
# prefix_match(req) = cached tokens for req from SGLang's radix cache (capped at len-1: the last token must be computed)
a='''        # Get priority queue
        self.policy.calc_priority(self.waiting_queue, self.running_batch)'''
b='''        # PREFIX_LIVE_ADMISSION_V1: opt-in, single-device non-batched ordering.
        if os.environ.get("PREFIX_ADMISSION_POLICY"):
            if self.enable_overlap or self.server_args.chunked_prefill_size != -1 or self.server_args.max_running_requests != 1 or self.tp_size != 1 or self.pp_size != 1:
                raise RuntimeError("live policy requires one GPU, one request, no overlap and no chunking")
            if not hasattr(self, "prefix_live_policy"):
                from prefix_scheduler.live_admission import from_environment
                self.prefix_live_policy = from_environment()
            from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
            from sglang.srt.mem_cache.radix_cache import RadixKey
            def prefix_match(req):
                tokens = req.origin_input_ids + req.output_ids
                max_len = max(len(tokens) - 1, 0)
                if req.return_logprob and req.logprob_start_len >= 0:
                    max_len = min(max_len, req.logprob_start_len)
                return len(self.tree_cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids=tokens[:max_len], extra_key=req.extra_key), req=req, cow_mamba=self.tree_cache.supports_mamba())).device_indices)
            self.prefix_live_policy.order(self.waiting_queue, prefix_match)
        else:
            self.policy.calc_priority(self.waiting_queue, self.running_batch)'''
# Hook 2 (after admission): c = original lines, d = same lines plus LiveAdmission.admitted() to log the pick
c='''        can_run_set = set(can_run_list)
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]'''
d='''        if os.environ.get("PREFIX_ADMISSION_POLICY"):
            self.prefix_live_policy.admitted(can_run_list)
        can_run_set = set(can_run_list)
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]'''
# Apply: original text must appear exactly once, back up the unpatched file, then write the patched one
if s.count(a)!=1 or s.count(c)!=1:raise SystemExit('Unexpected engine source; no changes made')
backup=pathlib.Path('patches/sglang-original');backup.mkdir(parents=True,exist_ok=True)
before=backup/'scheduler.before-live.py'
if before.exists() and before.read_text()!=s:raise SystemExit('Existing live-admission backup differs; refusing overwrite')
before.write_text(s)
p.write_text(s.replace(a,b).replace(c,d));print('Installed opt-in live admission hook')
