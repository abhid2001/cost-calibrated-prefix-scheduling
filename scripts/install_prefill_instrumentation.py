#!/usr/bin/env python3
"""Install opt-in timing into the pinned SGLang wheel; preserve original files."""
import importlib.metadata, pathlib, shutil, sys
# Purpose: add per-request GPU prefill timing (prefill_gpu_ms) to SGLang's reply; used by collect_calibration.py
# Only active when PREFIX_GPU_PREFILL_TIMING=1. Each oldN/newN pair = original SGLang text / replacement text.
if importlib.metadata.version('sglang') != '0.5.10.post1': raise SystemExit('Requires sglang 0.5.10.post1')
root=pathlib.Path(sys.prefix)/'lib64/python3.11/site-packages/sglang/srt/managers'
if not root.exists(): root=pathlib.Path(sys.prefix)/'lib/python3.11/site-packages/sglang/srt/managers'
worker=root/'tp_worker.py'; sched=root/'scheduler.py'; output=root/'scheduler_output_processor_mixin.py'
# old/new (tp_worker.py): start/stop CUDA events around the model forward pass on prefill => prefill_gpu_ms
old='''            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph'''
new='''            # PREFIX_GPU_PREFILL_TIMING_V1: opt-in, non-overlap single-request path.
            timing = os.environ.get("PREFIX_GPU_PREFILL_TIMING") == "1" and forward_batch.forward_mode.is_extend()
            if timing:
                if forward_batch.batch_size != 1 or self.enable_overlap:
                    raise RuntimeError("prefill timing requires one request and no overlap")
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(torch.cuda.current_stream())
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            if timing:
                end_event.record(torch.cuda.current_stream())
                end_event.synchronize()
                prefill_gpu_ms = float(start_event.elapsed_time(end_event))
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph'''
# old2/new2 (tp_worker.py): attach prefill_gpu_ms to the batch result
old2='''            if is_verify:
                # Skip sampling and return logits for target forward'''
new2='''            if timing:
                batch_result.prefix_prefill_gpu_ms = prefill_gpu_ms
            if is_verify:
                # Skip sampling and return logits for target forward'''
# old_guard/new_guard (scheduler.py): refuse timing unless 1 GPU, 1 running request, no chunking, no overlap
old_guard = "        self.forward_ct += 1\n\n        # Whether to run the profiler"
new_guard = """        self.forward_ct += 1
        if os.environ.get("PREFIX_GPU_PREFILL_TIMING") == "1":
            if self.enable_overlap or self.server_args.chunked_prefill_size != -1 or self.server_args.max_running_requests != 1 or self.tp_size != 1 or self.pp_size != 1:
                raise RuntimeError("GPU prefill timing requires single GPU, one request, no chunking and no overlap")

        # Whether to run the profiler"""
# old3/new3 (scheduler.py): copy the batch result's time into req.customized_info['prefill_gpu_ms']
old3='''                future_indices_or_next_token_ids = batch_result.next_token_ids
                self.update_cache_from_scheduler(batch, batch_result)'''
new3='''                if os.environ.get("PREFIX_GPU_PREFILL_TIMING") == "1" and batch.forward_mode == ForwardMode.EXTEND:
                    if self.server_args.chunked_prefill_size != -1 or len(batch.reqs) != 1:
                        raise RuntimeError("prefill timing requires unchunked single-request execution")
                    ms = getattr(batch_result, "prefix_prefill_gpu_ms", None)
                    if ms is None:
                        raise RuntimeError("GPU prefill timing missing from worker")
                    req = batch.reqs[0]
                    if req.customized_info is None: req.customized_info = {}
                    req.customized_info["prefill_gpu_ms"] = [ms]
                future_indices_or_next_token_ids = batch_result.next_token_ids
                self.update_cache_from_scheduler(batch, batch_result)'''
# old4/new4 (scheduler_output_processor_mixin.py): send prefill_gpu_ms back once in the reply's meta_info
old4='''                        customized_info[k].append(
                            v[send_token_offset : len(output_ids_)]
                        )'''
new4='''                        customized_info[k].append(
                            v[0] if k == "prefill_gpu_ms" and v else v[send_token_offset : len(output_ids_)]
                        )'''
# replace_once: target text must appear exactly once, otherwise stop before writing anything
def replace_once(s,a,b):
    if s.count(a)!=1: raise SystemExit('Patch target is missing or ambiguous; no files written')
    return s.replace(a,b,1)
paths=[worker,sched,output]; original=[p.read_text() for p in paths]
# Already installed: only add the guard if an older install lacks it, then exit
if 'PREFIX_GPU_PREFILL_TIMING_V1' in original[0]:
    if 'GPU prefill timing requires single GPU' not in original[1]:
        sched.write_text(replace_once(original[1], old_guard, new_guard))
    print('Instrumentation already installed; configuration guard verified');sys.exit(0)
w=replace_once(original[0],'import logging\n','import logging\nimport os\n'); w=replace_once(w,old,new); w=replace_once(w,old2,new2)
modified=[w,replace_once(replace_once(original[1],old3,new3),old_guard,new_guard),replace_once(original[2],old4,new4)]
# Back up the unpatched files (refuse to overwrite a different backup), then write the patched files
backup=pathlib.Path('patches/sglang-original');backup.mkdir(parents=True,exist_ok=True)
for p,text in zip(paths,original):
    dest=backup/p.name
    if dest.exists() and dest.read_text()!=text: raise SystemExit('Existing backup differs; refusing overwrite')
    dest.write_text(text)
for p,text in zip(paths,modified): p.write_text(text)
print('Installed opt-in GPU prefill instrumentation; originals saved to',backup)
