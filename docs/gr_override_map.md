# vllm-gr Override Map

This note summarizes where `vllm-gr-dev` overrides, wraps, or redirects
upstream `vllm` / `vllm-ascend` behavior.

It is meant to answer one practical question:

Which repo should carry the instrumentation for a given stage or function?

## Rule of thumb

- Request-stage beam-search timeline:
  instrument in `vllm-gr-dev`
- Generic engine / scheduler / worker hotspot functions:
  instrument in `vllm-0.14-dev` or `vllm-ascend-0.14-dev`
- Patch-layer probes:
  use only for debugging whether the override path is entered

## 1. OpenAI serving path overrides

### `OpenAIServing.beam_search`

- Patch entry:
  [D:\code\codexapp\vllm-gr-dev\vllm_gr\entrypoints\openai\beam_search_patch.py](D:\code\codexapp\vllm-gr-dev\vllm_gr\entrypoints\openai\beam_search_patch.py)
- Replacement target:
  `OpenAIServing.beam_search = beam_search`
- Actual implementation:
  [D:\code\codexapp\vllm-gr-dev\vllm_gr\entrypoints\openai\serving_engine.py](D:\code\codexapp\vllm-gr-dev\vllm_gr\entrypoints\openai\serving_engine.py)

Impact:

- The online OpenAI-compatible beam-search request path is no longer using the
  upstream `vllm.entrypoints.openai.serving_engine.OpenAIServing.beam_search`.
- Request-stage timing such as `add_batch_step`, `gather_beam_results`,
  `collect_beam_result`, and `reconstruct_and_decode` belongs in
  `vllm-gr-dev`.

### `ChatCompletionRequest.to_beam_search_params`

- Patch entry:
  [D:\code\codexapp\vllm-gr-dev\vllm_gr\entrypoints\openai\beam_search_patch.py](D:\code\codexapp\vllm-gr-dev\vllm_gr\entrypoints\openai\beam_search_patch.py)

Impact:

- Beam-search request parsing is customized in `vllm-gr-dev`.
- This is configuration / request construction logic, not a primary hotspot
  target.

### `OpenAIServingModels.__init__`

- Patch entry:
  [D:\code\codexapp\vllm-gr-dev\vllm_gr\entrypoints\openai\serving_models.py](D:\code\codexapp\vllm-gr-dev\vllm_gr\entrypoints\openai\serving_models.py)

Impact:

- Model-serving bootstrap is adjusted in `vllm-gr-dev`.
- Usually not a per-request hotspot target.

## 2. Offline GR entrypoint override

### `GRLLM.beam_search`

- Class:
  [D:\code\codexapp\vllm-gr-dev\vllm_gr\entrypoints\gr.py](D:\code\codexapp\vllm-gr-dev\vllm_gr\entrypoints\gr.py)
- Definition:
  `class GRLLM(LLM)` with its own `beam_search(...)`

Impact:

- Offline beam-search execution is also custom and does not follow the plain
  upstream `LLM.beam_search` behavior.
- If you analyze offline GR beam search, the request orchestration belongs to
  `vllm-gr-dev` too.

## 3. Engine-core pipeline overrides

### `EngineCoreProc.run_engine_core`

- Patch entry:
  [D:\code\codexapp\vllm-gr-dev\vllm_gr\entrypoints\openai\beam_search_patch.py](D:\code\codexapp\vllm-gr-dev\vllm_gr\entrypoints\openai\beam_search_patch.py)
- Replacement implementation:
  [D:\code\codexapp\vllm-gr-dev\vllm_gr\v1\engine\engine_core_patch.py](D:\code\codexapp\vllm-gr-dev\vllm_gr\v1\engine\engine_core_patch.py)

Impact:

- The engine core loop is wrapped so beam batch/fork behavior can be inserted.
- This is structural pipeline redirection, not the best place for persistent
  hotspot accounting.

### `EngineCoreProc.process_input_sockets`

- Patch implementation:
  [D:\code\codexapp\vllm-gr-dev\vllm_gr\v1\engine\engine_core_patch.py](D:\code\codexapp\vllm-gr-dev\vllm_gr\v1\engine\engine_core_patch.py)

Impact:

- Input socket handling is adjusted for beam-request traffic.
- Useful if debugging beam control-plane behavior.

### `Scheduler.schedule`

- Patch implementation:
  [D:\code\codexapp\vllm-gr-dev\vllm_gr\v1\engine\engine_core_patch.py](D:\code\codexapp\vllm-gr-dev\vllm_gr\v1\engine\engine_core_patch.py)

Impact:

- The scheduler is behaviorally patched for beam flow.
- For hotspot timing, direct instrumentation still belongs in upstream
  [D:\code\codexapp\vllm-0.14-dev\vllm\v1\core\sched\scheduler.py](D:\code\codexapp\vllm-0.14-dev\vllm\v1\core\sched\scheduler.py),
  because that captures the real scheduler body instead of patch-layer wrappers.

## 4. Worker / model runner overrides

### `GPUModelRunner.execute_model`
### `GPUModelRunner._prepare_inputs`
### `GPUModelRunner._bookkeeping_sync`

- Patch implementation:
  [D:\code\codexapp\vllm-gr-dev\vllm_gr\v1\engine\engine_core_patch.py](D:\code\codexapp\vllm-gr-dev\vllm_gr\v1\engine\engine_core_patch.py)

Impact:

- GPU runner paths are wrapped to propagate beam-related request context.
- For performance analysis, prefer upstream source instrumentation in
  `vllm-0.14-dev` for generic GPU runner hotspots.

### `NPUModelRunner.execute_model`
### `NPUModelRunner._prepare_inputs`
### `NPUModelRunner._bookkeeping_sync`
### `NPUModelRunner.sample_tokens`

- Patch implementation:
  [D:\code\codexapp\vllm-gr-dev\vllm_gr\v1\engine\engine_core_patch.py](D:\code\codexapp\vllm-gr-dev\vllm_gr\v1\engine\engine_core_patch.py)

Impact:

- NPU runner paths are wrapped for beam context propagation.
- For hotspot timing, direct instrumentation belongs in
  [D:\code\codexapp\vllm-ascend-0.14-dev\vllm_ascend\worker\model_runner_v1.py](D:\code\codexapp\vllm-ascend-0.14-dev\vllm_ascend\worker\model_runner_v1.py),
  because that is the real execution body.

### `NPUWorker.execute_model`
### `NPUWorker.sample_tokens`

- Patch entry:
  [D:\code\codexapp\vllm-gr-dev\vllm_gr\patch.py](D:\code\codexapp\vllm-gr-dev\vllm_gr\patch.py)

Impact:

- These were patch-level timing/probe hooks.
- For stable low-overhead profiling, the preferred instrumentation is now in
  [D:\code\codexapp\vllm-ascend-0.14-dev\vllm_ascend\worker\worker.py](D:\code\codexapp\vllm-ascend-0.14-dev\vllm_ascend\worker\worker.py).

### `AsyncOutput.get_output`

- Patch entry:
  [D:\code\codexapp\vllm-gr-dev\vllm_gr\patch.py](D:\code\codexapp\vllm-gr-dev\vllm_gr\patch.py)

Impact:

- Patch-layer timing exists, but the preferred low-overhead hotspot timing now
  lives directly in
  [D:\code\codexapp\vllm-0.14-dev\vllm\v1\worker\gpu\async_utils.py](D:\code\codexapp\vllm-0.14-dev\vllm\v1\worker\gpu\async_utils.py).

## 5. Other runtime patch families

### Attention backend / NPU backend / KV merge

- Main patch hub:
  [D:\code\codexapp\vllm-gr-dev\vllm_gr\patch.py](D:\code\codexapp\vllm-gr-dev\vllm_gr\patch.py)

This file also contains patches for:

- Yuanrong backend registration
- split QKV + RMSNorm + RoPE patching
- paged-KV-style merged KV cache compatibility
- NPU worker init re-patching
- optional LMCache adapter patching

Impact:

- These change runtime behavior and compatibility, but they are not the first
  place to instrument per-request or per-function latency unless you are
  debugging a patch-specific regression.

## 6. Where to instrument now

### Keep in `vllm-gr-dev`

- beam-search request timeline
- batch add / beam fork / merged result collection stages
- GR-specific orchestration

### Keep in `vllm-0.14-dev`

- `LLMEngine.step`
- `Scheduler.schedule`
- `AsyncOutput.get_output`

### Keep in `vllm-ascend-0.14-dev`

- `NPUWorker.execute_model`
- `NPUWorker.sample_tokens`
- `NPUModelRunner._prepare_inputs`
- `NPUModelRunner.execute_model`
- `NPUModelRunner.sample_tokens`
- `NPUModelRunner._bookkeeping_sync`

## 7. Practical interpretation

If a CSV row is one of these:

- `add_batch_step`
- `gather_beam_results`
- `collect_beam_result`
- `reconstruct_and_decode`

then it is part of `vllm-gr-dev` request orchestration.

If a CSV row is one of these:

- `vllm.v1.core.sched.scheduler.schedule`
- `vllm.v1.engine.llm_engine.step`
- `vllm_ascend.worker.worker.execute_model`
- `vllm_ascend.worker.model_runner_v1.execute_model`
- `vllm_ascend.worker.model_runner_v1.sample_tokens`

then it is part of the actual engine / worker execution path and should be
treated as the more trustworthy hotspot evidence.
