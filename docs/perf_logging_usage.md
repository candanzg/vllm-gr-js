# Performance Logging Usage

This profiling workflow is now split into two layers:

1. request timeline in `vllm-gr-dev`
2. low-overhead function hotspots in `vllm-0.14-dev` and `vllm-ascend-0.14-dev`

That split matters because `vllm-gr-dev` overrides part of the upstream serving
path, especially beam search, while the real worker-side hotspots live in
upstream `vllm` and `vllm-ascend`.

## 1. Request timeline

Use this when you want a per-request stage breakdown such as:

- `add_batch_step`
- `gather_beam_results`
- `collect_beam_result`
- `reconstruct_and_decode`

Enable it before starting the server:

```bash
export VLLM_GR_PERF_LOG=1
```

If you want to further split the prefill engine-compute path into worker-side
sub-stages, also enable the Ascend execution-time observer:

```bash
export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1
```

For this request-timeline workflow, `function_profile` is optional. You can keep
it off when you only need the structured JSON timeline:

```bash
export VLLM_GR_FUNCTION_PROFILE=0
unset VLLM_GR_FUNCTION_PROFILE_DIR
```

Then parse the JSON lines from `server.log`:

```bash
python -m vllm_gr.tools.parse_perf_logs \
  --input server.log \
  --output performance_timeline.csv
```

To isolate one request:

```bash
python -m vllm_gr.tools.parse_perf_logs \
  --input server.log \
  --request-id req-123 \
  --output performance_req_123.csv
```

If you want the request-side waiting block split into three clearer buckets,
add the stage summary view:

```bash
python -m vllm_gr.tools.parse_perf_logs \
  --input server.log \
  --stage-summary \
  --output performance_stage_summary.csv
```

That roll-up shows:

- `Request / queueing`
- `Worker / compute`
- `Request / collect+merge`

This CSV uses the sample-style columns:

- `Evidence group`
- `Segment / event`
- `Start us`
- `End us`
- `Avg us`
- `P99 us`
- `What happens`

## 2. Client-visible stream overhead

Use this when you need to explain the gap between:

- benchmark `TTFT`
- benchmark `E2EL`
- server-side `Avg Beam Time`

Enable this before starting the server:

```bash
export VLLM_GR_EMIT_STREAM_PERF=1
```

This path does not require `function_profile`.

With `vllm-gr-dev/benchmarks/open_one_rec/one_rec_patch.py` enabled, the
benchmark output will print:

- `Request -> response headers`
- `Server entry -> route start`
- `Server handler preprocess`
- `Server route wrapper`
- `Route return -> http.start`
- `Client/network residual to headers`
- `Headers -> first chunk`
- `First chunk -> first visible`

It will also split `Server handler preprocess` into:

- `check_model`
- `get_adapters`
- `get_tokenizer`
- `tool_setup`
- `validate_template`
- `preprocess_chat`
- `build_request_id`
- `sampling_params`
- `trace_headers`
- `generator_setup`
- `split total`
- `preprocess residual`

This is the fastest way to answer:

- why response headers are late
- whether the delay is server preprocess or client/network residual
- which `create_chat_completion()` sub-step dominates pre-header latency

## 3. Function hotspots

Use this when you want function-level profiling with much lower runtime impact
 than JSON logging on hot paths.

The worker-side profiler:

- is disabled by default
- only aggregates in memory during runtime
- writes one CSV per process at exit

Enable it before starting the server:

```bash
export VLLM_GR_FUNCTION_PROFILE=1
export VLLM_GR_FUNCTION_PROFILE_DIR=./function_profile
```

When the process exits, it will write files like:

```text
./function_profile/vllm_function_profile_<pid>.csv
```

Current function-level coverage includes:

- `vllm.v1.engine.llm_engine.step`
- `vllm.v1.core.sched.scheduler.schedule`
- `vllm.v1.worker.gpu.async_utils.get_output`
- `vllm_ascend.worker.worker.execute_model`
- `vllm_ascend.worker.worker.sample_tokens`
- `vllm_ascend.worker.model_runner_v1._prepare_inputs`
- `vllm_ascend.worker.model_runner_v1.execute_model`
- `vllm_ascend.worker.model_runner_v1.sample_tokens`
- `vllm_ascend.worker.model_runner_v1._bookkeeping_sync`
- `vllm.model_executor.models.qwen3.qwen3_for_causal_lm_forward`
- `vllm.model_executor.models.qwen3.qwen3_model_forward`
- `vllm.model_executor.models.qwen3.qwen3_model_layers`

## 4. Merge function-profile CSVs

After the run, merge the per-process hotspot CSVs into one sample-style table:

```bash
python -m vllm_gr.tools.merge_function_profile_csv \
  --input-dir ./function_profile \
  --output function_hotspots.csv
```

You can also pass explicit files:

```bash
python -m vllm_gr.tools.merge_function_profile_csv \
  --input ./function_profile/vllm_function_profile_1001.csv \
          ./function_profile/vllm_function_profile_1002.csv \
  --output function_hotspots.csv
```

The merged CSV still uses:

- `Evidence group`
- `Segment / event`
- `Start us`
- `End us`
- `Avg us`
- `P99 us`
- `What happens`

For function hotspots, rows are grouped more precisely than before:

- `Scheduler / driver`
- `Worker / GPU`
- `Worker / Ascend`
- `API / client`

## 5. Recommended usage

For OneRec beam-search analysis, use both layers together:

1. enable `VLLM_GR_PERF_LOG=1` to see request-stage waiting time
2. enable `VLLM_GR_EMIT_STREAM_PERF=1` to split request-to-headers latency
3. enable `VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1` to split prefill worker stages
4. optionally enable `VLLM_GR_FUNCTION_PROFILE=1` to see worker-side function hotspots
5. run the benchmark
6. parse request timeline CSV
7. merge function hotspot CSVs if `function_profile` is enabled
8. optionally generate the 3-way request-stage summary CSV

That gives you both:

- where the request spent time logically
- which concrete worker functions consumed time physically
- why client-visible `TTFT` / `E2EL` are larger than server-side beam timing

Recommended low-overhead setup for request-visible latency analysis:

```bash
export VLLM_GR_PERF_LOG=1
export VLLM_GR_EMIT_STREAM_PERF=1
export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1
export VLLM_GR_FUNCTION_PROFILE=0
unset VLLM_GR_FUNCTION_PROFILE_DIR
```

Only turn `function_profile` back on when you need function-level hotspots.

## 6. Why the old CSV only showed two groups

If you previously saw only `Scheduler / driver` and `API / client`, that means
the runtime logs only captured coarse request-stage events and did not capture
worker-side function events. That was the reason for moving hot-path profiling
into upstream `vllm` and `vllm-ascend` with in-memory aggregation.
