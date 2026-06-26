# Demo FuXi Model for vllm-GR

Minimal FuXi model configuration for testing and development.

## Purpose

The `demo-model` directory contains a `config.json` that defines a small
FuXi model architecture. **No weight files are included**.

To run this directory intentionally, the caller must now opt in to
dummy/random weight loading explicitly (for example,
`allow_missing_weights=True`). This keeps real model directories in
strict fail-fast mode while preserving a clear demo path.

Use cases:

- Validating the inference pipeline without trained weights
- Development and debugging
- CI / smoke testing

## Directory Structure

```text
demo-model/
├── config.json    # Model architecture + feature configuration
└── README.md
```

## Usage

```python
from vllm_gr import GRLLM, GRPrompt

llm = GRLLM(
    model="examples/demo-model",
    max_seq_len=64,
    max_candidates=50,
    allow_missing_weights=True,
)

prompt: GRPrompt = {
    "user_id": "demo_user",
    "user_history": [
        {"item_id": 1, "rating": 0, "timestamp": 1609459200},
        {"item_id": 2, "rating": 0, "timestamp": 1609462800},
    ],
    "candidates": [
        {"item_id": 100, "rating": 0, "timestamp": 1609470000},
        {"item_id": 200, "rating": 0, "timestamp": 1609470000},
    ],
}

outputs = llm.recommend(prompts=[prompt])
```

See `examples/fuxi_simple_inference.py` for a complete runnable example.

## config.json Fields

The config follows the `FuXiConfig` schema (a HuggingFace `PretrainedConfig`
subclass). Key sections:

| Section | Fields |
| --------- | -------- |
| Architecture | `num_blocks`, `num_heads`, `dqk`, `dv`, `item_embedding_dim`, ... |
| Features | `item_id_feature_name`, `rating_feature_name`, `timestamp_feature_name` |
| Runtime | `enable_kv_cache`, `use_rating_embedding`, `use_timestamp_payload` |
| Tasks | `task_names` (e.g. `["score"]`) |

## Notes

- Scores from random weights are **not meaningful** — the model runs a
  real FuXi forward pass but without learned parameters.
