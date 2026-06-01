# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate benchmark requests for vllm-gr beam-search serving.

Usage:
    python benchmarks/open_one_rec/tools/generate_benchmark_requests.py \
      --output data/bench/requests.jsonl \
      --input-len 256 \
      --beam-width 128 \
      --num-unique-requests 100 \
      --repeat-first-n 20 \
      --repeat-times 4 \
      --max-tokens 5

Notes:
    - --input-len means the number of SID items between <|sid_begin|> and
      <|sid_end|>.
    - The output file is JSONL. Each line is one request in the same format
      used by quickstart.md for /v1/chat/completions.
    - Repeated requests are copied from the first N unique requests and appended
      after all unique requests.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)
BASE = 8192
SID_BEGIN_TOKEN = "<|sid_begin|>"
SID_END_TOKEN = "<|sid_end|>"
DEFAULT_MODEL = "OpenOneRec/OneRec-1.7B"
DEFAULT_PROMPT_PREFIX = "这是一个视频："
DEFAULT_PROMPT_SUFFIX = "，帮我总结一下这个视频讲述了什么内容"


def _build_sid_triplet(value: int) -> tuple[str, str, str]:
    value_a = value // (BASE * BASE)
    value_b = (value % (BASE * BASE)) // BASE
    value_c = value % BASE
    return (f"<s_a_{value_a}>", f"<s_b_{value_b}>", f"<s_c_{value_c}>")


def _build_sequence(unique_index: int, input_len: int) -> str:
    if input_len <= 0:
        raise ValueError("input_len must be greater than 0")

    sid_tokens: list[str] = []
    start = unique_index * max(input_len, 1)
    for offset in range(input_len):
        sid_tokens.extend(_build_sid_triplet(start + offset))

    return "".join(
        [
            DEFAULT_PROMPT_PREFIX,
            SID_BEGIN_TOKEN,
            *sid_tokens,
            SID_END_TOKEN,
            DEFAULT_PROMPT_SUFFIX,
        ]
    )


def _build_request(model: str, content: str, beam_width: int, max_tokens: int) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "use_beam_search": beam_width > 1,
        "n": beam_width,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "ignore_eos": False,
        "length_penalty": 1.0,
        "vllm_xargs": {
            "begin_token": SID_BEGIN_TOKEN,
            "end_token": SID_END_TOKEN,
        },
    }


def generate_requests(
    model: str,
    input_len: int,
    beam_width: int,
    max_tokens: int,
    num_unique_requests: int,
    repeat_first_n: int,
    repeat_times: int,
) -> list[dict[str, object]]:
    if num_unique_requests <= 0:
        raise ValueError("num_unique_requests must be greater than 0")
    if repeat_first_n < 0:
        raise ValueError("repeat_first_n must be greater than or equal to 0")
    if repeat_times < 0:
        raise ValueError("repeat_times must be greater than or equal to 0")
    if repeat_first_n > num_unique_requests:
        raise ValueError("repeat_first_n cannot exceed num_unique_requests")
    if beam_width <= 0:
        raise ValueError("beam_width must be greater than 0")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    unique_requests: list[dict[str, object]] = []
    for unique_index in range(num_unique_requests):
        content = _build_sequence(unique_index, input_len)
        request = _build_request(model=model, content=content, beam_width=beam_width, max_tokens=max_tokens)
        request["request_id"] = f"unique-{unique_index:06d}"
        request["request_group"] = "unique"
        request["source_unique_index"] = unique_index
        unique_requests.append(request)

    requests = list(unique_requests)
    for repeat_round in range(repeat_times):
        for unique_index in range(repeat_first_n):
            duplicated_request = json.loads(json.dumps(unique_requests[unique_index], ensure_ascii=False))
            duplicated_request["request_id"] = (
                f"repeat-{repeat_round:03d}-from-{unique_index:06d}"
            )
            duplicated_request["request_group"] = "repeat"
            duplicated_request["repeat_round"] = repeat_round
            duplicated_request["source_unique_index"] = unique_index
            requests.append(duplicated_request)

    return requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate vllm-gr benchmark requests in JSONL format. "
            "input-len means the number of SID items between <|sid_begin|> and <|sid_end|>."
        )
    )
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name. Default: {DEFAULT_MODEL}.")
    parser.add_argument(
        "--input-len",
        type=int,
        required=True,
        help="Number of SID items in each unique request.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        required=True,
        help="Beam width written into use_beam_search/n fields.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5,
        help="max_tokens written into every request. Default: 5.",
    )
    parser.add_argument(
        "--num-unique-requests",
        type=int,
        required=True,
        help="Number of unique requests to generate.",
    )
    parser.add_argument(
        "--repeat-first-n",
        type=int,
        default=0,
        help="Repeat the first N unique requests.",
    )
    parser.add_argument(
        "--repeat-times",
        type=int,
        default=0,
        help="How many extra times to append those repeated requests.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Optional pretty-print indent for --preview output only.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the first generated request to stdout after writing the file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    requests = generate_requests(
        model=args.model,
        input_len=args.input_len,
        beam_width=args.beam_width,
        max_tokens=args.max_tokens,
        num_unique_requests=args.num_unique_requests,
        repeat_first_n=args.repeat_first_n,
        repeat_times=args.repeat_times,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for request in requests:
            file.write(json.dumps(request, ensure_ascii=False) + "\n")

    LOGGER.info(
        "Wrote %s requests to %s (unique=%s, repeated=%s, beam_width=%s, input_len=%s)",
        len(requests),
        output_path,
        args.num_unique_requests,
        args.repeat_first_n * args.repeat_times,
        args.beam_width,
        args.input_len,
    )

    if args.preview and requests:
        print(json.dumps(requests[0], ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
