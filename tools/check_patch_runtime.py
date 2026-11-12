# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Exercise real startup installation and export its process-local observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("runtime", "engine_core"), default="runtime")
    args = parser.parse_args()
    from vllm_gr.patch_report import get_runtime_patch_report
    from vllm_gr.plugin import initialize_runtime

    try:
        if args.phase == "runtime":
            initialize_runtime()
            initialize_runtime()
        else:
            from vllm_gr._env_check import (
                check_environment,
                configure_model_runner_default,
            )
            from vllm_gr.patch import initialize_engine_core_beam_patches

            configure_model_runner_default()
            check_environment()
            initialize_engine_core_beam_patches()
            initialize_engine_core_beam_patches()
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(get_runtime_patch_report(), indent=2) + "\n")


if __name__ == "__main__":
    main()
