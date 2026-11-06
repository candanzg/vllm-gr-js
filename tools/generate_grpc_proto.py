# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate the extended SMG bindings with package-relative imports/private pools.

Usage:
    python tools/generate_grpc_proto.py           # rewrite generated/ in place
    python tools/generate_grpc_proto.py --check   # fail if generated/ is stale

Requires ``grpcio-tools==1.84.0`` (declared in the ``dev`` extra). The bundled
protoc version is part of the output, so a different grpcio-tools release can
produce a spurious diff.
"""

import argparse
import difflib
import subprocess
import sys
import tempfile
from pathlib import Path

GRPC_ROOT = Path(__file__).resolve().parents[1] / "vllm_gr/entrypoints/grpc"
GENERATED_FILES = ("common_pb2.py", "vllm_engine_pb2.py", "vllm_engine_pb2_grpc.py")


def generate(output: Path) -> None:
    proto_dir = GRPC_ROOT / "proto"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={output}",
            f"--grpc_python_out={output}",
            str(proto_dir / "common.proto"),
            str(proto_dir / "vllm_engine.proto"),
        ],
        check=True,
    )
    # common.proto defines messages only; protoc emits no service bindings here.
    (output / "common_pb2_grpc.py").unlink(missing_ok=True)
    for path in output.glob("*_pb2*.py"):
        text = path.read_text()
        for module in ("common_pb2", "vllm_engine_pb2"):
            text = text.replace(f"import {module} as ", f"from . import {module} as ")
        if path.name.endswith("_pb2.py"):
            # Preserve the original package/service names without colliding with
            # the unmodified SMG descriptor in applications importing both APIs.
            pool = "_descriptor_pool.DescriptorPool()"
            if path.name == "vllm_engine_pb2.py":
                text = text.replace(
                    "DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(",
                    "_pool = " + pool + "\n"
                    "_pool.AddSerializedFile(common__pb2.DESCRIPTOR.serialized_pb)\n"
                    "DESCRIPTOR = _pool.AddSerializedFile(",
                )
            else:
                text = text.replace("_descriptor_pool.Default()", pool)
            module = "vllm_gr.entrypoints.grpc.generated." + path.stem
            text = text.replace(f"'{path.stem}'", repr(module))
        path.write_text(text)


def check() -> int:
    """Regenerate into a scratch directory and diff against the committed copy."""
    committed = GRPC_ROOT / "generated"
    with tempfile.TemporaryDirectory() as scratch:
        fresh = Path(scratch)
        generate(fresh)
        produced = sorted(path.name for path in fresh.glob("*.py"))
        if produced != sorted(GENERATED_FILES):
            print(f"unexpected generator output: {produced}", file=sys.stderr)
            return 1
        stale = []
        for name in GENERATED_FILES:
            expected = (fresh / name).read_text().splitlines(keepends=True)
            current_path = committed / name
            current = (
                current_path.read_text().splitlines(keepends=True) if current_path.exists() else []
            )
            if expected != current:
                stale.append(name)
                sys.stderr.writelines(
                    difflib.unified_diff(
                        current, expected, f"committed/{name}", f"regenerated/{name}", n=1
                    )
                )
    if stale:
        print(
            f"\nStale gRPC bindings: {', '.join(stale)}.\n"
            "Run `python tools/generate_grpc_proto.py` and commit the result.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated/ matches the .proto sources without modifying it",
    )
    args = parser.parse_args()
    if args.check:
        return check()
    generate(GRPC_ROOT / "generated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
