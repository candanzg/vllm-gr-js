# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compare an sdist-built wheel with a wheel built from an export of Git HEAD."""

import os
import subprocess
import sys
import tarfile
import tempfile
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from packaging.utils import parse_sdist_filename, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[2]
PROTO_DIR = Path("vllm_gr/entrypoints/grpc/proto")


def package_files(wheel: Path) -> set[str]:
    with ZipFile(wheel) as archive:
        return {
            name
            for name in archive.namelist()
            if name.startswith("vllm_gr/") and not name.endswith("/")
        }


def base_requires_dist(wheel: Path) -> set[str]:
    """Normalized base (non-extra) Requires-Dist entries of a wheel."""
    from packaging.requirements import Requirement

    with ZipFile(wheel) as archive:
        (metadata,) = (n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        text = archive.read(metadata).decode()
    requires = set()
    for line in text.splitlines():
        if not line.startswith("Requires-Dist: "):
            continue
        requirement = Requirement(line[len("Requires-Dist: ") :])
        if requirement.marker is None or not str(requirement.marker).count("extra"):
            requires.add(str(requirement))
    return requires


def expected_requirements(root: Path, device: str) -> set[str]:
    """Normalized requirements declared by requirements/<device>.txt (with -r)."""
    from packaging.requirements import Requirement

    def read(path: Path) -> list[str]:
        items: list[str] = []
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                items.extend(read(path.parent / line[3:].strip()))
            else:
                items.append(str(Requirement(line)))
        return items

    return set(read(root / "requirements" / f"{device}.txt"))


def main() -> None:
    if os.environ.get("VLLM_GR_TARGET_DEVICE") not in {"cpu", "cuda", "npu"}:
        raise SystemExit("Set VLLM_GR_TARGET_DEVICE to cpu, cuda or npu for both builds.")
    (wheel,) = (ROOT / "dist").glob("*.whl")
    (sdist,) = (ROOT / "dist").glob("*.tar.gz")
    name, version, _, _ = parse_wheel_filename(wheel.name)
    if parse_sdist_filename(sdist.name) != (name, version):
        raise SystemExit(f"Distribution names or versions differ: {sdist.name}, {wheel.name}")
    with tarfile.open(sdist) as source:
        member = source.extractfile(f"{sdist.name.removesuffix('.tar.gz')}/PKG-INFO")
        if member is None or BytesParser().parsebytes(member.read())["Version"] != str(version):
            raise SystemExit(f"Unexpected PKG-INFO version in {sdist.name}")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        export = work / "source"
        export.mkdir()
        tar = work / "source.tar"
        subprocess.run(["git", "archive", "--output", str(tar), "HEAD"], cwd=ROOT, check=True)
        with tarfile.open(tar) as archive:
            archive.extractall(export, filter="data")
        # The export has no Git metadata. Compare the same version and device.
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel"],
            cwd=export,
            check=True,
            env={**os.environ, "VLLM_GR_VERSION_OVERRIDE": str(version)},
        )
        archive_wheel = export / "dist" / wheel.name
        from_sdist = package_files(wheel)
        from_git = package_files(archive_wheel)
        if from_sdist != from_git:
            raise SystemExit(
                f"Only in sdist-built wheel: {sorted(from_sdist - from_git)}\n"
                f"Only in archive-built wheel: {sorted(from_git - from_sdist)}"
            )
        installed = work / "installed"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(installed),
                str(archive_wheel),
            ],
            cwd=work,
            check=True,
        )
        for proto in ("common.proto", "vllm_engine.proto"):
            relative = PROTO_DIR / proto
            if (installed / relative).read_bytes() != (export / relative).read_bytes():
                raise SystemExit(f"Installed protocol file differs from the source: {relative}")

    device = os.environ["VLLM_GR_TARGET_DEVICE"]
    actual = base_requires_dist(wheel)
    expected = expected_requirements(ROOT, device)
    if not actual:
        raise SystemExit(
            f"Wheel {wheel.name} declares no base dependencies; "
            f"requirements/{device}.txt was not applied."
        )
    if actual != expected:
        raise SystemExit(
            "Wheel base Requires-Dist does not match "
            f"requirements/{device}.txt\n"
            f"Only in wheel: {sorted(actual - expected)}\n"
            f"Only in requirements: {sorted(expected - actual)}"
        )
    print(
        f"Verified {len(from_sdist)} package files, {len(actual)} base dependencies "
        f"and installed gRPC protocols in {wheel.name}"
    )


if __name__ == "__main__":
    main()
