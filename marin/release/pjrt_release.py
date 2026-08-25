#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check that the XLA base and the jax revision agree, before the build runs.

This fork carries an XLA change on the upstream commit that a published jax revision pins. Only the
GPU PJRT plugin contains the change, so the build makes one wheel. Its stock siblings come from the
jax nightly index at the same generation.
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODULE_BAZEL_URL = "https://raw.githubusercontent.com/jax-ml/jax/{commit}/MODULE.bazel"
XLA_PREFIX = 'strip_prefix = "xla-'
JAX_NIGHTLY_VERSION = re.compile(r"^\d+\.\d+\.\d+\.dev(?P<date>\d{8})$")
SIBLING_TAG = "cp312-cp312-manylinux_2_27_aarch64"
PURE_TAG = "py3-none-manylinux_2_27_aarch64"


def note(message: str) -> None:
    """Progress goes to stderr. Stdout carries only the key=value pairs the workflow reads."""
    print(message, file=sys.stderr, flush=True)


def jax_build_date(jax_version: str) -> str:
    """Return the nightly's date, taken from its version rather than a separate field.

    jax stamps the wheel as ``<base>.dev<ML_WHEEL_BUILD_DATE>``. A separate field could only agree
    or disagree with the date already in ``jax_version``, and a disagreement makes a wheel that the
    stock plugin refuses.
    """
    match = JAX_NIGHTLY_VERSION.match(jax_version)
    if match is None:
        raise SystemExit(f"jax_version {jax_version!r} is not a nightly of the form X.Y.Z.devYYYYMMDD")
    return match.group("date")


def wheel_version(config: dict, fork_commit: str) -> str:
    """Return the version the build must make.

    ``jax-cuda13-plugin`` requires ``jax-cuda13-pjrt==<jax_version>``. PEP 440 ignores a local
    segment when the specifier has none. A local segment therefore keeps the wheel installable, and
    also records the XLA commit that the build used.
    """
    return f"{config['jax_version']}+marin.{fork_commit[:12]}"


def sibling_urls(config: dict) -> list[str]:
    """Return the stock wheels that share an ABI with the built wheel."""
    index, version = config["nightly_index"], config["jax_version"]
    return [
        f"{index}/jax/jax-{version}-py3-none-any.whl",
        f"{index}/jaxlib/jaxlib-{version}-{SIBLING_TAG}.whl",
        f"{index}/jax-cuda13-plugin/jax_cuda13_plugin-{version}-{SIBLING_TAG}.whl",
        f"{index}/jax-cuda13-pjrt/jax_cuda13_pjrt-{version}-{PURE_TAG}.whl",
    ]


def jax_pinned_xla(jax_commit: str) -> str:
    """Return the XLA commit that a jax revision pins, read from its MODULE.bazel."""
    with urllib.request.urlopen(MODULE_BAZEL_URL.format(commit=jax_commit), timeout=60) as response:
        text = response.read().decode()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(XLA_PREFIX):
            return stripped[len(XLA_PREFIX) :].rstrip('",')
    raise SystemExit(f"no XLA strip_prefix found in MODULE.bazel at {jax_commit}")


def check_base(config: dict, base: str) -> None:
    """Fail unless the base is the exact XLA commit that ``jax_commit`` pins.

    jax pins an exact XLA tarball by integrity hash, and builds its dependency graph from XLA module
    extensions. A base that no published jax generation pins can compile, and can then fail at run
    time. The check is symmetric. It fails if the overlay moved, and it fails if the jax revision
    moved.
    """
    pinned = jax_pinned_xla(config["jax_commit"])
    if pinned != base:
        raise SystemExit(
            f"upstream base {base} is not what jax {config['jax_commit'][:12]} pins ({pinned}).\n"
            "Rebase onto the pinned commit, or move jax_commit and jax_version together."
        )
    note(f"base {base[:12]} matches jax {config['jax_commit'][:12]}")


def check_siblings_published(config: dict) -> None:
    """Fail unless every stock sibling wheel is on the nightly index.

    A jax revision that was never cut as a nightly makes a wheel whose siblings nobody can install.
    Without this check that failure appears much later, in a training run.
    """
    missing = []
    for url in sibling_urls(config):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
                note(f"  {response.status}  {url.rsplit('/', 1)[1]}")
        except urllib.error.HTTPError as error:
            missing.append(f"  {error.code}  {url}")
        except (urllib.error.URLError, TimeoutError) as error:
            # Do not report a network fault as an absent wheel. They need different fixes.
            raise SystemExit(f"could not check {url}: {error}") from error
    if missing:
        raise SystemExit(
            f"jax_version {config['jax_version']} is not fully published on the nightly index:\n" + "\n".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fork-commit", required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if config.get("schema_version") != 1:
        raise SystemExit(f"{args.config}: unsupported schema_version {config.get('schema_version')!r}")

    build_date = jax_build_date(config["jax_version"])
    check_base(config, args.base)
    check_siblings_published(config)

    print(f"version={wheel_version(config, args.fork_commit)}")
    print(f"jax_build_date={build_date}")


if __name__ == "__main__":
    main()
