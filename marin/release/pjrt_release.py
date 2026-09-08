#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check that the XLA base and the jax revision agree, before the build runs.

This fork carries an XLA change on the upstream commit that a published jax revision pins. Only the
GPU PJRT plugin contains the change, so the build makes one wheel. Its stock siblings come from
PyPI at the same release.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODULE_BAZEL_URL = "https://raw.githubusercontent.com/jax-ml/jax/{commit}/MODULE.bazel"
XLA_PREFIX = 'strip_prefix = "xla-'
PYPI_RELEASE_URL = "https://pypi.org/pypi/{distribution}/{version}/json"
SIBLINGS = ("jax", "jaxlib", "jax-cuda13-plugin", "jax-cuda13-pjrt")


def note(message: str) -> None:
    """Progress goes to stderr. Stdout carries only the key=value pairs the workflow reads."""
    print(message, file=sys.stderr, flush=True)


def wheel_version(config: dict, fork_commit: str) -> str:
    """Return the version the build must make.

    ``jax-cuda13-plugin`` requires ``jax-cuda13-pjrt==<jax_version>``. PEP 440 ignores a local
    segment when the specifier has none. A local segment therefore keeps the wheel installable, and
    also records the XLA commit that the build used.
    """
    return f"{config['jax_version']}+marin.{fork_commit[:12]}"


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
    """Fail unless every stock sibling is released on PyPI at ``jax_version``.

    A jax revision that was never cut as a release makes a wheel whose siblings nobody can install.
    Without this check that failure appears much later, in a training run.
    """
    version = config["jax_version"]
    missing = []
    for distribution in SIBLINGS:
        url = PYPI_RELEASE_URL.format(distribution=distribution, version=version)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                note(f"  {response.status}  {distribution} {version}")
        except urllib.error.HTTPError as error:
            missing.append(f"  {error.code}  {distribution} {version}")
        except (urllib.error.URLError, TimeoutError) as error:
            # Do not report a network fault as an absent release. They need different fixes.
            raise SystemExit(f"could not check {url}: {error}") from error
    if missing:
        raise SystemExit(f"jax_version {version} is not fully released on PyPI:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fork-commit", required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if config.get("schema_version") != 1:
        raise SystemExit(f"{args.config}: unsupported schema_version {config.get('schema_version')!r}")

    check_base(config, args.base)
    check_siblings_published(config)

    print(f"version={wheel_version(config, args.fork_commit)}")


if __name__ == "__main__":
    main()
