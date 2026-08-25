#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the Marin PJRT lane: preflight checks and release rendering.

The fork carries a small XLA delta on the upstream commit a published jax revision pins. Only the
GPU PJRT plugin embeds that delta, so one wheel is built and its stock siblings come from the jax
nightly index at the same generation.

This is deliberately thin. The lane is a single linear workflow -- build, validate, publish -- so
there is no cross-workflow handoff to re-verify and no multi-architecture manifest to merge.
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

FORK_REPOSITORY = "https://github.com/marin-community/xla"
MODULE_BAZEL_URL = "https://raw.githubusercontent.com/jax-ml/jax/{commit}/MODULE.bazel"
XLA_PREFIX = 'strip_prefix = "xla-'
JAX_NIGHTLY_VERSION = re.compile(r"^\d+\.\d+\.\d+\.dev(?P<date>\d{8})$")
SIBLING_TAG = "cp312-cp312-manylinux_2_27_aarch64"
PURE_TAG = "py3-none-manylinux_2_27_aarch64"


def note(message: str) -> None:
    """Progress goes to stderr; stdout carries only the key=value pairs the workflow reads."""
    print(message, file=sys.stderr, flush=True)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise SystemExit(f"{path}: unsupported schema_version {config.get('schema_version')!r}")
    return config


def jax_build_date(jax_version: str) -> str:
    """The nightly's date, derived from its version rather than configured separately.

    jax's build stamps the wheel as ``<base>.dev<ML_WHEEL_BUILD_DATE>``, so this value must equal
    the date already embedded in ``jax_version``; a second configured field could only disagree
    with it, and a disagreement yields a wheel the stock plugin refuses.
    """
    match = JAX_NIGHTLY_VERSION.match(jax_version)
    if match is None:
        raise SystemExit(f"jax_version {jax_version!r} is not a nightly of the form X.Y.Z.devYYYYMMDD")
    return match.group("date")


def wheel_version(config: dict, fork_commit: str) -> str:
    """The version the build must produce for the stock plugin's exact pin to resolve.

    ``jax-cuda13-plugin`` requires ``jax-cuda13-pjrt==<jax_version>``. PEP 440 ignores a local
    segment when the specifier carries none, so appending one keeps the wheel installable while
    naming the exact XLA tip it was built from.
    """
    return f"{config['jax_version']}+marin.{fork_commit[:12]}"


def sibling_urls(config: dict) -> list[str]:
    """The stock wheels the built PJRT wheel shares an ABI with."""
    index, version = config["nightly_index"], config["jax_version"]
    return [
        f"{index}/jax/jax-{version}-py3-none-any.whl",
        f"{index}/jaxlib/jaxlib-{version}-{SIBLING_TAG}.whl",
        f"{index}/jax-cuda13-plugin/jax_cuda13_plugin-{version}-{SIBLING_TAG}.whl",
        f"{index}/jax-cuda13-pjrt/jax_cuda13_pjrt-{version}-{PURE_TAG}.whl",
    ]


def jax_pinned_xla(jax_commit: str) -> str:
    """Return the XLA commit a jax revision pins, read from its MODULE.bazel."""
    with urllib.request.urlopen(MODULE_BAZEL_URL.format(commit=jax_commit), timeout=60) as response:
        text = response.read().decode()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(XLA_PREFIX):
            return stripped[len(XLA_PREFIX) :].rstrip('",')
    raise SystemExit(f"no XLA strip_prefix found in MODULE.bazel at {jax_commit}")


def command_preflight(args: argparse.Namespace) -> None:
    """Check the pairing is coherent, then emit the values the build needs.

    Runs before the expensive compile so an incoherent pairing costs seconds rather than hours.
    """
    config = load_config(args.config)
    build_date = jax_build_date(config["jax_version"])

    # 1. The base must be exactly the XLA commit this jax revision pins. This is the invariant
    #    that makes the pairing safe and nothing else enforces it: jax archive_overrides an exact
    #    XLA tarball and draws its build graph from XLA-side module extensions, so a base no jax
    #    generation pins compiles cleanly and then mismatches at runtime. The check is symmetric --
    #    it fails whether the overlay was rebased without moving jax_commit, or jax_commit was
    #    moved without rebasing.
    pinned = jax_pinned_xla(config["jax_commit"])
    if pinned != args.base:
        raise SystemExit(
            f"upstream base {args.base} is not what jax {config['jax_commit'][:12]} pins ({pinned}).\n"
            "Rebase onto the pinned commit, or move jax_commit and jax_version together."
        )
    note(f"base {args.base[:12]} matches jax {config['jax_commit'][:12]}")

    # 2. The generation must actually have been published. A jax revision that was never cut as a
    #    nightly builds a wheel whose siblings cannot be installed, which would otherwise surface
    #    hours later in validation or, worse, in a training run.
    missing = []
    for url in sibling_urls(config):
        request = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                note(f"  {response.status}  {url.rsplit('/', 1)[1]}")
        except urllib.error.HTTPError as error:
            missing.append(f"  {error.code}  {url}")
    if missing:
        raise SystemExit(
            f"jax_version {config['jax_version']} is not fully published on the nightly index:\n"
            + "\n".join(missing)
        )

    print(f"version={wheel_version(config, args.fork_commit)}")
    print(f"jax_build_date={build_date}")


def command_render(args: argparse.Namespace) -> None:
    """Write the release manifest, the descriptor Marin pins, and the release notes."""
    config = load_config(args.config)
    validation = json.loads(args.validation.read_text())
    version = wheel_version(config, args.fork_commit)
    sm_targets = config["platforms"][validation["architecture"]]["sm_targets"]
    url = f"{FORK_REPOSITORY}/releases/download/{args.release_tag}/{quote(args.wheel_name, safe='')}"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "release_tag": args.release_tag,
        "published_at": args.published_at,
        "source": {
            "fork_repository": FORK_REPOSITORY,
            "fork_commit": args.fork_commit,
            "upstream_base": args.upstream_base,
        },
        "jax": {
            "commit": config["jax_commit"],
            "version": config["jax_version"],
            "nightly_index": config["nightly_index"],
        },
        "wheel": {
            "architecture": validation["architecture"],
            "filename": args.wheel_name,
            "sha256": args.wheel_sha256,
            "sm_targets": sm_targets,
            "version": version,
            "url": url,
        },
        "validation": validation,
    }
    (args.out_dir / "marin-xla-pjrt-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    descriptor = "\n".join(
        [
            "# Promoted Marin XLA PJRT release. Generated by `marin/release/pjrt_release.py render`",
            f"# in {FORK_REPOSITORY}; do not hand-edit. Only jax-cuda13-pjrt carries Marin's XLA delta;",
            "# its siblings install stock from the nightly index at `jax_version`.",
            "",
            f'release_tag = "{args.release_tag}"',
            f'source_commit = "{args.fork_commit}"',
            f'upstream_base = "{args.upstream_base}"',
            f'jax_commit = "{config["jax_commit"]}"',
            f'jax_version = "{config["jax_version"]}"',
            f'nightly_index = "{config["nightly_index"]}"',
            f'version = "{version}"',
            "",
            "[[wheels]]",
            f'architecture = "{validation["architecture"]}"',
            "sm_targets = [" + ", ".join(f'"{t}"' for t in sm_targets) + "]",
            f'url = "{url}"',
            f'sha256 = "{args.wheel_sha256}"',
            "",
        ]
    )
    (args.out_dir / "pjrt-release.toml").write_text(descriptor)

    notes = f"""Patched `jax-cuda13-pjrt` built from [`{args.fork_commit[:12]}`]({FORK_REPOSITORY}/commit/{args.fork_commit}) on upstream base [`{args.upstream_base[:12]}`](https://github.com/openxla/xla/commit/{args.upstream_base}), paired with jax `{config["jax_version"]}`.

Only the GPU PJRT plugin carries Marin's XLA delta. Install `jax`, `jaxlib`, and `jax-cuda13-plugin` stock at the same generation from the nightly index.

| Field | Value |
| --- | --- |
| Version | `{version}` |
| Architecture | `{validation["architecture"]}` ({", ".join(sm_targets)}) |
| SHA-256 | `{args.wheel_sha256}` |
| Validated on | {validation["hardware"]}, {validation["device_count"]} devices |
| Device kernel engaged | {validation["device_kernel_engaged"]} |

## Marin pin

Copy into `config/external/xla/pjrt-release.toml`, then run `uv run config/update-external.py xla`.

```toml
{descriptor}```
"""
    (args.out_dir / "release-notes.md").write_text(notes)
    print(f"wrote manifest, descriptor and notes to {args.out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="check the pairing and emit the build's values")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--fork-commit", required=True)
    preflight.add_argument("--base", required=True)
    preflight.set_defaults(func=command_preflight)

    render = sub.add_parser("render", help="write the release manifest, Marin descriptor and notes")
    render.add_argument("--config", type=Path, required=True)
    render.add_argument("--fork-commit", required=True)
    render.add_argument("--upstream-base", required=True)
    render.add_argument("--release-tag", required=True)
    render.add_argument("--wheel-name", required=True)
    render.add_argument("--wheel-sha256", required=True)
    render.add_argument("--validation", type=Path, required=True)
    render.add_argument("--published-at", required=True)
    render.add_argument("--out-dir", type=Path, required=True)
    render.set_defaults(func=command_render)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
