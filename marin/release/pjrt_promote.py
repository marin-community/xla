#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the release manifest that Marin promotes a pin from.

The shape here is a contract with ``render_pjrt_release_toml`` in Marin's
``config/update-external.py``. That function reads ``release.status``, ``release.repository``,
``validation.status``, ``validation.device_kernel_engaged``, ``distribution.version``, ``jax``, and
``platforms``, and refuses a manifest that does not clear every gate. It matches the shape
``marin-vllm-gpu-manifest.json`` already uses, so both forks promote the same way.
"""

import argparse
import json
from pathlib import Path

FORK_REPOSITORY = "marin-community/xla"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True, help="the candidate JSON from the prerelease")
    parser.add_argument("--validation", type=Path, required=True, help="the record from the hardware gate")
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--published-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    candidate = json.loads(args.candidate.read_text())
    validation = json.loads(args.validation.read_text())

    wheel = candidate["wheel"]
    if validation["wheel_sha256"] != wheel["sha256"]:
        raise SystemExit("the validation record does not describe the candidate's wheel")
    if validation["status"] != "passed":
        raise SystemExit(f"refusing to promote a candidate whose validation is {validation['status']!r}")
    if not validation["device_kernel_engaged"]:
        raise SystemExit("refusing to promote a wheel whose device kernel did not engage")

    manifest = {
        "release": {
            "status": "released",
            "tag": args.release_tag,
            "repository": FORK_REPOSITORY,
            "published_at": args.published_at,
            "promoted_from": candidate["candidate_tag"],
        },
        "validation": validation,
        "source": candidate["source"],
        "jax": {
            "commit": config["jax_commit"],
            "version": config["jax_version"],
        },
        "distribution": {"name": "jax-cuda13-pjrt", "version": wheel["version"]},
        "platforms": [
            {
                "architecture": wheel["architecture"],
                "sm_targets": wheel["sm_targets"],
                "wheel": {"filename": wheel["filename"], "sha256": wheel["sha256"]},
            }
        ],
    }
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
