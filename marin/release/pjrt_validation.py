#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prove on real hardware that a built PJRT wheel's XLA change is live.

``validate`` runs inside an Iris job. It fetches the candidate wheel by URL, installs it over the
stock sibling generation, runs a multi-device ``ragged_all_to_all``, and confirms the device-kernel
path engaged. ``extract`` pulls the resulting record back out of the job log, on the runner.

The device-kernel check is the point. A wheel whose change is inert still passes a correctness test
and still benchmarks, as a null result, so without it "the change did not help" and "the change was
never present" look the same.

Two things make that check subtle. Getting either wrong gives a gate that is permanently red rather
than one that is wrong but green:

* The device kernel is opt-in behind
  ``--xla_gpu_experimental_ragged_all_to_all_use_device_kernel``. Without the flag the thunk takes
  the one-shot or NCCL path, and never reaches the change.
* The only runtime evidence is an ``XLA_VLOG_DEVICE(3)`` line on **stderr**
  (``"Device kernel: lsa_size=..."``), which needs ``TF_CPP_MAX_VLOG_LEVEL``. XLA does not write it
  to a dump directory.

The exercise therefore runs in a child process whose stderr this script captures, rather than
trying to redirect the C++ runtime's stderr from inside the process that imported jax.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

MARKER = "MARIN_XLA_PJRT_VALIDATION"
RESULT_MARKER = "MARIN_RAGGED_RESULT"
DEVICE_KERNEL_LOG_PATTERN = "Device kernel: lsa_size="
DEVICE_KERNEL_FLAG = "--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true"
SIBLING_TAG = "cp312-cp312-manylinux_2_27_aarch64"
PURE_TAG = "py3-none-manylinux_2_27_aarch64"

# Offsets follow lib/levanter/src/levanter/grug/_moe/ep_common.py::_shard_a2a_params in the Marin
# repository: sender-side output offsets, one entry per destination shard. ragged_all_to_all needs
# rank-1 offset and size arrays, and shard_map to bind the axis name.
EXERCISE = f'''
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, NamedSharding, PartitionSpec

AXIS = "ep"
ROWS_PER_PEER = 512
WIDTH = 6144

devices = jax.devices()
n = len(devices)
if n < 2:
    raise SystemExit(f"need at least 2 devices, found {{n}}")

mesh = jax.make_mesh((n,), (AXIS,), axis_types=(AxisType.Explicit,))
rows = ROWS_PER_PEER * n
counts = np.full((n, n), ROWS_PER_PEER, dtype=np.int32)
rng = np.random.default_rng(0)
operand = rng.standard_normal((n * rows, WIDTH), dtype=np.float32)

def exchange(local):
    shard_id = jax.lax.axis_index(AXIS)
    shard_counts = jnp.asarray(counts)
    row = shard_counts[shard_id]
    input_offsets = jnp.cumsum(jnp.concatenate((jnp.zeros(1, row.dtype), row[:-1])))
    sender_output_offsets = jnp.cumsum(shard_counts, axis=0, dtype=shard_counts.dtype) - shard_counts
    return jax.lax.ragged_all_to_all(
        local,
        jnp.zeros((rows, WIDTH), dtype=local.dtype),
        input_offsets.astype(jnp.int32),
        row.astype(jnp.int32),
        sender_output_offsets[shard_id].astype(jnp.int32),
        shard_counts[:, shard_id].astype(jnp.int32),
        axis_name=AXIS,
    )

sharded = jax.device_put(operand, NamedSharding(mesh, PartitionSpec(AXIS)))
with jax.set_mesh(mesh):
    out = jax.jit(jax.shard_map(exchange, out_specs=PartitionSpec(AXIS)))(sharded)
    jax.block_until_ready(out)

# Shard s sends its block r to shard r, which receives it at slot s.
blocks = operand.reshape(n, n, ROWS_PER_PEER, WIDTH)
expected = blocks.transpose(1, 0, 2, 3).reshape(n * rows, WIDTH)
print("{RESULT_MARKER} " + json.dumps({{
    "correct": bool(np.array_equal(np.asarray(out), expected)),
    "device_count": int(jax.device_count()),
    "compute_capability": ".".join(str(p) for p in devices[0].compute_capability),
}}, sort_keys=True))
'''


def fetch_wheel(url: str, sha256: str, destination: Path) -> Path:
    """Download the candidate wheel and check it against the digest the build recorded.

    The wheel arrives by URL rather than in the Iris bundle. Iris lists bundle files with
    ``git ls-files --cached --others --exclude-standard`` under a 25 MB cap, which a >100 MB wheel
    cannot fit.
    """
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / url.rsplit("/", 1)[1]
    urllib.request.urlretrieve(url, path)  # noqa: S310 - a release asset URL from our own config
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise SystemExit(f"downloaded wheel digest {digest} does not match the candidate's {sha256}")
    return path


def install_runtime(config: dict, wheel: Path, python: str) -> None:
    """Install the stock sibling generation, then overlay the candidate wheel.

    The siblings install with full dependency resolution. Installing them ``--no-deps`` leaves the
    venv's baseline ``nvidia-*`` runtime in place, and with those the LSA path never engages:
    ``LsaSize()`` comes back empty and the thunk falls back with INVALID_ARGUMENT.
    """
    index, version = config["nightly_index"], config["jax_version"]
    siblings = [
        f"jax @ {index}/jax/jax-{version}-py3-none-any.whl",
        f"jaxlib @ {index}/jaxlib/jaxlib-{version}-{SIBLING_TAG}.whl",
        f"jax-cuda13-plugin[with-cuda] @ {index}/jax-cuda13-plugin/jax_cuda13_plugin-{version}-{SIBLING_TAG}.whl",
        f"jax-cuda13-pjrt @ {index}/jax-cuda13-pjrt/jax_cuda13_pjrt-{version}-{PURE_TAG}.whl",
    ]
    subprocess.run(["uv", "pip", "install", "--python", python, "--reinstall", *siblings], check=True)
    subprocess.run(["uv", "pip", "install", "--python", python, "--no-deps", "--reinstall", str(wheel)], check=True)


def command_validate(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text())
    wheel = fetch_wheel(args.wheel_url, args.expect_sha256, Path(args.download_dir))
    python = sys.executable
    install_runtime(config, wheel, python)

    versions = json.loads(
        subprocess.run(
            [
                python,
                "-c",
                "import json;from importlib.metadata import version;"
                "print(json.dumps({p: version(p) for p in "
                "('jax','jaxlib','jax-cuda13-plugin','jax-cuda13-pjrt','nvidia-nccl-cu13')}))",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    if versions["jax-cuda13-pjrt"] != args.expect_version:
        raise SystemExit(f"installed {versions['jax-cuda13-pjrt']}, candidate is {args.expect_version}")

    environment = dict(
        os.environ,
        XLA_FLAGS=" ".join(filter(None, (os.environ.get("XLA_FLAGS", ""), DEVICE_KERNEL_FLAG))),
        TF_CPP_MAX_VLOG_LEVEL="3",
    )
    completed = subprocess.run([python, "-c", EXERCISE], env=environment, capture_output=True, text=True, check=False)
    combined = completed.stdout + completed.stderr
    sys.stderr.write(combined)
    if completed.returncode != 0:
        raise SystemExit(f"ragged all-to-all exercise failed with status {completed.returncode}")

    record = None
    for line in completed.stdout.splitlines():
        index = line.find(RESULT_MARKER)
        if index >= 0:
            record = json.loads(line[index + len(RESULT_MARKER) :])
    if record is None:
        raise SystemExit("exercise produced no result record")

    engaged = DEVICE_KERNEL_LOG_PATTERN in combined
    result = {
        "architecture": args.architecture,
        "hardware": args.hardware,
        "wheel_sha256": args.expect_sha256,
        "installed_version": versions["jax-cuda13-pjrt"],
        "jax_version": versions["jax"],
        "nccl_version": versions["nvidia-nccl-cu13"],
        "compute_capability": record["compute_capability"],
        "device_count": record["device_count"],
        "ragged_all_to_all_ok": record["correct"],
        "device_kernel_engaged": engaged,
        "status": "passed" if (record["correct"] and engaged) else "failed",
    }
    print(f"{MARKER} {json.dumps(result, sort_keys=True)}", flush=True)
    if result["status"] != "passed":
        raise SystemExit("validation failed: " + ("device kernel did not engage" if record["correct"] else "incorrect"))


def command_extract(args: argparse.Namespace) -> None:
    record = None
    for line in args.log.read_text(errors="replace").splitlines():
        index = line.find(MARKER)
        if index >= 0:
            record = json.loads(line[index + len(MARKER) :])
    if record is None:
        raise SystemExit("no validation record found in the log")
    if record.get("status") != "passed":
        raise SystemExit(f"validation record reports status {record.get('status')!r}")
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="run the hardware gate, inside the Iris job")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--architecture", required=True)
    validate.add_argument("--hardware", required=True)
    validate.add_argument("--wheel-url", required=True)
    validate.add_argument("--expect-sha256", required=True)
    validate.add_argument("--expect-version", required=True)
    validate.add_argument("--download-dir", default="/tmp/marin-pjrt")
    validate.set_defaults(func=command_validate)

    extract = sub.add_parser("extract", help="pull the validation record out of a job log")
    extract.add_argument("--log", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.set_defaults(func=command_extract)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
