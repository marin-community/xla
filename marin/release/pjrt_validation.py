#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prove on real hardware that a built PJRT wheel's XLA delta is actually live.

``validate`` runs inside an Iris job: it installs the wheel over the stock sibling generation, runs
a multi-device ``ragged_all_to_all``, and confirms the device-kernel path engaged. ``extract`` pulls
the resulting record back out of the job log on the runner.

The device-kernel check is the point. A wheel whose delta is inert still passes a correctness test
and still benchmarks -- as a null result -- so without it "the change did not help" and "the change
was never present" are indistinguishable.

Two things make that check subtle, and getting either wrong yields a gate that is permanently red
rather than one that is wrong-but-green:

* The device kernel is opt-in behind
  ``--xla_gpu_experimental_ragged_all_to_all_use_device_kernel``. Without the flag the thunk takes
  the one-shot or NCCL path and the delta is never reached.
* The only runtime evidence is an ``XLA_VLOG_DEVICE(3)`` line on **stderr**
  (``"Device kernel: lsa_size=..."``), which needs ``TF_CPP_MAX_VLOG_LEVEL``. It is not written to
  an XLA dump directory.

The exercise therefore runs in a child process whose stderr this script captures, rather than
trying to redirect the C++ runtime's stderr from inside the process that imported jax.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

MARKER = "MARIN_XLA_PJRT_VALIDATION"
RESULT_MARKER = "MARIN_RAGGED_RESULT"
DEVICE_KERNEL_LOG_PATTERN = "Device kernel: lsa_size="
DEVICE_KERNEL_FLAG = "--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true"
SIBLING_TAG = "cp312-cp312-manylinux_2_27_aarch64"
PURE_TAG = "py3-none-manylinux_2_27_aarch64"

EXERCISE = f'''
import json
import numpy as np
import jax
from jax.sharding import NamedSharding, PartitionSpec

devices = jax.devices()
if len(devices) < 2:
    raise SystemExit(f"need at least 2 visible devices, found {{len(devices)}}")

mesh = jax.make_mesh((len(devices),), ("ep",))
rows, width = 512, 6144
per_peer = rows // len(devices)
sizes = np.full(len(devices), per_peer, dtype=np.int32)
offsets = np.arange(len(devices), dtype=np.int32) * per_peer
rng = np.random.default_rng(0)
operand = rng.standard_normal((len(devices) * rows, width), dtype=np.float32)

def shard(values):
    return jax.device_put(values, NamedSharding(mesh, PartitionSpec("ep")))

@jax.jit
def exchange(x, input_offsets, send_sizes, output_offsets, recv_sizes):
    return jax.lax.ragged_all_to_all(
        x, jax.numpy.zeros_like(x),
        input_offsets, send_sizes, output_offsets, recv_sizes,
        axis_name="ep",
    )

tiled_offsets = np.tile(offsets, (len(devices), 1))
tiled_sizes = np.tile(sizes, (len(devices), 1))
with mesh:
    out = exchange(shard(operand), shard(tiled_offsets), shard(tiled_sizes),
                   shard(tiled_offsets), shard(tiled_sizes))
    jax.block_until_ready(out)

out = np.asarray(out)
print("{RESULT_MARKER} " + json.dumps({{
    "correct": bool(np.all(np.isfinite(out)) and np.any(out != 0.0)),
    "device_count": int(jax.device_count()),
    "compute_capability": ".".join(str(p) for p in devices[0].compute_capability),
}}, sort_keys=True))
'''


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
    digest = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
    if digest != args.expect_sha256:
        raise SystemExit(f"wheel bytes {digest} do not match the built wheel {args.expect_sha256}")

    python = sys.executable
    install_runtime(config, args.wheel, python)

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

    environment = dict(
        os.environ,
        XLA_FLAGS=" ".join(filter(None, (os.environ.get("XLA_FLAGS", ""), DEVICE_KERNEL_FLAG))),
        TF_CPP_MAX_VLOG_LEVEL="3",
    )
    completed = subprocess.run([python, "-c", EXERCISE], env=environment, capture_output=True, text=True, check=False)
    combined = completed.stdout + completed.stderr
    if completed.returncode != 0:
        sys.stderr.write(combined)
        raise SystemExit(f"ragged all-to-all exercise failed with status {completed.returncode}")

    record = None
    for line in completed.stdout.splitlines():
        index = line.find(RESULT_MARKER)
        if index >= 0:
            record = json.loads(line[index + len(RESULT_MARKER) :])
    if record is None:
        sys.stderr.write(combined)
        raise SystemExit("exercise produced no result record")

    engaged = DEVICE_KERNEL_LOG_PATTERN in combined
    result = {
        "architecture": args.architecture,
        "hardware": args.hardware,
        "wheel_sha256": digest,
        "installed_version": versions["jax-cuda13-pjrt"],
        "jax_version": versions["jax"],
        "nccl_version": versions["nvidia-nccl-cu13"],
        "compute_capability": record["compute_capability"],
        "device_count": record["device_count"],
        "ragged_all_to_all_ok": record["correct"],
        "device_kernel_engaged": engaged,
    }
    print(f"{MARKER} {json.dumps(result, sort_keys=True)}", flush=True)
    if not (record["correct"] and engaged):
        raise SystemExit("validation failed: " + ("device kernel did not engage" if record["correct"] else "incorrect"))


def command_extract(args: argparse.Namespace) -> None:
    record = None
    for line in args.log.read_text(errors="replace").splitlines():
        index = line.find(MARKER)
        if index >= 0:
            record = json.loads(line[index + len(MARKER) :])
    if record is None:
        raise SystemExit("no validation record found in the log")
    if not record["device_kernel_engaged"]:
        raise SystemExit("validation record reports the device kernel did not engage")
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="run the hardware gate (inside the Iris job)")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--architecture", required=True)
    validate.add_argument("--hardware", required=True)
    validate.add_argument("--wheel", type=Path, required=True)
    validate.add_argument("--expect-sha256", required=True)
    validate.set_defaults(func=command_validate)

    extract = sub.add_parser("extract", help="pull the validation record out of a job log")
    extract.add_argument("--log", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.set_defaults(func=command_extract)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
