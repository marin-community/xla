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

The exercise therefore runs in child processes whose stderr this script captures, rather than
trying to redirect the C++ runtime's stderr from inside the process that imported jax.

It runs one child per GPU. Symmetric memory is registered per process, so a single process holding
several devices never resolves the buffers and the device kernel silently declines. The children
form a JAX distributed world through a coordinator on loopback, which is enough because the whole
job is one Iris task on one node.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

MARKER = "MARIN_XLA_PJRT_VALIDATION"
RESULT_MARKER = "MARIN_RAGGED_RESULT"
DEVICE_KERNEL_LOG_PATTERN = "Device kernel: lsa_size="
# The device kernel is gated on more than its own flag. ragged_all_to_all_thunk.cc engages it only
# when collective memory is present and both the source and destination buffers resolve through
# FindSymmetricMemory. Ragged all-to-all defaults to COLLECTIVES_PRIVATE_MEMORY, where they never
# do, so the mode has to be switched as well: symmetric memory is what the thunk's comment calls
# "device-initiated collectives that run as device kernels".
DEVICE_KERNEL_FLAGS = (
    "--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true",
    "--xla_gpu_ragged_all_to_all_mode=symmetric",
)
SIBLING_TAG = "cp312-cp312-manylinux_2_27_aarch64"
PURE_TAG = "py3-none-manylinux_2_27_aarch64"

# Offsets follow lib/levanter/src/levanter/grug/_moe/ep_common.py::_shard_a2a_params in the Marin
# repository: sender-side output offsets, one entry per destination shard. ragged_all_to_all needs
# rank-1 offset and size arrays, and shard_map to bind the axis name.
EXERCISE = f'''
import json
import os
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, NamedSharding, PartitionSpec

process_id = int(os.environ["MARIN_PROCESS_ID"])
num_processes = int(os.environ["MARIN_NUM_PROCESSES"])

# One device per process. Symmetric memory is registered per process, so a process holding both
# devices leaves FindSymmetricMemory empty and the device kernel declines without saying so.
jax.distributed.initialize(
    coordinator_address=os.environ["MARIN_COORDINATOR"],
    num_processes=num_processes,
    process_id=process_id,
    local_device_ids=[process_id],
)

AXIS = "ep"
ROWS_PER_PEER = 512
WIDTH = 6144

devices = jax.devices()
n = len(devices)
if n != num_processes:
    raise SystemExit(f"expected {{num_processes}} global devices, found {{n}}")

mesh = jax.make_mesh((n,), (AXIS,), axis_types=(AxisType.Explicit,))
rows = ROWS_PER_PEER * n
counts = np.full((n, n), ROWS_PER_PEER, dtype=np.int32)
# Every process builds the same operand from the same seed, then contributes its own shard.
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

sharding = NamedSharding(mesh, PartitionSpec(AXIS))
local_rows = slice(process_id * rows, (process_id + 1) * rows)
sharded = jax.make_array_from_process_local_data(sharding, operand[local_rows], (n * rows, WIDTH))
with jax.set_mesh(mesh):
    out = jax.jit(jax.shard_map(exchange, out_specs=PartitionSpec(AXIS)))(sharded)
    jax.block_until_ready(out)

# Shard s sends its block r to shard r, which receives it at slot s. Compare this process's own
# shard: the global array is not addressable from any single process.
blocks = operand.reshape(n, n, ROWS_PER_PEER, WIDTH)
expected = blocks.transpose(1, 0, 2, 3).reshape(n * rows, WIDTH)[local_rows]
actual = np.asarray(out.addressable_shards[0].data)
print("{RESULT_MARKER} " + json.dumps({{
    "process_id": process_id,
    "correct": bool(np.array_equal(actual, expected)),
    "device_count": int(jax.device_count()),
    "compute_capability": ".".join(str(p) for p in devices[0].compute_capability),
}}, sort_keys=True), flush=True)

'''


def fetch_wheel(url: str, sha256: str, destination: Path) -> Path:
    """Download the candidate wheel and check it against the digest the build recorded.

    The wheel arrives by URL rather than in the Iris bundle. Iris lists bundle files with
    ``git ls-files --cached --others --exclude-standard`` under a 25 MB cap, which a >100 MB wheel
    cannot fit.

    The URL percent-encodes the ``+`` that opens the version's local segment, so the basename has
    to be decoded before it names a file. ``uv pip install`` reads the version out of the filename
    and rejects ``...dev20260824%2Bmarin...`` as invalid.
    """
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / urllib.parse.unquote(url.rsplit("/", 1)[1])
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


def free_port() -> int:
    """Reserve a loopback port for the JAX coordinator, then release it for JAX to bind."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_exercise(python: str, processes: int) -> tuple[str, list[dict]]:
    """Run one exercise process per GPU and return their combined output and result records."""
    environment = dict(
        os.environ,
        XLA_FLAGS=" ".join(filter(None, (os.environ.get("XLA_FLAGS", ""), *DEVICE_KERNEL_FLAGS))),
        TF_CPP_MAX_VLOG_LEVEL="3",
        MARIN_COORDINATOR=f"127.0.0.1:{free_port()}",
        MARIN_NUM_PROCESSES=str(processes),
    )
    children = [
        subprocess.Popen(  # noqa: S603 - our own interpreter and source
            [python, "-c", EXERCISE],
            env=dict(environment, MARIN_PROCESS_ID=str(index)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for index in range(processes)
    ]
    outputs = [child.communicate()[0] for child in children]
    combined = "".join(outputs)
    sys.stderr.write(combined)

    failed = [index for index, child in enumerate(children) if child.returncode != 0]
    if failed:
        raise SystemExit(f"ragged all-to-all exercise failed in processes {failed}")

    records = []
    for output in outputs:
        for line in output.splitlines():
            index = line.find(RESULT_MARKER)
            if index >= 0:
                records.append(json.loads(line[index + len(RESULT_MARKER) :]))
    return combined, records


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

    processes = config["validation"]["processes"]
    combined, records = run_exercise(python, processes)
    if len(records) != processes:
        raise SystemExit(f"expected {processes} result records, got {len(records)}")
    record = records[0]
    correct = all(entry["correct"] for entry in records)

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
        "ragged_all_to_all_ok": correct,
        "device_kernel_engaged": engaged,
        "status": "passed" if (correct and engaged) else "failed",
    }
    print(f"{MARKER} {json.dumps(result, sort_keys=True)}", flush=True)
    if not correct:
        wrong = [entry["process_id"] for entry in records if not entry["correct"]]
        raise SystemExit(f"validation failed: the ragged all-to-all result is wrong in processes {wrong}")
    if not engaged:
        # The thunk logs lsa_size before it decides, and logs a separate line when it declines, so
        # these say how far it got. Without them "did not engage" costs a whole run to localize.
        trace = [
            line.strip()
            for line in combined.splitlines()
            if "lsa_size" in line or "Device kernel" in line or "SupportsDeviceComm" in line
        ]
        detail = "\n  ".join(trace) if trace else "the thunk logged nothing about lsa_size"
        raise SystemExit(f"validation failed: device kernel did not engage\n  {detail}")


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
