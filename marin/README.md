# Marin's XLA fork

This is `marin-community/xla`, a fork of [openxla/xla](https://github.com/openxla/xla).

The fork carries one change to the GPU ragged all-to-all device kernel. Marin's expert-parallel MoE
training needs that change, and it is proposed upstream as
[openxla/xla#47928](https://github.com/openxla/xla/pull/47928) but not merged. `main` is the
overlay: an upstream XLA commit, plus Marin's commits on top. `main` therefore stays behind upstream
`main`, which is expected. Everything Marin adds is under `marin/`, so a rebase does not conflict
with these files.

Every upstream XLA workflow is disabled here. 21 of them start on a push to `main` or on a schedule,
and they would run on Marin's account. Disable any that a rebase adds, because they arrive enabled:

```sh
gh api -X PUT repos/marin-community/xla/actions/workflows/<id>/disable
```

## CI

Two workflows. Building is automatic. Promoting is not.

`.github/workflows/marin-pjrt.yaml` runs on a push to `main`, or on demand. It builds the wheel,
then publishes it as a prerelease tagged `marin-xla-pjrt-candidate-<sha12>`, together with a JSON
record of the fork commit, the upstream base, the jax pairing, the filename and the SHA-256.

`.github/workflows/marin-pjrt-promote.yaml` is dispatched by hand against one candidate tag:

```sh
gh workflow run marin-pjrt-promote.yaml --repo marin-community/xla \
  -f candidate_tag=marin-xla-pjrt-candidate-<sha12>
```

It validates that candidate on a GB200 through Iris, then republishes the same bytes as
`marin-xla-pjrt-<date>-<sha12>` with the manifest Marin pins from.

The second workflow never compiles. Everything it needs is in the candidate record, so a promotion
takes minutes. Keeping it out of the build matters here: a full build is about three and a half
hours, the bazel cache for this repository is larger than GitHub retains, and a promotion tied to a
build would therefore pay for a compile of bytes that already exist.

A candidate is unvalidated. Marin must pin a promoted release, never a candidate.

Validation is what makes that distinction mean something. A wheel that does not carry the change
still passes a correctness test, and still benchmarks as a null result, so the gate checks that the
device kernel actually started. The kernel is opt-in behind
`--xla_gpu_experimental_ragged_all_to_all_use_device_kernel` and reports only through an
`XLA_VLOG_DEVICE(3)` line on stderr.

Promotion writes `marin-xla-pjrt-manifest.json`. Its shape is a contract with
`render_pjrt_release_toml` in Marin's `config/update-external.py`, which refuses a manifest that is
not `released`, whose validation did not pass, that does not record the device kernel engaging, or
that names another repository. Pin a release with:

```sh
uv run config/update-external.py --promote-pjrt-release marin-xla-pjrt-manifest.json
```

Marin's change compiles into the GPU PJRT plugin, so `jax-cuda13-pjrt` is the only wheel built here.
It is built for `aarch64` and `sm_100` (GB200). The other three wheels are stock: install
`jax[cuda13]` from PyPI at the `jax_version` in `marin/release/config.json`. Do not change one
alone. `jax-cuda13-plugin` pins an exact `jax-cuda13-pjrt` version and shares an ABI with it, and a
different generation also changes NCCL.

`jax_version` is a jax release rather than a nightly, so it is the version Marin's own workspace
already installs, the siblings resolve from PyPI with no index overlay, and the pairing stays
reconstructible after a nightly registry prunes.

The wheel version is `<jax_version>+marin.<xla_short_sha>`. PEP 440 ignores a local segment when the
specifier has none, so this version satisfies the plugin's pin and still records the XLA commit.
Under `ML_WHEEL_TYPE=release` both files that compose it — `python_wheel.bzl`, which names the
wheel, and `build/build.py`, which globs it out of `bazel-bin` afterwards — append
`ML_WHEEL_VERSION_SUFFIX` verbatim, so the suffix carries its own `+`. `marin/build_pjrt.sh`
composes the version the way both files will and refuses a build that disagrees with the preflight,
because the version is otherwise materialized only by the last action of a multi-hour compile.

## How to refresh onto a newer upstream

Marin drives a refresh through `.agents/skills/refresh-fork/SKILL.md`, against the `[xla]` section of
`config/external/migration.toml`.

The base is derived. Do not rebase onto an upstream `main` tip. jax pins an exact XLA tarball by
integrity hash, and builds its dependency graph from XLA module extensions, so the two change
together most days. A base that no published jax generation pins can compile and then fail at run
time.

1. Take the jax release tag Marin's workspace pins.
2. Read the XLA commit it pins, from `MODULE.bazel` in the jax repository.
3. Rebase Marin's commits onto that XLA commit, on a branch named
   `auto-refresh/<YYYYMMDD>/<base-id>-<shortsha>`.
4. Set `jax_commit` and `jax_version` in `marin/release/config.json` to that jax revision, in the
   same commit as the rebase.
5. Merge into `main`. CI then builds the wheel.

`marin/release/pjrt_release.py` checks steps 2 to 4 before the compile, and fails in seconds. It
rejects a base that is not what `jax_commit` pins, in either direction. It also rejects a
`jax_version` whose siblings are not released on PyPI.

An upstream commit the device kernel needs can land after that base. Carry it as its own
cherry-picked commit under the upstream author, ahead of Marin's change, and drop it at the refresh
that moves past it.
