# Marin's XLA fork

This is `marin-community/xla`, a fork of [openxla/xla](https://github.com/openxla/xla).

The fork carries one change to the GPU ragged all-to-all device kernel. Marin's expert-parallel MoE
training needs that change, and it is not upstream. `main` is the overlay: an upstream XLA commit,
plus Marin's commits on top. `main` therefore stays behind upstream `main`, which is expected.
Everything Marin adds is under `marin/`, so a rebase does not conflict with these files.

Every upstream XLA workflow is disabled here. 21 of them start on a push to `main` or on a schedule,
and they would run on Marin's account. Disable any that a rebase adds, because they arrive enabled:

```sh
gh api -X PUT repos/marin-community/xla/actions/workflows/<id>/disable
```

## CI

`.github/workflows/marin-pjrt.yaml` runs on a push to `main`, or on demand. It has four jobs:

| Job | Runs on | Does |
| --- | --- | --- |
| `build` | `ubuntu-24.04-arm` | checks the pairing, builds the wheel, records its SHA-256 |
| `publish_candidate` | hosted | publishes the wheel as a prerelease `marin-xla-pjrt-candidate-<sha12>` |
| `validate` | GB200, through Iris | proves the device kernel engages on those exact bytes |
| `promote` | hosted | publishes the same bytes as `marin-xla-pjrt-<date>-<sha12>`, with the manifest Marin pins from |

The prerelease is the durable copy, and the wheel reaches the GB200 from its URL. The build
artifact between jobs expires after 14 days and needs credentials, and it is far past the 25 MB
limit Iris puts on a job's workspace bundle.

A candidate is unvalidated. Marin must pin a promoted release, never a candidate.

`validate` is what makes that distinction mean something. A wheel that does not carry the change
still passes a correctness test, and still benchmarks as a null result, so the gate checks that the
device kernel actually started. The kernel is opt-in behind
`--xla_gpu_experimental_ragged_all_to_all_use_device_kernel` and reports only through an
`XLA_VLOG_DEVICE(3)` line on stderr.

`promote` writes `marin-xla-pjrt-manifest.json`. Its shape is a contract with
`render_pjrt_release_toml` in Marin's `config/update-external.py`, which refuses a manifest that is
not `released`, whose validation did not pass, that does not record the device kernel engaging, or
that names another repository. Pin a release with:

```sh
uv run config/update-external.py --promote-pjrt-release marin-xla-pjrt-manifest.json
```

Marin's change compiles into the GPU PJRT plugin, so `jax-cuda13-pjrt` is the only wheel built here.
It is built for `aarch64` and `sm_100` (GB200). The other three wheels are stock: install `jax`,
`jaxlib`, and `jax-cuda13-plugin` from the jax nightly index at the `jax_version` in
`marin/release/config.json`. Do not change one alone. `jax-cuda13-plugin` pins an exact
`jax-cuda13-pjrt` version and shares an ABI with it, and a different generation also changes NCCL.

The wheel version is `<jax_version>+marin.<xla_short_sha>`. PEP 440 ignores a local segment when the
specifier has none, so this version satisfies the plugin's pin and still records the XLA commit.

## How to refresh onto a newer upstream

Marin drives a refresh through `.agents/skills/refresh-fork/SKILL.md`, against the `[xla]` section of
`config/external/migration.toml`.

The base is derived. Do not rebase onto an upstream `main` tip. jax pins an exact XLA tarball by
integrity hash, and builds its dependency graph from XLA module extensions, so the two change
together most days. A base that no published jax generation pins can compile and then fail at run
time.

1. Choose a jax revision that has published nightly wheels.
2. Read the XLA commit it pins, from `MODULE.bazel` in the jax repository.
3. Rebase Marin's commits onto that XLA commit, on a branch named
   `auto-refresh/<YYYYMMDD>/<base-id>-<shortsha>`.
4. Set `jax_commit` and `jax_version` in `marin/release/config.json` to that jax revision, in the
   same commit as the rebase.
5. Merge into `main`. CI then builds the wheel.

`marin/release/pjrt_release.py` checks steps 2 to 4 before the compile, and fails in seconds. It
rejects a base that is not what `jax_commit` pins, in either direction. It also rejects a
`jax_version` whose sibling wheels are absent from the nightly index.
