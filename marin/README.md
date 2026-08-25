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

`.github/workflows/marin-pjrt.yaml` runs on a push to `main`, or on demand.

| Job | Runs on | Does |
| --- | --- | --- |
| `build` | `ubuntu-24.04-arm` | checks the pairing, builds the wheel, records its SHA-256 |
| `validate` | GB200, through Iris | installs the wheel and shows that the device kernel starts |
| `publish` | hosted | publishes the wheel and the pin |

Marin's change compiles into the GPU PJRT plugin, so `jax-cuda13-pjrt` is the only wheel built here.
It is built for `aarch64` and `sm_100` (GB200). The other three wheels are stock: install `jax`,
`jaxlib`, and `jax-cuda13-plugin` from the jax nightly index at the `jax_version` in
`marin/release/config.json`. Do not change one alone. `jax-cuda13-plugin` pins an exact
`jax-cuda13-pjrt` version and shares an ABI with it, and a different generation also changes NCCL.

The wheel moves between jobs as a build artifact. If `validate` fails, run that job again on its
own. It downloads the same artifact and does not build again.

`validate` checks that the ragged all-to-all device kernel started. A wheel that does not carry the
change still passes a correctness test, and still benchmarks as a null result.

## Where the wheel is stored

`publish` creates a GitHub release on this repository, tagged `marin-xla-pjrt-<date>-<sha12>`, which
holds the wheel, a `marin-xla-pjrt-manifest.json` that records the build inputs and the validation
result, and a `pjrt-release.toml`. A release is never overwritten, and no alias moves.

`pjrt-release.toml` is the descriptor Marin pins. To pin a release, copy it to
`config/external/xla/pjrt-release.toml` in the Marin repository, then run:

```sh
uv run config/update-external.py xla
```

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
5. Merge into `main`. CI builds, validates, and publishes.
6. Pin the new release in Marin.

The `preflight` step in `build` checks steps 2 to 4 and fails in seconds. It rejects a base that is
not what `jax_commit` pins, in either direction, and a `jax_version` whose sibling wheels are not on
the nightly index.
