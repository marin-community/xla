#!/usr/bin/env bash
# Build jax-cuda13-pjrt from this XLA checkout.
#
# The changed XLA sources (xla/backends/gpu/runtime/ragged_all_to_all*,
# xla/stream_executor/gpu/ragged_all_to_all_device_kernel_lib.cu.h) compile into the GPU PJRT
# plugin, so the PJRT wheel is the only artifact that carries Marin's delta. jax, jaxlib and
# jax-cuda13-plugin are consumed stock from the nightly index at JAX_VERSION.
#
# The built wheel must satisfy the stock plugin's exact `jax-cuda13-pjrt==<JAX_VERSION>` pin.
# jax's build derives the version as `<base>.dev<ML_WHEEL_BUILD_DATE>+<ML_WHEEL_GIT_HASH>` under
# ML_WHEEL_TYPE=custom, and PEP 440 ignores a local segment when the specifier has none, so
# `0.11.2.dev20260824+marin.<sha12>` resolves against `==0.11.2.dev20260824`. Getting the date
# wrong silently produces a wheel the plugin will refuse.
#
# usage: XLA_SOURCE=<path> JAX_COMMIT=<sha> JAX_WHEEL_BUILD_DATE=YYYYMMDD \
#        MARIN_XLA_COMMIT=<sha> OUT_DIR=<path> [CUDA_COMPUTE_CAPABILITIES=sm_100] \
#        [BAZEL_JOBS=4] build_pjrt.sh
set -euo pipefail

XLA_SOURCE="${XLA_SOURCE:?set XLA_SOURCE (path to this XLA checkout)}"
JAX_COMMIT="${JAX_COMMIT:?set JAX_COMMIT (jax revision to build against)}"
JAX_WHEEL_BUILD_DATE="${JAX_WHEEL_BUILD_DATE:?set JAX_WHEEL_BUILD_DATE (YYYYMMDD of the sibling nightly)}"
MARIN_XLA_COMMIT="${MARIN_XLA_COMMIT:?set MARIN_XLA_COMMIT (the fork tip being built)}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
CUDA_COMPUTE_CAPABILITIES="${CUDA_COMPUTE_CAPABILITIES:-sm_100}"
BAZEL_JOBS="${BAZEL_JOBS:-4}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_MAJOR_VERSION="${CUDA_MAJOR_VERSION:-13}"

XLA_SOURCE="$(cd "$XLA_SOURCE" && pwd)"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# The local segment records the exact XLA tip the wheel was built from, so an installed wheel
# names its own provenance without a manifest lookup.
SHORT_XLA="${MARIN_XLA_COMMIT:0:12}"

work="${WORK_DIR:-$(mktemp -d)}"
mkdir -p "$work"
jax_dir="$work/jax"

if [ ! -d "$jax_dir/.git" ]; then
  git clone --filter=blob:none https://github.com/jax-ml/jax.git "$jax_dir"
fi
git -C "$jax_dir" fetch --no-tags origin "$JAX_COMMIT"
git -C "$jax_dir" checkout --detach "$JAX_COMMIT"

echo "xla:  $(git -C "$XLA_SOURCE" rev-parse HEAD)"
echo "jax:  $(git -C "$jax_dir" rev-parse HEAD)"

# Assert the delta is actually present. A silently-absent patch produces a wheel that looks
# healthy and benchmarks as a null result, which is indistinguishable from the change not helping.
if ! grep -q "kGridSmMultiplier" "$XLA_SOURCE/xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"; then
  echo "XLA checkout does not carry the ragged a2a device-kernel grid delta." >&2
  exit 1
fi

cd "$jax_dir"
python3 build/build.py build \
  --wheels=jax-cuda-pjrt \
  --cuda_major_version="$CUDA_MAJOR_VERSION" \
  --python_version="$PYTHON_VERSION" \
  --cuda_compute_capabilities="$CUDA_COMPUTE_CAPABILITIES" \
  --local_xla_path="$XLA_SOURCE" \
  --bazel_options=--jobs="$BAZEL_JOBS" \
  --bazel_options=--repo_env=ML_WHEEL_TYPE=custom \
  --bazel_options=--repo_env=ML_WHEEL_BUILD_DATE="$JAX_WHEEL_BUILD_DATE" \
  --bazel_options=--repo_env=ML_WHEEL_GIT_HASH="marin.${SHORT_XLA}" \
  --bazel_options=--define=ynn_enable_arm64_neonfp8=false \
  --verbose

shopt -s nullglob
wheels=(dist/jax_cuda"${CUDA_MAJOR_VERSION}"_pjrt-*.whl)
if [ "${#wheels[@]}" -ne 1 ]; then
  echo "expected exactly one PJRT wheel in dist/, found ${#wheels[@]}" >&2
  ls -la dist/ >&2
  exit 1
fi

cp "${wheels[0]}" "$OUT_DIR/"
built="$OUT_DIR/$(basename "${wheels[0]}")"
echo "wheel:  $built"
echo "sha256: $(sha256sum "$built" | cut -d' ' -f1)"
