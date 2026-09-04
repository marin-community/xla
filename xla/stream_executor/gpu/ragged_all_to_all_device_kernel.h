/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_H_

#include <cstdint>

#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"

namespace xla {
class SymmetricMemory;
namespace gpu {
class GpuDeviceCommunicator;
}  // namespace gpu
}  // namespace xla

namespace stream_executor::gpu {

// BENCH BRANCH ONLY. Not for upstream.
//
// The launch geometry and the CTA assignment are selected at run time so that a
// single wheel can measure every cell of the geometry x assignment matrix in one
// interleaved session. Placement variance across sessions is larger than some of
// the deltas we are trying to resolve, so the cells have to be drawn against each
// other rather than compared across builds.
//
// Geometry is a compile-time property of the kernel, because __launch_bounds__
// fixes both the block size and the register budget. Two instantiations cover the
// three geometries: kStock reproduces upstream (512-thread CTAs, no
// minBlocksPerMultiprocessor, so up to 128 registers/thread), while kNarrow and
// kWide share the 128-thread instantiation whose launch bounds cap registers at
// 64/thread and differ only in the grid the thunk launches.
inline constexpr int kRaggedAllToAllStockThreadsPerCta = 512;
inline constexpr int kRaggedAllToAllStockCtasPerSm = 1;
inline constexpr int kRaggedAllToAllDeviceKernelThreadsPerCta = 128;
inline constexpr int kRaggedAllToAllDeviceKernelCtasPerSm = 8;
// kUncapped: the narrow grid on an instantiation whose launch bounds ask for
// one block per SM, so the register cap is lifted. ptxas -v on the PR head
// (sm_100) gives 61-62 registers/thread at minBlocks=8 and 84-86 at 1, with no
// spills either way; this arm prices that difference.
inline constexpr int kRaggedAllToAllUncappedCtasPerSm = 1;

enum class RaggedAllToAllGeometry {
  kStock,   // 512-thread CTAs; grid derived from the active update count.
  kNarrow,  // 128-thread CTAs, one per SM.
  kWide,    // 128-thread CTAs, kRaggedAllToAllDeviceKernelCtasPerSm per SM.
  kUncapped,  // As kNarrow, but launch bounds (128, 1): no register cap.
};

// How the LSA copy is spread over the grid. Passed as a kernel argument rather
// than a template parameter: the branch is taken once per kernel, ahead of
// multi-MB copies, so specializing it would only multiply instantiations.
enum class RaggedAllToAllAssignment : int64_t {
  kFixed = 0,     // Upstream: a fixed CTA count per update.
  kBalanced = 1,  // Work-proportional: an equal element share per CTA.
};

template <int64_t kVectorSize, int kThreadsPerCta, int kCtasPerSm>
struct RaggedAllToAllDeviceKernel {
  using KernelType = stream_executor::TypedKernel<
      xla::gpu::GpuDeviceCommunicator*,    // dev_comm
      xla::SymmetricMemory*,               // send_win (input buffer)
      xla::SymmetricMemory*,               // recv_win (output buffer)
      stream_executor::DeviceAddressBase,  // input_offsets
      stream_executor::DeviceAddressBase,  // send_sizes
      stream_executor::DeviceAddressBase,  // output_offsets
      int64_t,                             // num_updates_per_replica
      int64_t,                             // num_row_elements
      int64_t,                             // input_buffer_offset_bytes
      int64_t,                             // output_buffer_offset_bytes
      int64_t>;                            // assignment
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_H_
