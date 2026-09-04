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

// Launch shape of the device kernel. The thunk sizes its grid to
// kRaggedAllToAllDeviceKernelCtasPerSm CTAs per SM, and the in-kernel
// cross-rank barriers deadlock unless the whole grid is resident at once.
// The kernel's __launch_bounds__ passes kRaggedAllToAllDeviceKernelCtasPerSm
// as minBlocksPerMultiprocessor, so the register allocator must preserve
// that residency (64 registers/thread at 8x128 on a 64K-register SM).
inline constexpr int kRaggedAllToAllDeviceKernelThreadsPerCta = 128;
inline constexpr int kRaggedAllToAllDeviceKernelCtasPerSm = 8;

// Layout of the device kernel's cross-rank barrier timeout record.
//
// The in-kernel barriers have no way to report a stall: NCCL leaves
// `abortFlag == nullptr` for user-created device comms, so the no-timeout
// barrier waits are uninterruptible and a lost peer wedges the kernel with no
// NCCL watchdog, no abort and no CUDA error. The kernel therefore uses NCCL's
// timeout barrier overloads and, on expiry, publishes a single 64-bit record
// naming the stall before exiting.
//
// One aligned 64-bit word is used (rather than a struct) so that the host's
// unsynchronized readback cannot observe a torn record: the word is written
// once by atomicCAS from 0, and is only ever interpreted when the tag byte is
// present. The record is sticky for the life of the process; the first CTA on
// the device to time out wins, later ones leave it alone.
//
//   bits [63:56]  tag, kRaggedAllToAllBarrierTimeoutTag, marks the word set
//   bits [55:48]  phase, one of the kRaggedAllToAllBarrierPhase* values below
//   bits [47:32]  rank of the recording device within its LSA team
//   bits [31:16]  rank of the recording device within the world team
//   bits [15:0]   blockIdx.x, i.e. the barrier slot that timed out
//
// Rank fields are truncated to 16 bits; the barrier slot count is bounded by
// the SM count, so it always fits.
inline constexpr uint64_t kRaggedAllToAllBarrierTimeoutTag = 0xA5;
inline constexpr int kRaggedAllToAllBarrierTimeoutTagShift = 56;
inline constexpr int kRaggedAllToAllBarrierTimeoutPhaseShift = 48;
inline constexpr int kRaggedAllToAllBarrierTimeoutLsaRankShift = 32;
inline constexpr int kRaggedAllToAllBarrierTimeoutWorldRankShift = 16;

// Which of the kernel's two barrier waits expired. "Pre-copy" is the rendezvous
// that publishes buffer readiness before any peer store; "post-copy" is the
// release barrier that publishes the stores.
inline constexpr uint64_t kRaggedAllToAllBarrierPhasePreCopy = 1;
inline constexpr uint64_t kRaggedAllToAllBarrierPhasePostCopy = 2;

// Layout of the record buffer the kernel publishes into. Three aligned 64-bit
// words, each written independently and read back together:
//
//   [0] the timeout record described above
//   [1] the longest pre-copy barrier wait observed on this device, in cycles
//   [2] the longest post-copy barrier wait observed on this device, in cycles
//
// The wait words exist because a stall is the tail of a distribution rather
// than a separate phenomenon: a run that never times out still carries evidence
// in how long its barriers waited. Recording the high-water mark per phase is
// what makes a hang-free run informative instead of a single "no timeout" bit,
// which is the whole difficulty in #8870 -- eight reproduction attempts have
// each returned exactly that bit.
//
// Both wait words are monotone high-water marks, sticky for the life of the
// process, so an unsynchronized host readback of either is always meaningful.
inline constexpr int kRaggedAllToAllBarrierRecordWords = 3;
inline constexpr int kRaggedAllToAllBarrierTimeoutWord = 0;
inline constexpr int kRaggedAllToAllBarrierMaxPreCopyWord = 1;
inline constexpr int kRaggedAllToAllBarrierMaxPostCopyWord = 2;

template <int64_t kVectorSize>
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
      int64_t,                             // barrier_timeout_cycles
      stream_executor::DeviceAddressBase>;  // barrier_timeout_record
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_H_
