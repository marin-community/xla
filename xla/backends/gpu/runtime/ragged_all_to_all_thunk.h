/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_RAGGED_ALL_TO_ALL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_RAGGED_ALL_TO_ALL_THUNK_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_device_kernel.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/util/env_var.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/util/tied_ref.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct RaggedAllToAllConfig {
  CollectiveConfig config;
  int64_t num_total_updates = 1;
  int64_t num_input_rows = 1;
  int64_t num_row_elements = 1;

  // Whether the one-shot kernel is enabled. If true, the thunk will use the
  // one-shot kernel when possible.
  bool one_shot_kernel_enabled = false;

  // If true, the thunk will use the MultiGpuBarrierWithNcclKernel in the
  // one-shot kernel for multi-host synchronization. Works when devices are on
  // multiple hosts connected via a fast interconnect (e.g., MNNVL).
  bool use_multi_gpu_barrier_with_nccl_in_one_shot_kernel = false;

  // If true, the thunk will use the fallback NCCL ragged all-to-all kernel.
  bool allow_fallback_to_nccl = false;

  CollectiveThunk::CollectivesMode collectives_mode =
      DebugOptions::COLLECTIVES_PRIVATE_MEMORY;

  // If true, the thunk will use the device-initiated (NCCL GIN + LSA) kernel
  // for ragged-all-to-all when symmetric buffers are available.
  bool use_device_kernel = false;

  // If set, this will be used to determine if optimized kernels that assume a
  // fast interconnect can be used.
  std::optional<int64_t> fast_interconnect_slice_size_override = std::nullopt;
};

// Contains the values that are passed between host threads with rendezvous.
struct RaggedAllToAllRendezvousValue {
  RankId rank;
  se::DeviceAddressBase output_buffer;

  // Exchange the address of the SIGNAL BUFFER array.
  // Peers will write to their_rank's-th cell in the signals array.
  se::DeviceAddressBase barrier_signal_buffer;

  bool operator<(const RaggedAllToAllRendezvousValue& other) const {
    return rank < other.rank;
  }
};

struct RaggedAllToAllStreamState {
  int device_ordinal;
  RankId rank;
  std::optional<int64_t> lsa_size;
  GpuCliqueKey clique_key;

  // Host memory allocations for ragged metadata.
  absl::InlinedVector<std::unique_ptr<se::MemoryAllocation>, 8>
      host_buffer_allocs;

  // Device memory buffer for output offsets.
  se::ScopedDeviceAddress<uint8_t> output_offsets_device_buffer;

  // MultiGpuBarrier: Device memory buffer for signal values (one per peer).
  // Peers write specific slots in this array to signal this device.
  std::unique_ptr<se::MemoryAllocation> barrier_signal_buffer;

  // Reference to the symmetric memory handler for the barrier signal buffer.
  tsl::TiedRef<xla::SymmetricMemory> barrier_signal_symmetric_memory;

  // MultiGpuBarrier: Device memory for the current local step counter.
  // This value is incremented locally by the kernel after every barrier.
  std::unique_ptr<se::MemoryAllocation> barrier_signal_value;

  // Device memory buffer to store the output buffer pointers.
  std::unique_ptr<se::MemoryAllocation> output_buffer_ptr_storage;

  // Device kernel barrier timeout diagnostics (see #8870). The kernel bounds
  // its cross-rank barrier waits and, on expiry, publishes a single 64-bit
  // record here naming the rank, the barrier slot and the barrier phase that
  // stalled; the record is sticky for the life of the process. Allocated only
  // on the device-kernel path.
  se::ScopedDeviceAddress<uint8_t> barrier_timeout_record;

  // Pinned host destination for the readback of `barrier_timeout_record`. The
  // readback is enqueued after every device-kernel launch and never waited on,
  // so the host copy lags the device by at least one launch. That is fine: a
  // stalled barrier is sticky, and one aligned 64-bit word cannot be observed
  // torn.
  std::unique_ptr<se::MemoryAllocation> barrier_timeout_host_record;

  // Contains the output buffer pointers and barrier signal buffers for all
  // peers.
  std::shared_ptr<std::vector<RaggedAllToAllRendezvousValue>> participants;

  RaggedAllToAllStreamState(int device_ordinal, RankId rank,
                            GpuCliqueKey clique_key)
      : device_ordinal(device_ordinal),
        rank(rank),
        clique_key(std::move(clique_key)) {}
};

// Thunk that performs a NCCL-based Ragged-All-to-All among CUDA GPU-based
// replicas.
class RaggedAllToAllThunk : public CollectiveThunk {
 public:
  RaggedAllToAllThunk(ThunkInfo thunk_info,
                      const HloRaggedAllToAllInstruction* instr,
                      std::vector<Buffer> buffers, bool p2p_memcpy_enabled);
  RaggedAllToAllThunk(ThunkInfo thunk_info, const RaggedAllToAllConfig& config,
                      std::vector<CollectiveThunk::Buffer> buffers);

  // Returns whether the given instruction can be lowered to a nccl
  // ragged-all-to-all call.
  static absl::Status CheckImplementable(
      const HloRaggedAllToAllInstruction* instr, int64_t replica_count,
      int64_t partition_count);

  CollectiveCliqueRequests::CliqueRequirements GetCliqueRequirements(
      const GpuCliqueKey& clique_key, const PrepareParams& params) override;

  absl::Status Initialize(const InitializeParams& params) override;

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const ExecuteParams& execute_params, const RecordParams& record_params,
      RecordAction record_action, se::CommandBuffer* command_buffer) override;

  static absl::string_view GetHloOpName() { return "ragged-all-to-all-start"; }

  static CollectiveOpGroupMode GetGroupMode(
      const HloRaggedAllToAllInstruction* instr);

  const CollectiveConfig& config() const override { return config_.config; }

  bool CanUseSymmetricBuffer() const override { return true; }

  const RaggedAllToAllConfig& ragged_all_to_all_config() const {
    return config_;
  }

  bool is_one_shot_kernel_enabled() const {
    return config_.one_shot_kernel_enabled;
  }

  bool use_multi_gpu_barrier_with_nccl_in_one_shot_kernel() const {
    return config_.use_multi_gpu_barrier_with_nccl_in_one_shot_kernel;
  }

  bool UsesDeviceKernel() const {
    return config_.use_device_kernel && config_.config.use_symmetric_buffer;
  }

  // Launch grid for the device kernel, and the number of per-CTA
  // barrier/signal slots reserved when creating the device communicator (the
  // kernel indexes its barriers by blockIdx.x, so registration must cover the
  // launched grid). One CTA per SM: the copies are link-bound at these message
  // sizes, so a wider grid buys little transport latency, and the kernel holds
  // its CTAs for the whole transport including the barrier spins, which starves
  // compute scheduled against it. Callers pass the SM count from
  // se::DeviceDescription::core_count(); all participating ranks are expected
  // to be homogeneous so every rank arrives at the same value.
  //
  // XLA_RAGGED_ALL_TO_ALL_CTA_HEADROOM reserves that many SMs by shrinking the
  // grid. The LSA barrier makes CTA k wait on CTA k of every peer, so a CTA
  // that is never scheduled stalls its peers with no timeout and no error. A
  // grid sized to the whole device leaves no room for a concurrently resident
  // kernel, and at multi-rack scale the cross-rack AllReduce is exactly that.
  // Zero keeps the current behaviour.
  static int32_t DeviceKernelCtaCount(int core_count) {
    int64_t headroom = 0;
    // Diagnostic knob for marin-community/marin#8870; not a supported flag.
    tsl::ReadInt64FromEnvVar("XLA_RAGGED_ALL_TO_ALL_CTA_HEADROOM", 0, &headroom)
        .IgnoreError();
    const int32_t reserved =
        static_cast<int32_t>(std::clamp<int64_t>(headroom, 0, core_count));
    return std::max<int32_t>(core_count - reserved, kMinDeviceKernelCtaCount);
  }

  // Per-wait budget for the device kernel's cross-rank barriers, in SM clock
  // cycles (what clock64() counts, and what NCCL's timeout barrier overloads
  // compare against).
  //
  // Targets ~45s of wall time. That is far longer than any legitimate skew
  // between ranks of one collective - the whole grid is co-resident and the
  // peers are on the same NVLink domain - but short enough that a wedged
  // collective reports itself well inside the job's watchdog window instead of
  // hanging silently. The SM clock throttles, so the realised timeout is only
  // ever longer than the target, never shorter.
  static int64_t DeviceKernelBarrierTimeoutCycles(
      const se::DeviceDescription& device_description) {
    constexpr double kTargetSeconds = 45.0;
    // clock_rate_ghz() is kUninitialized<float> (-1) on platforms that do not
    // report it. Fall back to a deliberately high clock so the fallback errs
    // towards a longer timeout rather than a spurious one.
    constexpr double kFallbackClockGhz = 3.0;
    const double clock_ghz = device_description.clock_rate_ghz() > 0.0f
                                 ? device_description.clock_rate_ghz()
                                 : kFallbackClockGhz;
    return static_cast<int64_t>(kTargetSeconds * clock_ghz * 1e9);
  }

  GpuDeviceCommunicator::Requirements DeviceKernelLsaDevCommRequirements(
      int core_count) const {
    GpuDeviceCommunicator::Requirements requirements;
    requirements.lsa_barrier_count = DeviceKernelCtaCount(core_count);
    return requirements;
  }

  GpuDeviceCommunicator::Requirements DeviceKernelDevCommRequirements(
      int core_count) const {
    GpuDeviceCommunicator::Requirements requirements;
    const int32_t c = DeviceKernelCtaCount(core_count);
    requirements.barrier_count = c;
    requirements.lsa_barrier_count = c;
    requirements.rail_gin_barrier_count = c;
    requirements.gin_signal_count = c;
    requirements.gin_connection_full = true;
    return requirements;
  }

  // Returns true if one shot kernel is supported
  bool IsOneShotKernelSupported() const;

  static absl::StatusOr<std::unique_ptr<RaggedAllToAllThunk>> FromProto(
      ThunkInfo thunk_info, const RaggedAllToAllThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::StatusOr<ThunkProto> ToProto() const override;

 protected:
  // No rendezvous needed when using one-shot kernel in local mode instead of
  // NCCL.
  bool RequiresRendezvous() const override {
    return !is_one_shot_kernel_enabled();
  }

  absl::Status PrepareCollective(const PrepareParams& params,
                                 const GpuCliqueKey& clique_key) override;

  absl::Status RunCollective(const ExecuteParams& params,
                             const GpuCliqueKey& clique_key, se::Stream& stream,
                             Communicator& comm) override;

 private:
  bool is_local(int device_count) const;

  const RaggedAllToAllConfig config_;

  // Floor on the launch grid so small shapes still get some parallelism.
  // The upper bound is derived from the executor's SM count at Prepare /
  // Initialize / Run time via DeviceKernelCtaCount().
  static constexpr int32_t kMinDeviceKernelCtaCount = 8;

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<RaggedAllToAllStreamState>>
      per_stream_states_ ABSL_GUARDED_BY(mutex_);

  absl::StatusOr<RaggedAllToAllStreamState*> InitializeOnce(
      const InitializeParams& params);
};

// Executes the rendezvous to exchange buffer addresses and barrier signal
// buffers.
absl::StatusOr<std::shared_ptr<std::vector<RaggedAllToAllRendezvousValue>>>
RendezvousResources(int device_ordinal, RankId rank,
                    const GpuCliqueKey& clique_key,
                    const se::DeviceAddressBase& output_buffer,
                    const se::DeviceAddressBase& barrier_signal_buffer);

// Executes an optimized "One-Shot" Ragged All-to-All collective.
//
// Unlike the standard implementation, this approach consolidates the
// coordination and data movement into a single execution path (typically a
// custom kernel or specialized P2P sequence) to reduce host-device
// synchronization overhead.
//
// It utilizes `MultiGpuBarrierKernel` to enforce device-side synchronization.
// This ensures input/output buffers are safe to access without requiring
// Event-based coordination, enabling compatibility with CUDA Graphs.
absl::Status RunOneShotRaggedAllToAll(
    const GpuCliqueKey& clique_key, se::Stream& stream, RankId rank,
    const se::DeviceAddressBase& barrier_signal_buffer,
    const se::DeviceAddressBase& barrier_signal_value,
    int64_t num_total_updates, int64_t num_input_rows, int64_t num_row_elements,
    absl::Span<DeviceBufferPair const> buffers,
    const std::vector<RaggedAllToAllRendezvousValue>& participants);

// It utilizes `MultiGpuBarrierWithNcclKernel` to enforce device-side
// synchronization. This ensures input/output buffers are safe to access without
// requiring Event-based coordination, enabling compatibility with CUDA Graphs.
absl::Status RunOneShotRaggedAllToAllWithNccl(
    const GpuCliqueKey& clique_key, se::Stream& stream, RankId rank,
    std::shared_ptr<xla::SymmetricMemory> barrier_signal_symmetric_memory,
    const se::DeviceAddressBase& barrier_signal_value,
    SymmetricMemory* output_sym_mem, size_t output_sym_offset,
    int64_t num_total_updates, int64_t num_input_rows, int64_t num_row_elements,
    absl::Span<DeviceBufferPair const> buffers);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_RAGGED_ALL_TO_ALL_THUNK_H_
