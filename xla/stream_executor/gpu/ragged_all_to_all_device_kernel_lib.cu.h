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

#ifndef XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_LIB_CU_H_
#define XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_LIB_CU_H_

#include <cstdint>

#include "xla/stream_executor/gpu/ragged_all_to_all_device_kernel.h"

#if NCCL_VERSION_CODE >= 22900
#include "third_party/nccl/nccl_device.h"
#endif

namespace stream_executor::gpu {

template <int64_t kSize>
struct alignas(kSize) DeviceVec {
  uint8_t data[kSize];
};

#if NCCL_VERSION_CODE >= 22900

template <int64_t kVectorSize>
struct RaggedAllToAllUpdateMetadata {
  int peer;
  int update;
  int64_t meta_idx;
  int64_t send_size;
  int64_t src_byte_offset;
  int64_t dst_byte_offset;
  int64_t byte_count;
};

template <int64_t kVectorSize>
__device__ bool LoadRaggedAllToAllUpdateMetadata(
    int64_t flat_idx, int64_t num_updates_per_replica, int64_t num_row_elements,
    int64_t input_buffer_offset_bytes, int64_t output_buffer_offset_bytes,
    const int64_t* __restrict__ input_offsets_ptr,
    const int64_t* __restrict__ send_sizes_ptr,
    const int64_t* __restrict__ output_offsets_ptr,
    RaggedAllToAllUpdateMetadata<kVectorSize>* meta) {
  meta->peer = flat_idx / num_updates_per_replica;
  meta->update = flat_idx % num_updates_per_replica;
  meta->meta_idx = meta->peer * num_updates_per_replica + meta->update;
  meta->send_size = send_sizes_ptr[meta->meta_idx];
  if (meta->send_size == 0) {
    return false;
  }

  const int64_t input_offset = input_offsets_ptr[meta->meta_idx];
  const int64_t output_offset = output_offsets_ptr[meta->meta_idx];
  meta->src_byte_offset =
      input_buffer_offset_bytes + input_offset * num_row_elements * kVectorSize;
  meta->dst_byte_offset = output_buffer_offset_bytes +
                          output_offset * num_row_elements * kVectorSize;
  meta->byte_count = meta->send_size * num_row_elements * kVectorSize;
  return true;
}

template <int64_t kVectorSize>
__device__ void RaggedAllToAllCopy(
    ncclWindow_t send_win, ncclWindow_t recv_win,
    const int64_t* __restrict__ input_offsets_ptr,
    const int64_t* __restrict__ send_sizes_ptr,
    const int64_t* __restrict__ output_offsets_ptr,
    int64_t num_updates_per_replica, int64_t num_row_elements,
    int64_t input_buffer_offset_bytes, int64_t output_buffer_offset_bytes,
    int start_lsa, int lsa_size, int num_ranks, ncclGin* gin, ncclTeam world,
    unsigned int signal_index) {
  using T = DeviceVec<kVectorSize>;

  if (lsa_size > 0) {
    // Work-proportional CTA assignment: CTA k copies the global element range
    // [total*k/grid, total*(k+1)/grid), walking update boundaries as needed.
    // This keeps CTA work balanced regardless of how transferred bytes are
    // distributed across updates. Every CTA redundantly scans the (small)
    // update-size array; the loads hit L2 and the scan is trivially cheap
    // next to the multi-MB copies. Updates are walked in peer-major order, so
    // adjacent CTAs write to adjacent peers.
    const int64_t grid = static_cast<int64_t>(gridDim.x);
    const int64_t num_lsa_updates =
        static_cast<int64_t>(lsa_size) * num_updates_per_replica;
    const int64_t meta_base =
        static_cast<int64_t>(start_lsa) * num_updates_per_replica;

    int64_t total_elements = 0;
    for (int64_t update = 0; update < num_lsa_updates; ++update) {
      total_elements += send_sizes_ptr[meta_base + update] * num_row_elements;
    }

    const int64_t cta_begin =
        total_elements * static_cast<int64_t>(blockIdx.x) / grid;
    const int64_t cta_end =
        total_elements * (static_cast<int64_t>(blockIdx.x) + 1) / grid;

    // Element offset of the current update within the concatenated element
    // space that cta_begin/cta_end index into.
    int64_t update_begin = 0;
    for (int64_t update = 0;
         update < num_lsa_updates && update_begin < cta_end; ++update) {
      RaggedAllToAllUpdateMetadata<kVectorSize> meta;
      if (!LoadRaggedAllToAllUpdateMetadata<kVectorSize>(
              meta_base + update, num_updates_per_replica, num_row_elements,
              input_buffer_offset_bytes, output_buffer_offset_bytes,
              input_offsets_ptr, send_sizes_ptr, output_offsets_ptr, &meta)) {
        continue;
      }
      const int64_t update_end = update_begin + meta.byte_count / kVectorSize;
      if (update_end > cta_begin) {
        const int64_t lo =
            cta_begin > update_begin ? cta_begin - update_begin : 0;
        const int64_t hi =
            (cta_end < update_end ? cta_end : update_end) - update_begin;
        const int lsa_peer =
            static_cast<int>(update / num_updates_per_replica);
        const T* src = static_cast<const T*>(
            ncclGetLocalPointer(send_win, meta.src_byte_offset));
        T* dst = static_cast<T*>(
            ncclGetLsaPointer(recv_win, meta.dst_byte_offset, lsa_peer));
        for (int64_t i = lo + static_cast<int64_t>(threadIdx.x); i < hi;
             i += static_cast<int64_t>(blockDim.x)) {
          dst[i] = src[i];
        }
      }
      update_begin = update_end;
    }
  }

  if (gin == nullptr) {
    return;
  }

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
  const int64_t total_updates = num_updates_per_replica * num_ranks;

  for (int64_t flat_idx = tid; flat_idx < total_updates; flat_idx += nthreads) {
    const int peer = flat_idx / num_updates_per_replica;
    if (peer >= start_lsa && peer < start_lsa + lsa_size) {
      continue;
    }

    RaggedAllToAllUpdateMetadata<kVectorSize> meta;
    if (!LoadRaggedAllToAllUpdateMetadata<kVectorSize>(
            flat_idx, num_updates_per_replica, num_row_elements,
            input_buffer_offset_bytes, output_buffer_offset_bytes,
            input_offsets_ptr, send_sizes_ptr, output_offsets_ptr, &meta)) {
      continue;
    }

    gin->put(world, meta.peer, recv_win, meta.dst_byte_offset, send_win,
             meta.src_byte_offset, meta.byte_count,
             ncclGin_SignalInc{signal_index});
  }
}

// Reduces a per-thread barrier result to a CTA-uniform verdict.
//
// NCCL's timeout barriers report the timeout per thread: a thread whose share
// of the peer loop is empty, or whose peers all arrived, returns ncclSuccess
// even though a sibling thread gave up. (NCCL notes the same in
// ncclBarrierSession::sync's timeout overload.) The whole CTA has to agree,
// because leaving the barrier scope runs ~ncclLsaBarrierSession, which ends in
// coop.sync() == __syncthreads(): a partial exit would hang the survivors.
//
// Safe to call here because every timeout overload ends in coop.sync(), so all
// threads of the CTA reach this point together.
__device__ inline bool RaggedAllToAllBarrierTimedOut(ncclResult_t result) {
  return __syncthreads_or(result != ncclSuccess) != 0;
}

// Publishes the first barrier timeout seen on this device. See
// ragged_all_to_all_device_kernel.h for the record layout.
//
// A timeout is terminal, not recoverable: NCCL's wait skips its epoch
// increment on the timeout path, so the session destructor persists a stale
// epoch and the barrier's shared state is left inconsistent for every
// subsequent launch on that comm. The host is expected to fail the execution
// once it observes a record rather than to retry.
__device__ inline void RecordRaggedAllToAllBarrierTimeout(
    uint64_t* __restrict__ record_ptr, uint64_t phase, int world_rank,
    int lsa_rank) {
  if (record_ptr == nullptr || threadIdx.x != 0) {
    return;
  }
  const uint64_t record =
      (kRaggedAllToAllBarrierTimeoutTag
       << kRaggedAllToAllBarrierTimeoutTagShift) |
      (phase << kRaggedAllToAllBarrierTimeoutPhaseShift) |
      (static_cast<uint64_t>(lsa_rank & 0xFFFF)
       << kRaggedAllToAllBarrierTimeoutLsaRankShift) |
      (static_cast<uint64_t>(world_rank & 0xFFFF)
       << kRaggedAllToAllBarrierTimeoutWorldRankShift) |
      static_cast<uint64_t>(blockIdx.x & 0xFFFF);
  // First writer wins, so the record names the CTA that stalled rather than the
  // last one to notice.
  atomicCAS(reinterpret_cast<unsigned long long*>(record_ptr), 0ULL,
            static_cast<unsigned long long>(record));
}

template <int64_t kVectorSize>
__global__ void __launch_bounds__(kRaggedAllToAllDeviceKernelThreadsPerCta,
                                  kRaggedAllToAllDeviceKernelCtasPerSm)
    RaggedAllToAllDeviceKernelImpl(
    struct ncclDevComm dev_comm, ncclWindow_t send_win, ncclWindow_t recv_win,
    const int64_t* __restrict__ input_offsets_ptr,
    const int64_t* __restrict__ send_sizes_ptr,
    const int64_t* __restrict__ output_offsets_ptr,
    int64_t num_updates_per_replica, int64_t num_row_elements,
    int64_t input_buffer_offset_bytes, int64_t output_buffer_offset_bytes,
    int64_t barrier_timeout_cycles,
    uint64_t* __restrict__ barrier_timeout_record) {
  // NCCL device barrier/GIN APIs emit scope-qualified atomics that require
  // sm_60+. Lower architectures compile to an empty stub; the kernel is only
  // launched when the device supports NCCL device comms.
#if __CUDA_ARCH__ >= 600
  ncclTeam world = ncclTeamWorld(dev_comm);
  ncclTeam lsa = ncclTeamLsa(dev_comm);
  const int start_lsa = world.rank - lsa.rank;
  const int lsa_size = lsa.nRanks;
  const int num_ranks = world.nRanks;
  const bool has_remote_peers = (lsa_size < num_ranks);
  // Budget for each individual barrier wait, in SM clock cycles. The host
  // derives it from the device clock rate; see
  // RaggedAllToAllThunk::DeviceKernelBarrierTimeoutCycles.
  const uint64_t timeout_cycles =
      static_cast<uint64_t>(barrier_timeout_cycles);

  if (has_remote_peers) {
    const int gin_context = 0;
    const unsigned int signal_index = 0;

    ncclGin gin{dev_comm, gin_context};
    uint64_t signal_value =
        (blockIdx.x == 0) ? gin.readSignal(signal_index) : 0;

    ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld(), gin,
                                        blockIdx.x};
    // NB: gin.waitSignal below has no timeout overload in NCCL, so the GIN
    // path is only partially bounded. The reported stall (#8870) is on the
    // pure-LSA path.
    if (RaggedAllToAllBarrierTimedOut(bar.sync(
            ncclCoopCta(), ::cuda::memory_order_acquire,
            ncclGinFenceLevel::Relaxed, timeout_cycles))) {
      RecordRaggedAllToAllBarrierTimeout(barrier_timeout_record,
                                         kRaggedAllToAllBarrierPhasePreCopy,
                                         world.rank, lsa.rank);
      return;
    }

    RaggedAllToAllCopy<kVectorSize>(
        send_win, recv_win, input_offsets_ptr, send_sizes_ptr,
        output_offsets_ptr, num_updates_per_replica, num_row_elements,
        input_buffer_offset_bytes, output_buffer_offset_bytes, start_lsa,
        lsa_size, num_ranks, &gin, world, signal_index);

    const int num_remote_peers =
        (num_ranks - lsa_size) * num_updates_per_replica;
    if (blockIdx.x == 0) {
      gin.waitSignal(ncclCoopCta(), signal_index,
                     signal_value + num_remote_peers);
    }

    gin.flush(ncclCoopCta());
    if (RaggedAllToAllBarrierTimedOut(bar.sync(
            ncclCoopCta(), ::cuda::memory_order_release,
            ncclGinFenceLevel::Relaxed, timeout_cycles))) {
      RecordRaggedAllToAllBarrierTimeout(barrier_timeout_record,
                                         kRaggedAllToAllBarrierPhasePostCopy,
                                         world.rank, lsa.rank);
      return;
    }
  } else {
    ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), dev_comm,
                                           ncclTeamTagLsa{}, blockIdx.x};
    if (RaggedAllToAllBarrierTimedOut(bar.sync(
            ncclCoopCta(), ::cuda::memory_order_relaxed, timeout_cycles))) {
      RecordRaggedAllToAllBarrierTimeout(barrier_timeout_record,
                                         kRaggedAllToAllBarrierPhasePreCopy,
                                         world.rank, lsa.rank);
      return;
    }

    RaggedAllToAllCopy<kVectorSize>(
        send_win, recv_win, input_offsets_ptr, send_sizes_ptr,
        output_offsets_ptr, num_updates_per_replica, num_row_elements,
        input_buffer_offset_bytes, output_buffer_offset_bytes, start_lsa,
        lsa_size, num_ranks, /*gin=*/nullptr, world, /*signal_index=*/0);

    if (RaggedAllToAllBarrierTimedOut(bar.sync(
            ncclCoopCta(), ::cuda::memory_order_release, timeout_cycles))) {
      RecordRaggedAllToAllBarrierTimeout(barrier_timeout_record,
                                         kRaggedAllToAllBarrierPhasePostCopy,
                                         world.rank, lsa.rank);
      return;
    }
  }
#endif  // __CUDA_ARCH__ >= 600
}

#else  // NCCL_VERSION_CODE < 22900

template <int64_t kVectorSize>
__global__ void RaggedAllToAllDeviceKernelImpl(
    void* dev_comm, void* send_win, void* recv_win,
    const int64_t* input_offsets_ptr, const int64_t* send_sizes_ptr,
    const int64_t* output_offsets_ptr, int64_t num_updates_per_replica,
    int64_t num_row_elements, int64_t input_buffer_offset_bytes,
    int64_t output_buffer_offset_bytes, int64_t barrier_timeout_cycles,
    uint64_t* barrier_timeout_record) {}

#endif  // NCCL_VERSION_CODE >= 22900

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_LIB_CU_H_
