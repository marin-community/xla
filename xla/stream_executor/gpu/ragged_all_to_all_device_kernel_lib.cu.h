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

#if __CUDA_ARCH__ >= 900
// Four warp leaders stream through separate buffers. Read completion permits
// buffer reuse; full completion before return makes remote stores visible to
// the subsequent NCCL release barrier.
__device__ __forceinline__ void RaggedAllToAllBulkCopy(const DeviceVec<16>* src,
                                                       DeviceVec<16>* dst,
                                                       int64_t lo, int64_t hi) {
  constexpr int kStreams = 4;
  constexpr int kChunkBytes = 8192;
  constexpr int kChunkElements = kChunkBytes / 16;
  __shared__ __align__(16) uint8_t buffers[kStreams][kChunkBytes];
  __shared__ __align__(8) uint64_t barriers[kStreams];
  const int warp = threadIdx.x / 32;
  if (threadIdx.x % 32 == 0 && warp < kStreams) {
    const uint32_t buffer =
        static_cast<uint32_t>(__cvta_generic_to_shared(buffers[warp]));
    const uint32_t barrier =
        static_cast<uint32_t>(__cvta_generic_to_shared(&barriers[warp]));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(barrier)
                 : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    uint32_t phase = 0;
    for (int64_t offset = lo + warp * kChunkElements; offset < hi;
         offset += kStreams * kChunkElements) {
      const int64_t remaining = hi - offset;
      const uint32_t bytes = static_cast<uint32_t>(
          (remaining < kChunkElements ? remaining : kChunkElements) * 16);
      asm volatile(
          "{ .reg .b64 state; "
          "mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1; }" ::"r"(
              barrier),
          "r"(bytes)
          : "memory");
      asm volatile(
          "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
          "[%0], [%1], %2, [%3];" ::"r"(buffer),
          "l"(src + offset), "r"(bytes), "r"(barrier)
          : "memory");
      uint32_t complete;
      do {
        asm volatile(
            "{ .reg .pred ready; "
            "mbarrier.try_wait.parity.shared::cta.b64 ready, [%1], %2; "
            "selp.u32 %0, 1, 0, ready; }"
            : "=r"(complete)
            : "r"(barrier), "r"(phase)
            : "memory");
      } while (!complete);
      phase ^= 1;
      asm volatile(
          "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(
              dst + offset),
          "r"(buffer), "r"(bytes)
          : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
      // Keep at most one remote write outstanding while loading the next tile.
      asm volatile("cp.async.bulk.wait_group 1;" ::: "memory");
    }
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(barrier)
                 : "memory");
  }
  __syncthreads();
}

// Prefetch two tiles per stream. All configurations use 32 KiB of buffers;
// streams and tile size vary independently of the CTA's metadata threads.
template <int kStreams, int kChunkBytes>
__device__ __forceinline__ void RaggedAllToAllPipelinedBulkCopy(
    const DeviceVec<16>* src, DeviceVec<16>* dst, int64_t lo, int64_t hi) {
  constexpr int kStages = 2;
  constexpr int kChunkElements = kChunkBytes / 16;
  static_assert(kStreams * kStages * kChunkBytes == 32768);
  __shared__ __align__(16) uint8_t buffers[kStreams][kStages][kChunkBytes];
  __shared__ __align__(8) uint64_t barriers[kStreams][kStages];
  const int warp = threadIdx.x / 32;
  if (threadIdx.x % 32 == 0 && warp < kStreams) {
    uint32_t buffer[kStages];
    uint32_t barrier[kStages];
    uint32_t phase[kStages] = {};
#pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
      buffer[stage] =
          static_cast<uint32_t>(__cvta_generic_to_shared(buffers[warp][stage]));
      barrier[stage] = static_cast<uint32_t>(
          __cvta_generic_to_shared(&barriers[warp][stage]));
      asm volatile(
          "mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(barrier[stage])
          : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    const int64_t stride = kStreams * kChunkElements;
    auto prefetch = [&](int stage, int64_t offset) {
      if (offset >= hi) return;
      const int64_t remaining = hi - offset;
      const uint32_t bytes = static_cast<uint32_t>(
          (remaining < kChunkElements ? remaining : kChunkElements) * 16);
      asm volatile(
          "{ .reg .b64 state; "
          "mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1; }" ::"r"(
              barrier[stage]),
          "r"(bytes)
          : "memory");
      asm volatile(
          "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
          "[%0], [%1], %2, [%3];" ::"r"(buffer[stage]),
          "l"(src + offset), "r"(bytes), "r"(barrier[stage])
          : "memory");
    };
    const int64_t begin = lo + warp * kChunkElements;
#pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
      prefetch(stage, begin + stage * stride);
    }
    for (int64_t base = begin; base < hi; base += kStages * stride) {
#pragma unroll
      for (int stage = 0; stage < kStages; ++stage) {
        const int64_t offset = base + stage * stride;
        if (offset >= hi) break;
        uint32_t complete;
        do {
          asm volatile(
              "{ .reg .pred ready; "
              "mbarrier.try_wait.parity.shared::cta.b64 ready, [%1], %2; "
              "selp.u32 %0, 1, 0, ready; }"
              : "=r"(complete)
              : "r"(barrier[stage]), "r"(phase[stage])
              : "memory");
        } while (!complete);
        phase[stage] ^= 1;
        const int64_t remaining = hi - offset;
        const uint32_t bytes = static_cast<uint32_t>(
            (remaining < kChunkElements ? remaining : kChunkElements) * 16);
        asm volatile(
            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(
                dst + offset),
            "r"(buffer[stage]), "r"(bytes)
            : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        // Reuse this buffer only after the outgoing copy has read it. The next
        // stage's source load can complete while this stage is being forwarded.
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 1;" ::: "memory");
        prefetch(stage, offset + kStages * stride);
      }
    }
    // No prefetch extends past hi, so consuming all tiles also drains every
    // mbarrier transaction. Complete remote stores before the NCCL release.
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
#pragma unroll
    for (int index = 0; index < kStages; ++index) {
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(barrier[index])
                   : "memory");
    }
  }
  __syncthreads();
}
#endif  // __CUDA_ARCH__ >= 900

// Policies: 0/1 use the original vector loop, 2/3 unroll four vectors, 4/5
// unroll eight vectors, and 6/7 use bulk copies for 16-byte vectors. Policies
// 8/9, 10/11, and 12/13 prefetch two tiles with 4/2/1 streams and 4/8/16 KiB
// tiles, respectively. Odd policies visit peers relative to the sender, self
// last.
template <int64_t kVectorSize, int kCopyPolicy>
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
    // distributed across updates. Every CTA scans the update-size array,
    // then walks updates in peer-major order to locate its range.
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
    for (int64_t update = 0; update < num_lsa_updates && update_begin < cta_end;
         ++update) {
      int64_t metadata_update = update;
      if constexpr (kCopyPolicy % 2 == 1) {
        const int64_t peer = update / num_updates_per_replica;
        const int64_t slot = update % num_updates_per_replica;
        const int64_t rotated_peer =
            (peer + world.rank - start_lsa + 1) % lsa_size;
        metadata_update = rotated_peer * num_updates_per_replica + slot;
      }
      RaggedAllToAllUpdateMetadata<kVectorSize> meta;
      if (!LoadRaggedAllToAllUpdateMetadata<kVectorSize>(
              meta_base + metadata_update, num_updates_per_replica,
              num_row_elements, input_buffer_offset_bytes,
              output_buffer_offset_bytes, input_offsets_ptr, send_sizes_ptr,
              output_offsets_ptr, &meta)) {
        continue;
      }
      const int64_t update_end = update_begin + meta.byte_count / kVectorSize;
      if (update_end > cta_begin) {
        const int64_t lo =
            cta_begin > update_begin ? cta_begin - update_begin : 0;
        const int64_t hi =
            (cta_end < update_end ? cta_end : update_end) - update_begin;
        const int lsa_peer =
            static_cast<int>(metadata_update / num_updates_per_replica);
        const T* src = static_cast<const T*>(
            ncclGetLocalPointer(send_win, meta.src_byte_offset));
        T* dst = static_cast<T*>(
            ncclGetLsaPointer(recv_win, meta.dst_byte_offset, lsa_peer));
#if __CUDA_ARCH__ >= 900
        if constexpr (kCopyPolicy >= 6 && kVectorSize == 16) {
          if ((reinterpret_cast<uintptr_t>(src) % 16 == 0) &&
              (reinterpret_cast<uintptr_t>(dst) % 16 == 0)) {
            if constexpr (kCopyPolicy < 8) {
              RaggedAllToAllBulkCopy(src, dst, lo, hi);
            } else if constexpr (kCopyPolicy < 10) {
              RaggedAllToAllPipelinedBulkCopy<4, 4096>(src, dst, lo, hi);
            } else if constexpr (kCopyPolicy < 12) {
              RaggedAllToAllPipelinedBulkCopy<2, 8192>(src, dst, lo, hi);
            } else {
              RaggedAllToAllPipelinedBulkCopy<1, 16384>(src, dst, lo, hi);
            }
            update_begin = update_end;
            continue;
          }
        }
#endif
        if constexpr (kCopyPolicy < 2) {
          for (int64_t i = lo + static_cast<int64_t>(threadIdx.x); i < hi;
               i += static_cast<int64_t>(blockDim.x)) {
            dst[i] = src[i];
          }
        } else {
          // Separate independent loads from stores, as in NCCL tests' optimized
          // NVLink all-to-all. Keep the original copy for the partial tail.
          constexpr int kUnroll = kCopyPolicy < 4 ? 4 : 8;
          const int64_t stride = static_cast<int64_t>(blockDim.x);
          int64_t i = lo + static_cast<int64_t>(threadIdx.x);
          for (; i + (kUnroll - 1) * stride < hi; i += kUnroll * stride) {
            T values[kUnroll];
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
              values[u] = src[i + u * stride];
            }
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
              dst[i + u * stride] = values[u];
            }
          }
          for (; i < hi; i += stride) {
            dst[i] = src[i];
          }
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

// Compile each width with matching launch bounds.
template <int64_t kVectorSize, int kThreadsPerCta, int kCopyPolicy>
__global__ void __launch_bounds__(kThreadsPerCta, 1)
    RaggedAllToAllDeviceKernelImpl(
        struct ncclDevComm dev_comm, ncclWindow_t send_win,
        ncclWindow_t recv_win, const int64_t* __restrict__ input_offsets_ptr,
        const int64_t* __restrict__ send_sizes_ptr,
        const int64_t* __restrict__ output_offsets_ptr,
        int64_t num_updates_per_replica, int64_t num_row_elements,
        int64_t input_buffer_offset_bytes, int64_t output_buffer_offset_bytes) {
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

  if (has_remote_peers) {
    const int gin_context = 0;
    const unsigned int signal_index = 0;

    ncclGin gin{dev_comm, gin_context};
    uint64_t signal_value =
        (blockIdx.x == 0) ? gin.readSignal(signal_index) : 0;

    ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld(), gin,
                                        blockIdx.x};
    bar.sync(ncclCoopCta(), ::cuda::memory_order_acquire,
             ncclGinFenceLevel::Relaxed);

    RaggedAllToAllCopy<kVectorSize, kCopyPolicy>(
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
    bar.sync(ncclCoopCta(), ::cuda::memory_order_release,
             ncclGinFenceLevel::Relaxed);
  } else {
    ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), dev_comm,
                                           ncclTeamTagLsa{}, blockIdx.x};
    bar.sync(ncclCoopCta(), ::cuda::memory_order_relaxed);

    RaggedAllToAllCopy<kVectorSize, kCopyPolicy>(
        send_win, recv_win, input_offsets_ptr, send_sizes_ptr,
        output_offsets_ptr, num_updates_per_replica, num_row_elements,
        input_buffer_offset_bytes, output_buffer_offset_bytes, start_lsa,
        lsa_size, num_ranks, /*gin=*/nullptr, world, /*signal_index=*/0);

    bar.sync(ncclCoopCta(), ::cuda::memory_order_release);
  }
#endif  // __CUDA_ARCH__ >= 600
}

#else  // NCCL_VERSION_CODE < 22900

template <int64_t kVectorSize, int kThreadsPerCta, int kCopyPolicy>
__global__ void RaggedAllToAllDeviceKernelImpl(
    void* dev_comm, void* send_win, void* recv_win,
    const int64_t* input_offsets_ptr, const int64_t* send_sizes_ptr,
    const int64_t* output_offsets_ptr, int64_t num_updates_per_replica,
    int64_t num_row_elements, int64_t input_buffer_offset_bytes,
    int64_t output_buffer_offset_bytes) {}

#endif  // NCCL_VERSION_CODE >= 22900

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_LIB_CU_H_
