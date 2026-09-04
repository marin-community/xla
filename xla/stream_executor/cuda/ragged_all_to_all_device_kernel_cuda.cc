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

#include "absl/base/casts.h"
#include "third_party/nccl/nccl.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_device_kernel.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_device_kernel_lib.cu.h"

#define SINGLE_ARG(...) __VA_ARGS__

// BENCH BRANCH ONLY. Both launch geometries are registered so the thunk can
// pick one per run. NAME must differ per (vector size, geometry) pair: the
// registry keys on the trait type, and the loader spec keys on the symbol name.
#define REGISTER_RAGGED_ALL_TO_ALL_DEVICE_KERNEL(NAME, VECTOR_SIZE, THREADS,  \
                                                 CTAS_PER_SM)                 \
  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                             \
      RaggedAllToAllDeviceKernelCuda##NAME,                                   \
      SINGLE_ARG(stream_executor::gpu::RaggedAllToAllDeviceKernel<            \
                 VECTOR_SIZE, THREADS, CTAS_PER_SM>),                         \
      stream_executor::cuda::kCudaPlatformId, ([](size_t arity) {             \
        return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(  \
            absl::bit_cast<void*>(&SINGLE_ARG(                                \
                stream_executor::gpu::RaggedAllToAllDeviceKernelImpl<         \
                    VECTOR_SIZE, THREADS, CTAS_PER_SM>)),                     \
            "ragged_all_to_all_device_kernel_" #NAME, arity);                 \
      }));

#define REGISTER_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_GEOMETRIES(VECTOR_SIZE)      \
  REGISTER_RAGGED_ALL_TO_ALL_DEVICE_KERNEL(                                   \
      VECTOR_SIZE##BytesNarrow, VECTOR_SIZE,                                  \
      stream_executor::gpu::kRaggedAllToAllDeviceKernelThreadsPerCta,         \
      stream_executor::gpu::kRaggedAllToAllDeviceKernelCtasPerSm)             \
  REGISTER_RAGGED_ALL_TO_ALL_DEVICE_KERNEL(                                   \
      VECTOR_SIZE##BytesStock, VECTOR_SIZE,                                   \
      stream_executor::gpu::kRaggedAllToAllStockThreadsPerCta,                \
      stream_executor::gpu::kRaggedAllToAllStockCtasPerSm)                    \
  REGISTER_RAGGED_ALL_TO_ALL_DEVICE_KERNEL(                                   \
      VECTOR_SIZE##BytesUncapped, VECTOR_SIZE,                                \
      stream_executor::gpu::kRaggedAllToAllDeviceKernelThreadsPerCta,         \
      stream_executor::gpu::kRaggedAllToAllUncappedCtasPerSm)

REGISTER_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_GEOMETRIES(1);
REGISTER_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_GEOMETRIES(2);
REGISTER_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_GEOMETRIES(4);
REGISTER_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_GEOMETRIES(8);
REGISTER_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_GEOMETRIES(16);
