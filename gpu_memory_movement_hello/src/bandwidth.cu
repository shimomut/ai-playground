// gpu_memory_movement_hello
//
// The foundation under every form of "GPU-GPU memory exchange": before bytes
// can travel between GPUs or across a network, they move between host memory,
// device memory, and DMA engines on a single node. This program measures the
// bandwidth of those local paths and shows two effects that matter for
// distributed training:
//
//   1. Pinned (page-locked) host memory is dramatically faster to copy to/from
//      the GPU than ordinary pageable memory, because the GPU's DMA engine can
//      read it directly. Pageable memory must first be staged through a
//      pinned bounce buffer by the driver.
//
//   2. Pinned memory is also the precondition for *asynchronous* copies that
//      overlap with compute (and, later, for RDMA: a NIC can only DMA out of
//      memory that is pinned and registered). We demonstrate the overlap with
//      a copy/compute pipeline across CUDA streams.
//
// Everything here runs on a single GPU. The concepts (registered memory + DMA
// engines moving bytes without the CPU) are exactly what rdma_libfabric_hello
// and nccl_collectives_hello build on.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,  \
                   __LINE__, cudaGetErrorString(err__));                     \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                         \
  } while (0)

namespace {

constexpr int kIters = 50;       // timed iterations per measurement
constexpr int kWarmup = 5;       // untimed warmup iterations

// Returns the average milliseconds for a copy of `bytes` in `direction`,
// timed with CUDA events on the default stream.
float TimeCopy(void *dst, const void *src, size_t bytes,
               cudaMemcpyKind direction) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < kWarmup; ++i) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, direction));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < kIters; ++i) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, direction));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / kIters;
}

double GiBPerSec(size_t bytes, float ms) {
  const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  return gib / (ms / 1000.0);
}

// A trivial kernel that touches every element, used purely to occupy the GPU
// while a copy runs concurrently on another stream.
__global__ void ScaleKernel(float *data, size_t n, float factor) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < n; i += stride) {
    data[i] = data[i] * factor + 1.0f;
  }
}

// Demonstrates that pinned memory enables H2D copies to overlap with compute.
// We split a buffer into chunks and pipeline copy(chunk i+1) with
// kernel(chunk i) across two streams, then compare to doing it serially.
void OverlapDemo(size_t total_bytes) {
  const size_t n = total_bytes / sizeof(float);
  const int chunks = 8;
  const size_t chunk_n = n / chunks;
  const size_t chunk_bytes = chunk_n * sizeof(float);

  float *h_pinned = nullptr;
  float *d_buf = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_pinned, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_buf, n * sizeof(float)));
  for (size_t i = 0; i < n; ++i) h_pinned[i] = 1.0f;

  cudaStream_t s_copy, s_compute;
  CUDA_CHECK(cudaStreamCreate(&s_copy));
  CUDA_CHECK(cudaStreamCreate(&s_compute));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  const int threads = 256;
  const int blocks = 256;

  // Serial: copy everything, then run the kernel over everything.
  CUDA_CHECK(cudaEventRecord(start));
  CUDA_CHECK(cudaMemcpyAsync(d_buf, h_pinned, n * sizeof(float),
                             cudaMemcpyHostToDevice, s_copy));
  ScaleKernel<<<blocks, threads, 0, s_copy>>>(d_buf, n, 1.001f);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float serial_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&serial_ms, start, stop));

  // Pipelined: per chunk, copy on s_copy then compute on s_compute, using
  // events so compute waits only on its own chunk's copy. Chunk i+1's copy
  // overlaps chunk i's compute.
  std::vector<cudaEvent_t> copied(chunks);
  for (int c = 0; c < chunks; ++c) CUDA_CHECK(cudaEventCreate(&copied[c]));

  CUDA_CHECK(cudaEventRecord(start));
  for (int c = 0; c < chunks; ++c) {
    const size_t off = static_cast<size_t>(c) * chunk_n;
    CUDA_CHECK(cudaMemcpyAsync(d_buf + off, h_pinned + off, chunk_bytes,
                               cudaMemcpyHostToDevice, s_copy));
    CUDA_CHECK(cudaEventRecord(copied[c], s_copy));
    CUDA_CHECK(cudaStreamWaitEvent(s_compute, copied[c], 0));
    ScaleKernel<<<blocks, threads, 0, s_compute>>>(d_buf + off, chunk_n,
                                                    1.001f);
  }
  CUDA_CHECK(cudaEventRecord(stop, s_compute));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float pipelined_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&pipelined_ms, start, stop));

  std::printf("\nCopy/compute overlap (pinned memory, %d chunks):\n", chunks);
  std::printf("  serial    copy-then-compute : %8.3f ms\n", serial_ms);
  std::printf("  pipelined copy||compute     : %8.3f ms\n", pipelined_ms);
  std::printf("  speedup from overlap        : %8.2fx\n",
              serial_ms / pipelined_ms);

  for (int c = 0; c < chunks; ++c) CUDA_CHECK(cudaEventDestroy(copied[c]));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(s_copy));
  CUDA_CHECK(cudaStreamDestroy(s_compute));
  CUDA_CHECK(cudaFreeHost(h_pinned));
  CUDA_CHECK(cudaFree(d_buf));
}

}  // namespace

int main(int argc, char **argv) {
  size_t mib = 256;
  if (argc > 1) mib = static_cast<size_t>(std::atoll(argv[1]));
  const size_t bytes = mib * 1024 * 1024;

  int device = 0;
  CUDA_CHECK(cudaSetDevice(device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("device: %s (compute %d.%d)\n", prop.name, prop.major,
              prop.minor);
  std::printf("transfer size: %zu MiB, %d timed iters\n\n", mib, kIters);

  // Allocate the three memory kinds we want to compare.
  void *h_pageable = std::malloc(bytes);
  if (!h_pageable) {
    std::fprintf(stderr, "malloc of %zu bytes failed\n", bytes);
    return EXIT_FAILURE;
  }
  std::memset(h_pageable, 1, bytes);

  void *h_pinned = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));  // page-locked
  std::memset(h_pinned, 1, bytes);

  void *d_a = nullptr;
  void *d_b = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));

  std::printf("%-34s %12s %12s\n", "path", "ms", "GiB/s");
  std::printf("%-34s %12s %12s\n", "----", "--", "-----");

  float ms;
  ms = TimeCopy(d_a, h_pageable, bytes, cudaMemcpyHostToDevice);
  std::printf("%-34s %12.3f %12.2f\n", "H2D pageable", ms, GiBPerSec(bytes, ms));

  ms = TimeCopy(h_pageable, d_a, bytes, cudaMemcpyDeviceToHost);
  std::printf("%-34s %12.3f %12.2f\n", "D2H pageable", ms, GiBPerSec(bytes, ms));

  ms = TimeCopy(d_a, h_pinned, bytes, cudaMemcpyHostToDevice);
  const float pinned_h2d = ms;
  std::printf("%-34s %12.3f %12.2f\n", "H2D pinned", ms, GiBPerSec(bytes, ms));

  ms = TimeCopy(h_pinned, d_a, bytes, cudaMemcpyDeviceToHost);
  std::printf("%-34s %12.3f %12.2f\n", "D2H pinned", ms, GiBPerSec(bytes, ms));

  ms = TimeCopy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice);
  std::printf("%-34s %12.3f %12.2f\n", "D2D (on-device)", ms,
              GiBPerSec(bytes, ms));

  // Headline comparison: how much pinning helps the H2D path.
  float pageable_h2d = TimeCopy(d_a, h_pageable, bytes, cudaMemcpyHostToDevice);
  std::printf("\npinned vs pageable H2D speedup: %.2fx\n",
              pageable_h2d / pinned_h2d);

  OverlapDemo(bytes);

  std::free(h_pageable);
  CUDA_CHECK(cudaFreeHost(h_pinned));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));

  std::printf("\nOK\n");
  return 0;
}
