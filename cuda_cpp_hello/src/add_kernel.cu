// CUDA translation unit. Compiled by nvcc.
//
// Implements launch_vector_add() declared in add_kernel.cuh. This file is
// the only place that touches the CUDA runtime API or __global__ code, so
// the host side stays a plain C++ program.

#include "add_kernel.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace {

// Minimal error-check helper. Aborts on the first CUDA error so the example
// stays short; real code would surface errors back to the caller instead.
inline void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Demonstrates device-side printf. Every thread that reaches a printf runs
// it, so a real launch can flood stdout; here we guard on the global index
// to print from just the first few threads. Note also that output ordering
// between threads is not guaranteed, even though each individual printf call
// is atomic.
__global__ void printf_demo_kernel() {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Print from the first 4 threads only to keep the output short.
    if (idx < 4) {
        printf("hello from global thread %d (block %d, thread %d)\n",
               idx, blockIdx.x, threadIdx.x);
    }

    // Have a single thread print the launch geometry as well.
    if (idx == 0) {
        printf("launch geometry: gridDim.x=%d, blockDim.x=%d\n",
               gridDim.x, blockDim.x);
    }
}

}  // namespace

void launch_vector_add(const float* a, const float* b, float* c, int n) {
    if (n <= 0) {
        return;
    }

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    check(cudaMalloc(&d_a, bytes), "cudaMalloc(d_a)");
    check(cudaMalloc(&d_b, bytes), "cudaMalloc(d_b)");
    check(cudaMalloc(&d_c, bytes), "cudaMalloc(d_c)");

    check(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice), "cudaMemcpy(H2D a)");
    check(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice), "cudaMemcpy(H2D b)");

    constexpr int kThreadsPerBlock = 256;
    const int blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
    vector_add_kernel<<<blocks, kThreadsPerBlock>>>(d_a, d_b, d_c, n);
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H c)");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void launch_printf_demo() {
    // Two blocks of four threads = 8 threads total.
    printf_demo_kernel<<<2, 4>>>();
    check(cudaGetLastError(), "printf_demo kernel launch");

    // Device printf output is buffered and only flushed to the host stdout at
    // synchronization points (here, the explicit sync). Without this, the
    // program could exit before any output appears.
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(printf_demo)");
}
