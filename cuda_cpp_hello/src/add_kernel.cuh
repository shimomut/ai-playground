#pragma once

// Plain C++ entry point exposed by the CUDA translation unit.
//
// The host code (main.cpp) intentionally knows nothing about CUDA: it does
// not include <cuda_runtime.h> and is compiled with a plain C++ compiler.
// All device memory management and kernel launches live behind this
// declaration in add_kernel.cu, which is compiled with nvcc.
//
// Computes c[i] = a[i] + b[i] for i in [0, n) on the GPU.
void launch_vector_add(const float* a, const float* b, float* c, int n);
