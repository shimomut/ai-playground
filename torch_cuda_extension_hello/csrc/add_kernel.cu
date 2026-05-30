// CUDA C++ kernel wrapped as a PyTorch extension.
//
// This is the path nearly every hand-written production AI kernel follows:
// write the kernel in CUDA C++, accept and return torch::Tensor, and let
// PyTorch own memory, devices, streams, and dtype dispatch.
//
// The function is registered with PyTorch via PYBIND11_MODULE / TORCH_LIBRARY
// at the bottom. Here we use the simple pybind11 module form, which is what
// torch.utils.cpp_extension.load expects.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__global__ void vector_add_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int64_t n) {
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

}  // namespace

// c = a + b, elementwise. Inputs must be CUDA tensors of the same shape.
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have the same dtype");

    // Make inputs contiguous and operate on the flattened view.
    a = a.contiguous();
    b = b.contiguous();
    auto c = torch::empty_like(a);

    const int64_t n = a.numel();
    if (n == 0) {
        return c;
    }

    // Ensure kernels launch on the input's device, and use the current
    // stream PyTorch has set up so this composes with the rest of a model.
    const at::cuda::CUDAGuard device_guard(a.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int kThreadsPerBlock = 256;
    const int blocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        a.scalar_type(), "vector_add", [&] {
            vector_add_kernel<scalar_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                c.data_ptr<scalar_t>(),
                n);
        });

    // Surface launch errors as a Python exception instead of a silent crash.
    C10_CUDA_CHECK(cudaGetLastError());
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "Elementwise a + b (CUDA)");
}
