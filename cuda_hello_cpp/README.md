# cuda_hello_cpp

Minimal example of calling a CUDA kernel from a plain C++ host program.

The point of this experiment is to keep the C++ ↔ CUDA boundary as explicit
as possible:

- `src/main.cpp` is compiled with `g++`. It does **not** include any CUDA
  headers. It only sees a plain C++ function declaration.
- `src/add_kernel.cu` is compiled with `nvcc`. It owns the kernel
  (`__global__ void vector_add_kernel`) and a host-callable wrapper
  (`launch_vector_add`) that handles `cudaMalloc` / `cudaMemcpy` / launch /
  copy-back.
- `src/add_kernel.cuh` is the plain C++ header that bridges the two.

The kernel itself is a textbook elementwise vector add: `c[i] = a[i] + b[i]`.

A sibling project for a Python-driven main program will live in its own
top-level directory (e.g. `cuda_hello_python/`).

## Prerequisites

- An NVIDIA GPU with a recent driver.
- CUDA Toolkit installed at `/usr/local/cuda` (override with `CUDA_HOME`).
- `g++` with C++17 support.

This was developed against CUDA 12.9 on an A10G (compute capability 8.6).

## Build and run

```bash
make            # builds build/cuda_hello
make run        # builds and runs
make clean      # removes build/
```

To target a different GPU architecture, override `SM_ARCH`:

```bash
make SM_ARCH=sm_80     # A100
make SM_ARCH=sm_90     # H100
```

To use a non-default CUDA install:

```bash
make CUDA_HOME=/opt/cuda-12.8
```

Expected output:

```
vector_add: n=1048576, c[0]=0.0, c[1]=3.0, c[1048575]=3145725.0
OK
```

## Layout

```
cuda_hello_cpp/
├── Makefile
├── README.md
├── .gitignore
└── src/
    ├── add_kernel.cu      # __global__ kernel + host-side launcher
    ├── add_kernel.cuh     # plain C++ declaration shared by host and device TUs
    └── main.cpp           # plain C++ host program, no CUDA headers
```

## How the build is wired

1. `g++` compiles `main.cpp` into `build/main.o`. It only needs
   `add_kernel.cuh`, which is plain C++.
2. `nvcc` compiles `add_kernel.cu` into `build/add_kernel.o`.
3. `nvcc` links both object files into the final executable, pulling in
   `libcudart` so the CUDA runtime calls from the `.cu` translation unit
   resolve.

This split is what a larger project would look like: a small CUDA "shim"
exposing C++-friendly entry points, with the rest of the codebase blissfully
unaware of CUDA.
