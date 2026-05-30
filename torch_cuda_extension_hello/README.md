# torch_cuda_extension_hello

A CUDA C++ kernel wrapped as a PyTorch extension so it accepts and returns
`torch.Tensor`.

This is the path that nearly every hand-written production AI kernel follows
(FlashAttention, Apex, vLLM, Megatron, xFormers, ...). You write the kernel in
CUDA C++ and let PyTorch own memory, device placement, streams, dtype
dispatch, and error propagation. Compared to the raw C-ABI boundary in
[`cuda_cpp_hello`](../cuda_cpp_hello), the win is integration: the kernel
operates on live GPU tensors with no host round-trip and raises real Python
exceptions on failure.

Compared to [`triton_hello`](../triton_hello), this is the heavier but more
powerful path: you get full CUDA C++ (tensor cores, custom layouts, async
copies, CUTLASS interop) at the cost of writing and building C++.

## What's here

- `csrc/add_kernel.cu` — a templated `vector_add` kernel plus a
  `torch::Tensor vector_add(...)` wrapper. It uses `AT_DISPATCH_FLOATING_TYPES_AND2`
  (covering float/double plus half and bfloat16) for dtype dispatch,
  `CUDAGuard` + the current stream for correct device and stream behavior, and
  `TORCH_CHECK` / `C10_CUDA_CHECK` to raise Python exceptions on bad input or
  launch errors.
- `torch_ext_hello/__init__.py` — loads the extension. It prefers an
  ahead-of-time built module if one exists, otherwise JIT-compiles the `.cu`
  on first import.
- `setup.py` — ahead-of-time build using `CUDAExtension`, for the AOT path.
- `main.py` — runs the kernel for float32/float16/bfloat16 and checks each
  against `a + b`.

## Two build paths

**JIT (default, simplest).** The first import calls
`torch.utils.cpp_extension.load`, which invokes `nvcc`, builds the extension,
and caches it under `~/.cache/torch_extensions`. No `setup.py` involved.

```bash
make venv     # create .venv, install torch
make run      # first run compiles the kernel (slow), then runs main.py
```

**AOT (ahead-of-time).** Build the extension as a real installed module. The
package then imports the prebuilt module instead of JIT-compiling. This is how
shipped libraries work.

```bash
make build    # pip install -e . -> builds torch_ext_hello_cuda
make run
```

## Prerequisites

- An NVIDIA GPU with a recent driver.
- A CUDA Toolkit whose **major version matches the PyTorch CUDA build** is
  needed to compile the extension. The current PyPI `torch` is a CUDA 13
  build, so the Makefile defaults `CUDA_HOME` to `/usr/local/cuda-13.0`.
  Override it if your torch wheel targets a different major version, e.g.
  `make run CUDA_HOME=/usr/local/cuda-12.6`. The Makefile puts
  `$(CUDA_HOME)/bin` on `PATH` for its recipes so `nvcc` is found.
- Python 3 with `venv`.

Developed against the CUDA 13.0 toolkit / driver 580, with PyTorch 2.12
(CUDA 13 build), on an A10G (compute capability 8.6).

Expected output (half/bfloat16 error is larger by design):

```
device: NVIDIA A10G
vector_add[ torch.float32]: max_abs_err=0.00e+00 -> OK
vector_add[ torch.float16]: max_abs_err=...      -> OK
vector_add[torch.bfloat16]: max_abs_err=...      -> OK
OK
```

## Layout

```
torch_cuda_extension_hello/
├── Makefile
├── README.md
├── requirements.txt
├── setup.py                 # AOT build (CUDAExtension)
├── .gitignore
├── csrc/
│   └── add_kernel.cu        # CUDA kernel + torch::Tensor wrapper + pybind
├── main.py                  # driver + validation across dtypes
└── torch_ext_hello/
    └── __init__.py          # loads extension (prefers AOT, else JIT)
```

## Why `nvcc` builds the extension

Same reason as in `cuda_cpp_hello`: the `.cu` file contains device code that
only `nvcc` can compile, and PyTorch's build helpers drive `nvcc` for the
device side while using the host compiler for the C++ glue, linking against
`libtorch` and the CUDA runtime. `torch.utils.cpp_extension` wraps all of
that so you don't write the compile/link lines by hand.
