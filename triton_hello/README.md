# triton_hello

Writing custom GPU kernels in [Triton](https://triton-lang.org/) and running
them on PyTorch CUDA tensors.

Triton is the highest-leverage way to write custom GPU code for modern AI
work: the kernel is plain Python, it JIT-compiles to PTX on first call, it
operates directly on PyTorch tensors, and it integrates with `torch.compile`.
A large and growing share of research kernels (FlashAttention reference
implementations, Liger Kernel, Mamba, and others) are written in Triton.

There is no `nvcc` step and no C++ in this project. That contrast with
[`cuda_cpp_hello`](../cuda_cpp_hello) and
[`torch_cuda_extension_hello`](../torch_cuda_extension_hello) is the point:
same kind of compute, very different authoring experience.

## What's here

- `triton_hello/kernels.py`
  - `vector_add` — elementwise `c = a + b`, the GPU "hello world".
  - `softmax` — a numerically-stable row-wise softmax. This is the canonical
    Triton teaching kernel and a real operation used in attention and
    classification heads.
- `main.py` — runs both kernels and checks them against the PyTorch
  references (`a + b` and `torch.softmax`).

## Prerequisites

- An NVIDIA GPU with a recent driver.
- Python 3 with `venv`.

PyTorch (which bundles a matching Triton on Linux) is installed into a local
`.venv` by the Makefile. Developed against an A10G (compute capability 8.6),
driver 580, CUDA 13 runtime.

## Build and run

```bash
make venv    # create .venv and install torch (+ triton)
make run     # run main.py: executes the kernels and validates them
make clean   # remove .venv and caches
```

Expected output (error magnitudes will vary slightly run to run):

```
device: NVIDIA A10G
vector_add: n=1048576, max_abs_err=0.00e+00 -> OK
softmax: shape=(1823,781), max_abs_err=5.96e-08 -> OK
OK
```

The first `make run` is slow: it downloads PyTorch and JIT-compiles the
kernels. Subsequent runs reuse the cached compilation.

## Layout

```
triton_hello/
├── Makefile
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                  # driver + validation against PyTorch
└── triton_hello/
    ├── __init__.py
    └── kernels.py           # @triton.jit kernels + Python wrappers
```

## Notes on the kernels

- A Triton kernel is a `@triton.jit` Python function. Inside it you work with
  `tl.program_id`, `tl.arange`, `tl.load`/`tl.store`, and masks rather than
  raw thread indices. Triton handles the intra-block parallelism for you.
- The launch grid is a Python tuple or a `lambda meta: (...)` that can read
  the `constexpr` values (like `BLOCK_SIZE`) Triton selects.
- The softmax kernel assigns one program per row and sizes `BLOCK_SIZE` to
  the next power of two `>= n_cols`, so a whole row fits in one block. That
  is why it handles non-power-of-two column counts via masking.
