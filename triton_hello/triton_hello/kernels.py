"""Triton kernels and thin Python wrappers.

Triton lets you write GPU kernels in Python. Each kernel is decorated with
@triton.jit and compiled to PTX on first call. Kernels operate on raw
pointers (PyTorch tensors are passed directly and decay to their data
pointer), and the launch grid is a normal Python callable/tuple.

Two kernels here:

- add_kernel:     elementwise c = a + b, the "hello world" of GPU code.
- softmax_kernel: a numerically-stable row-wise softmax, the canonical
                  Triton teaching example and a real operation used in
                  attention and classification heads.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# Elementwise add
# --------------------------------------------------------------------------
@triton.jit
def add_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "inputs must be CUDA tensors"
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty_like(a)
    n = a.numel()

    # 1-D launch grid: one program per BLOCK_SIZE chunk. `meta` carries the
    # constexpr values Triton picks, so the grid adapts to BLOCK_SIZE.
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](a, b, c, n, BLOCK_SIZE=1024)
    return c


# --------------------------------------------------------------------------
# Row-wise softmax
# --------------------------------------------------------------------------
@triton.jit
def softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # One program handles one row. BLOCK_SIZE is the next power of two >=
    # n_cols, so an entire row fits in a single block of threads.
    row_idx = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    in_row = in_ptr + row_idx * in_row_stride + col_offsets
    # Masked-out lanes read -inf so they never win the max and contribute 0
    # to the sum after exp.
    row = tl.load(in_row, mask=mask, other=-float("inf"))

    # Numerically stable softmax: subtract the row max before exp.
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    out_row = out_ptr + row_idx * out_row_stride + col_offsets
    tl.store(out_row, softmax_out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the last dim of a 2-D tensor."""
    assert x.is_cuda, "input must be a CUDA tensor"
    assert x.ndim == 2, f"expected 2-D input, got {x.ndim}-D"
    x = x.contiguous()
    n_rows, n_cols = x.shape

    block_size = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)

    # One program per row.
    softmax_kernel[(n_rows,)](
        out, x,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
    )
    return out
