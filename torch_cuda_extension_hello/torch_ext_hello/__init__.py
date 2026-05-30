"""JIT-loaded PyTorch CUDA extension wrapper.

This package exposes `vector_add`, backed by the CUDA C++ kernel in
csrc/add_kernel.cu. By default the kernel is compiled on first import using
torch.utils.cpp_extension.load (the "JIT" path) -- no separate build step is
required. If the extension was instead built ahead of time with setup.py
(producing an importable `torch_ext_hello_cuda` module), that compiled module
is used directly.
"""

from __future__ import annotations

from pathlib import Path

import torch

_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "add_kernel.cu"


def _load_extension():
    # Prefer an ahead-of-time built module if present (from `make build`).
    try:
        import torch_ext_hello_cuda as ext  # type: ignore
        return ext
    except ImportError:
        pass

    # Otherwise JIT-compile from source. The first call is slow (it invokes
    # nvcc and caches the result under TORCH_EXTENSIONS_DIR); later calls are
    # fast.
    from torch.utils.cpp_extension import load

    if not _CSRC.exists():
        raise FileNotFoundError(f"CUDA source not found: {_CSRC}")

    return load(
        name="torch_ext_hello_cuda",
        sources=[str(_CSRC)],
        verbose=True,
    )


_EXT = _load_extension()


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise a + b computed by the CUDA C++ kernel."""
    return _EXT.vector_add(a, b)


__all__ = ["vector_add"]
