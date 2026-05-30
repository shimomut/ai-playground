"""Driver: run the Triton kernels and check them against PyTorch references."""

from __future__ import annotations

import sys

import torch

from triton_hello import softmax, vector_add


def check_vector_add() -> bool:
    n = 1 << 20
    a = torch.rand(n, device="cuda")
    b = torch.rand(n, device="cuda")

    c = vector_add(a, b)
    ref = a + b
    ok = torch.allclose(c, ref, atol=1e-6)
    max_err = (c - ref).abs().max().item()
    print(f"vector_add: n={n}, max_abs_err={max_err:.2e} -> {'OK' if ok else 'FAIL'}")
    return ok


def check_softmax() -> bool:
    n_rows, n_cols = 1823, 781  # deliberately non-power-of-two
    x = torch.randn(n_rows, n_cols, device="cuda")

    y = softmax(x)
    ref = torch.softmax(x, dim=1)
    ok = torch.allclose(y, ref, atol=1e-6)
    max_err = (y - ref).abs().max().item()
    print(
        f"softmax: shape=({n_rows},{n_cols}), max_abs_err={max_err:.2e} "
        f"-> {'OK' if ok else 'FAIL'}"
    )
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available to PyTorch", file=sys.stderr)
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")

    ok = check_vector_add()
    ok &= check_softmax()

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
