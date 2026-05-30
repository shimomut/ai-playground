"""Driver: run the CUDA C++ extension kernel and check it against PyTorch."""

from __future__ import annotations

import sys

import torch

from torch_ext_hello import vector_add


def check(dtype: torch.dtype) -> bool:
    n = 1 << 20
    a = torch.rand(n, device="cuda", dtype=dtype)
    b = torch.rand(n, device="cuda", dtype=dtype)

    c = vector_add(a, b)
    ref = a + b

    # half/bfloat16 accumulate more error; loosen tolerance accordingly.
    atol = 1e-6 if dtype == torch.float32 else 1e-2
    ok = torch.allclose(c, ref, atol=atol)
    max_err = (c.float() - ref.float()).abs().max().item()
    print(f"vector_add[{str(dtype):>14}]: max_abs_err={max_err:.2e} -> {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available to PyTorch", file=sys.stderr)
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")

    ok = True
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        ok &= check(dtype)

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
