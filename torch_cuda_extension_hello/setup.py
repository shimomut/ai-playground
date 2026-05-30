"""Ahead-of-time (AOT) build of the CUDA extension.

This is the alternative to the JIT path in torch_ext_hello/__init__.py. Run
it with `make build` (which calls `pip install -e .`). It produces an
importable `torch_ext_hello_cuda` module, which the package will then prefer
over JIT compilation.

Production libraries (FlashAttention, etc.) ship this way: a setup.py with
CUDAExtension so users get a prebuilt wheel or a one-time compile at install.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="torch_ext_hello_cuda",
    ext_modules=[
        CUDAExtension(
            name="torch_ext_hello_cuda",
            sources=["csrc/add_kernel.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
