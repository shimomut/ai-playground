# PyTorch CUDA extension "hello world": a CUDA C++ kernel wrapped so it
# accepts and returns torch.Tensor.
#
# Two ways to build the extension:
#   make run    -> JIT path. First import compiles csrc/add_kernel.cu via
#                  torch.utils.cpp_extension.load and caches it. No setup.py.
#   make build  -> AOT path. pip install -e . builds torch_ext_hello_cuda
#                  ahead of time; the package then prefers it over JIT.

CUDA_HOME ?= /usr/local/cuda-13.0

VENV       := .venv
VENV_PY    := $(VENV)/bin/python
VENV_STAMP := $(VENV)/.installed

# Pick the latest python3.X interpreter on PATH, falling back to python3.
# Entries like python3.10-config are filtered out by the trailing-digit regex.
PYTHON ?= $(shell \
	ls /usr/bin/python3.[0-9]* /usr/local/bin/python3.[0-9]* 2>/dev/null \
		| grep -E '/python3\.[0-9]+$$' \
		| sort -V \
		| tail -n1 \
		| grep . \
		|| command -v python3)

# nvcc must be on PATH for both the JIT and AOT builds, and torch's extension
# builder shells out to `ninja`, which lives in the venv's bin dir. Put both
# on PATH for recipes so a bare `make` works without manual venv activation.
export PATH := $(CURDIR)/$(VENV)/bin:$(CUDA_HOME)/bin:$(PATH)

.PHONY: all venv build run clean

all: venv

venv: $(VENV_STAMP)

$(VENV_STAMP): requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt
	touch $@

# Optional ahead-of-time build of the extension module.
build: $(VENV_STAMP)
	$(VENV_PY) -m pip install -e .

run: $(VENV_STAMP)
	$(VENV_PY) main.py

clean:
	rm -rf $(VENV) build *.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
