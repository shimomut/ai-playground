# Triton "hello world": custom GPU kernels written in Python, run on PyTorch
# CUDA tensors. No nvcc, no C++ build step -- Triton JIT-compiles to PTX on
# first call.

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

.PHONY: all venv run clean

all: venv

venv: $(VENV_STAMP)

$(VENV_STAMP): requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt
	touch $@

run: $(VENV_STAMP)
	$(VENV_PY) main.py

clean:
	rm -rf $(VENV) .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
