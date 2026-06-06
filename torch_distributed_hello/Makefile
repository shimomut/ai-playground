# torch_distributed_hello: torch.distributed collectives + a DDP step, launched
# with torchrun.
#
#   make venv      create .venv and install torch
#   make run       gloo (CPU) backend with NPROC ranks -> real multi-rank
#                  collectives even on a single-GPU box
#   make run-gpu   nccl backend on the GPU(s); NPROC defaults to the GPU count
#   make clean

VENV       := .venv
VENV_PY    := $(VENV)/bin/python
TORCHRUN   := $(VENV)/bin/torchrun
VENV_STAMP := $(VENV)/.installed

# Pick the latest python3.X interpreter on PATH, falling back to python3.
PYTHON ?= $(shell \
	ls /usr/bin/python3.[0-9]* /usr/local/bin/python3.[0-9]* 2>/dev/null \
		| grep -E '/python3\.[0-9]+$$' \
		| sort -V \
		| tail -n1 \
		| grep . \
		|| command -v python3)

# Ranks for the gloo (CPU) demo. Override: make run NPROC=8
NPROC ?= 4

.PHONY: all venv run run-gpu clean

all: venv

venv: $(VENV_STAMP)

$(VENV_STAMP): requirements.txt
	$(PYTHON) -m venv $(VENV) 2>/dev/null || virtualenv -p $(PYTHON) $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt
	touch $@

# Multi-rank collectives + DDP on CPU via gloo. Runs real cross-rank traffic
# regardless of how many GPUs are present.
run: $(VENV_STAMP)
	BACKEND=gloo $(TORCHRUN) --standalone --nnodes=1 --nproc_per_node=$(NPROC) main.py

# NCCL on the GPU(s): one rank per visible GPU (so 1 rank on a single-GPU box).
run-gpu: $(VENV_STAMP)
	BACKEND=nccl $(TORCHRUN) --standalone --nnodes=1 \
		--nproc_per_node=$$(nvidia-smi -L | wc -l) main.py

clean:
	rm -rf $(VENV) .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
