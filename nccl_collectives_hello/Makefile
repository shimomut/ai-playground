# nccl_collectives_hello: the four core collectives (AllReduce, Broadcast,
# AllGather, ReduceScatter) against the NCCL C API, one process per rank.
#
#   make             build build/nccl_collectives
#   make run         launch WORLD_SIZE ranks (default 2) on this node
#   make run-verbose same, with NCCL_DEBUG=INFO (topology + transport logs)
#   make clean
#
# NCCL ships inside the CUDA install here (nccl.h + libnccl in $(CUDA_HOME)).
# If you use a standalone NCCL, point NCCL_HOME at it.

CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= $(CUDA_HOME)
NVCC      ?= $(CUDA_HOME)/bin/nvcc

# L4 is sm_89. Override for other GPUs, e.g. make SM_ARCH=sm_90 (H100).
SM_ARCH ?= sm_89

NVCCFLAGS ?= -O2 -std=c++17 -arch=$(SM_ARCH) -I$(NCCL_HOME)/include
# nvcc forwards linker flags via -Xlinker (it rejects bare -Wl,...).
LDFLAGS   ?= -L$(NCCL_HOME)/lib -L$(NCCL_HOME)/lib64 \
             -Xlinker -rpath -Xlinker $(NCCL_HOME)/lib \
             -Xlinker -rpath -Xlinker $(NCCL_HOME)/lib64
LDLIBS    ?= -lnccl

BUILD_DIR := build
TARGET    := $(BUILD_DIR)/nccl_collectives

# Number of ranks for `make run`. If unset, run.sh defaults to the number of
# visible GPUs (NCCL needs one GPU per rank). Override: make run WORLD_SIZE=4
WORLD_SIZE ?=

.PHONY: all run run-verbose clean

all: $(TARGET)

$(TARGET): src/collectives.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: $(TARGET)
	$(if $(WORLD_SIZE),WORLD_SIZE=$(WORLD_SIZE) )./run.sh

run-verbose: $(TARGET)
	$(if $(WORLD_SIZE),WORLD_SIZE=$(WORLD_SIZE) )NCCL_DEBUG=INFO ./run.sh

clean:
	rm -rf $(BUILD_DIR)
