# gpu_memory_movement_hello: measure local host<->device memory bandwidth and
# show why pinned (page-locked) memory matters for fast transfers and overlap.
#
# Single translation unit compiled and linked with nvcc.

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc

# L4 is sm_89. Override on the command line for other GPUs, e.g.
#   make SM_ARCH=sm_80   # A100
#   make SM_ARCH=sm_90   # H100
SM_ARCH ?= sm_89

NVCCFLAGS ?= -O2 -std=c++17 -arch=$(SM_ARCH) -Isrc

BUILD_DIR := build
TARGET    := $(BUILD_DIR)/bandwidth

# Transfer size in MiB for `make run`. Override: make run SIZE_MIB=512
SIZE_MIB ?= 256

.PHONY: all run clean

all: $(TARGET)

$(TARGET): src/bandwidth.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: $(TARGET)
	$(TARGET) $(SIZE_MIB)

clean:
	rm -rf $(BUILD_DIR)
