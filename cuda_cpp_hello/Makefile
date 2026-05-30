# Minimal CUDA-from-C++ build.
#
# Layout:
#   src/main.cpp       -> compiled with $(CXX), no CUDA headers
#   src/add_kernel.cu  -> compiled with $(NVCC), holds the kernel + launcher
#   src/add_kernel.cuh -> plain C++ declaration shared by both
#
# The host object and device object are linked together with $(NVCC) so the
# CUDA runtime is pulled in automatically.

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CXX       ?= g++

# A10G is sm_86. Override on the command line for other GPUs, e.g.
#   make SM_ARCH=sm_80
SM_ARCH ?= sm_86

CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra -Isrc
NVCCFLAGS ?= -O2 -std=c++17 -arch=$(SM_ARCH) -Isrc
LDFLAGS   ?= -L$(CUDA_HOME)/lib64
LDLIBS    ?= -lcudart

BUILD_DIR := build
TARGET    := $(BUILD_DIR)/cuda_hello

CPP_OBJS := $(BUILD_DIR)/main.o
CU_OBJS  := $(BUILD_DIR)/add_kernel.o
OBJS     := $(CPP_OBJS) $(CU_OBJS)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/main.o: src/main.cpp src/add_kernel.cuh | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/add_kernel.o: src/add_kernel.cu src/add_kernel.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: $(TARGET)
	$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
