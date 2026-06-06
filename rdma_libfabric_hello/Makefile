# rdma_libfabric_hello: a libfabric (OFI) tagged send/recv ping-pong over EFA.
#
#   make            build the host-buffer binary build/rdma_pingpong
#   make run        launch two processes on this node and ping-pong over EFA
#   make run-gpu    build the CUDA (FI_HMEM) binary and ping-pong GPU buffers
#   make clean
#
# EFA's libfabric ships under /opt/amazon/efa on AWS. Override EFA_HOME if your
# install differs.

EFA_HOME  ?= /opt/amazon/efa
CC        ?= cc
# gnu11 (not c11) so POSIX helpers like clock_gettime/usleep/strdup are declared.
CFLAGS    ?= -O2 -std=gnu11 -Wall -Wextra -I$(EFA_HOME)/include
LDFLAGS   ?= -L$(EFA_HOME)/lib -Wl,-rpath,$(EFA_HOME)/lib
LDLIBS    ?= -lfabric

CUDA_HOME ?= /usr/local/cuda

BUILD_DIR  := build
TARGET     := $(BUILD_DIR)/rdma_pingpong       # host buffers
TARGET_GPU := $(BUILD_DIR)/rdma_pingpong_gpu   # CUDA device buffers (FI_HMEM)
EXCH_DIR   := $(BUILD_DIR)/exch

# Ping-pong parameters for `make run`. Override e.g. make run MSG=1048576 ITERS=200
MSG   ?= 65536
ITERS ?= 1000

.PHONY: all run run-gpu clean

all: $(TARGET)

$(TARGET): src/rdma_pingpong.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

# Separate artifact for the GPU build so its extra flags never collide with the
# host build's object/binary (and a flag change always produces a fresh build).
$(TARGET_GPU): src/rdma_pingpong.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DWITH_CUDA -I$(CUDA_HOME)/include $< -o $@ \
		$(LDFLAGS) -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 \
		$(LDLIBS) -lcudart

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Launch rank 0 and rank 1 on this node. They discover each other through the
# files in EXCH_DIR, send to the EFA device, and loop back.
run: $(TARGET)
	@rm -rf $(EXCH_DIR) && mkdir -p $(EXCH_DIR)
	@echo "launching two ranks (msg=$(MSG) bytes, iters=$(ITERS))..."
	@$(TARGET) 0 $(EXCH_DIR) $(MSG) $(ITERS) & \
	 $(TARGET) 1 $(EXCH_DIR) $(MSG) $(ITERS) ; \
	 wait

run-gpu: $(TARGET_GPU)
	@rm -rf $(EXCH_DIR) && mkdir -p $(EXCH_DIR)
	@echo "launching two ranks with GPU buffers (msg=$(MSG) bytes, iters=$(ITERS))..."
	@$(TARGET_GPU) 0 $(EXCH_DIR) $(MSG) $(ITERS) --gpu & \
	 $(TARGET_GPU) 1 $(EXCH_DIR) $(MSG) $(ITERS) --gpu ; \
	 wait

clean:
	rm -rf $(BUILD_DIR)
