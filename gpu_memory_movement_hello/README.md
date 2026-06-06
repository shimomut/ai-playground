# gpu_memory_movement_hello

The foundation under every form of GPU-to-GPU memory exchange. Before bytes
can travel between GPUs or across a network, they move between **host memory**,
**device memory**, and the **DMA engines** that copy them. This project
measures the bandwidth of those local paths and demonstrates the one fact that
the rest of this learning family depends on: **pinned (page-locked) host memory
is the precondition for fast, asynchronous, DMA-driven transfers** — and, later,
for RDMA.

This is the first project in a distributed-communication family that climbs the
stack:

- `gpu_memory_movement_hello` (this project) — local host ↔ device movement.
- [`rdma_libfabric_hello`](../rdma_libfabric_hello) — moving bytes between
  processes over the network with RDMA (libfabric / EFA), including GPUDirect.
- [`nccl_collectives_hello`](../nccl_collectives_hello) — collective
  communication (AllReduce, Broadcast, ...) with NCCL.
- [`torch_distributed_hello`](../torch_distributed_hello) — the same
  collectives as a training framework uses them, via `torch.distributed`.

## The mental model

A GPU cannot read ordinary (pageable) host memory directly. Pageable pages can
be moved or swapped by the OS at any time, so the GPU's copy engine has no
stable physical address to DMA from. To copy pageable memory the driver first
stages it through a hidden **pinned bounce buffer**, then DMAs from there.

**Pinned** (page-locked) memory, allocated with `cudaMallocHost`, is locked to
fixed physical pages. The DMA engine can read it directly, which:

1. removes the staging copy, so H2D/D2H transfers are faster (sometimes much
   faster, depending on platform), and
2. lets transfers run **asynchronously** (`cudaMemcpyAsync`) and overlap with
   compute on other streams.

That same property — memory locked to a fixed physical address so a DMA engine
can touch it without the CPU — is exactly what a NIC requires to perform RDMA.
This project is where that idea is introduced; `rdma_libfabric_hello` puts it on
the wire.

## What it measures

- H2D and D2H bandwidth for **pageable** memory (`malloc`).
- H2D and D2H bandwidth for **pinned** memory (`cudaMallocHost`).
- D2D (on-device) bandwidth, which reflects HBM/GDDR bandwidth rather than PCIe.
- The pinned-vs-pageable H2D speedup.
- A **copy/compute overlap** demo: a chunked pipeline that copies chunk *i+1*
  while a kernel processes chunk *i* across two streams, compared to doing the
  copy and compute serially.

## Prerequisites

- An NVIDIA GPU with a recent driver.
- CUDA Toolkit at `/usr/local/cuda` (override with `CUDA_HOME`).

Developed against CUDA 12.9 on an NVIDIA L4 (compute capability 8.9).

## Build and run

```bash
make            # builds build/bandwidth
make run        # builds and runs (default 256 MiB transfers)
make run SIZE_MIB=512   # larger transfers
make clean
```

Target a different GPU architecture with `SM_ARCH`:

```bash
make SM_ARCH=sm_80   # A100
make SM_ARCH=sm_90   # H100
```

Example output (NVIDIA L4, numbers vary by instance and PCIe generation):

```
device: NVIDIA L4 (compute 8.9)
transfer size: 256 MiB, 50 timed iters

path                                         ms        GiB/s
H2D pageable                             20.434        12.23
D2H pageable                             20.739        12.05
H2D pinned                               19.937        12.54
D2H pinned                               20.319        12.30
D2D (on-device)                           2.334       107.13

pinned vs pageable H2D speedup: 1.03x

Copy/compute overlap (pinned memory, 8 chunks):
  serial    copy-then-compute :   22.502 ms
  pipelined copy||compute     :   20.274 ms
  speedup from overlap        :     1.11x
```

### Reading the numbers

- **D2D is ~10x faster than H2D/D2H.** On-device copies run at HBM/GDDR
  bandwidth (hundreds of GiB/s); host transfers are capped by PCIe (~12 GiB/s
  here). This is *why* keeping data resident on the GPU matters, and why
  collective libraries work hard to avoid host round-trips.
- **The pinned-vs-pageable gap depends on the platform.** On systems where the
  driver's bounce-buffer staging is efficient and PCIe is the bottleneck, the
  two can be close (as above). On many systems pinned is 2–3x faster. Either
  way, the *async overlap* benefit of pinned memory is the bigger practical win,
  and pinned memory is mandatory for RDMA.
- **The overlap speedup** shows DMA engines and compute units working at the
  same time. The copy of one chunk proceeds on the copy stream while the kernel
  for the previous chunk runs on the compute stream. This is the same
  pipelining that lets communication overlap with computation in distributed
  training.

## Layout

```
gpu_memory_movement_hello/
├── Makefile
├── README.md
├── .gitignore
└── src/
    └── bandwidth.cu     # bandwidth measurements + copy/compute overlap demo
```

## Where this leads

The next project, `rdma_libfabric_hello`, takes the "DMA engine reads pinned,
registered memory" idea and applies it to a NIC: a network adapter DMAs
straight out of a registered buffer (host *or* GPU memory) and places it in a
registered buffer on another process, with no CPU copy in the middle. That is
RDMA, and GPUDirect RDMA is the variant where the registered buffer lives in
GPU memory.
