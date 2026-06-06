# nccl_collectives_hello

The four collective-communication primitives that distributed training is built
from — **AllReduce, Broadcast, AllGather, ReduceScatter** — written directly
against the **NCCL C API**, with one process per rank and a file-based
`ncclUniqueId` bootstrap.

This is the third project in the distributed-communication family:

- [`gpu_memory_movement_hello`](../gpu_memory_movement_hello) — local host ↔
  device movement.
- [`rdma_libfabric_hello`](../rdma_libfabric_hello) — point-to-point RDMA over
  EFA (the transport NCCL uses on AWS).
- `nccl_collectives_hello` (this project) — collectives on top of those
  transports.
- [`torch_distributed_hello`](../torch_distributed_hello) — the same
  collectives as a framework calls them, and the place to see **real
  multi-rank** traffic on this single-GPU box (via the CPU/gloo backend).

## What collectives are

A collective is a communication pattern over a group of ranks:

- **AllReduce** — element-wise reduce (e.g. sum) of every rank's buffer, with
  the result delivered to all ranks. This is the workhorse of data-parallel
  training: it averages gradients across replicas.
- **Broadcast** — one root rank's buffer is copied to all ranks (e.g.
  distributing initial weights).
- **AllGather** — each rank's chunk is concatenated and delivered to all ranks
  (used in sharded optimizers / FSDP to reassemble parameters).
- **ReduceScatter** — reduce across ranks, then scatter disjoint chunks of the
  result to each rank (the other half of FSDP; AllReduce ≈ ReduceScatter +
  AllGather).

NCCL implements these with bandwidth-optimal **ring** and latency-optimal
**tree** algorithms, choosing transports per link: NVLink / PCIe peer-to-peer
within a node, and the network (InfiniBand, or EFA via the `aws-ofi-nccl`
plugin) between nodes.

## What the program does

Each rank is its own process — the real multi-GPU / multi-node pattern:

1. Rank 0 calls `ncclGetUniqueId` and writes it to `NCCL_ID_FILE`; the other
   ranks read it. (This out-of-band exchange is the NCCL analogue of the
   address exchange in `rdma_libfabric_hello`. A real job uses MPI or a
   rendezvous; a file keeps this dependency-free.)
2. Each rank picks `device = local_rank % visibleGPUs`, then
   `ncclCommInitRank(comm, world, id, rank)`.
3. It runs each collective on **deterministic inputs** so the output can be
   checked exactly. Rank *r* contributes the value `r+1`, so for `N` ranks an
   AllReduce/ReduceScatter sum is `N(N+1)/2`, AllGather block *r* is `r+1`, and
   Broadcast delivers a fixed constant. Every rank verifies its result.

## Hardware reality: one GPU per rank

**NCCL requires a distinct GPU for each rank in a communicator.** Two ranks on
the same physical GPU are rejected with `Duplicate GPU detected`. So:

- On a **single-GPU** box (like the L4 this was developed on), `make run`
  defaults to `WORLD_SIZE=1`. All four collectives still execute and validate,
  exercising the full NCCL init + call + completion path, but with one rank
  they are degenerate (no cross-rank traffic).
- On a **multi-GPU** box, set `WORLD_SIZE` up to the number of GPUs to get real
  ring/tree collectives across NVLink/PCIe.
- Across **nodes**, launch one process per GPU on each node sharing the unique
  id (e.g. via a shared filesystem path or a launcher like MPI/torchrun), and
  NCCL will route inter-node traffic over EFA through `aws-ofi-nccl`.

To see real multi-rank collective *semantics* on this single-GPU instance, use
`torch_distributed_hello` with the gloo (CPU) backend, which happily runs many
ranks on one machine.

## Prerequisites

- An NVIDIA GPU with a recent driver.
- CUDA Toolkit including NCCL (`nccl.h` + `libnccl`). Here both live under
  `/usr/local/cuda` (NCCL 2.28.3). Override `CUDA_HOME` / `NCCL_HOME` for other
  layouts.

Developed against CUDA 12.9 / NCCL 2.28.3 on a single NVIDIA L4 (sm_89).

## Build and run

```bash
make                  # build build/nccl_collectives
make run              # WORLD_SIZE defaults to the number of visible GPUs
make run WORLD_SIZE=4 # force 4 ranks (needs >= 4 GPUs)
make run-verbose      # NCCL_DEBUG=INFO: watch topology + transport selection
make clean
```

Example output on a single L4 (`WORLD_SIZE=1`):

```
[rank 0/1] device 0 (NVIDIA L4)
NCCL version 2.28.3+cuda12.9
  AllReduce(sum)   : got 1, want 1 -> OK
  Broadcast        : got 7, want 7 -> OK
  AllGather        : blocks=[1..1] -> OK
  ReduceScatter    : got 1, want 1 -> OK
[rank 0] OK
```

On two GPUs (`WORLD_SIZE=2`) the AllReduce/ReduceScatter sums become 3, the
AllGather shows blocks `[1..2]`, and `make run-verbose` prints the rings NCCL
built and the transport chosen for each link.

### Reading `NCCL_DEBUG=INFO`

With more than one GPU, the verbose log is the real payoff. Look for:

- `NCCL INFO ... via P2P/IPC` or `via SHM` — intra-node transport between GPUs.
- `NCCL INFO Channel .. : ring` / tree structures — the algorithm graph.
- On multi-node, `NCCL INFO NET/OFI ...` — the `aws-ofi-nccl` plugin selecting
  EFA, i.e. NCCL reaching the same libfabric layer that
  `rdma_libfabric_hello` uses directly.

## Layout

```
nccl_collectives_hello/
├── Makefile
├── README.md
├── run.sh                # launches WORLD_SIZE rank processes
├── .gitignore
└── src/
    └── collectives.cu    # unique-id bootstrap + the four collectives + checks
```

## Where this leads

`torch_distributed_hello` shows the layer above: PyTorch's `torch.distributed`
calls exactly these NCCL collectives under the hood (`all_reduce`, `broadcast`,
`all_gather`, `reduce_scatter`) and a DDP training step is essentially an
AllReduce of gradients. It also runs real multi-rank collectives on CPU here.
