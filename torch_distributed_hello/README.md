# torch_distributed_hello

The collectives from `nccl_collectives_hello`, one layer up: as
**`torch.distributed`** exposes them, plus the canonical thing that wraps them —
**DistributedDataParallel (DDP)**, whose backward pass is an AllReduce of
gradients. This is what real distributed training actually calls.

Final project in the distributed-communication family:

- [`gpu_memory_movement_hello`](../gpu_memory_movement_hello) — local host ↔
  device movement.
- [`rdma_libfabric_hello`](../rdma_libfabric_hello) — point-to-point RDMA over
  EFA.
- [`nccl_collectives_hello`](../nccl_collectives_hello) — the collectives
  against the raw NCCL C API.
- `torch_distributed_hello` (this project) — the framework layer, and the place
  to run **real multi-rank** collectives on this single-GPU box.

## Why this runs real collectives on one GPU

`nccl_collectives_hello` can't show multi-rank traffic on a single GPU because
NCCL needs one GPU per rank. PyTorch ships a second backend, **gloo**, that runs
on CPU and happily places many ranks on one machine. So here we can launch 4
ranks and watch an AllReduce actually combine 4 contributions — on the same box
that has only one GPU. The program picks the backend automatically:

- **nccl** when CUDA is available and `WORLD_SIZE <= ` visible GPUs (one GPU per
  rank). This is the path used in GPU training; on multi-node it reaches EFA via
  `aws-ofi-nccl`.
- **gloo** otherwise — real multi-rank collectives on CPU.

Override with `BACKEND=nccl` or `BACKEND=gloo`.

## What it demonstrates

Run on **deterministic** inputs (rank *r* contributes `r+1`) so every result is
checkable:

- `all_reduce(SUM)` → `N(N+1)/2` on all ranks.
- `broadcast` from rank 0 → a constant everywhere.
- `all_gather` → every rank's block, in order.
- `reduce_scatter` → reduced chunk per rank (NCCL only; gloo lacks it, so it's
  skipped with a note).
- **DDP** → build a tiny `nn.Linear`, wrap in `DistributedDataParallel`, run a
  forward/backward on per-rank-different data, and confirm the gradient is
  **identical on every rank** afterward. That equality is DDP's gradient
  AllReduce doing its job; it is the entire reason data-parallel training stays
  in sync.

## How launching works

`torchrun` spawns the rank processes and sets `RANK`, `WORLD_SIZE`,
`LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT`. The program reads them and calls
`dist.init_process_group(init_method="env://")`. This is the same launcher used
for real multi-GPU and multi-node jobs — only the `--nnodes` / `--nproc_per_node`
(and a multi-node rendezvous endpoint) change.

## Prerequisites

- Python 3 with a working `venv` **or** `virtualenv` (the Makefile falls back to
  `virtualenv` when `python -m venv` can't bootstrap pip).
- For `run-gpu`: an NVIDIA GPU + driver. The PyPI `torch` wheel bundles a
  matching NCCL, so no system NCCL is required.

Developed against PyTorch 2.12 on a single NVIDIA L4. The first `make venv`
downloads PyTorch and is slow.

## Build and run

```bash
make venv              # create .venv and install torch
make run               # gloo (CPU), NPROC=4 ranks -> real multi-rank collectives
make run NPROC=8       # more CPU ranks
make run-gpu           # nccl backend, one rank per visible GPU
make clean
```

Example output, `make run` (4 CPU ranks via gloo):

```
backend=gloo (CPU), world_size=4
  all_reduce(sum)  : 10 (want 10) -> OK
  broadcast        : 7 (want 7) -> OK
  all_gather       : blocks=[1, 2, 3, 4] -> OK
  reduce_scatter   : skipped (not supported by gloo)
  DDP grad sync    : max spread across ranks=0.00e+00 -> OK
OK
```

Example output, `make run-gpu` (nccl, single L4 so one rank):

```
backend=nccl (GPU), world_size=1
  all_reduce(sum)  : 1 (want 1) -> OK
  broadcast        : 7 (want 7) -> OK
  all_gather       : blocks=[1] -> OK
  reduce_scatter   : 1 (want 1) -> OK
  DDP grad sync    : max spread across ranks=0.00e+00 -> OK
OK
```

To watch the NCCL transport selection on a multi-GPU host, run with
`NCCL_DEBUG=INFO`:

```bash
BACKEND=nccl NCCL_DEBUG=INFO .venv/bin/torchrun --standalone \
    --nproc_per_node=2 main.py
```

## Layout

```
torch_distributed_hello/
├── Makefile
├── README.md
├── requirements.txt
├── .gitignore
└── main.py     # backend selection, the four collectives, and a DDP step
```

## The full arc

You've now climbed the stack: bytes moving host↔device over PCIe
(`gpu_memory_movement_hello`), the same bytes moving between processes over an
RDMA NIC (`rdma_libfabric_hello`), those point-to-point transfers composed into
collective algorithms (`nccl_collectives_hello`), and finally a training
framework issuing those collectives to keep replicas in sync (this project).
That is the foundation of how distributed training exchanges GPU memory and
performs collective communication.
