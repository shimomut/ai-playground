"""torch_distributed_hello

The collective primitives from nccl_collectives_hello, one level up: as
torch.distributed calls them, plus the canonical use that wraps them --
DistributedDataParallel (DDP), whose backward pass is an AllReduce of gradients.

Run with torchrun, which sets RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR /
MASTER_PORT in the environment:

    torchrun --standalone --nproc_per_node=4 main.py        # gloo (CPU)
    torchrun --standalone --nproc_per_node=1 main.py        # nccl (1 GPU)

Backend selection (override with BACKEND=nccl|gloo):
  - nccl  when CUDA is available and WORLD_SIZE <= visible GPUs (one GPU/rank).
  - gloo  otherwise. gloo runs many ranks on CPU, so it demonstrates real
          multi-rank collective semantics even on this single-GPU instance.
"""

from __future__ import annotations

import datetime
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def pick_backend(world_size: int) -> str:
    forced = os.environ.get("BACKEND")
    if forced:
        return forced
    if torch.cuda.is_available() and world_size <= torch.cuda.device_count():
        return "nccl"
    return "gloo"


def main() -> int:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = pick_backend(world)

    if backend == "nccl":
        if world > torch.cuda.device_count():
            print(
                f"ERROR: nccl backend needs one GPU per rank, but WORLD_SIZE="
                f"{world} > {torch.cuda.device_count()} GPU(s). Use BACKEND=gloo "
                f"or fewer ranks.",
                file=sys.stderr,
            )
            return 1
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # torchrun provides MASTER_ADDR/PORT; init_method="env://" reads them.
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=60),
    )

    if rank == 0:
        where = "GPU" if backend == "nccl" else "CPU"
        print(f"backend={backend} ({where}), world_size={world}")

    expect_sum = world * (world + 1) / 2.0  # sum_{r=1..N} r
    ok = True

    # ---- all_reduce(sum): each rank contributes (rank+1) ----
    t = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    pass_ = torch.allclose(t, torch.full_like(t, expect_sum))
    ok &= pass_
    if rank == 0:
        print(f"  all_reduce(sum)  : {t[0].item():.0f} (want {expect_sum:.0f}) "
              f"-> {'OK' if pass_ else 'FAIL'}")

    # ---- broadcast: root 0 sends 7 to all ----
    b = torch.full((4,), 7.0 if rank == 0 else -1.0, device=device)
    dist.broadcast(b, src=0)
    pass_ = torch.allclose(b, torch.full_like(b, 7.0))
    ok &= pass_
    if rank == 0:
        print(f"  broadcast        : {b[0].item():.0f} (want 7) "
              f"-> {'OK' if pass_ else 'FAIL'}")

    # ---- all_gather: rank r contributes (r+1); all ranks get every block ----
    src = torch.full((4,), float(rank + 1), device=device)
    gathered = [torch.empty_like(src) for _ in range(world)]
    dist.all_gather(gathered, src)
    pass_ = all(
        torch.allclose(gathered[r], torch.full_like(src, float(r + 1)))
        for r in range(world)
    )
    ok &= pass_
    if rank == 0:
        blocks = [int(g[0].item()) for g in gathered]
        print(f"  all_gather       : blocks={blocks} "
              f"-> {'OK' if pass_ else 'FAIL'}")

    # ---- reduce_scatter(sum): each rank gets the reduced version of its chunk
    # gloo does not implement reduce_scatter, so guard it to nccl.
    if backend == "nccl":
        inp = torch.full((world * 4,), float(rank + 1), device=device)
        out = torch.empty(4, device=device)
        dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
        pass_ = torch.allclose(out, torch.full_like(out, expect_sum))
        ok &= pass_
        if rank == 0:
            print(f"  reduce_scatter   : {out[0].item():.0f} "
                  f"(want {expect_sum:.0f}) -> {'OK' if pass_ else 'FAIL'}")
    elif rank == 0:
        print("  reduce_scatter   : skipped (not supported by gloo)")

    # ---- DDP: the collective that matters for training ----
    # DDP broadcasts initial weights at construction (so all ranks start equal),
    # then AllReduces gradients during backward. We verify that by checking the
    # gradient is identical on every rank after backward.
    torch.manual_seed(1234 + rank)  # different input data per rank
    model = nn.Linear(8, 8).to(device)
    ddp = DDP(model, device_ids=[device.index] if backend == "nccl" else None)
    x = torch.randn(16, 8, device=device)
    loss = ddp(x).pow(2).mean()
    loss.backward()

    g0 = next(ddp.module.parameters()).grad.flatten()[0].clone()
    grads = [torch.empty_like(g0) for _ in range(world)]
    dist.all_gather(grads, g0)
    spread = max(abs(grads[r].item() - grads[0].item()) for r in range(world))
    pass_ = spread < 1e-5
    ok &= pass_
    if rank == 0:
        print(f"  DDP grad sync    : max spread across ranks={spread:.2e} "
              f"-> {'OK' if pass_ else 'FAIL'}")

    # Reduce the per-rank pass/fail to a global verdict.
    verdict = torch.tensor([1 if ok else 0], device=device)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("OK" if verdict.item() == 1 else "FAILED")

    dist.destroy_process_group()
    return 0 if verdict.item() == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
