# rdma_libfabric_hello

A minimal RDMA ping-pong between two processes, programmed against
**libfabric (OFI)** and run over the AWS **Elastic Fabric Adapter (EFA)**. It
moves bytes directly between registered buffers in two processes with no CPU
copy in the data path — and, with `make run-gpu`, straight out of and into GPU
memory (**GPUDirect RDMA**).

This is the second project in the distributed-communication family:

- [`gpu_memory_movement_hello`](../gpu_memory_movement_hello) — local host ↔
  device movement; introduces pinned/registered memory + DMA.
- `rdma_libfabric_hello` (this project) — that same idea over the network: a
  NIC DMAs out of registered memory on one node into registered memory on
  another.
- [`nccl_collectives_hello`](../nccl_collectives_hello) — collectives built on
  top of point-to-point transports like this one.
- [`torch_distributed_hello`](../torch_distributed_hello) — the framework layer.

## Why libfabric and not "classic" ibverbs RDMA?

Most RDMA tutorials use the InfiniBand verbs API (`ibv_*`) with **Reliable
Connection (RC)** queue pairs and `IBV_WR_RDMA_WRITE`. **That does not work on
EFA.** EFA is not an RC device: it exposes only reliable-*datagram*
(`FI_EP_RDM`) endpoints and is designed to be driven through libfabric. This is
exactly why AWS ships the `aws-ofi-nccl` plugin so NCCL can reach EFA via
libfabric (see `nccl_collectives_hello`).

libfabric is also the portable choice: the same code targets InfiniBand, RoCE,
and EFA by selecting a different provider. Here we request the `efa` provider
explicitly. On a Mellanox/RoCE box you would request `verbs;ofi_rxm`, and the
rest of the program is unchanged.

If you want to see the classic ibverbs RC path for contrast, the `rdma-core`
package installs `ibv_rc_pingpong` — but it will fail to run against the EFA
device for the reason above. The conceptual mapping is:

| concept              | ibverbs (RC)            | libfabric (this project)        |
| -------------------- | ----------------------- | ------------------------------- |
| device/context       | `ibv_open_device`       | `fi_fabric` / `fi_domain`       |
| endpoint             | queue pair (`ibv_qp`)   | `fi_endpoint` (`FI_EP_RDM`)     |
| completion queue     | `ibv_cq`                | `fid_cq`                        |
| memory registration  | `ibv_reg_mr`            | `fi_mr_reg` / `fi_mr_regattr`   |
| addressing           | LID/QPN exchange        | address vector (`fi_av_insert`) |
| post send / recv     | `ibv_post_send/recv`    | `fi_tsend` / `fi_trecv`         |

## What the program does

1. **Discovery.** `fi_getinfo` with hints (`FI_EP_RDM`, `FI_MSG | FI_TAGGED`,
   provider `efa`) selects the EFA fabric.
2. **Setup.** Opens a fabric, domain, endpoint, two completion queues (tx/rx),
   and an address vector.
3. **Memory registration.** Each message buffer is registered with `fi_mr_reg`
   (host) or `fi_mr_regattr` with `iface=FI_HMEM_CUDA` (GPU) so the EFA DMA
   engine is permitted to touch it directly. This is the network analogue of
   `cudaMallocHost` pinning host pages in the previous project.
4. **Address exchange.** Each rank writes its endpoint address to a file in a
   shared directory and reads its peer's, then inserts it into the address
   vector. (A real job uses a socket/MPI bootstrap; files keep this single-node
   demo dependency-free. To run across two nodes, point both at a shared
   directory or swap in a TCP exchange.)
5. **Ping-pong.** Tagged send/recv (`fi_tsend`/`fi_trecv`) bounces a message
   back and forth `ITERS` times; completions are reaped from the CQs. Rank 1
   (ping) verifies the echoed bytes.

### A note on `-FI_EAGAIN`

`fi_tsend`/`fi_trecv` can return `-FI_EAGAIN`, especially on the first message
while EFA drives its handshake. The correct response is to call `fi_cq_read`
(to advance the progress engine) and retry the post — see `post_send` /
`post_recv`. Treating `EAGAIN` as fatal is the most common first-timer bug with
libfabric.

## Prerequisites

- An AWS instance with EFA enabled and the EFA software stack installed under
  `/opt/amazon/efa` (libfabric, headers). Check with
  `/opt/amazon/efa/bin/fi_info -p efa`.
- A C compiler.
- For `make run-gpu`: CUDA Toolkit (`CUDA_HOME`, default `/usr/local/cuda`) and
  the `efa-nv-peermem` / GPUDirect support that ships with the EFA installer.

Developed against libfabric 1.30 (EFA provider, API 2.4) and CUDA 12.9 on an
NVIDIA L4 instance with a single EFA device.

## Build and run

```bash
make            # build host-buffer binary
make run        # two ranks ping-pong over EFA using host memory
make run-gpu    # two ranks ping-pong using CUDA device buffers (GPUDirect RDMA)
make clean

# tune the message size and iteration count:
make run MSG=1048576 ITERS=200
```

Both ranks run on this one node and the traffic loops back through the EFA
device, so this works on a single-GPU, single-NIC instance. To run across two
nodes, build on both and launch rank 0 on one and rank 1 on the other, sharing
the exchange directory (or replace the file exchange with a socket bootstrap).

Example output (single NVIDIA L4 instance, loopback over one EFA device):

```
ping-pong over EFA (host buffers)
  msg size        : 65536 bytes
  iterations      : 1000
  round-trip      : 30.16 us
  one-way latency : 15.08 us
  one-way bw      : 4.05 GiB/s
  data check      : OK

ping-pong over EFA (GPU buffers)
  one-way latency : 78.72 us
  one-way bw      : 0.78 GiB/s
  data check      : OK
```

### Reading the numbers

These are loopback figures on one node with one NIC, not a real two-node
fabric benchmark — treat them as relative, not absolute.

- The **host** path is the baseline EFA messaging latency for this instance.
- The **GPU** path is slower *here* because both ranks share the single L4, so
  the GPUDirect loopback shuttles within one device rather than between peers.
  On a real multi-GPU/multi-node job, GPUDirect RDMA is a large win because it
  removes the device→host→NIC staging entirely. The point demonstrated on this
  box is correctness: `data check: OK` means the EFA NIC read the payload
  directly from one CUDA allocation and delivered it into another.

## Layout

```
rdma_libfabric_hello/
├── Makefile
├── README.md
├── .gitignore
└── src/
    └── rdma_pingpong.c   # libfabric setup, MR, address exchange, ping-pong
```

## Where this leads

`nccl_collectives_hello` is the next step up. NCCL builds collectives
(AllReduce, etc.) on top of point-to-point transports just like this one — and
on EFA it reaches the network through the same libfabric layer, via the
`aws-ofi-nccl` plugin. Turning on `NCCL_DEBUG=INFO` there shows NCCL selecting
the EFA/OFI transport the same way this program selects the `efa` provider.
