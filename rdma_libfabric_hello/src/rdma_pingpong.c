/* rdma_libfabric_hello
 *
 * A minimal RDMA ping-pong between two processes on the same node, programmed
 * against libfabric (OFI) and run over the AWS Elastic Fabric Adapter (EFA).
 *
 * Why libfabric and not "classic" ibverbs? EFA is not an InfiniBand RC device.
 * It exposes only reliable-datagram (FI_EP_RDM) endpoints and is designed to be
 * driven through libfabric, which is exactly why AWS ships the aws-ofi-nccl
 * plugin so NCCL can use EFA. The textbook ibverbs RC "RDMA WRITE pingpong"
 * does not run on EFA. So we use the libfabric API, which is also the portable
 * way to target InfiniBand, RoCE, and EFA from one codebase.
 *
 * What this shows:
 *   - opening a fabric / domain / endpoint and an address vector,
 *   - registering memory so the NIC's DMA engine may touch it (the same
 *     "pinned + registered" idea from gpu_memory_movement_hello, now for a NIC),
 *   - exchanging raw endpoint addresses out of band (here via files), and
 *   - a tagged send/recv ping-pong with completion-queue polling.
 *
 * Two processes are launched on one node; traffic goes out to the EFA device
 * and loops back. The same binary scales to two nodes unchanged once the
 * address-exchange directory is shared (or replaced with a socket bootstrap).
 *
 * Optional GPUDirect RDMA path: build with `make WITH_CUDA=1` to register CUDA
 * device buffers (FI_HMEM) so the NIC DMAs straight out of GPU memory. See the
 * README.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_errno.h>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

#define PING_TAG 0xA1u
#define PONG_TAG 0xB2u

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FATAL %s:%d: %s\n", __FILE__, __LINE__, (msg));         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define FI_CHECK(call)                                                         \
  do {                                                                         \
    int rc__ = (int)(call);                                                    \
    if (rc__) {                                                                \
      fprintf(stderr, "FATAL %s:%d: %s = %d (%s)\n", __FILE__, __LINE__,       \
              #call, rc__, fi_strerror(-rc__));                                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

struct ctx {
  struct fi_info *info;
  struct fid_fabric *fabric;
  struct fid_domain *domain;
  struct fid_ep *ep;
  struct fid_av *av;
  struct fid_cq *txcq;
  struct fid_cq *rxcq;

  /* Communication buffers and their memory registrations. */
  char *sbuf;
  char *rbuf;
  struct fid_mr *smr;
  struct fid_mr *rmr;
  void *sdesc;
  void *rdesc;
  size_t buf_len;

  fi_addr_t peer; /* address-vector index of the remote endpoint */
};

/* libfabric may require a per-operation context object (FI_CONTEXT2). We have
 * at most one outstanding send and one outstanding recv, so two suffice. */
static struct fi_context2 send_ctx;
static struct fi_context2 recv_ctx;

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Block until exactly one completion is reaped from `cq`, or die on error. */
static void wait_one(struct fid_cq *cq) {
  struct fi_cq_tagged_entry comp;
  for (;;) {
    ssize_t rc = fi_cq_read(cq, &comp, 1);
    if (rc == 1) return;
    if (rc == -FI_EAGAIN) continue; /* nothing ready yet, keep polling */
    if (rc == -FI_EAVAIL) {
      struct fi_cq_err_entry err = {0};
      fi_cq_readerr(cq, &err, 0);
      fprintf(stderr, "cq error: %s\n",
              fi_cq_strerror(cq, err.prov_errno, err.err_data, NULL, 0));
      exit(EXIT_FAILURE);
    }
    fprintf(stderr, "fi_cq_read failed: %s\n", fi_strerror((int)-rc));
    exit(EXIT_FAILURE);
  }
}

/* Posting an operation can return -FI_EAGAIN when the provider's queue is
 * momentarily full or, on EFA, while the first-message handshake is still being
 * driven. The contract is to retry, calling fi_cq_read to advance the progress
 * engine. We have no outstanding completion of this kind in flight when we
 * post (the ping-pong reaps each completion before posting the next op of the
 * same type), so draining for progress here cannot drop a real completion. */
static void post_send(struct ctx *c, void *buf, void *desc, uint64_t tag,
                      void *op_ctx) {
  ssize_t rc;
  do {
    rc = fi_tsend(c->ep, buf, c->buf_len, desc, c->peer, tag, op_ctx);
    if (rc == -FI_EAGAIN) {
      struct fi_cq_tagged_entry tmp;
      fi_cq_read(c->txcq, &tmp, 1); /* drive progress, then retry */
    }
  } while (rc == -FI_EAGAIN);
  FI_CHECK((int)rc);
}

static void post_recv(struct ctx *c, void *buf, void *desc, uint64_t tag,
                      void *op_ctx) {
  ssize_t rc;
  do {
    rc = fi_trecv(c->ep, buf, c->buf_len, desc, c->peer, tag, 0, op_ctx);
    if (rc == -FI_EAGAIN) {
      struct fi_cq_tagged_entry tmp;
      fi_cq_read(c->rxcq, &tmp, 1); /* drive progress, then retry */
    }
  } while (rc == -FI_EAGAIN);
  FI_CHECK((int)rc);
}

/* --- out-of-band address exchange via files ---------------------------------
 * Each rank writes its own endpoint address to <dir>/rank<r>.addr (atomically,
 * via a temp file + rename) and polls until the peer's file appears. This is a
 * single-node-friendly stand-in for the socket/MPI bootstrap a real job uses.
 */
static void write_addr(const char *dir, int rank, const void *addr,
                       size_t len) {
  char tmp[512], fin[512];
  snprintf(tmp, sizeof(tmp), "%s/rank%d.addr.tmp", dir, rank);
  snprintf(fin, sizeof(fin), "%s/rank%d.addr", dir, rank);
  FILE *f = fopen(tmp, "wb");
  CHECK(f != NULL, "fopen addr tmp");
  CHECK(fwrite(&len, sizeof(len), 1, f) == 1, "write addr len");
  CHECK(fwrite(addr, 1, len, f) == len, "write addr bytes");
  fclose(f);
  CHECK(rename(tmp, fin) == 0, "rename addr file");
}

static void read_addr(const char *dir, int rank, void *addr, size_t *len) {
  char fin[512];
  snprintf(fin, sizeof(fin), "%s/rank%d.addr", dir, rank);
  for (;;) {
    FILE *f = fopen(fin, "rb");
    if (f) {
      size_t stored = 0;
      if (fread(&stored, sizeof(stored), 1, f) == 1 && stored <= *len) {
        if (fread(addr, 1, stored, f) == stored) {
          *len = stored;
          fclose(f);
          return;
        }
      }
      fclose(f);
    }
    usleep(2000); /* peer hasn't published its address yet */
  }
}

static void setup(struct ctx *c, size_t buf_len, int use_gpu) {
  /* Describe what we need: a reliable-datagram endpoint that can do tagged
   * messaging, served by the EFA provider. */
  struct fi_info *hints = fi_allocinfo();
  CHECK(hints != NULL, "fi_allocinfo");
  hints->ep_attr->type = FI_EP_RDM;
  hints->caps = FI_MSG | FI_TAGGED;
  hints->domain_attr->mr_mode =
      FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
  hints->fabric_attr->prov_name = strdup("efa");
  if (use_gpu) {
    /* Requesting FI_HMEM (GPU-memory transfers) requires also acknowledging
     * FI_MR_HMEM in mr_mode, otherwise the HMEM-capable provider is filtered
     * out and fi_getinfo returns -FI_ENODATA. */
    hints->caps |= FI_HMEM;
    hints->domain_attr->mr_mode |= FI_MR_HMEM;
  }

  FI_CHECK(fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &c->info));
  fi_freeinfo(hints);

  printf("[setup] provider=%s fabric=%s domain=%s\n",
         c->info->fabric_attr->prov_name, c->info->fabric_attr->name,
         c->info->domain_attr->name);

  FI_CHECK(fi_fabric(c->info->fabric_attr, &c->fabric, NULL));
  FI_CHECK(fi_domain(c->fabric, c->info, &c->domain, NULL));

  struct fi_cq_attr cq_attr = {0};
  cq_attr.format = FI_CQ_FORMAT_TAGGED;
  cq_attr.size = c->info->tx_attr->size;
  FI_CHECK(fi_cq_open(c->domain, &cq_attr, &c->txcq, NULL));
  cq_attr.size = c->info->rx_attr->size;
  FI_CHECK(fi_cq_open(c->domain, &cq_attr, &c->rxcq, NULL));

  struct fi_av_attr av_attr = {0};
  av_attr.type = FI_AV_TABLE;
  FI_CHECK(fi_av_open(c->domain, &av_attr, &c->av, NULL));

  FI_CHECK(fi_endpoint(c->domain, c->info, &c->ep, NULL));
  FI_CHECK(fi_ep_bind(c->ep, &c->txcq->fid, FI_TRANSMIT));
  FI_CHECK(fi_ep_bind(c->ep, &c->rxcq->fid, FI_RECV));
  FI_CHECK(fi_ep_bind(c->ep, &c->av->fid, 0));
  FI_CHECK(fi_enable(c->ep));

  c->buf_len = buf_len;

  /* Allocate the message buffers (host or GPU) and register them so the EFA
   * DMA engine is permitted to read/write them directly. This registration is
   * the network analogue of cudaMallocHost pinning host pages. */
  if (use_gpu) {
#ifdef WITH_CUDA
    CHECK(cudaMalloc((void **)&c->sbuf, buf_len) == cudaSuccess, "cudaMalloc s");
    CHECK(cudaMalloc((void **)&c->rbuf, buf_len) == cudaSuccess, "cudaMalloc r");
    struct fi_mr_attr mr_attr = {0};
    struct iovec iov;
    iov.iov_base = c->sbuf;
    iov.iov_len = buf_len;
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_SEND | FI_RECV;
    mr_attr.iface = FI_HMEM_CUDA;
    FI_CHECK(fi_mr_regattr(c->domain, &mr_attr, 0, &c->smr));
    iov.iov_base = c->rbuf;
    FI_CHECK(fi_mr_regattr(c->domain, &mr_attr, 0, &c->rmr));
#else
    CHECK(0, "rebuild with WITH_CUDA=1 for the GPU path");
#endif
  } else {
    c->sbuf = calloc(1, buf_len);
    c->rbuf = calloc(1, buf_len);
    CHECK(c->sbuf && c->rbuf, "calloc buffers");
    FI_CHECK(fi_mr_reg(c->domain, c->sbuf, buf_len, FI_SEND | FI_RECV, 0, 0, 0,
                       &c->smr, NULL));
    FI_CHECK(fi_mr_reg(c->domain, c->rbuf, buf_len, FI_SEND | FI_RECV, 0, 0, 0,
                       &c->rmr, NULL));
  }
  c->sdesc = fi_mr_desc(c->smr);
  c->rdesc = fi_mr_desc(c->rmr);
}

static void exchange_addresses(struct ctx *c, const char *dir, int rank) {
  /* Query our own endpoint address. */
  char local[256];
  size_t local_len = sizeof(local);
  FI_CHECK(fi_getname(&c->ep->fid, local, &local_len));

  int peer_rank = rank ^ 1;
  write_addr(dir, rank, local, local_len);

  char peer[256];
  size_t peer_len = sizeof(peer);
  read_addr(dir, peer_rank, peer, &peer_len);

  int inserted = fi_av_insert(c->av, peer, 1, &c->peer, 0, NULL);
  CHECK(inserted == 1, "fi_av_insert");
  printf("[rank %d] exchanged addresses with rank %d\n", rank, peer_rank);
}

static void teardown(struct ctx *c, int use_gpu) {
  fi_close(&c->smr->fid);
  fi_close(&c->rmr->fid);
  fi_close(&c->ep->fid);
  fi_close(&c->txcq->fid);
  fi_close(&c->rxcq->fid);
  fi_close(&c->av->fid);
  fi_close(&c->domain->fid);
  fi_close(&c->fabric->fid);
  fi_freeinfo(c->info);
  if (use_gpu) {
#ifdef WITH_CUDA
    cudaFree(c->sbuf);
    cudaFree(c->rbuf);
#endif
  } else {
    free(c->sbuf);
    free(c->rbuf);
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage: %s <rank 0|1> <exchange_dir> [msg_bytes] [iters] [--gpu]\n",
            argv[0]);
    return EXIT_FAILURE;
  }
  int rank = atoi(argv[1]);
  const char *dir = argv[2];
  size_t msg_bytes = (argc > 3) ? (size_t)atoll(argv[3]) : 65536;
  int iters = (argc > 4) ? atoi(argv[4]) : 1000;
  int use_gpu = 0;
  for (int i = 5; i < argc; ++i)
    if (strcmp(argv[i], "--gpu") == 0) use_gpu = 1;

  CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");

  struct ctx c = {0};
  setup(&c, msg_bytes, use_gpu);
  exchange_addresses(&c, dir, rank);

  const int warmup = 20;
  const uint64_t send_tag = (rank == 1) ? PING_TAG : PONG_TAG;
  const uint64_t recv_tag = (rank == 1) ? PONG_TAG : PING_TAG;

  /* Fill the send buffer with a checkable pattern. For the GPU path the buffer
   * lives in device memory, so we stage the pattern through host memory with
   * cudaMemcpy. */
  if (!use_gpu) {
    for (size_t i = 0; i < msg_bytes; ++i) c.sbuf[i] = (char)(i & 0xff);
  } else {
#ifdef WITH_CUDA
    char *seed = malloc(msg_bytes);
    CHECK(seed != NULL, "malloc seed");
    for (size_t i = 0; i < msg_bytes; ++i) seed[i] = (char)(i & 0xff);
    CHECK(cudaMemcpy(c.sbuf, seed, msg_bytes, cudaMemcpyHostToDevice) ==
              cudaSuccess,
          "cudaMemcpy seed H2D");
    free(seed);
#endif
  }

  double t0 = 0.0;
  for (int i = 0; i < warmup + iters; ++i) {
    if (i == warmup) t0 = now_sec();

    if (rank == 1) {
      /* ping side: post recv for the echo, send the ping, wait for both. */
      post_recv(&c, c.rbuf, c.rdesc, recv_tag, &recv_ctx);
      post_send(&c, c.sbuf, c.sdesc, send_tag, &send_ctx);
      wait_one(c.txcq);
      wait_one(c.rxcq);
    } else {
      /* pong side: receive the ping, echo it straight back. */
      post_recv(&c, c.rbuf, c.rdesc, recv_tag, &recv_ctx);
      wait_one(c.rxcq);
      post_send(&c, c.rbuf, c.rdesc, send_tag, &send_ctx);
      wait_one(c.txcq);
    }
  }
  double elapsed = now_sec() - t0;

  if (rank == 1) {
    /* Verify the echoed bytes match what we sent. The GPU path copies the
     * device receive buffer back to host first. */
    int ok = 1;
    if (!use_gpu) {
      for (size_t i = 0; i < msg_bytes; ++i)
        if (c.rbuf[i] != (char)(i & 0xff)) { ok = 0; break; }
    } else {
#ifdef WITH_CUDA
      char *check = malloc(msg_bytes);
      CHECK(check != NULL, "malloc check");
      CHECK(cudaMemcpy(check, c.rbuf, msg_bytes, cudaMemcpyDeviceToHost) ==
                cudaSuccess,
            "cudaMemcpy check D2H");
      for (size_t i = 0; i < msg_bytes; ++i)
        if (check[i] != (char)(i & 0xff)) { ok = 0; break; }
      free(check);
#endif
    }

    double rtt_us = elapsed / iters * 1e6;
    double owlat_us = rtt_us / 2.0;
    double bw_gib = ((double)msg_bytes / (1024.0 * 1024.0 * 1024.0)) /
                    (owlat_us * 1e-6);
    printf("\nping-pong over EFA (%s buffers)\n", use_gpu ? "GPU" : "host");
    printf("  msg size        : %zu bytes\n", msg_bytes);
    printf("  iterations      : %d\n", iters);
    printf("  round-trip      : %.2f us\n", rtt_us);
    printf("  one-way latency : %.2f us\n", owlat_us);
    printf("  one-way bw      : %.2f GiB/s\n", bw_gib);
    printf("  data check      : %s\n", ok ? "OK" : "MISMATCH");
    printf("%s\n", ok ? "OK" : "FAILED");
  }

  teardown(&c, use_gpu);
  return 0;
}
