// nccl_collectives_hello
//
// The four collective-communication primitives that distributed training is
// built from, written against the NCCL C API:
//
//   AllReduce, Broadcast, AllGather, ReduceScatter
//
// Each rank is a separate process (the real multi-GPU / multi-node pattern).
// Rank 0 creates an ncclUniqueId and publishes it to a file; the other ranks
// read it. Every rank then joins the communicator with ncclCommInitRank and
// runs each collective on deterministic inputs so the result can be checked
// exactly.
//
// Device selection is `local_rank % visibleDevices`, so this runs unchanged on:
//   - a single-GPU box (all ranks share device 0 -- NCCL uses its SHM/P2P
//     transport; good for learning the API and reading NCCL_DEBUG logs), and
//   - a multi-GPU box or multi-node job (one rank per GPU, NCCL picks NVLink /
//     PCIe / network transports, using the aws-ofi-nccl plugin over EFA).
//
// Config comes from the environment (set by run.sh, torchrun-style):
//   RANK, WORLD_SIZE, LOCAL_RANK, NCCL_ID_FILE, [COUNT]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

#include <unistd.h>

#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e__ = (call);                                                  \
    if (e__ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(e__));                                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define NCCL_CHECK(call)                                                       \
  do {                                                                         \
    ncclResult_t r__ = (call);                                                 \
    if (r__ != ncclSuccess) {                                                  \
      std::fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,       \
                   ncclGetErrorString(r__));                                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

namespace {

int env_int(const char *name, int fallback) {
  const char *v = std::getenv(name);
  return v ? std::atoi(v) : fallback;
}

// Rank 0 writes the unique id atomically (temp + rename); everyone else spins
// until the final file exists and is the right size, then reads it.
void publish_id(const std::string &path, const ncclUniqueId &id) {
  std::string tmp = path + ".tmp";
  FILE *f = std::fopen(tmp.c_str(), "wb");
  if (!f) { std::perror("fopen id tmp"); std::exit(EXIT_FAILURE); }
  std::fwrite(&id, sizeof(id), 1, f);
  std::fclose(f);
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    std::perror("rename id file");
    std::exit(EXIT_FAILURE);
  }
}

ncclUniqueId read_id(const std::string &path) {
  ncclUniqueId id;
  for (;;) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f) {
      size_t n = std::fread(&id, 1, sizeof(id), f);
      std::fclose(f);
      if (n == sizeof(id)) return id;
    }
    usleep(2000);
  }
}

}  // namespace

int main() {
  const int rank = env_int("RANK", 0);
  const int world = env_int("WORLD_SIZE", 1);
  const int local_rank = env_int("LOCAL_RANK", rank);
  const size_t count = (size_t)env_int("COUNT", 1 << 20);
  const char *id_file = std::getenv("NCCL_ID_FILE");
  if (!id_file) {
    std::fprintf(stderr, "NCCL_ID_FILE must be set\n");
    return EXIT_FAILURE;
  }

  int n_dev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n_dev));
  if (n_dev <= 0) {
    std::fprintf(stderr, "no CUDA devices visible\n");
    return EXIT_FAILURE;
  }
  // NCCL requires a distinct GPU per rank within a communicator: two ranks on
  // the same physical GPU are rejected ("Duplicate GPU detected"). So a real
  // multi-rank collective needs WORLD_SIZE <= number of visible GPUs.
  if (world > n_dev) {
    if (rank == 0)
      std::fprintf(stderr,
                   "ERROR: WORLD_SIZE=%d but only %d GPU(s) visible.\n"
                   "NCCL needs one GPU per rank. Set WORLD_SIZE<=%d here, or\n"
                   "run on a host with more GPUs. (For real multi-rank "
                   "collectives on a single GPU, see torch_distributed_hello,\n"
                   "which can use the CPU/gloo backend.)\n",
                   world, n_dev, n_dev);
    return EXIT_FAILURE;
  }
  const int device = local_rank % n_dev;
  CUDA_CHECK(cudaSetDevice(device));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("[rank %d/%d] device %d (%s)\n", rank, world, device, prop.name);
  std::fflush(stdout);

  // --- bootstrap: exchange the NCCL unique id out of band ---
  ncclUniqueId id;
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
    publish_id(id_file, id);
  } else {
    id = read_id(id_file);
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));

  const size_t bytes = count * sizeof(float);
  const float expect_sum = (float)world * (world + 1) / 2.0f;  // sum_{r=1..N} r
  bool ok = true;

  std::vector<float> host(count);
  auto sample = [&](float *dev) -> float {
    float v;
    CUDA_CHECK(cudaMemcpy(&v, dev, sizeof(float), cudaMemcpyDeviceToHost));
    return v;
  };
  auto check_all = [&](float *dev, size_t n, float want) -> bool {
    host.resize(n);
    CUDA_CHECK(cudaMemcpy(host.data(), dev, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i)
      if (host[i] != want) return false;
    return true;
  };

  // ---- AllReduce(sum): each rank contributes (rank+1); result = sum ----
  {
    float *send, *recv;
    CUDA_CHECK(cudaMalloc(&send, bytes));
    CUDA_CHECK(cudaMalloc(&recv, bytes));
    std::fill(host.begin(), host.end(), (float)(rank + 1));
    CUDA_CHECK(cudaMemcpy(send, host.data(), bytes, cudaMemcpyHostToDevice));

    NCCL_CHECK(ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm,
                             stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    bool pass = check_all(recv, count, expect_sum);
    ok &= pass;
    if (rank == 0)
      std::printf("  AllReduce(sum)   : got %.0f, want %.0f -> %s\n",
                  sample(recv), expect_sum, pass ? "OK" : "FAIL");
    CUDA_CHECK(cudaFree(send));
    CUDA_CHECK(cudaFree(recv));
  }

  // ---- Broadcast: root 0 sends 7.0 to everyone ----
  {
    const float val = 7.0f;
    float *buf;
    CUDA_CHECK(cudaMalloc(&buf, bytes));
    std::fill(host.begin(), host.end(), rank == 0 ? val : -1.0f);
    CUDA_CHECK(cudaMemcpy(buf, host.data(), bytes, cudaMemcpyHostToDevice));

    NCCL_CHECK(ncclBroadcast(buf, buf, count, ncclFloat, 0, comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    bool pass = check_all(buf, count, val);
    ok &= pass;
    if (rank == 0)
      std::printf("  Broadcast        : got %.0f, want %.0f -> %s\n",
                  sample(buf), val, pass ? "OK" : "FAIL");
    CUDA_CHECK(cudaFree(buf));
  }

  // ---- AllGather: rank r contributes (r+1); each rank ends with all blocks --
  {
    float *send, *recv;
    CUDA_CHECK(cudaMalloc(&send, bytes));
    CUDA_CHECK(cudaMalloc(&recv, bytes * world));
    std::fill(host.begin(), host.end(), (float)(rank + 1));
    CUDA_CHECK(cudaMemcpy(send, host.data(), bytes, cudaMemcpyHostToDevice));

    NCCL_CHECK(ncclAllGather(send, recv, count, ncclFloat, comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // block r of the output should be all (r+1)
    std::vector<float> out(count * world);
    CUDA_CHECK(cudaMemcpy(out.data(), recv, bytes * world,
                          cudaMemcpyDeviceToHost));
    bool pass = true;
    for (int r = 0; r < world && pass; ++r)
      for (size_t i = 0; i < count; ++i)
        if (out[(size_t)r * count + i] != (float)(r + 1)) { pass = false; break; }
    ok &= pass;
    if (rank == 0)
      std::printf("  AllGather        : blocks=[%.0f..%.0f] -> %s\n", out[0],
                  out[(size_t)(world - 1) * count], pass ? "OK" : "FAIL");
    CUDA_CHECK(cudaFree(send));
    CUDA_CHECK(cudaFree(recv));
  }

  // ---- ReduceScatter(sum): each rank gets the sum of its chunk = expect_sum -
  {
    float *send, *recv;
    CUDA_CHECK(cudaMalloc(&send, bytes * world));
    CUDA_CHECK(cudaMalloc(&recv, bytes));
    std::vector<float> in(count * world, (float)(rank + 1));
    CUDA_CHECK(cudaMemcpy(send, in.data(), bytes * world,
                          cudaMemcpyHostToDevice));

    NCCL_CHECK(ncclReduceScatter(send, recv, count, ncclFloat, ncclSum, comm,
                                 stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    bool pass = check_all(recv, count, expect_sum);
    ok &= pass;
    if (rank == 0)
      std::printf("  ReduceScatter    : got %.0f, want %.0f -> %s\n",
                  sample(recv), expect_sum, pass ? "OK" : "FAIL");
    CUDA_CHECK(cudaFree(send));
    CUDA_CHECK(cudaFree(recv));
  }

  NCCL_CHECK(ncclCommDestroy(comm));
  CUDA_CHECK(cudaStreamDestroy(stream));

  std::printf("[rank %d] %s\n", rank, ok ? "OK" : "FAILED");
  if (rank == 0) std::remove(id_file);
  return ok ? 0 : 1;
}
