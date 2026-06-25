// nccl_backend.cu — real NCCL all-reduce + TCP rendezvous, exposed via a small
// C ABI so the rest of the dist code can stay plain C++ and the NCCL/CUDA
// headers don't bleed into public headers.
//
// Build inclusion: this TU is appended to ZWT_CUDA_SOURCES only when CMake
// finds both nccl.h and libnccl. So if we're compiling, NCCL is available.
//
// Wire protocol (rank 0 ↔ all other ranks, single TCP socket per rank):
//   1. Rank 0 opens a listening socket on $MASTER_ADDR:$MASTER_PORT.
//   2. Non-zero ranks connect (with retry/backoff: rank 0 may still be racing
//      to listen()).
//   3. Rank 0 sends sizeof(ncclUniqueId) bytes of its generated ID to every
//      child connection. Non-zero ranks read.
//   4. Everyone calls ncclCommInitRank with that ID + their rank.
//
// This is the same shape as PyTorch's TCPStore-based init, just inlined.
// Keeps zwt free of an extra rendezvous library and is forward-compatible to
// multinode (the only knob is "what hostname does rank 0 bind on").

#include <cuda_runtime.h>
#include <nccl.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace {

void set_err(char* dst, size_t cap, const char* msg) {
  if (!dst || cap == 0) return;
  std::snprintf(dst, cap, "%s", msg);
}

void set_errf(char* dst, size_t cap, const char* fmt, ...) {
  if (!dst || cap == 0) return;
  va_list ap; va_start(ap, fmt);
  std::vsnprintf(dst, cap, fmt, ap);
  va_end(ap);
}

// ---- TCP rendezvous ------------------------------------------------------

constexpr int  kAcceptBacklog = 64;
constexpr int  kConnectRetryMaxMs = 60'000;   // 60s ceiling — caller fail-fast
constexpr int  kConnectRetryStepMs = 200;

int set_socket_opts(int fd) {
  int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  return 0;
}

// Rank 0: bind+listen+accept(world-1)+send(uid)+close.
int rendezvous_server(int port, int world,
                      const ncclUniqueId& uid,
                      char* errbuf, size_t errlen) {
  int srv = ::socket(AF_INET, SOCK_STREAM, 0);
  if (srv < 0) {
    set_errf(errbuf, errlen, "socket() failed: %s", std::strerror(errno));
    return -1;
  }
  set_socket_opts(srv);
  sockaddr_in addr{};
  addr.sin_family      = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port        = htons(static_cast<uint16_t>(port));
  if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    set_errf(errbuf, errlen, "bind(%d) failed: %s", port, std::strerror(errno));
    ::close(srv); return -1;
  }
  if (::listen(srv, kAcceptBacklog) < 0) {
    set_errf(errbuf, errlen, "listen() failed: %s", std::strerror(errno));
    ::close(srv); return -1;
  }
  for (int peer = 1; peer < world; ++peer) {
    int cli = ::accept(srv, nullptr, nullptr);
    if (cli < 0) {
      set_errf(errbuf, errlen, "accept(peer=%d) failed: %s",
               peer, std::strerror(errno));
      ::close(srv); return -1;
    }
    set_socket_opts(cli);
    const char* p = reinterpret_cast<const char*>(&uid);
    size_t left = sizeof(uid);
    while (left > 0) {
      ssize_t w = ::send(cli, p, left, 0);
      if (w <= 0) {
        set_errf(errbuf, errlen, "send(uid → peer=%d) failed: %s",
                 peer, std::strerror(errno));
        ::close(cli); ::close(srv); return -1;
      }
      p += w; left -= static_cast<size_t>(w);
    }
    ::close(cli);
  }
  ::close(srv);
  return 0;
}

// Non-zero ranks: connect with retry, then read sizeof(uid) bytes.
int rendezvous_client(const char* host, int port,
                      ncclUniqueId* out_uid,
                      char* errbuf, size_t errlen) {
  // Resolve once; retry connect.
  addrinfo hints{}; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  char portstr[16]; std::snprintf(portstr, sizeof(portstr), "%d", port);
  int gai = ::getaddrinfo(host, portstr, &hints, &res);
  if (gai != 0 || !res) {
    set_errf(errbuf, errlen, "getaddrinfo(%s:%d) failed: %s",
             host, port, gai_strerror(gai));
    if (res) ::freeaddrinfo(res);
    return -1;
  }

  int fd = -1;
  auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) {
      set_errf(errbuf, errlen, "socket() failed: %s", std::strerror(errno));
      ::freeaddrinfo(res); return -1;
    }
    set_socket_opts(fd);
    if (::connect(fd, res->ai_addr, res->ai_addrlen) == 0) break;
    int e = errno;
    ::close(fd); fd = -1;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    if (elapsed_ms > kConnectRetryMaxMs) {
      set_errf(errbuf, errlen, "connect(%s:%d) timed out after %lldms: %s",
               host, port, (long long)elapsed_ms, std::strerror(e));
      ::freeaddrinfo(res); return -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kConnectRetryStepMs));
  }
  ::freeaddrinfo(res);

  char* p = reinterpret_cast<char*>(out_uid);
  size_t left = sizeof(*out_uid);
  while (left > 0) {
    ssize_t r = ::recv(fd, p, left, 0);
    if (r <= 0) {
      set_errf(errbuf, errlen, "recv(uid) failed: %s", std::strerror(errno));
      ::close(fd); return -1;
    }
    p += r; left -= static_cast<size_t>(r);
  }
  ::close(fd);
  return 0;
}

}  // namespace

// ---- C ABI ---------------------------------------------------------------

extern "C" {

void* zwt_nccl_init(int rank, int world_size,
                    const char* master_addr, int master_port,
                    int device_index,
                    char* errbuf, size_t errlen) {
  if (world_size < 1 || rank < 0 || rank >= world_size) {
    set_err(errbuf, errlen, "invalid rank/world_size");
    return nullptr;
  }
  if (cudaSetDevice(device_index) != cudaSuccess) {
    set_errf(errbuf, errlen, "cudaSetDevice(%d) failed: %s",
             device_index, cudaGetErrorString(cudaGetLastError()));
    return nullptr;
  }

  ncclUniqueId uid;
  if (rank == 0) {
    ncclResult_t r = ncclGetUniqueId(&uid);
    if (r != ncclSuccess) {
      set_errf(errbuf, errlen, "ncclGetUniqueId: %s", ncclGetErrorString(r));
      return nullptr;
    }
    if (rendezvous_server(master_port, world_size, uid, errbuf, errlen) != 0) {
      return nullptr;
    }
  } else {
    if (rendezvous_client(master_addr, master_port, &uid, errbuf, errlen) != 0) {
      return nullptr;
    }
  }

  ncclComm_t comm = nullptr;
  ncclResult_t r = ncclCommInitRank(&comm, world_size, uid, rank);
  if (r != ncclSuccess) {
    set_errf(errbuf, errlen, "ncclCommInitRank: %s", ncclGetErrorString(r));
    return nullptr;
  }
  return reinterpret_cast<void*>(comm);
}

int zwt_nccl_allreduce_avg_f32(void* handle, void* buf, size_t count,
                               void* stream,
                               char* errbuf, size_t errlen) {
  if (!handle || !buf || count == 0) {
    set_err(errbuf, errlen, "allreduce: null handle/buf or zero count");
    return -1;
  }
  ncclComm_t comm = reinterpret_cast<ncclComm_t>(handle);
  cudaStream_t s  = reinterpret_cast<cudaStream_t>(stream);
  // ncclAvg performs sum-then-divide-by-world_size in one pass. Available
  // since NCCL 2.10 (CUDA 12 ships much newer). This lets BucketManager
  // skip a separate divide kernel during scatter.
  ncclResult_t r = ncclAllReduce(buf, buf, count, ncclFloat32, ncclAvg, comm, s);
  if (r != ncclSuccess) {
    set_errf(errbuf, errlen, "ncclAllReduce: %s", ncclGetErrorString(r));
    return -1;
  }
  return 0;
}

void zwt_nccl_destroy(void* handle) {
  if (!handle) return;
  ncclComm_t comm = reinterpret_cast<ncclComm_t>(handle);
  ncclCommDestroy(comm);
}

void zwt_nccl_abort(void* handle) {
  if (!handle) return;
  ncclComm_t comm = reinterpret_cast<ncclComm_t>(handle);
  ncclCommAbort(comm);
}

}  // extern "C"
