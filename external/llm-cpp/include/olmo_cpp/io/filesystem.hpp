#pragma once

/**
 * include/olmo_cpp/io/filesystem.hpp
 *
 * Pluggable filesystem abstraction used by the trainer for checkpoints, data,
 * and config artefacts. A single FileSystem interface is implemented by Local
 * (std::filesystem), S3 (shells out to `aws s3`), GCS (shells out to `gsutil`),
 * and HTTP (shells out to `curl`). The factory `FileSystem::create(uri)`
 * dispatches by scheme so callers can work with `s3://bucket/key` or a plain
 * `/local/path` interchangeably.
 *
 * --- Includes from this project ---
 *   - none beyond STL: this header is intentionally self-contained so it can
 *     be pulled into checkpointing, data prep, and config loaders without
 *     dragging LibTorch.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/io/filesystem.cpp: implements every method declared here.
 *   (Direct callers in the trainer not located via quick grep — searched
 *   src/checkpoint and src/train without a hit; the API is intended for
 *   future remote checkpoint loaders.)
 *
 * --- Role in training pipeline ---
 *   Provides a uniform read/write/list/exists API for code paths that may run
 *   against local disk during dev and against object storage in production.
 *   Keeping this in one place avoids littering the trainer with `if cloud`
 *   branches.
 */

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <utility>

namespace olmo_cpp {

/// Abstract filesystem interface implemented by Local/S3/GCS/HTTP backends.
/// All methods take a backend-relative `path`; the concrete subclass knows
/// how to translate that into a URL or absolute path.
class FileSystem {
 public:
  virtual ~FileSystem() = default;

  /// Slurp the entire file into a byte buffer. Throws on missing file.
  virtual std::vector<uint8_t> read_file(const std::string& path) = 0;
  /// Overwrite (or create) the file at `path`. For tree-shaped backends,
  /// implementations are expected to create parent directories.
  virtual void write_file(const std::string& path, const std::vector<uint8_t>& data) = 0;
  /// Cheap existence probe; falls back to `ls` for cloud backends.
  virtual bool exists(const std::string& path) = 0;
  /// Non-recursive directory listing returning leaf names (no trailing '/').
  virtual std::vector<std::string> list_dir(const std::string& path) = 0;
  /// Create directory (no-op on object-store backends that lack real dirs).
  virtual void mkdir(const std::string& path) = 0;
  /// Recursive delete. Cloud backends use `--recursive`/`-r` flags.
  virtual void remove(const std::string& path) = 0;
  /// Size of a single file in bytes. Cloud backends parse CLI output.
  virtual size_t file_size(const std::string& path) = 0;

  /// Factory: parse the URI scheme (file://, s3://, gs://, http(s)://) and
  /// return the corresponding concrete backend; bare paths default to Local.
  static std::unique_ptr<FileSystem> create(const std::string& uri);
};

/// Local filesystem implementation backed entirely by C++17 <filesystem>.
/// This is the fast path used for on-node training; no shelling out.
class LocalFileSystem : public FileSystem {
 public:
  std::vector<uint8_t> read_file(const std::string& path) override;
  void write_file(const std::string& path, const std::vector<uint8_t>& data) override;
  bool exists(const std::string& path) override;
  std::vector<std::string> list_dir(const std::string& path) override;
  void mkdir(const std::string& path) override;
  void remove(const std::string& path) override;
  size_t file_size(const std::string& path) override;
};

/// S3 backend that shells out to the `aws` CLI rather than linking the
/// AWS C++ SDK. This keeps the build dependency surface tiny — the user
/// just needs `aws` on PATH and credentials in the environment.
class S3FileSystem : public FileSystem {
 public:
  /// `bucket` is the S3 bucket name (no scheme), `prefix` is an optional
  /// key-prefix appended before `path` on every operation.
  S3FileSystem(std::string bucket, std::string prefix = "");
  std::vector<uint8_t> read_file(const std::string& path) override;
  void write_file(const std::string& path, const std::vector<uint8_t>& data) override;
  bool exists(const std::string& path) override;
  std::vector<std::string> list_dir(const std::string& path) override;
  void mkdir(const std::string& path) override;
  void remove(const std::string& path) override;
  size_t file_size(const std::string& path) override;
 private:
  std::string bucket_, prefix_;
  /// Build a fully-qualified s3://bucket/prefix/path URI for CLI calls.
  std::string s3_uri(const std::string& path) const;
  /// Run a shell command (e.g., `aws s3 cp ...`); optionally captures stdout.
  int run_cli(const std::string& cmd, std::string* output = nullptr) const;
};

/// GCS backend, same shell-out pattern as S3FileSystem but using `gsutil`.
class GCSFileSystem : public FileSystem {
 public:
  /// `bucket` and optional `prefix` are joined into gs://bucket/prefix/path.
  GCSFileSystem(std::string bucket, std::string prefix = "");
  std::vector<uint8_t> read_file(const std::string& path) override;
  void write_file(const std::string& path, const std::vector<uint8_t>& data) override;
  bool exists(const std::string& path) override;
  std::vector<std::string> list_dir(const std::string& path) override;
  void mkdir(const std::string& path) override;
  void remove(const std::string& path) override;
  size_t file_size(const std::string& path) override;
 private:
  std::string bucket_, prefix_;
  /// Compose the gs://bucket/prefix/path URI for `gsutil`.
  std::string gs_uri(const std::string& path) const;
  /// Run a `gsutil ...` command; captures stdout if requested.
  int run_cli(const std::string& cmd, std::string* output = nullptr) const;
};

/// HTTP(S) read-only backend useful for fetching remote tokenized blobs and
/// public checkpoints. Write operations throw — this is intentionally a
/// pull-only filesystem.
class HTTPFileSystem : public FileSystem {
 public:
  /// `base_url` is the URL prefix; subsequent `path` arguments are appended.
  explicit HTTPFileSystem(std::string base_url);
  std::vector<uint8_t> read_file(const std::string& path) override;
  void write_file(const std::string& path, const std::vector<uint8_t>& data) override;
  bool exists(const std::string& path) override;
  std::vector<std::string> list_dir(const std::string& path) override;
  void mkdir(const std::string& path) override;
  void remove(const std::string& path) override;
  size_t file_size(const std::string& path) override;
 private:
  std::string base_url_;
  /// Fetch `url` via `curl -f` into a temp file, then slurp into memory.
  std::vector<uint8_t> curl_get(const std::string& url) const;
};

/// Split a URI into (scheme, remainder). For example:
///   "s3://bucket/key" -> {"s3", "bucket/key"}
///   "/local/path"     -> {"",   "/local/path"}
/// Used by FileSystem::create to dispatch to the right backend.
std::pair<std::string, std::string> parse_uri(const std::string& uri);

}  // namespace olmo_cpp
