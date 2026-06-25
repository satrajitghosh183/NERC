/**
 * src/io/filesystem.cpp
 *
 * Implementations of the FileSystem hierarchy declared in
 * include/olmo_cpp/io/filesystem.hpp. Local filesystem uses C++17
 * `<filesystem>`; cloud and HTTP backends shell out to `aws`, `gsutil`,
 * and `curl` so the trainer doesn't have to link the AWS/GCS C++ SDKs.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/io/filesystem.hpp: declarations of every class implemented
 *     below; everything else is STL.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   Direct callers not located via quick grep — the API is plumbed for
 *   future remote-checkpoint loaders; today only LocalFileSystem is
 *   exercised on the hot path.
 *
 * --- Role in training pipeline ---
 *   Provides the read/write/exists/list primitives the rest of the codebase
 *   uses for non-tensor I/O (configs, JSONL eval data, checkpoint metadata).
 */

#include "olmo_cpp/io/filesystem.hpp"

#include <filesystem>     // std::filesystem for LocalFileSystem
#include <fstream>        // ifstream/ofstream for binary read/write
#include <sstream>        // istringstream for parsing CLI output
#include <stdexcept>      // runtime_error on I/O failure
#include <cstdio>         // popen/pclose/fgets for run_command
#include <cstring>        // (legacy include; kept for portability)
#include <algorithm>      // std::sort on directory listings
#include <array>          // fixed-size buffer for the popen pipe

// Short alias so the LocalFileSystem methods read more like Python.
namespace fs = std::filesystem;

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// URI parsing
// ---------------------------------------------------------------------------

/// Split a URI on the canonical "://" separator. Anything without it is
/// treated as a bare local path and gets an empty scheme returned.
std::pair<std::string, std::string> parse_uri(const std::string& uri) {
  // Look for the scheme delimiter; absence means we have a plain path.
  auto pos = uri.find("://");
  if (pos == std::string::npos) return {"", uri};  // local path
  // Skip the three "://" characters when extracting the remainder.
  return {uri.substr(0, pos), uri.substr(pos + 3)};
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Factory: dispatch on URI scheme. Bucket+prefix are split at the first '/'
/// so callers can pass `s3://my-bucket/some/prefix` and get future paths
/// joined relative to that prefix.
std::unique_ptr<FileSystem> FileSystem::create(const std::string& uri) {
  auto [scheme, rest] = parse_uri(uri);
  if (scheme.empty() || scheme == "file") {
    // No scheme or explicit file:// => local disk.
    return std::make_unique<LocalFileSystem>();
  } else if (scheme == "s3") {
    // s3://bucket  -> bucket only, no prefix
    // s3://bucket/foo/bar -> bucket="bucket", prefix="foo/bar"
    auto slash = rest.find('/');
    if (slash == std::string::npos) return std::make_unique<S3FileSystem>(rest);
    return std::make_unique<S3FileSystem>(rest.substr(0, slash), rest.substr(slash + 1));
  } else if (scheme == "gs") {
    // Same split logic as S3 but for Google Cloud Storage.
    auto slash = rest.find('/');
    if (slash == std::string::npos) return std::make_unique<GCSFileSystem>(rest);
    return std::make_unique<GCSFileSystem>(rest.substr(0, slash), rest.substr(slash + 1));
  } else if (scheme == "http" || scheme == "https") {
    // HTTP(S) preserves the full URI as the base; subsequent reads append
    // their relative path with `/`.
    return std::make_unique<HTTPFileSystem>(uri);
  }
  throw std::runtime_error("Unsupported URI scheme: " + scheme);
}

// ---------------------------------------------------------------------------
// Helper: run shell command, capture stdout
// ---------------------------------------------------------------------------

/// Fork+exec a shell command and capture its stdout into `output` (when
/// non-null). Returns the child's exit status. Used by every cloud backend
/// because we deliberately avoid linking the AWS/GCS C++ SDKs.
static int run_command(const std::string& cmd, std::string* output) {
  std::array<char, 4096> buf;     // page-sized read buffer
  std::string result;             // accumulated stdout
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) return -1;           // popen failure (no fork etc.)
  // Drain stdout one chunk at a time; null-terminator from fgets is fine
  // because we append via std::string operator+=.
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
    result += buf.data();
  }
  int status = pclose(pipe);      // reaps the child; returns wait status
  if (output) *output = std::move(result);
  // WEXITSTATUS pulls the actual exit code from the wait status word.
  return WEXITSTATUS(status);
}

// ---------------------------------------------------------------------------
// LocalFileSystem
// ---------------------------------------------------------------------------

/// Slurp `path` into a byte vector. Opens at end (`ios::ate`) to size the
/// buffer in one tellg() call, then seeks back and bulk-reads.
std::vector<uint8_t> LocalFileSystem::read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("Cannot open file: " + path);
  auto sz = in.tellg();                                  // file size
  in.seekg(0);                                           // rewind for read
  std::vector<uint8_t> data(static_cast<size_t>(sz));    // single allocation
  in.read(reinterpret_cast<char*>(data.data()), sz);     // bulk binary read
  return data;
}

/// Atomic-ish write: ensure parent dirs exist, then dump bytes. We don't
/// write-through-rename; that's a TODO for crash-safe checkpoints.
void LocalFileSystem::write_file(const std::string& path, const std::vector<uint8_t>& data) {
  // Make every missing component of the parent path; safe if it exists.
  fs::create_directories(fs::path(path).parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot write file: " + path);
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

/// Existence probe — wraps std::filesystem::exists.
bool LocalFileSystem::exists(const std::string& path) {
  return fs::exists(path);
}

/// Non-recursive listing returning sorted leaf names. Returns empty vector
/// when the directory is missing rather than throwing — easier for callers.
std::vector<std::string> LocalFileSystem::list_dir(const std::string& path) {
  std::vector<std::string> entries;
  if (!fs::exists(path)) return entries;
  for (auto& e : fs::directory_iterator(path)) {
    // Push only the leaf name, not the full path — matches cloud behavior.
    entries.push_back(e.path().filename().string());
  }
  // Lexicographic sort gives deterministic ordering across runs.
  std::sort(entries.begin(), entries.end());
  return entries;
}

/// `create_directories` is idempotent and creates intermediate parents.
void LocalFileSystem::mkdir(const std::string& path) {
  fs::create_directories(path);
}

/// Recursive delete (file or whole subtree).
void LocalFileSystem::remove(const std::string& path) {
  // Non-throwing: prune() runs on every rank, so two ranks can race to remove
  // the same checkpoint dir. The throwing remove_all() would make the loser
  // throw "No such file or directory" mid-recursion, killing that rank and
  // deadlocking the survivor on the next NCCL collective. error_code tolerates
  // already-gone / racily-vanishing paths.
  std::error_code ec;
  fs::remove_all(path, ec);
}

/// File size in bytes. Throws via std::filesystem on missing/non-regular.
size_t LocalFileSystem::file_size(const std::string& path) {
  return static_cast<size_t>(fs::file_size(path));
}

// ---------------------------------------------------------------------------
// S3FileSystem
// ---------------------------------------------------------------------------

/// Move-construct bucket and prefix; both are appended verbatim into URIs.
S3FileSystem::S3FileSystem(std::string bucket, std::string prefix)
    : bucket_(std::move(bucket)), prefix_(std::move(prefix)) {}

/// Build an `s3://bucket[/prefix][/path]` URI for the AWS CLI.
std::string S3FileSystem::s3_uri(const std::string& path) const {
  std::string full = "s3://" + bucket_;
  if (!prefix_.empty()) full += "/" + prefix_;  // optional bucket sub-path
  if (!path.empty()) full += "/" + path;        // caller-supplied key
  return full;
}

/// Thin forwarder so the cmd construction stays inside the class while the
/// process plumbing lives in the file-local helper.
int S3FileSystem::run_cli(const std::string& cmd, std::string* output) const {
  return run_command(cmd, output);
}

/// Download via `aws s3 cp` to a temp file, slurp it, then delete the temp.
/// The temp filename uses a hash of the key so concurrent reads of distinct
/// keys don't collide.
std::vector<uint8_t> S3FileSystem::read_file(const std::string& path) {
  std::string tmp = "/tmp/olmo_s3_" + std::to_string(std::hash<std::string>{}(path));
  std::string cmd = "aws s3 cp " + s3_uri(path) + " " + tmp + " 2>/dev/null";
  if (run_cli(cmd) != 0) throw std::runtime_error("S3 read failed: " + path);
  LocalFileSystem local;
  auto data = local.read_file(tmp);
  std::remove(tmp.c_str());                 // best-effort cleanup
  return data;
}

/// Symmetric write: stage to /tmp, then push with `aws s3 cp`.
void S3FileSystem::write_file(const std::string& path, const std::vector<uint8_t>& data) {
  std::string tmp = "/tmp/olmo_s3_" + std::to_string(std::hash<std::string>{}(path));
  LocalFileSystem local;
  local.write_file(tmp, data);              // serialize bytes locally first
  std::string cmd = "aws s3 cp " + tmp + " " + s3_uri(path) + " 2>/dev/null";
  if (run_cli(cmd) != 0) throw std::runtime_error("S3 write failed: " + path);
  std::remove(tmp.c_str());
}

/// Existence: `aws s3 ls` exits 0 iff the key matched. We don't capture
/// stdout because we only care about the exit code.
bool S3FileSystem::exists(const std::string& path) {
  std::string cmd = "aws s3 ls " + s3_uri(path) + " 2>/dev/null";
  return run_cli(cmd) == 0;
}

/// Parse the columnar output of `aws s3 ls`. Each line is either
/// "<date> <time> <size> <key>" for files or "PRE <dir>/" for sub-prefixes.
/// We just take the last whitespace-separated token and strip a trailing '/'.
std::vector<std::string> S3FileSystem::list_dir(const std::string& path) {
  std::string output;
  std::string cmd = "aws s3 ls " + s3_uri(path) + "/ 2>/dev/null";
  run_cli(cmd, &output);
  std::vector<std::string> entries;
  std::istringstream iss(output);
  std::string line;
  while (std::getline(iss, line)) {
    // aws s3 ls output: "2024-01-01 00:00:00  1234 filename" or "PRE dirname/"
    auto last_space = line.rfind(' ');
    if (last_space != std::string::npos) {
      std::string name = line.substr(last_space + 1);
      // Sub-prefix lines look like "PRE foo/" — drop the trailing slash so
      // the returned name matches the local-FS convention.
      if (!name.empty() && name.back() == '/') name.pop_back();
      if (!name.empty()) entries.push_back(name);
    }
  }
  return entries;
}

/// S3 has no real directories; creating one is a no-op. Keys with '/' just
/// happen to *display* hierarchically in many tools.
void S3FileSystem::mkdir(const std::string& /*path*/) {
  // S3 doesn't have real directories; no-op
}

/// Recursive object delete via `aws s3 rm --recursive`.
void S3FileSystem::remove(const std::string& path) {
  std::string cmd = "aws s3 rm --recursive " + s3_uri(path) + " 2>/dev/null";
  run_cli(cmd);
}

/// Pull the size column out of `aws s3 ls`. The CLI prints
/// `<date> <time> <bytes> <key>`; we extract the third whitespace token.
size_t S3FileSystem::file_size(const std::string& path) {
  std::string output;
  std::string cmd = "aws s3 ls " + s3_uri(path) + " 2>/dev/null";
  run_cli(cmd, &output);
  // Parse "2024-01-01 00:00:00  1234 filename"
  std::istringstream iss(output);
  std::string d, t;        // date, time — discarded
  size_t sz = 0;
  iss >> d >> t >> sz;     // operator>> on size_t parses the byte count
  return sz;
}

// ---------------------------------------------------------------------------
// GCSFileSystem
// ---------------------------------------------------------------------------

GCSFileSystem::GCSFileSystem(std::string bucket, std::string prefix)
    : bucket_(std::move(bucket)), prefix_(std::move(prefix)) {}

std::string GCSFileSystem::gs_uri(const std::string& path) const {
  std::string full = "gs://" + bucket_;
  if (!prefix_.empty()) full += "/" + prefix_;
  if (!path.empty()) full += "/" + path;
  return full;
}

int GCSFileSystem::run_cli(const std::string& cmd, std::string* output) const {
  return run_command(cmd, output);
}

std::vector<uint8_t> GCSFileSystem::read_file(const std::string& path) {
  std::string tmp = "/tmp/olmo_gcs_" + std::to_string(std::hash<std::string>{}(path));
  std::string cmd = "gsutil cp " + gs_uri(path) + " " + tmp + " 2>/dev/null";
  if (run_cli(cmd) != 0) throw std::runtime_error("GCS read failed: " + path);
  LocalFileSystem local;
  auto data = local.read_file(tmp);
  std::remove(tmp.c_str());
  return data;
}

void GCSFileSystem::write_file(const std::string& path, const std::vector<uint8_t>& data) {
  std::string tmp = "/tmp/olmo_gcs_" + std::to_string(std::hash<std::string>{}(path));
  LocalFileSystem local;
  local.write_file(tmp, data);
  std::string cmd = "gsutil cp " + tmp + " " + gs_uri(path) + " 2>/dev/null";
  if (run_cli(cmd) != 0) throw std::runtime_error("GCS write failed: " + path);
  std::remove(tmp.c_str());
}

bool GCSFileSystem::exists(const std::string& path) {
  std::string cmd = "gsutil ls " + gs_uri(path) + " 2>/dev/null";
  return run_cli(cmd) == 0;
}

std::vector<std::string> GCSFileSystem::list_dir(const std::string& path) {
  std::string output;
  std::string cmd = "gsutil ls " + gs_uri(path) + "/ 2>/dev/null";
  run_cli(cmd, &output);
  std::vector<std::string> entries;
  std::istringstream iss(output);
  std::string line;
  while (std::getline(iss, line)) {
    auto last_slash = line.rfind('/');
    if (last_slash != std::string::npos && last_slash < line.size() - 1) {
      entries.push_back(line.substr(last_slash + 1));
    } else if (last_slash == line.size() - 1 && line.size() > 1) {
      auto prev_slash = line.rfind('/', last_slash - 1);
      if (prev_slash != std::string::npos) {
        entries.push_back(line.substr(prev_slash + 1, last_slash - prev_slash - 1));
      }
    }
  }
  return entries;
}

void GCSFileSystem::mkdir(const std::string& /*path*/) {
  // GCS doesn't have real directories
}

void GCSFileSystem::remove(const std::string& path) {
  std::string cmd = "gsutil -m rm -r " + gs_uri(path) + " 2>/dev/null";
  run_cli(cmd);
}

size_t GCSFileSystem::file_size(const std::string& path) {
  std::string output;
  std::string cmd = "gsutil du -s " + gs_uri(path) + " 2>/dev/null";
  run_cli(cmd, &output);
  size_t sz = 0;
  std::istringstream iss(output);
  iss >> sz;
  return sz;
}

// ---------------------------------------------------------------------------
// HTTPFileSystem (read-only)
// ---------------------------------------------------------------------------

HTTPFileSystem::HTTPFileSystem(std::string base_url) : base_url_(std::move(base_url)) {
  // Strip trailing slash
  while (!base_url_.empty() && base_url_.back() == '/') base_url_.pop_back();
}

std::vector<uint8_t> HTTPFileSystem::curl_get(const std::string& url) const {
  std::string tmp = "/tmp/olmo_http_" + std::to_string(std::hash<std::string>{}(url));
  std::string cmd = "curl -sS -f -o " + tmp + " '" + url + "' 2>/dev/null";
  if (run_command(cmd, nullptr) != 0) {
    throw std::runtime_error("HTTP GET failed: " + url);
  }
  LocalFileSystem local;
  auto data = local.read_file(tmp);
  std::remove(tmp.c_str());
  return data;
}

std::vector<uint8_t> HTTPFileSystem::read_file(const std::string& path) {
  std::string url = base_url_;
  if (!path.empty()) url += "/" + path;
  return curl_get(url);
}

void HTTPFileSystem::write_file(const std::string& /*path*/, const std::vector<uint8_t>& /*data*/) {
  throw std::runtime_error("HTTPFileSystem is read-only");
}

bool HTTPFileSystem::exists(const std::string& path) {
  std::string url = base_url_;
  if (!path.empty()) url += "/" + path;
  std::string cmd = "curl -sS -f -I '" + url + "' >/dev/null 2>&1";
  return run_command(cmd, nullptr) == 0;
}

std::vector<std::string> HTTPFileSystem::list_dir(const std::string& /*path*/) {
  throw std::runtime_error("HTTPFileSystem does not support list_dir");
}

void HTTPFileSystem::mkdir(const std::string& /*path*/) {
  throw std::runtime_error("HTTPFileSystem is read-only");
}

void HTTPFileSystem::remove(const std::string& /*path*/) {
  throw std::runtime_error("HTTPFileSystem is read-only");
}

size_t HTTPFileSystem::file_size(const std::string& path) {
  std::string url = base_url_;
  if (!path.empty()) url += "/" + path;
  std::string output;
  std::string cmd = "curl -sS -I '" + url + "' 2>/dev/null | grep -i content-length";
  run_command(cmd, &output);
  // Parse "Content-Length: 12345"
  auto pos = output.find(':');
  if (pos == std::string::npos) return 0;
  return static_cast<size_t>(std::stoull(output.substr(pos + 1)));
}

}  // namespace olmo_cpp
