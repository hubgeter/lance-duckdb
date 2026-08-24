#pragma once

#include "duckdb.hpp"
#include "lance_ffi.hpp"

#include <cstdint>
#include <memory>

namespace duckdb {

enum class LanceLeaseKind : uint8_t {
  Snapshot = LANCE_LEASE_SNAPSHOT,
  Mutation = LANCE_LEASE_MUTATION,
  Vacuum = LANCE_LEASE_VACUUM,
};

//! RAII owner for a storage-backed Lance lease. The remote record is retained
//! when release fails, so an outcome-unknown mutation cannot be retried as if
//! it had never started.
class LanceLease {
public:
  static unique_ptr<LanceLease>
  Acquire(ClientContext &context, const string &path, LanceLeaseKind kind,
          const string &operation_id, const vector<string> &option_keys = {},
          const vector<string> &option_values = {}, uint64_t wait_ms = 0);
  static unique_ptr<LanceLease> AcquireForDataset(void *dataset,
                                                  LanceLeaseKind kind,
                                                  const string &operation_id,
                                                  uint64_t wait_ms = 0);

  LanceLease(const LanceLease &) = delete;
  LanceLease &operator=(const LanceLease &) = delete;
  ~LanceLease();

  void Release();
  void RetainForReconciliation() { release_on_destroy = false; }
  bool IsReleased() const { return handle == nullptr; }
  const string &Token() const { return token; }

private:
  explicit LanceLease(void *handle_p, string token_p)
      : handle(handle_p), token(std::move(token_p)) {}

  void *handle = nullptr;
  string token;
  bool release_on_destroy = true;
};

} // namespace duckdb
