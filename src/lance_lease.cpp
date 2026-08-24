#include "lance_lease.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"
#include "lance_session_state.hpp"

#include "duckdb/common/exception.hpp"

namespace duckdb {

static const char *LanceLeaseKindName(LanceLeaseKind kind) {
  switch (kind) {
  case LanceLeaseKind::Snapshot:
    return "snapshot";
  case LanceLeaseKind::Mutation:
    return "mutation";
  case LanceLeaseKind::Vacuum:
    return "vacuum";
  default:
    return "unknown";
  }
}

unique_ptr<LanceLease>
LanceLease::Acquire(ClientContext &context, const string &path,
                    LanceLeaseKind kind, const string &operation_id,
                    const vector<string> &option_keys,
                    const vector<string> &option_values, uint64_t wait_ms) {
  if (option_keys.size() != option_values.size()) {
    throw InternalException(
        "Lance lease storage option key/value size mismatch");
  }

  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                  value_ptrs);
  auto *handle = lance_acquire_lease_with_storage_options(
      path.c_str(), static_cast<uint8_t>(kind), operation_id.c_str(),
      key_ptrs.empty() ? nullptr : key_ptrs.data(),
      value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size(),
      LanceGetSessionHandle(context), wait_ms);
  if (!handle) {
    throw IOException("Failed to acquire Lance " +
                      string(LanceLeaseKindName(kind)) + " lease for '" + path +
                      "'" + LanceFormatErrorSuffix());
  }

  auto *token_ptr = lance_lease_token(handle);
  if (!token_ptr) {
    lance_close_lease(handle);
    throw IOException("Lance lease was acquired but its token could not be "
                      "read for '" +
                      path + "'" + LanceFormatErrorSuffix());
  }
  string token(token_ptr);
  lance_free_string(token_ptr);
  return unique_ptr<LanceLease>(new LanceLease(handle, std::move(token)));
}

unique_ptr<LanceLease> LanceLease::AcquireForDataset(void *dataset,
                                                     LanceLeaseKind kind,
                                                     const string &operation_id,
                                                     uint64_t wait_ms) {
  auto *handle = lance_acquire_dataset_lease(
      dataset, static_cast<uint8_t>(kind), operation_id.c_str(), wait_ms);
  if (!handle) {
    throw IOException("Failed to acquire Lance " +
                      string(LanceLeaseKindName(kind)) + " lease for dataset" +
                      LanceFormatErrorSuffix());
  }
  auto *token_ptr = lance_lease_token(handle);
  if (!token_ptr) {
    lance_close_lease(handle);
    throw IOException("Lance dataset lease was acquired but its token could "
                      "not be read" +
                      LanceFormatErrorSuffix());
  }
  string token(token_ptr);
  lance_free_string(token_ptr);
  return unique_ptr<LanceLease>(new LanceLease(handle, std::move(token)));
}

void LanceLease::Release() {
  if (!handle) {
    return;
  }
  if (lance_release_lease(handle) != 0) {
    // Keep the opaque handle alive so a caller can retry or explicitly
    // reconcile it. In particular, do not let the destructor turn a failed
    // release into an apparent success.
    throw IOException("Failed to release Lance lease token '" + token + "'" +
                      LanceFormatErrorSuffix());
  }
  lance_close_lease(handle);
  handle = nullptr;
  release_on_destroy = false;
}

LanceLease::~LanceLease() {
  if (!handle) {
    return;
  }
  if (release_on_destroy) {
    // Destructors cannot report an error. A failed release intentionally leaves
    // the storage record in place; recovery tooling must remove it after
    // proving that the owner is gone.
    (void)lance_release_lease(handle);
  }
  lance_close_lease(handle);
  handle = nullptr;
}

} // namespace duckdb
