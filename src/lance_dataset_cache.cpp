#include "lance_dataset_cache.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"
#include "lance_session_state.hpp"
#include "lance_table_entry.hpp"

#include "duckdb/common/types/hash.hpp"
#include "duckdb/main/client_context_state.hpp"

#include <functional>

namespace duckdb {

static constexpr const char *LANCE_DATASET_CACHE_STATE_KEY =
    "lance_dataset_cache_state";

class LanceDatasetCacheState final : public ClientContextState {
public:
  shared_ptr<LanceDatasetCacheEntry> Get(const string &key) {
    lock_guard<mutex> guard(lock);
    auto entry = entries.find(key);
    if (entry == entries.end()) {
      query_misses++;
      return nullptr;
    }
    query_hits++;
    return entry->second;
  }

  shared_ptr<LanceDatasetCacheEntry>
  PutOrGetExisting(const string &key,
                   shared_ptr<LanceDatasetCacheEntry> entry) {
    lock_guard<mutex> guard(lock);
    auto existing = entries.find(key);
    if (existing != entries.end()) {
      return existing->second;
    }
    entries[key] = entry;
    return entry;
  }

  shared_ptr<LanceDatasetCacheEntry>
  PutOrGetNewest(const string &key, shared_ptr<LanceDatasetCacheEntry> entry,
                 bool &out_cache_hit) {
    lock_guard<mutex> guard(lock);
    auto existing = entries.find(key);
    if (existing != entries.end()) {
      auto existing_version = lance_dataset_version(existing->second->Handle());
      auto opened_version = lance_dataset_version(entry->Handle());
      bool keep_existing =
          existing_version != 0 && existing_version == opened_version &&
          existing->second->GenerationId() == entry->GenerationId();
      if (!keep_existing && existing_version > opened_version &&
          opened_version != 0) {
        // Two concurrent latest opens can complete out of order.  Check whether
        // the handle that appeared older can still reach the cached snapshot:
        // if it can, both handles belong to the same dataset lineage and the
        // higher version wins.  If it cannot (or the generation differs), the
        // dataset was replaced at the same cache key and the freshly opened
        // handle must replace the stale cache entry even though its version
        // reset.
        auto *same_version =
            lance_dataset_checkout_version(entry->Handle(), existing_version);
        if (same_version) {
          auto *generation = lance_dataset_generation_id(same_version);
          keep_existing =
              generation && existing->second->GenerationId() == generation;
          if (generation) {
            lance_free_string(generation);
          }
          lance_close_dataset(same_version);
        } else {
          // Clear the expected checkout error before returning to unrelated
          // callers; a successful FFI call clears the thread-local error state.
          (void)lance_dataset_version(entry->Handle());
        }
      }
      if (keep_existing) {
        query_hits++;
        out_cache_hit = true;
        return existing->second;
      }
      existing->second = std::move(entry);
    } else {
      entries[key] = std::move(entry);
    }
    query_misses++;
    out_cache_hit = false;
    return entries[key];
  }

  void Invalidate(const string &key) {
    lock_guard<mutex> guard(lock);
    entries.erase(key);
  }

  void QueryBegin(ClientContext &) override {
    lock_guard<mutex> guard(lock);
    query_hits = 0;
    query_misses = 0;
  }

  void WriteProfilingInformation(std::ostream &ss) override {
    lock_guard<mutex> guard(lock);
    ss << "Lance Dataset Cache: entries=" << entries.size()
       << " hits=" << query_hits << " misses=" << query_misses << "\n";
  }

private:
  mutex lock;
  unordered_map<string, shared_ptr<LanceDatasetCacheEntry>> entries;
  idx_t query_hits = 0;
  idx_t query_misses = 0;
};

LanceDatasetCacheEntry::LanceDatasetCacheEntry(void *dataset_p,
                                               string display_uri_p)
    : dataset(dataset_p), display_uri(std::move(display_uri_p)) {
  auto *generation_id_ptr = lance_dataset_generation_id(dataset);
  if (!generation_id_ptr) {
    auto suffix = LanceFormatErrorSuffix();
    lance_close_dataset(dataset);
    dataset = nullptr;
    throw IOException("Failed to identify Lance dataset generation" + suffix);
  }
  generation_id = generation_id_ptr;
  lance_free_string(generation_id_ptr);
  if (generation_id.empty()) {
    lance_close_dataset(dataset);
    dataset = nullptr;
    throw IOException("Lance dataset generation identity is empty");
  }
}

LanceDatasetCacheEntry::~LanceDatasetCacheEntry() {
  if (dataset) {
    lance_close_dataset(dataset);
    dataset = nullptr;
  }
}

static shared_ptr<LanceDatasetCacheState>
GetOrCreateLanceDatasetCacheState(ClientContext &context) {
  return context.registered_state->GetOrCreate<LanceDatasetCacheState>(
      LANCE_DATASET_CACHE_STATE_KEY);
}

static void AppendCacheKeyPart(string &key, const string &value) {
  key += to_string(value.size());
  key += ':';
  key += value;
  key += ';';
}

static void AppendCacheKeyPart(string &key, idx_t value) {
  AppendCacheKeyPart(key, to_string(value));
}

static string FingerprintCacheKeyPart(const string &value) {
  return to_string(Hash(value.c_str(), value.size()));
}

string
LanceBuildResolvedPathDatasetCacheKey(const string &open_path,
                                      const vector<string> &option_keys,
                                      const vector<string> &option_values) {
  if (option_keys.size() != option_values.size()) {
    throw InternalException(
        "Storage option keys/values size mismatch for Lance dataset cache");
  }

  string key = "path|";
  AppendCacheKeyPart(key, open_path);
  AppendCacheKeyPart(key, option_keys.size());
  for (idx_t i = 0; i < option_keys.size(); i++) {
    AppendCacheKeyPart(key, option_keys[i]);
    // Storage option values can contain credentials.  They still need to
    // participate in cache identity, but must never be retained verbatim in
    // a long-lived ClientContext state or exposed through diagnostics.
    AppendCacheKeyPart(key, FingerprintCacheKeyPart(option_values[i]));
  }
  return key;
}

string LanceBuildPathDatasetCacheKey(ClientContext &context,
                                     const string &path) {
  string open_path;
  vector<string> option_keys;
  vector<string> option_values;
  ResolveLanceStorageOptions(context, path, open_path, option_keys,
                             option_values);
  return LanceBuildResolvedPathDatasetCacheKey(open_path, option_keys,
                                               option_values);
}

string LanceBuildNamespaceDatasetCacheKey(
    const string &endpoint, const string &table_id, const string &bearer_token,
    const string &api_key, const string &delimiter, const string &headers_tsv) {
  string key = "namespace|";
  AppendCacheKeyPart(key, endpoint);
  AppendCacheKeyPart(key, table_id);
  AppendCacheKeyPart(key, FingerprintCacheKeyPart(bearer_token));
  AppendCacheKeyPart(key, FingerprintCacheKeyPart(api_key));
  AppendCacheKeyPart(key, delimiter);
  AppendCacheKeyPart(key, FingerprintCacheKeyPart(headers_tsv));
  return key;
}

static string
LanceBuildDirNamespaceDatasetCacheKey(const LanceNamespaceTableConfig &cfg) {
  return LanceBuildResolvedPathDatasetCacheKey(
      LanceDirectoryNamespaceDatasetUri(cfg), cfg.option_keys,
      cfg.option_values);
}

static void *OpenResolvedPathDataset(ClientContext &context,
                                     const string &open_path,
                                     const vector<string> &option_keys,
                                     const vector<string> &option_values) {
  auto *session = LanceGetSessionHandle(context);
  if (option_keys.empty()) {
    return lance_open_dataset_with_session(open_path.c_str(), session);
  }

  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                  value_ptrs);
  return lance_open_dataset_with_storage_options_and_session(
      open_path.c_str(), key_ptrs.data(), value_ptrs.data(), option_keys.size(),
      session);
}

static void *
OpenNamespaceDataset(ClientContext &context, const string &endpoint,
                     const string &table_id, const string &bearer_token,
                     const string &api_key, const string &delimiter,
                     const string &headers_tsv, string &out_table_uri) {
  out_table_uri.clear();
  auto *session = LanceGetSessionHandle(context);
  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();
  const char *headers_ptr = headers_tsv.empty() ? nullptr : headers_tsv.c_str();

  const char *uri_ptr = nullptr;
  auto *dataset = lance_open_dataset_in_namespace_with_session(
      endpoint.c_str(), table_id.c_str(), bearer_ptr, api_key_ptr,
      delimiter_ptr, headers_ptr, session, &uri_ptr);
  if (uri_ptr) {
    out_table_uri = uri_ptr;
    lance_free_string(uri_ptr);
  }
  return dataset;
}

static void *OpenDirNamespaceDataset(ClientContext &context, const string &root,
                                     const string &table_id,
                                     const vector<string> &option_keys,
                                     const vector<string> &option_values,
                                     string &out_table_uri) {
  out_table_uri.clear();
  auto *session = LanceGetSessionHandle(context);

  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                  value_ptrs);

  const char *uri_ptr = nullptr;
  auto *dataset = lance_open_dataset_in_dir_namespace_with_session(
      root.c_str(), table_id.c_str(),
      key_ptrs.empty() ? nullptr : key_ptrs.data(),
      value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size(),
      session, &uri_ptr);
  if (uri_ptr) {
    out_table_uri = uri_ptr;
    lance_free_string(uri_ptr);
  }
  return dataset;
}

static shared_ptr<LanceDatasetCacheEntry> GetOrOpenDatasetCacheEntry(
    ClientContext &context, const string &cache_key,
    const std::function<shared_ptr<LanceDatasetCacheEntry>()> &open_dataset,
    bool *out_cache_hit) {
  auto state = GetOrCreateLanceDatasetCacheState(context);
  auto entry = state->Get(cache_key);
  if (entry) {
    if (out_cache_hit) {
      *out_cache_hit = true;
    }
    return entry;
  }

  auto opened = open_dataset();
  if (!opened) {
    return nullptr;
  }
  if (out_cache_hit) {
    *out_cache_hit = false;
  }
  return state->PutOrGetExisting(cache_key, opened);
}

static shared_ptr<LanceDatasetCacheEntry> GetOrRefreshLatestDatasetCacheEntry(
    ClientContext &context, const string &cache_key,
    const std::function<shared_ptr<LanceDatasetCacheEntry>()> &open_dataset,
    bool *out_cache_hit) {
  // A latest-version handle is mutable external state. Always reopen it so a
  // commit from another connection, process, or Ray cluster is visible to the
  // next bind. The cache still reuses the existing immutable handle when the
  // freshly resolved version is unchanged; version-qualified entries use the
  // cheaper GetOrOpenDatasetCacheEntry path above.
  auto opened = open_dataset();
  if (!opened) {
    return nullptr;
  }
  bool cache_hit = false;
  auto entry = GetOrCreateLanceDatasetCacheState(context)->PutOrGetNewest(
      cache_key, std::move(opened), cache_hit);
  if (out_cache_hit) {
    *out_cache_hit = cache_hit;
  }
  return entry;
}

shared_ptr<LanceDatasetCacheEntry>
LanceGetOrOpenDatasetEntry(ClientContext &context, const string &path,
                           bool *out_cache_hit) {
  string open_path;
  vector<string> option_keys;
  vector<string> option_values;
  ResolveLanceStorageOptions(context, path, open_path, option_keys,
                             option_values);
  auto cache_key = LanceBuildResolvedPathDatasetCacheKey(open_path, option_keys,
                                                         option_values);

  return GetOrRefreshLatestDatasetCacheEntry(
      context, cache_key,
      [&]() {
        auto *dataset = OpenResolvedPathDataset(context, open_path, option_keys,
                                                option_values);
        if (!dataset) {
          return shared_ptr<LanceDatasetCacheEntry>();
        }
        return make_shared_ptr<LanceDatasetCacheEntry>(dataset, open_path);
      },
      out_cache_hit);
}

shared_ptr<LanceDatasetCacheEntry> LanceGetOrOpenDatasetEntryAtVersion(
    ClientContext &context, const string &path, uint64_t version,
    const string &generation_id, bool *out_cache_hit) {
  if (version == 0) {
    throw InvalidInputException(
        "Lance dataset version must be greater than zero");
  }
  if (generation_id.empty()) {
    throw InvalidInputException(
        "Lance dataset generation identity cannot be empty");
  }
  string open_path;
  vector<string> option_keys;
  vector<string> option_values;
  ResolveLanceStorageOptions(context, path, open_path, option_keys,
                             option_values);
  auto latest = LanceGetOrOpenDatasetEntry(context, path);
  if (!latest) {
    return nullptr;
  }
  auto cache_key = LanceBuildResolvedPathDatasetCacheKey(open_path, option_keys,
                                                         option_values);
  AppendCacheKeyPart(cache_key, "generation");
  AppendCacheKeyPart(cache_key, generation_id);
  AppendCacheKeyPart(cache_key, "version");
  AppendCacheKeyPart(cache_key, NumericCast<idx_t>(version));

  return GetOrOpenDatasetCacheEntry(
      context, cache_key,
      [&]() {
        auto *dataset =
            lance_dataset_checkout_version(latest->Handle(), version);
        if (!dataset) {
          return shared_ptr<LanceDatasetCacheEntry>();
        }
        auto entry =
            make_shared_ptr<LanceDatasetCacheEntry>(dataset, open_path);
        if (entry->GenerationId() != generation_id) {
          throw IOException("Checked-out Lance dataset generation does not "
                            "match the serialized snapshot: " +
                            path);
        }
        return entry;
      },
      out_cache_hit);
}

shared_ptr<LanceDatasetCacheEntry> LanceGetOrOpenDatasetEntryInNamespace(
    ClientContext &context, const string &endpoint, const string &table_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, string &out_display_uri, bool *out_cache_hit) {
  auto cache_key = LanceBuildNamespaceDatasetCacheKey(
      endpoint, table_id, bearer_token, api_key, delimiter, headers_tsv);
  auto entry = GetOrRefreshLatestDatasetCacheEntry(
      context, cache_key,
      [&]() {
        string table_uri;
        auto *dataset =
            OpenNamespaceDataset(context, endpoint, table_id, bearer_token,
                                 api_key, delimiter, headers_tsv, table_uri);
        if (!dataset) {
          return shared_ptr<LanceDatasetCacheEntry>();
        }

        string display_uri = table_uri.empty() ? endpoint + "/" + table_id
                                               : std::move(table_uri);
        return make_shared_ptr<LanceDatasetCacheEntry>(dataset,
                                                       std::move(display_uri));
      },
      out_cache_hit);
  if (entry) {
    out_display_uri = entry->DisplayUri();
  } else {
    out_display_uri.clear();
  }
  return entry;
}

shared_ptr<LanceDatasetCacheEntry>
LanceGetOrOpenDatasetEntryInNamespaceAtVersion(
    ClientContext &context, const string &endpoint, const string &table_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, uint64_t version, const string &generation_id,
    string &out_display_uri, bool *out_cache_hit) {
  if (version == 0) {
    throw InvalidInputException(
        "Lance dataset version must be greater than zero");
  }
  if (generation_id.empty()) {
    throw InvalidInputException(
        "Lance dataset generation identity cannot be empty");
  }
  string latest_display_uri;
  auto latest = LanceGetOrOpenDatasetEntryInNamespace(
      context, endpoint, table_id, bearer_token, api_key, delimiter,
      headers_tsv, latest_display_uri);
  if (!latest) {
    out_display_uri.clear();
    return nullptr;
  }
  auto cache_key = LanceBuildNamespaceDatasetCacheKey(
      endpoint, table_id, bearer_token, api_key, delimiter, headers_tsv);
  AppendCacheKeyPart(cache_key, "generation");
  AppendCacheKeyPart(cache_key, generation_id);
  AppendCacheKeyPart(cache_key, "version");
  AppendCacheKeyPart(cache_key, NumericCast<idx_t>(version));

  auto entry = GetOrOpenDatasetCacheEntry(
      context, cache_key,
      [&]() {
        auto *dataset =
            lance_dataset_checkout_version(latest->Handle(), version);
        if (!dataset) {
          return shared_ptr<LanceDatasetCacheEntry>();
        }
        auto display_uri = latest_display_uri.empty()
                               ? endpoint + "/" + table_id
                               : std::move(latest_display_uri);
        auto entry = make_shared_ptr<LanceDatasetCacheEntry>(
            dataset, std::move(display_uri));
        if (entry->GenerationId() != generation_id) {
          throw IOException("Checked-out Lance namespace dataset generation "
                            "does not match the serialized "
                            "snapshot: " +
                            endpoint + "/" + table_id);
        }
        return entry;
      },
      out_cache_hit);
  if (entry) {
    out_display_uri = entry->DisplayUri();
  } else {
    out_display_uri.clear();
  }
  return entry;
}

shared_ptr<LanceDatasetCacheEntry> LanceGetOrOpenDatasetEntryForTable(
    ClientContext &context, const LanceTableEntry &table,
    string &out_display_uri, bool *out_cache_hit) {
  out_display_uri = table.DatasetUri();
  if (!table.IsNamespaceBacked()) {
    auto entry =
        LanceGetOrOpenDatasetEntry(context, table.DatasetUri(), out_cache_hit);
    if (entry) {
      out_display_uri = entry->DisplayUri();
    }
    return entry;
  }

  auto &cfg = table.NamespaceConfig();
  if (cfg.IsDirectory()) {
    auto display_uri = LanceDirectoryNamespaceDatasetUri(cfg);
    auto cache_key = LanceBuildDirNamespaceDatasetCacheKey(cfg);
    auto entry = GetOrRefreshLatestDatasetCacheEntry(
        context, cache_key,
        [&]() {
          string table_uri;
          auto *dataset = OpenDirNamespaceDataset(context, cfg.root,
                                                  cfg.table_id, cfg.option_keys,
                                                  cfg.option_values, table_uri);
          if (!dataset) {
            return shared_ptr<LanceDatasetCacheEntry>();
          }
          string entry_display_uri = !table_uri.empty()
                                         ? LanceNormalizeS3Scheme(table_uri)
                                         : display_uri;
          return make_shared_ptr<LanceDatasetCacheEntry>(
              dataset, std::move(entry_display_uri));
        },
        out_cache_hit);
    if (entry) {
      out_display_uri = entry->DisplayUri();
    } else {
      out_display_uri.clear();
    }
    return entry;
  }

  string bearer_token;
  string api_key;
  string headers_tsv;
  ResolveLanceNamespaceTableAuth(context, cfg, bearer_token, api_key,
                                 headers_tsv);
  return LanceGetOrOpenDatasetEntryInNamespace(
      context, cfg.endpoint, cfg.table_id, bearer_token, api_key, cfg.delimiter,
      headers_tsv, out_display_uri, out_cache_hit);
}

string LanceBuildDatasetCacheKeyForTable(ClientContext &context,
                                         const LanceTableEntry &table) {
  if (!table.IsNamespaceBacked()) {
    return LanceBuildPathDatasetCacheKey(context, table.DatasetUri());
  }

  auto &cfg = table.NamespaceConfig();
  if (cfg.IsDirectory()) {
    return LanceBuildDirNamespaceDatasetCacheKey(cfg);
  }

  string bearer_token;
  string api_key;
  string headers_tsv;
  ResolveLanceNamespaceTableAuth(context, cfg, bearer_token, api_key,
                                 headers_tsv);
  return LanceBuildNamespaceDatasetCacheKey(cfg.endpoint, cfg.table_id,
                                            bearer_token, api_key,
                                            cfg.delimiter, headers_tsv);
}

void LanceInvalidateDatasetCache(ClientContext &context,
                                 const string &cache_key) noexcept {
  // Cache invalidation is a best-effort resource-release optimization, not
  // part of the external mutation. Latest-version binds always reopen the
  // dataset and compare its version/generation before reusing an entry, so a
  // stale entry cannot change query results. Never turn an already durable
  // Lance mutation into a retryable SQL failure because this cleanup failed.
  try {
    auto state = context.registered_state->Get<LanceDatasetCacheState>(
        LANCE_DATASET_CACHE_STATE_KEY);
    if (state) {
      state->Invalidate(cache_key);
    }
  } catch (...) {
  }
}

void LanceInvalidateDatasetCacheForPath(ClientContext &context,
                                        const string &path) {
  LanceInvalidateDatasetCache(context,
                              LanceBuildPathDatasetCacheKey(context, path));
}

void LanceInvalidateDatasetCacheForTable(ClientContext &context,
                                         const LanceTableEntry &table) {
  LanceInvalidateDatasetCache(
      context, LanceBuildDatasetCacheKeyForTable(context, table));
}

} // namespace duckdb
