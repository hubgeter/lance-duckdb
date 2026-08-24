#pragma once

#include "duckdb.hpp"

struct LanceNamespaceQueryConfig;

namespace duckdb {

struct LanceLastError {
  int32_t code = 0;
  string message;

  string ToString() const;
};

LanceLastError LanceConsumeLastErrorDetail();
string LanceConsumeLastError();
string LanceFormatErrorSuffix();

enum class LanceComputedSearchColumns : uint8_t {
  None = 0,
  Distance = 1,
  Score = 2,
  HybridScore = 4,
  Hybrid = 7,
};

bool IsComputedSearchColumn(const string &name);
bool IsComputedSearchColumn(const string &name,
                            LanceComputedSearchColumns columns);

void LanceValidateAndReorderArrowBatch(
    ClientContext &context, ArrowSchema &schema, ArrowArray &array,
    const vector<string> &expected_names,
    const vector<LogicalType> &expected_types, const string &source_name);

void ApplyDuckDBFilters(ClientContext &context, TableFilterSet &filters,
                        DataChunk &chunk, SelectionVector &sel);

void *LanceOpenDataset(ClientContext &context, const string &path);

//! Build the stable identity used by DuckDB's serialized logical write
//! target contract. The Lance generation is part of the identity so a plan
//! bound before DROP/recreate cannot silently write to the replacement table.
string LanceBuildLogicalWriteTargetIdentity(const string &dataset_identity,
                                            void *dataset);

string LanceNormalizeS3Scheme(const string &path);
string LanceNormalizeDatasetPath(ClientContext &context, const string &path);
void ValidateLanceCString(const string &value, const string &label);
vector<pair<string, string>> ParseLanceKeyValueRows(const char *ptr,
                                                    const string &description);
void LanceFillStorageOptions(ClientContext &context, const string &path,
                             vector<string> &out_keys,
                             vector<string> &out_values);
bool LanceStorageOptionsAreWorkerReplayable(
    ClientContext &context, const string &path,
    const vector<string> &resolved_keys, const vector<string> &resolved_values);
bool LanceStorageOptionsAreWorkerReplayable(ClientContext &context,
                                            const string &path);
void ResolveLanceStorageOptions(ClientContext &context, const string &path,
                                string &out_open_path,
                                vector<string> &out_option_keys,
                                vector<string> &out_option_values);
void BuildStorageOptionPointerArrays(const vector<string> &option_keys,
                                     const vector<string> &option_values,
                                     vector<const char *> &out_key_ptrs,
                                     vector<const char *> &out_value_ptrs);

static constexpr uint64_t LANCE_DEFAULT_MAX_ROWS_PER_FILE = 1024ULL * 1024ULL;
static constexpr uint64_t LANCE_DEFAULT_MAX_ROWS_PER_GROUP = 1024ULL;
static constexpr uint64_t LANCE_DEFAULT_MAX_BYTES_PER_FILE =
    90ULL * 1024ULL * 1024ULL * 1024ULL;
static constexpr const char *LANCE_DEFAULT_DATA_STORAGE_VERSION = "2.2";

struct LanceNamespaceTableConfig;
class LanceTableEntry;

void ResolveLanceNamespaceAuth(ClientContext &context, const string &endpoint,
                               const unordered_map<string, Value> &options,
                               string &out_bearer_token, string &out_api_key);
void ResolveLanceNamespaceAuth(ClientContext &context, const string &endpoint,
                               const named_parameter_map_t &options,
                               string &out_bearer_token, string &out_api_key);
void ResolveLanceNamespaceAuthOverrides(
    const unordered_map<string, Value> &options, string &out_bearer_token,
    string &out_api_key);
void ResolveLanceNamespaceTableAuth(ClientContext &context,
                                    const LanceNamespaceTableConfig &cfg,
                                    string &out_bearer_token,
                                    string &out_api_key,
                                    string &out_headers_tsv);
uint64_t ResolveLanceNamespaceTableVersion(ClientContext &context,
                                           LanceNamespaceTableConfig &cfg);

bool TryLanceNamespaceListTables(ClientContext &context, const string &endpoint,
                                 const string &namespace_id,
                                 const string &bearer_token,
                                 const string &api_key, const string &delimiter,
                                 const string &headers_tsv,
                                 vector<string> &out_tables, string &out_error);

bool TryLanceDirNamespaceListTables(ClientContext &context, const string &root,
                                    vector<string> &out_tables,
                                    string &out_error);

void *
LanceOpenDatasetInNamespace(ClientContext &context, const string &endpoint,
                            const string &table_id, const string &bearer_token,
                            const string &api_key, const string &delimiter,
                            const string &headers_tsv, string &out_table_uri);

bool TryLanceNamespaceDescribeTable(
    ClientContext &context, const string &endpoint, const string &table_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, string &out_location,
    vector<string> &out_option_keys, vector<string> &out_option_values,
    string &out_error);

bool TryLanceNamespaceCreateEmptyTable(
    ClientContext &context, const string &endpoint, const string &table_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, string &out_location,
    vector<string> &out_option_keys, vector<string> &out_option_values,
    string &out_error, bool &inout_namespace_mutated);

bool TryLanceNamespaceDropTable(ClientContext &context, const string &endpoint,
                                const string &table_id,
                                const string &bearer_token,
                                const string &api_key, const string &delimiter,
                                const string &headers_tsv, string &out_error,
                                bool &inout_namespace_mutated);

void FillLanceNamespaceQueryConfig(
    ClientContext &context, const LanceNamespaceTableConfig &cfg,
    uint64_t dataset_version, uint64_t k, bool prefilter, const string &filter,
    const vector<string> &columns, vector<string> &resolved_option_keys,
    vector<string> &resolved_option_values,
    vector<const char *> &option_key_ptrs,
    vector<const char *> &option_value_ptrs, vector<const char *> &column_ptrs,
    string &bearer_token, string &api_key, string &headers_tsv,
    ::LanceNamespaceQueryConfig &out_config);

string LanceDirectoryNamespaceDatasetUri(const LanceNamespaceTableConfig &cfg);

// Resolve a string like "catalog.schema.table" to a LanceTableEntry* if it
// names a Lance-backed table in an attached catalog.  Returns nullptr for
// inputs that look like filesystem / URL paths, or when the lookup does not
// yield a Lance table (missing entry, non-Lance table, malformed qualified
// name).  This is the canonical way to re-resolve a dataset from its
// original first-argument literal so we can take the namespace-aware
// LanceOpenDatasetForTable() path instead of passing the virtual
// namespace URI directly to lance-io.
LanceTableEntry *TryResolveLanceTableEntry(ClientContext &context,
                                           const string &input);

void RequireLanceTableWritable(const LanceTableEntry &table,
                               const string &operation);

void *LanceOpenDatasetForTable(ClientContext &context,
                               const LanceTableEntry &table,
                               string &out_display_uri);

void ResolveLanceStorageOptionsForTable(ClientContext &context,
                                        const LanceTableEntry &table,
                                        string &out_open_path,
                                        vector<string> &out_option_keys,
                                        vector<string> &out_option_values,
                                        string &out_display_uri);

int64_t LanceTruncateDatasetWithStorageOptions(
    ClientContext &context, const string &open_path,
    const vector<string> &option_keys, const vector<string> &option_values,
    const string &display_uri);

int64_t LanceTruncateDataset(ClientContext &context, const string &dataset_uri);

} // namespace duckdb
