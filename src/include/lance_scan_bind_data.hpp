#pragma once

#include "lance_dataset_cache.hpp"

#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"

#include <cstdint>

namespace duckdb {
class TableCatalogEntry;
struct LanceNamespaceTableConfig;

struct LanceScanBindData : public TableFunctionData {
  string file_path;
  bool explain_verbose = false;
  shared_ptr<LanceDatasetCacheEntry> dataset_entry;
  void *dataset = nullptr;
  bool dataset_cache_hit = false;
  ArrowSchemaWrapper schema_root;
  ArrowSchemaWrapper scan_schema_root;
  ArrowTableSchema arrow_table;
  ArrowTableSchema scan_arrow_table;
  vector<string> names;
  vector<LogicalType> types;
  uint64_t dataset_version = 0;
  string dataset_generation_id;
  bool non_replayable_storage_options = false;

  //! A target scan nested inside UPDATE/DELETE/MERGE is protected by the
  //! enclosing mutation lease.  Acquiring a second, conflicting snapshot
  //! lease for that same write would self-deadlock before the sink can obtain
  //! its mutation lease.
  bool mutation_scan = false;

  //! Process-independent distributed scan state. A split contains only a
  //! fragment id; URI, version, projections and filters live in serialized
  //! bind data. Search/query-table paths use one explicit global split.
  bool scan_splits_applied = false;
  bool force_global_scan = false;
  vector<uint64_t> selected_fragment_ids;

  //! Namespace identity is retained without resolved credentials. Workers use
  //! replayed connection settings where the extension supports them.
  bool reopen_via_namespace = false;
  bool namespace_requires_worker_auth = false;
  string namespace_endpoint;
  string namespace_table_id;
  string namespace_delimiter;
  vector<string> lance_pushed_filter_ir_parts;
  vector<string> duckdb_pushed_filter_sql_parts;
  optional_ptr<TableCatalogEntry> table_entry = nullptr;
  unique_ptr<LanceNamespaceTableConfig> namespace_query_config;

  bool sampling_pushed_down = false;
  double sample_percentage = 0.0;
  int64_t sample_seed = -1;
  bool sample_repeatable = false;
  vector<uint64_t> take_row_ids;

  bool limit_offset_pushed_down = false;
  optional_idx pushed_limit = optional_idx::Invalid();
  idx_t pushed_offset = 0;

  bool UsesNamespaceQuery() const { return namespace_query_config != nullptr; }

  unique_ptr<FunctionData> Copy() const override;

  ~LanceScanBindData() override;
};

} // namespace duckdb
