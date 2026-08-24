#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value.hpp"
#ifdef LANCE_VANE_DISTRIBUTED
#include "duckdb/function/distributed_table_function.hpp"
#endif
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/table_filter.hpp"

#include "lance_arrow_compat.hpp"
#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_ffi.hpp"
#include "lance_filter_ir.hpp"
#include "lance_lease.hpp"
#include "lance_resolver.hpp"
#include "lance_table_entry.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace duckdb {

#ifdef LANCE_VANE_DISTRIBUTED
static constexpr const char *LANCE_SEARCH_GLOBAL_SPLIT =
    "lance-search-global-v1";
static constexpr const char *LANCE_SEARCH_SPLIT_CODEC = "lance.search-split";
static constexpr idx_t LANCE_SEARCH_PROTOCOL_VERSION = 2;
static constexpr idx_t LANCE_SEARCH_SPLIT_CODEC_VERSION = 1;

static DistributedScanSplit LanceGlobalSearchSplit(uint64_t k) {
  DistributedScanSplit split;
  split.split_id = "global-search";
  split.payload = LANCE_SEARCH_GLOBAL_SPLIT;
  split.estimated_cardinality = optional_idx(NumericCast<idx_t>(k));
  return split;
}

struct LanceGlobalSearchSplitState {
  virtual ~LanceGlobalSearchSplitState() = default;
  bool assigned_empty_scan = false;
};

static bool
ValidateLanceGlobalSearchSplits(const vector<DistributedScanSplit> &splits) {
  if (splits.empty()) {
    return false;
  }
  for (auto &split : splits) {
    if (split.split_id != "global-search" ||
        split.payload != LANCE_SEARCH_GLOBAL_SPLIT) {
      throw SerializationException(
          "Lance search received an invalid global search split");
    }
  }
  // FTE queues may merge replayed copies of the same indivisible global
  // split. They are one idempotent assignment, not independent searches.
  return true;
}

static void
LanceApplyGlobalSearchSplits(FunctionData &bind_data,
                             const vector<DistributedScanSplit> &splits) {
  auto *split_state = dynamic_cast<LanceGlobalSearchSplitState *>(&bind_data);
  if (!split_state) {
    throw InternalException(
        "Lance global search bind data is missing split state");
  }
  split_state->assigned_empty_scan = !ValidateLanceGlobalSearchSplits(splits);
}
#else
struct LanceGlobalSearchSplitState {
  bool assigned_empty_scan = false;
};
#endif

static void PopulateSearchSchemaFromTypes(ClientContext &context,
                                          ArrowSchemaWrapper &schema_root,
                                          ArrowTableSchema &arrow_table,
                                          const vector<string> &names,
                                          const vector<LogicalType> &types) {
  if (names.empty() || names.size() != types.size()) {
    throw SerializationException(
        "Serialized Lance search has an invalid output schema");
  }
  memset(&schema_root.arrow_schema, 0, sizeof(schema_root.arrow_schema));
  auto properties = context.GetClientProperties();
  ArrowConverter::ToArrowSchema(&schema_root.arrow_schema, types, names,
                                properties);
  LanceCoerceArrowSchemaForDuckDB(&schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(context, arrow_table,
                                               schema_root.arrow_schema);
}

static bool TryLanceExplainKnn(void *dataset, const string &vector_column,
                               const vector<float> &query, uint64_t k,
                               uint64_t nprobes, uint64_t refine_factor,
                               const string *filter_ir, bool prefilter,
                               bool use_index, bool verbose, string &out_plan,
                               string &out_error) {
  out_plan.clear();
  out_error.clear();

  if (!dataset) {
    out_error = "dataset is null";
    return false;
  }
  if (query.empty()) {
    out_error = "query is empty";
    return false;
  }

  const uint8_t *filter_ptr = nullptr;
  idx_t filter_len = 0;
  if (filter_ir && !filter_ir->empty()) {
    filter_ptr = reinterpret_cast<const uint8_t *>(filter_ir->data());
    filter_len = NumericCast<idx_t>(filter_ir->size());
  }

  auto *plan_ptr = lance_explain_knn_scan_ir(
      dataset, vector_column.c_str(), query.data(), query.size(), k, nprobes,
      refine_factor, filter_ptr, NumericCast<size_t>(filter_len),
      prefilter ? 1 : 0, use_index ? 1 : 0, verbose ? 1 : 0);
  if (!plan_ptr) {
    out_error = LanceConsumeLastError();
    if (out_error.empty()) {
      out_error = "unknown error";
    }
    return false;
  }

  out_plan = plan_ptr;
  lance_free_string(plan_ptr);
  return true;
}

// TryResolveLanceTableEntry() is now defined in lance_common.cpp so that
// the maintenance helpers (compact / cleanup / optimize_index /
// auto_cleanup) can share the same "catalog.schema.table" -> entry
// resolution logic.

static shared_ptr<LanceDatasetCacheEntry>
OpenSearchDatasetEntry(ClientContext &context, const Value &input,
                       const string &function_name, string &out_display_uri,
                       bool *out_cache_hit) {
  out_display_uri = ResolveLanceDatasetUri(
      context, input, LanceResolvePolicy::FALLBACK_TO_PATH, function_name);
  auto input_str = input.GetValue<string>();
  if (auto *table = TryResolveLanceTableEntry(context, input_str)) {
    if (StringUtil::CIEquals(out_display_uri, table->DatasetUri())) {
      return LanceGetOrOpenDatasetEntryForTable(context, *table,
                                                out_display_uri, out_cache_hit);
    }
  }

  auto entry =
      LanceGetOrOpenDatasetEntry(context, out_display_uri, out_cache_hit);
  if (entry) {
    out_display_uri = entry->DisplayUri();
  }
  return entry;
}

static bool SearchStorageOptionsAreWorkerReplayable(ClientContext &context,
                                                    const Value &input,
                                                    const string &display_uri) {
  auto input_str = input.GetValue<string>();
  auto *table = TryResolveLanceTableEntry(context, input_str);
  if (table && table->IsNamespaceBacked() &&
      table->NamespaceConfig().IsDirectory()) {
    auto &cfg = table->NamespaceConfig();
    return LanceStorageOptionsAreWorkerReplayable(
        context, LanceDirectoryNamespaceDatasetUri(cfg), cfg.option_keys,
        cfg.option_values);
  }
  return LanceStorageOptionsAreWorkerReplayable(context, display_uri);
}

static vector<float> ParseQueryVector(const Value &value,
                                      const string &function_name) {
  if (value.IsNull()) {
    throw InvalidInputException(function_name +
                                " requires a non-null query vector");
  }
  if (value.type().id() != LogicalTypeId::LIST &&
      value.type().id() != LogicalTypeId::ARRAY) {
    throw InvalidInputException(function_name +
                                " requires query vector to be a LIST or ARRAY");
  }
  vector<Value> children;
  if (value.type().id() == LogicalTypeId::LIST) {
    children = ListValue::GetChildren(value);
  } else {
    children = ArrayValue::GetChildren(value);
  }
  if (children.empty()) {
    throw InvalidInputException(function_name +
                                " requires a non-empty query vector");
  }

  auto cast_f32 = [&function_name](double v) {
    if (!std::isfinite(v)) {
      throw InvalidInputException(function_name +
                                  " query vector contains non-finite value");
    }
    auto max_v = static_cast<double>(std::numeric_limits<float>::max());
    if (v > max_v || v < -max_v) {
      throw InvalidInputException(
          function_name + " query vector value is out of float32 range");
    }
    return static_cast<float>(v);
  };

  vector<float> out;
  out.reserve(children.size());
  for (auto &child : children) {
    if (child.IsNull()) {
      throw InvalidInputException(function_name +
                                  " query vector contains NULL");
    }
    switch (child.type().id()) {
    case LogicalTypeId::FLOAT:
      out.push_back(cast_f32(child.GetValue<float>()));
      break;
    case LogicalTypeId::DOUBLE:
      out.push_back(cast_f32(child.GetValue<double>()));
      break;
    case LogicalTypeId::TINYINT:
    case LogicalTypeId::SMALLINT:
    case LogicalTypeId::INTEGER:
    case LogicalTypeId::BIGINT:
      out.push_back(cast_f32(static_cast<double>(child.GetValue<int64_t>())));
      break;
    case LogicalTypeId::UTINYINT:
    case LogicalTypeId::USMALLINT:
    case LogicalTypeId::UINTEGER:
    case LogicalTypeId::UBIGINT:
      out.push_back(cast_f32(static_cast<double>(child.GetValue<uint64_t>())));
      break;
    default:
      try {
        auto dbl = child.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
        out.push_back(cast_f32(dbl));
      } catch (Exception &) {
        throw InvalidInputException(function_name +
                                    " query vector elements must be numeric");
      }
    }
  }
  return out;
}

static void ValidateSerializedQueryVector(const vector<float> &query,
                                          const string &function_name) {
  if (query.empty()) {
    throw SerializationException("Serialized " + function_name +
                                 " has an empty query vector");
  }
  for (auto value : query) {
    if (!std::isfinite(value)) {
      throw SerializationException("Serialized " + function_name +
                                   " has a non-finite query vector");
    }
  }
}

static void ValidateSerializedSearchCString(const string &value,
                                            const string &field_name) {
  if (value.find('\0') != string::npos) {
    throw SerializationException("Serialized Lance search " + field_name +
                                 " contains a NUL byte");
  }
}

static void ValidateSerializedSearchCStrings(const vector<string> &values,
                                             const string &field_name) {
  for (auto &value : values) {
    ValidateSerializedSearchCString(value, field_name);
  }
}

static string ParseOptionalNamedString(const TableFunctionBindInput &input,
                                       const string &name) {
  auto it = input.named_parameters.find(name);
  if (it == input.named_parameters.end() || it->second.IsNull()) {
    return string();
  }
  return it->second.DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
}

static LanceTableEntry *
TryResolveNamespaceBackedSearchTable(ClientContext &context,
                                     const Value &input) {
  auto input_str = input.GetValue<string>();
  auto *table = TryResolveLanceTableEntry(context, input_str);
  if (!table || !table->IsNamespaceBacked()) {
    return nullptr;
  }
  return table;
}

static string RequireNamespaceSearchColumn(const LanceTableEntry &table,
                                           const string &column,
                                           const string &function_name,
                                           const string &argument_name) {
  for (auto &col : table.GetColumns().Physical()) {
    if (StringUtil::CIEquals(col.Name(), column)) {
      return col.Name();
    }
  }
  throw InvalidInputException(function_name + " requires " + argument_name +
                              " to name an existing column on " +
                              "namespace-backed table: " + column);
}

static void PopulateNamespaceSearchSchema(
    ClientContext &context, const LanceTableEntry &table,
    const string &metric_name, ArrowSchemaWrapper &schema_root,
    ArrowTableSchema &arrow_table, vector<string> &result_names,
    vector<LogicalType> &result_types, vector<string> &names,
    vector<LogicalType> &return_types) {
  vector<string> field_names;
  vector<LogicalType> field_types;
  field_names.reserve(table.GetColumns().PhysicalColumnCount() + 1);
  field_types.reserve(table.GetColumns().PhysicalColumnCount() + 1);
  for (auto &col : table.GetColumns().Physical()) {
    if (StringUtil::CIEquals(col.Name(), metric_name)) {
      throw InvalidInputException("Lance namespace search output column '" +
                                  metric_name +
                                  "' conflicts with an existing table column");
    }
    field_names.push_back(col.Name());
    field_types.push_back(col.Type());
  }
  field_names.push_back(metric_name);
  field_types.push_back(LogicalType::FLOAT);

  memset(&schema_root.arrow_schema, 0, sizeof(schema_root.arrow_schema));
  auto props = context.GetClientProperties();
  ArrowConverter::ToArrowSchema(&schema_root.arrow_schema, field_types,
                                field_names, props);
  LanceCoerceArrowSchemaForDuckDB(&schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(context, arrow_table,
                                               schema_root.arrow_schema);
  result_names = arrow_table.GetNames();
  result_types = arrow_table.GetTypes();
  names = result_names;
  return_types = result_types;
}

static void RejectSearchComputedColumnCollisions(
    void *dataset, LanceComputedSearchColumns computed_columns,
    const string &function_name) {
  auto *schema_handle = lance_get_schema(dataset);
  if (!schema_handle) {
    throw IOException("Failed to inspect Lance schema for " + function_name +
                      LanceFormatErrorSuffix());
  }

  ArrowSchemaWrapper schema_root;
  memset(&schema_root.arrow_schema, 0, sizeof(schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &schema_root.arrow_schema) != 0) {
    lance_free_schema(schema_handle);
    throw IOException("Failed to export Lance schema for " + function_name +
                      LanceFormatErrorSuffix());
  }
  lance_free_schema(schema_handle);

  auto &schema = schema_root.arrow_schema;
  if (schema.n_children < 0 || (schema.n_children > 0 && !schema.children)) {
    throw IOException("Lance returned an invalid schema for " + function_name);
  }
  for (int64_t i = 0; i < schema.n_children; i++) {
    auto *field = schema.children[i];
    if (!field || !field->name) {
      throw IOException("Lance returned an unnamed schema field for " +
                        function_name);
    }
    if (IsComputedSearchColumn(field->name, computed_columns)) {
      throw InvalidInputException(
          function_name + " output column '" + field->name +
          "' conflicts with an existing dataset column");
    }
  }
}

struct LanceKnnBindData : public TableFunctionData,
                          public LanceGlobalSearchSplitState {
  string file_path;
  string vector_column;
  vector<float> query;
  uint64_t k = 0;
  uint64_t nprobes = 0;
  uint64_t refine_factor = 0;
  bool prefilter = true;
  bool use_index = true;
  bool explain_verbose = false;
  bool namespace_backed = false;
  LanceNamespaceTableConfig namespace_config;
  string namespace_filter;

  shared_ptr<LanceDatasetCacheEntry> dataset_entry;
  void *dataset = nullptr;
  bool dataset_cache_hit = false;
  uint64_t dataset_version = 0;
  string dataset_generation_id;
  bool non_replayable_storage_options = false;
  ArrowSchemaWrapper schema_root;
  ArrowTableSchema arrow_table;
  vector<string> names;
  vector<LogicalType> types;
  vector<string> lance_pushed_filter_ir_parts;
  bool complex_filter_pushdown_failed = false;

  unique_ptr<FunctionData> Copy() const override {
    auto result = make_uniq<LanceKnnBindData>();
    result->column_ids = column_ids;
    result->file_path = file_path;
    result->vector_column = vector_column;
    result->query = query;
    result->k = k;
    result->nprobes = nprobes;
    result->refine_factor = refine_factor;
    result->prefilter = prefilter;
    result->use_index = use_index;
    result->explain_verbose = explain_verbose;
    result->namespace_backed = namespace_backed;
    result->namespace_config = namespace_config;
    result->namespace_filter = namespace_filter;
    result->dataset_entry = dataset_entry;
    result->dataset = dataset;
    result->dataset_cache_hit = dataset_cache_hit;
    result->dataset_version = dataset_version;
    result->dataset_generation_id = dataset_generation_id;
    result->non_replayable_storage_options = non_replayable_storage_options;
    result->arrow_table = arrow_table;
    result->names = names;
    result->types = types;
    result->lance_pushed_filter_ir_parts = lance_pushed_filter_ir_parts;
    result->complex_filter_pushdown_failed = complex_filter_pushdown_failed;
    result->assigned_empty_scan = assigned_empty_scan;
    return std::move(result);
  }
};

#ifdef LANCE_VANE_DISTRIBUTED
static LanceNamespaceTableConfig
LanceCreateDistributedNamespaceConfig(const LanceNamespaceTableConfig &source) {
  LanceNamespaceTableConfig result;
  result.kind = source.kind;
  result.root = source.root;
  result.endpoint = source.endpoint;
  result.table_id = source.table_id;
  result.delimiter = source.delimiter;
  result.display_uri = source.display_uri;
  return result;
}

static unique_ptr<FunctionData> LanceCreateDistributedKnnWorkerBind(
    const TableFunctionDistributedScanInput &input) {
  auto &source = input.bind_data.Cast<LanceKnnBindData>();
  if (source.namespace_backed && source.namespace_config.requires_worker_auth) {
    throw NotImplementedException(
        "Distributed Lance REST namespace vector search does not transport "
        "bearer tokens, API keys, or custom headers to workers");
  }
  if (source.non_replayable_storage_options) {
    throw NotImplementedException(
        "Distributed Lance vector search cannot transport TYPE LANCE secrets "
        "or arbitrary storage options to workers; configure equivalent "
        "explicit s3_* connection settings or run locally");
  }
  auto result = make_uniq<LanceKnnBindData>();
  result->file_path = source.file_path;
  result->vector_column = source.vector_column;
  result->query = source.query;
  result->k = source.k;
  result->nprobes = source.nprobes;
  result->refine_factor = source.refine_factor;
  result->prefilter = source.prefilter;
  result->use_index = source.use_index;
  result->explain_verbose = source.explain_verbose;
  result->namespace_backed = source.namespace_backed;
  result->namespace_config =
      LanceCreateDistributedNamespaceConfig(source.namespace_config);
  result->namespace_filter = source.namespace_filter;
  result->dataset_version = source.dataset_version;
  result->dataset_generation_id = source.dataset_generation_id;
  result->names = source.names;
  result->types = source.types;
  result->lance_pushed_filter_ir_parts = source.lance_pushed_filter_ir_parts;
  result->complex_filter_pushdown_failed =
      source.complex_filter_pushdown_failed;
  return std::move(result);
}

static vector<DistributedScanSplit> LancePlanDistributedKnnSearchSplits(
    const TableFunctionDistributedScanPlanningInput &input) {
  auto &bind_data = input.bind_data.Cast<LanceKnnBindData>();
  return {LanceGlobalSearchSplit(bind_data.k)};
}

static TableFunctionDistributedScanCallbacks
LanceKnnDistributedScanCallbacks() {
  TableFunctionDistributedScanCallbacks callbacks;
  callbacks.protocol_version = LANCE_SEARCH_PROTOCOL_VERSION;
  callbacks.split_codec = {LANCE_SEARCH_SPLIT_CODEC,
                           LANCE_SEARCH_SPLIT_CODEC_VERSION};
  callbacks.plan_splits = LancePlanDistributedKnnSearchSplits;
  callbacks.create_worker_bind = LanceCreateDistributedKnnWorkerBind;
  callbacks.apply_splits = LanceApplyGlobalSearchSplits;
  return callbacks;
}
#endif

struct LanceKnnGlobalState : public GlobalTableFunctionState {
  unique_ptr<LanceLease> snapshot_lease;
  std::atomic<idx_t> lines_read{0};
  std::atomic<idx_t> record_batches{0};
  std::atomic<idx_t> record_batch_rows{0};
  string lance_filter_ir;
  bool filter_pushed_down = false;
  std::atomic<idx_t> filter_pushdown_fallbacks{0};

  vector<idx_t> projection_ids;
  vector<LogicalType> scanned_types;
  vector<string> namespace_columns;

  std::atomic<bool> explain_computed{false};
  string explain_plan;
  string explain_error;
  std::mutex explain_mutex;

  idx_t MaxThreads() const override { return 1; }
  bool CanRemoveFilterColumns() const { return !projection_ids.empty(); }
};

struct LanceKnnLocalState : public ArrowScanLocalState {
  explicit LanceKnnLocalState(unique_ptr<ArrowArrayWrapper> current_chunk,
                              ClientContext &context)
      : ArrowScanLocalState(std::move(current_chunk), context),
        filter_sel(STANDARD_VECTOR_SIZE) {}

  void *stream = nullptr;
  LanceKnnGlobalState *global_state = nullptr;
  bool filter_pushed_down = false;
  SelectionVector filter_sel;

  ~LanceKnnLocalState() override {
    if (stream) {
      lance_close_stream(stream);
    }
  }
};

static void
LancePushdownComplexFilter(ClientContext &, LogicalGet &get,
                           FunctionData *bind_data,
                           vector<unique_ptr<Expression>> &filters) {
  if (!bind_data || filters.empty()) {
    return;
  }
  auto &scan_bind = bind_data->Cast<LanceKnnBindData>();
  if (scan_bind.namespace_backed) {
    return;
  }

  for (auto &expr : filters) {
    if (!expr || expr->HasParameter() || expr->IsVolatile()) {
      scan_bind.complex_filter_pushdown_failed = true;
      continue;
    }
    if (expr->expression_class == ExpressionClass::BOUND_COMPARISON) {
      auto &cmp = expr->Cast<BoundComparisonExpression>();
      if (cmp.type == ExpressionType::COMPARE_DISTINCT_FROM ||
          cmp.type == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
        auto is_constant = [](const unique_ptr<Expression> &node) -> bool {
          if (!node) {
            return false;
          }
          if (node->expression_class == ExpressionClass::BOUND_CONSTANT) {
            return true;
          }
          if (node->expression_class == ExpressionClass::BOUND_CAST) {
            auto &cast = node->Cast<BoundCastExpression>();
            return !cast.try_cast && cast.child &&
                   cast.child->expression_class ==
                       ExpressionClass::BOUND_CONSTANT;
          }
          return false;
        };

        auto is_column = [](const unique_ptr<Expression> &node) -> bool {
          if (!node) {
            return false;
          }
          return node->expression_class == ExpressionClass::BOUND_COLUMN_REF ||
                 node->expression_class == ExpressionClass::BOUND_REF;
        };

        if ((is_column(cmp.left) && is_constant(cmp.right)) ||
            (is_column(cmp.right) && is_constant(cmp.left))) {
          continue;
        }
      }
    }
    string filter_ir;
    if (!TryBuildLanceExprFilterIR(get, scan_bind.names, scan_bind.types,
                                   LanceComputedSearchColumns::Distance, *expr,
                                   filter_ir)) {
      scan_bind.complex_filter_pushdown_failed = true;
      continue;
    }
    scan_bind.lance_pushed_filter_ir_parts.push_back(std::move(filter_ir));
  }
}

static bool LancePushdownExpression(ClientContext &, const LogicalGet &,
                                    Expression &expr) {
  if (expr.expression_class != ExpressionClass::BOUND_COMPARISON) {
    return false;
  }
  auto &cmp = expr.Cast<BoundComparisonExpression>();
  return cmp.type == ExpressionType::COMPARE_DISTINCT_FROM ||
         cmp.type == ExpressionType::COMPARE_NOT_DISTINCT_FROM;
}

static unique_ptr<FunctionData>
LanceSearchVectorBind(ClientContext &context, TableFunctionBindInput &input,
                      vector<LogicalType> &return_types,
                      vector<string> &names) {
  if (input.inputs.size() < 3) {
    throw InvalidInputException(
        "lance_vector_search requires (path, vector_column, vector)");
  }
  if (input.inputs[0].IsNull()) {
    throw InvalidInputException(
        "lance_vector_search requires a dataset root path");
  }
  if (input.inputs[1].IsNull()) {
    throw InvalidInputException(
        "lance_vector_search requires a non-null vector_column");
  }
  if (input.inputs[2].IsNull()) {
    throw InvalidInputException(
        "lance_vector_search requires a non-null query vector");
  }

  auto result = make_uniq<LanceKnnBindData>();
  result->vector_column = input.inputs[1].GetValue<string>();
  result->query = ParseQueryVector(input.inputs[2], "lance_vector_search");
  result->prefilter = false;
  result->namespace_filter = ParseOptionalNamedString(input, "filter");
  ValidateLanceCString(result->vector_column, "Lance vector column");
  ValidateLanceCString(result->namespace_filter, "Lance search filter");

  auto verbose_it = input.named_parameters.find("explain_verbose");
  if (verbose_it != input.named_parameters.end() &&
      !verbose_it->second.IsNull()) {
    result->explain_verbose =
        verbose_it->second.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
  }

  int64_t k_val = 10;
  auto k_named = input.named_parameters.find("k");
  if (k_named != input.named_parameters.end() && !k_named->second.IsNull()) {
    k_val =
        k_named->second.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
  }
  if (k_val <= 0) {
    throw InvalidInputException("lance_vector_search requires k > 0");
  }
  result->k = NumericCast<uint64_t>(k_val);

  bool has_nprobes = false;
  int64_t nprobes_val = 0;
  auto nprobes_named = input.named_parameters.find("nprobs");
  if (nprobes_named != input.named_parameters.end() &&
      !nprobes_named->second.IsNull()) {
    has_nprobes = true;
    nprobes_val = nprobes_named->second.DefaultCastAs(LogicalType::BIGINT)
                      .GetValue<int64_t>();
  }
  if (has_nprobes && nprobes_val <= 0) {
    throw InvalidInputException("lance_vector_search requires nprobs > 0");
  }
  result->nprobes = has_nprobes ? NumericCast<uint64_t>(nprobes_val) : 0;

  bool has_refine_factor = false;
  int64_t refine_factor_val = 0;
  auto refine_factor_named = input.named_parameters.find("refine_factor");
  if (refine_factor_named != input.named_parameters.end() &&
      !refine_factor_named->second.IsNull()) {
    has_refine_factor = true;
    refine_factor_val =
        refine_factor_named->second.DefaultCastAs(LogicalType::BIGINT)
            .GetValue<int64_t>();
  }
  if (has_refine_factor && refine_factor_val <= 0) {
    throw InvalidInputException(
        "lance_vector_search requires refine_factor > 0");
  }
  result->refine_factor =
      has_refine_factor ? NumericCast<uint64_t>(refine_factor_val) : 0;

  auto prefilter_named = input.named_parameters.find("prefilter");
  if (prefilter_named != input.named_parameters.end() &&
      !prefilter_named->second.IsNull()) {
    result->prefilter =
        prefilter_named->second.DefaultCastAs(LogicalType::BOOLEAN)
            .GetValue<bool>();
  }
  auto use_index_named = input.named_parameters.find("use_index");
  if (use_index_named != input.named_parameters.end() &&
      !use_index_named->second.IsNull()) {
    result->use_index =
        use_index_named->second.DefaultCastAs(LogicalType::BOOLEAN)
            .GetValue<bool>();
  }

  auto *namespace_table =
      TryResolveNamespaceBackedSearchTable(context, input.inputs[0]);
  if (namespace_table && namespace_table->NamespaceConfig().IsRest()) {
    auto *table = namespace_table;
    result->namespace_backed = true;
    result->namespace_config = table->NamespaceConfig();
    result->file_path = table->DatasetUri();
    result->dataset_version =
        ResolveLanceNamespaceTableVersion(context, result->namespace_config);
    result->vector_column = RequireNamespaceSearchColumn(
        *table, result->vector_column, "lance_vector_search", "vector_column");
    if (result->prefilter && result->namespace_filter.empty()) {
      throw InvalidInputException(
          "lance_vector_search requires explicit filter when prefilter=true "
          "on namespace-backed tables");
    }
    PopulateNamespaceSearchSchema(
        context, *table, "_distance", result->schema_root, result->arrow_table,
        result->names, result->types, names, return_types);
    return std::move(result);
  }

  if (!result->namespace_filter.empty() &&
      (!namespace_table || !namespace_table->NamespaceConfig().IsDirectory())) {
    throw InvalidInputException(
        "lance_vector_search filter parameter is only supported for "
        "namespace-backed tables");
  }

  result->file_path.clear();
  result->dataset_entry =
      OpenSearchDatasetEntry(context, input.inputs[0], "lance_vector_search",
                             result->file_path, &result->dataset_cache_hit);
  result->dataset =
      result->dataset_entry ? result->dataset_entry->Handle() : nullptr;
  if (!result->dataset) {
    throw IOException("Failed to open Lance dataset: " + result->file_path +
                      LanceFormatErrorSuffix());
  }
  result->non_replayable_storage_options =
      !SearchStorageOptionsAreWorkerReplayable(context, input.inputs[0],
                                               result->file_path);
  RejectSearchComputedColumnCollisions(result->dataset,
                                       LanceComputedSearchColumns::Distance,
                                       "lance_vector_search");
  result->dataset_version = lance_dataset_version(result->dataset);
  if (result->dataset_version == 0) {
    throw IOException("Failed to capture Lance dataset version: " +
                      result->file_path + LanceFormatErrorSuffix());
  }
  result->dataset_generation_id = result->dataset_entry->GenerationId();

  auto *schema_handle = lance_get_knn_schema(
      result->dataset, result->vector_column.c_str(), result->query.data(),
      result->query.size(), result->k, result->nprobes, result->refine_factor,
      result->prefilter ? 1 : 0, result->use_index ? 1 : 0);
  if (!schema_handle) {
    throw IOException("Failed to get Lance KNN schema: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  memset(&result->schema_root.arrow_schema, 0,
         sizeof(result->schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &result->schema_root.arrow_schema) !=
      0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance KNN schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(schema_handle);
  LanceCoerceArrowSchemaForDuckDB(&result->schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(
      context, result->arrow_table, result->schema_root.arrow_schema);
  result->names = result->arrow_table.GetNames();
  result->types = result->arrow_table.GetTypes();
  names = result->names;
  return_types = result->types;
  return std::move(result);
}

static unique_ptr<GlobalTableFunctionState>
LanceKnnInitGlobal(ClientContext &, TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceKnnBindData>();
  auto state = make_uniq_base<GlobalTableFunctionState, LanceKnnGlobalState>();
  auto &global = state->Cast<LanceKnnGlobalState>();

  global.projection_ids = input.projection_ids;
  if (!input.projection_ids.empty()) {
    global.scanned_types.reserve(input.column_ids.size());
    for (auto col_id : input.column_ids) {
      if (col_id >= bind_data.types.size()) {
        throw IOException("Invalid column id in projection");
      }
      global.scanned_types.push_back(bind_data.types[col_id]);
    }
  }

  if (bind_data.namespace_backed) {
    if (bind_data.names.empty()) {
      throw InternalException(
          "Lance namespace vector search has no output columns");
    }
    global.namespace_columns.assign(bind_data.names.begin(),
                                    bind_data.names.end() - 1);
    return state;
  }

  if (bind_data.dataset) {
    auto operation_id = "vector-search:" + bind_data.file_path + ":" +
                        to_string(bind_data.dataset_version) + ":" +
                        bind_data.dataset_generation_id;
    global.snapshot_lease = LanceLease::AcquireForDataset(
        bind_data.dataset, LanceLeaseKind::Snapshot, operation_id);
  }

  auto table_filters =
      BuildLanceTableFilterIRParts(bind_data.names, bind_data.types, input,
                                   LanceComputedSearchColumns::Distance);
  if (bind_data.prefilter && bind_data.complex_filter_pushdown_failed) {
    throw InvalidInputException(
        "lance_vector_search cannot apply every complex predicate before top-k "
        "when prefilter=true");
  }
  if (bind_data.prefilter && !table_filters.all_prefilterable_filters_pushed) {
    throw InvalidInputException("lance_vector_search requires filter pushdown "
                                "for prefilterable columns when "
                                "prefilter=true");
  }

  bool has_table_filter_parts = !table_filters.parts.empty();
  auto filter_parts = std::move(table_filters.parts);
  if (!bind_data.lance_pushed_filter_ir_parts.empty()) {
    filter_parts.reserve(filter_parts.size() +
                         bind_data.lance_pushed_filter_ir_parts.size());
    for (auto &part : bind_data.lance_pushed_filter_ir_parts) {
      filter_parts.push_back(part);
    }
  }

  string filter_ir_msg;
  if (!filter_parts.empty()) {
    if (!TryEncodeLanceFilterIRMessage(filter_parts, filter_ir_msg)) {
      filter_ir_msg.clear();
    }
    global.lance_filter_ir = std::move(filter_ir_msg);
  }
  if (bind_data.prefilter && has_table_filter_parts &&
      global.lance_filter_ir.empty()) {
    throw IOException("Failed to encode Lance filter IR");
  }
  global.filter_pushed_down =
      table_filters.all_filters_pushed && !global.lance_filter_ir.empty();
  return state;
}

static unique_ptr<LocalTableFunctionState>
LanceKnnLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                  GlobalTableFunctionState *global_state) {
  auto &bind_data = input.bind_data->Cast<LanceKnnBindData>();
  auto &global = global_state->Cast<LanceKnnGlobalState>();

  auto chunk = make_uniq<ArrowArrayWrapper>();
  auto result = make_uniq<LanceKnnLocalState>(std::move(chunk), context.client);
  result->column_ids = input.column_ids;
  result->filters = input.filters.get();
  result->global_state = &global;
  result->filter_pushed_down = global.filter_pushed_down;
  if (global.CanRemoveFilterColumns()) {
    result->all_columns.Initialize(context.client, global.scanned_types);
  }
  if (bind_data.assigned_empty_scan) {
    return std::move(result);
  }

  if (bind_data.namespace_backed) {
    vector<string> resolved_option_keys;
    vector<string> resolved_option_values;
    vector<const char *> option_key_ptrs;
    vector<const char *> option_value_ptrs;
    vector<const char *> column_ptrs;
    string bearer_token;
    string api_key;
    string headers_tsv;
    LanceNamespaceQueryConfig config;
    FillLanceNamespaceQueryConfig(
        context.client, bind_data.namespace_config, bind_data.dataset_version,
        bind_data.k, bind_data.prefilter, bind_data.namespace_filter,
        global.namespace_columns, resolved_option_keys, resolved_option_values,
        option_key_ptrs, option_value_ptrs, column_ptrs, bearer_token, api_key,
        headers_tsv, config);
    vector<const char *> expected_column_ptrs;
    expected_column_ptrs.reserve(bind_data.names.size());
    for (auto &name : bind_data.names) {
      expected_column_ptrs.push_back(name.c_str());
    }
    config.expected_columns = expected_column_ptrs.data();
    config.expected_columns_len = expected_column_ptrs.size();
    LanceNamespaceVectorSearchOptions options;
    options.vector_column = bind_data.vector_column.c_str();
    options.query_values = bind_data.query.data();
    options.query_len = bind_data.query.size();
    options.nprobes = bind_data.nprobes;
    options.refine_factor = bind_data.refine_factor;
    options.use_index = bind_data.use_index ? 1 : 0;
    result->stream =
        lance_create_namespace_vector_search_stream(&config, &options);
    if (!result->stream) {
      throw IOException("Failed to create Lance namespace vector search "
                        "stream" +
                        LanceFormatErrorSuffix());
    }
    return std::move(result);
  }

  const uint8_t *filter_ir =
      global.lance_filter_ir.empty()
          ? nullptr
          : reinterpret_cast<const uint8_t *>(global.lance_filter_ir.data());
  auto filter_ir_len = global.lance_filter_ir.size();
  const char *filter_sql = bind_data.namespace_filter.empty()
                               ? nullptr
                               : bind_data.namespace_filter.c_str();
  result->stream = lance_create_knn_stream_ir(
      bind_data.dataset, bind_data.vector_column.c_str(),
      bind_data.query.data(), bind_data.query.size(), bind_data.k,
      bind_data.nprobes, bind_data.refine_factor, filter_ir, filter_ir_len,
      filter_sql, bind_data.prefilter ? 1 : 0, bind_data.use_index ? 1 : 0);
  if (!result->stream && filter_ir && !bind_data.prefilter) {
    // Best-effort: if filter pushdown failed, retry without it and rely on
    // DuckDB-side filter execution for correctness.
    global.filter_pushdown_fallbacks.fetch_add(1);
    global.filter_pushed_down = false;
    result->filter_pushed_down = false;
    result->stream = lance_create_knn_stream_ir(
        bind_data.dataset, bind_data.vector_column.c_str(),
        bind_data.query.data(), bind_data.query.size(), bind_data.k,
        bind_data.nprobes, bind_data.refine_factor, nullptr, 0, filter_sql,
        bind_data.prefilter ? 1 : 0, bind_data.use_index ? 1 : 0);
  }
  if (!result->stream) {
    throw IOException("Failed to create Lance KNN stream" +
                      LanceFormatErrorSuffix());
  }

  return std::move(result);
}

static bool LanceKnnLoadNextBatch(ClientContext &context,
                                  LanceKnnLocalState &local_state,
                                  const LanceKnnBindData &bind_data) {
  if (!local_state.stream) {
    return false;
  }

  void *batch = nullptr;
  auto rc = lance_stream_next(local_state.stream, &batch);
  if (rc == 1) {
    lance_close_stream(local_state.stream);
    local_state.stream = nullptr;
    return false;
  }
  if (rc != 0) {
    throw IOException("Failed to read next Lance RecordBatch" +
                      LanceFormatErrorSuffix());
  }

  auto new_chunk = make_shared_ptr<ArrowArrayWrapper>();
  memset(&new_chunk->arrow_array, 0, sizeof(new_chunk->arrow_array));
  ArrowSchemaWrapper tmp_schema;
  memset(&tmp_schema.arrow_schema, 0, sizeof(tmp_schema.arrow_schema));

  if (lance_batch_to_arrow(batch, &new_chunk->arrow_array,
                           &tmp_schema.arrow_schema) != 0) {
    lance_free_batch(batch);
    throw IOException(
        "Failed to export Lance RecordBatch to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }

  lance_free_batch(batch);

  LanceValidateAndReorderArrowBatch(context, tmp_schema.arrow_schema,
                                    new_chunk->arrow_array, bind_data.names,
                                    bind_data.types, "Lance KNN search");

  if (local_state.global_state) {
    local_state.global_state->record_batches.fetch_add(1);
    auto rows = NumericCast<idx_t>(new_chunk->arrow_array.length);
    local_state.global_state->record_batch_rows.fetch_add(rows);
  }

  local_state.chunk = std::move(new_chunk);
  local_state.Reset();
  return true;
}

static void LanceKnnFunc(ClientContext &context, TableFunctionInput &data,
                         DataChunk &output) {
  if (!data.local_state) {
    return;
  }

  auto &bind_data = data.bind_data->Cast<LanceKnnBindData>();
  if (bind_data.assigned_empty_scan) {
    return;
  }
  auto &global_state = data.global_state->Cast<LanceKnnGlobalState>();
  auto &local_state = data.local_state->Cast<LanceKnnLocalState>();

  while (true) {
    if (local_state.chunk_offset >=
        NumericCast<idx_t>(local_state.chunk->arrow_array.length)) {
      if (!LanceKnnLoadNextBatch(context, local_state, bind_data)) {
        return;
      }
    }

    auto remaining = NumericCast<idx_t>(local_state.chunk->arrow_array.length) -
                     local_state.chunk_offset;
    auto output_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, remaining);
    global_state.lines_read.fetch_add(output_size);

    if (global_state.CanRemoveFilterColumns()) {
      local_state.all_columns.Reset();
      local_state.all_columns.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(local_state,
                                        bind_data.arrow_table.GetColumns(),
                                        local_state.all_columns, false);
      local_state.chunk_offset += output_size;
      if (local_state.filters && !local_state.filter_pushed_down) {
        ApplyDuckDBFilters(context, *local_state.filters,
                           local_state.all_columns, local_state.filter_sel);
      }
      output.ReferenceColumns(local_state.all_columns,
                              global_state.projection_ids);
      output.SetCardinality(local_state.all_columns);
    } else {
      output.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(
          local_state, bind_data.arrow_table.GetColumns(), output, false);
      local_state.chunk_offset += output_size;
      if (local_state.filters && !local_state.filter_pushed_down) {
        ApplyDuckDBFilters(context, *local_state.filters, output,
                           local_state.filter_sel);
      }
    }

    if (output.size() == 0) {
      continue;
    }
    output.Verify();
    return;
  }
}

static InsertionOrderPreservingMap<string>
LanceKnnToString(TableFunctionToStringInput &input) {
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceKnnBindData>();

  result["Lance Path"] = bind_data.file_path;
  result["Lance Search Backend"] =
      bind_data.namespace_backed ? "namespace_query_table" : "dataset_scan";
  result["Lance Vector Column"] = bind_data.vector_column;
  result["Lance K"] = to_string(bind_data.k);
  result["Lance Nprobes"] = to_string(bind_data.nprobes);
  result["Lance Refine Factor"] = to_string(bind_data.refine_factor);
  result["Lance Query Dim"] = to_string(bind_data.query.size());
  result["Lance Prefilter"] = bind_data.prefilter ? "true" : "false";
  result["Lance Use Index"] = bind_data.use_index ? "true" : "false";
  result["Lance Explain Verbose"] =
      bind_data.explain_verbose ? "true" : "false";
  result["Lance Dataset Cache Hit"] =
      bind_data.dataset_cache_hit ? "true" : "false";
  if (!bind_data.namespace_filter.empty()) {
    result["Lance Namespace Filter"] = bind_data.namespace_filter;
  }

  if (bind_data.namespace_backed) {
    return result;
  }

  result["Lance Pushed Filter Parts"] =
      to_string(bind_data.lance_pushed_filter_ir_parts.size());
  string filter_ir_msg;
  if (!bind_data.lance_pushed_filter_ir_parts.empty()) {
    TryEncodeLanceFilterIRMessage(bind_data.lance_pushed_filter_ir_parts,
                                  filter_ir_msg);
  }
  result["Lance Filter IR Bytes (Bind)"] = to_string(filter_ir_msg.size());

  string plan;
  string error;
  if (TryLanceExplainKnn(
          bind_data.dataset, bind_data.vector_column, bind_data.query,
          bind_data.k, bind_data.nprobes, bind_data.refine_factor,
          filter_ir_msg.empty() ? nullptr : &filter_ir_msg, bind_data.prefilter,
          bind_data.use_index, bind_data.explain_verbose, plan, error)) {
    result["Lance Plan (Bind)"] = plan;
  } else if (!error.empty()) {
    result["Lance Plan Error (Bind)"] = error;
  }

  return result;
}

static InsertionOrderPreservingMap<string>
LanceKnnDynamicToString(TableFunctionDynamicToStringInput &input) {
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceKnnBindData>();
  auto &global_state = input.global_state->Cast<LanceKnnGlobalState>();

  result["Lance Path"] = bind_data.file_path;
  result["Lance Search Backend"] =
      bind_data.namespace_backed ? "namespace_query_table" : "dataset_scan";
  result["Lance Vector Column"] = bind_data.vector_column;
  result["Lance K"] = to_string(bind_data.k);
  result["Lance Nprobes"] = to_string(bind_data.nprobes);
  result["Lance Refine Factor"] = to_string(bind_data.refine_factor);
  result["Lance Query Dim"] = to_string(bind_data.query.size());
  result["Lance Prefilter"] = bind_data.prefilter ? "true" : "false";
  result["Lance Use Index"] = bind_data.use_index ? "true" : "false";
  result["Lance Explain Verbose"] =
      bind_data.explain_verbose ? "true" : "false";
  result["Lance Dataset Cache Hit"] =
      bind_data.dataset_cache_hit ? "true" : "false";
  if (!bind_data.namespace_filter.empty()) {
    result["Lance Namespace Filter"] = bind_data.namespace_filter;
  }

  result["Lance Filter Pushed Down"] =
      global_state.filter_pushed_down ? "true" : "false";
  result["Lance Filter Pushdown Fallbacks"] =
      to_string(global_state.filter_pushdown_fallbacks.load());
  result["Lance Filter IR Bytes"] =
      to_string(global_state.lance_filter_ir.size());

  result["Lance Record Batches"] =
      to_string(global_state.record_batches.load());
  result["Lance Record Batch Rows"] =
      to_string(global_state.record_batch_rows.load());
  result["Lance Rows Out"] = to_string(global_state.lines_read.load());

  if (bind_data.namespace_backed) {
    return result;
  }

  if (!global_state.explain_computed.load()) {
    std::lock_guard<std::mutex> guard(global_state.explain_mutex);
    if (!global_state.explain_computed.load()) {
      string plan;
      string error;
      auto ok = TryLanceExplainKnn(
          bind_data.dataset, bind_data.vector_column, bind_data.query,
          bind_data.k, bind_data.nprobes, bind_data.refine_factor,
          global_state.lance_filter_ir.empty() ? nullptr
                                               : &global_state.lance_filter_ir,
          bind_data.prefilter, bind_data.use_index, bind_data.explain_verbose,
          plan, error);
      if (ok) {
        global_state.explain_plan = std::move(plan);
      } else {
        global_state.explain_error = std::move(error);
      }
      global_state.explain_computed.store(true);
    }
  }

  if (!global_state.explain_plan.empty()) {
    result["Lance Plan"] = global_state.explain_plan;
  } else if (!global_state.explain_error.empty()) {
    result["Lance Plan Error"] = global_state.explain_error;
  }

  return result;
}

static void LanceKnnSerialize(Serializer &serializer,
                              const optional_ptr<FunctionData> bind_data_p,
                              const TableFunction &) {
  auto &bind_data = bind_data_p->Cast<LanceKnnBindData>();
  serializer.WriteProperty(100, "file_path", bind_data.file_path);
  serializer.WriteProperty(101, "vector_column", bind_data.vector_column);
  serializer.WriteProperty(102, "query", bind_data.query);
  serializer.WriteProperty(103, "k", bind_data.k);
  serializer.WriteProperty(104, "nprobes", bind_data.nprobes);
  serializer.WriteProperty(105, "refine_factor", bind_data.refine_factor);
  serializer.WriteProperty(106, "prefilter", bind_data.prefilter);
  serializer.WriteProperty(107, "use_index", bind_data.use_index);
  serializer.WriteProperty(108, "explain_verbose", bind_data.explain_verbose);
  serializer.WriteProperty(109, "namespace_backed", bind_data.namespace_backed);
  serializer.WriteProperty(110, "namespace_filter", bind_data.namespace_filter);
  serializer.WriteProperty(111, "dataset_version", bind_data.dataset_version);
  serializer.WriteProperty(112, "names", bind_data.names);
  serializer.WriteProperty(113, "types", bind_data.types);
  serializer.WriteProperty(114, "lance_filter_ir_parts",
                           bind_data.lance_pushed_filter_ir_parts);
  if (bind_data.namespace_backed) {
    auto &cfg = bind_data.namespace_config;
    serializer.WriteProperty(115, "namespace_kind",
                             static_cast<uint8_t>(cfg.kind));
    serializer.WriteProperty(116, "namespace_root", cfg.root);
    serializer.WriteProperty(117, "namespace_endpoint", cfg.endpoint);
    serializer.WriteProperty(118, "namespace_table_id", cfg.table_id);
    serializer.WriteProperty(119, "namespace_delimiter", cfg.delimiter);
    serializer.WriteProperty(120, "namespace_display_uri", cfg.display_uri);
  }
  serializer.WriteProperty(122, "dataset_generation_id",
                           bind_data.dataset_generation_id);
  serializer.WriteProperty(123, "complex_filter_pushdown_failed",
                           bind_data.complex_filter_pushdown_failed);
  serializer.WriteProperty(124, "namespace_requires_worker_auth",
                           bind_data.namespace_backed &&
                               bind_data.namespace_config.requires_worker_auth);
  serializer.WriteProperty(125, "non_replayable_storage_options",
                           bind_data.non_replayable_storage_options);
}

static unique_ptr<FunctionData> LanceKnnDeserialize(Deserializer &deserializer,
                                                    TableFunction &) {
  auto result = make_uniq<LanceKnnBindData>();
  auto &context = deserializer.Get<ClientContext &>();
  result->file_path = deserializer.ReadProperty<string>(100, "file_path");
  result->vector_column =
      deserializer.ReadProperty<string>(101, "vector_column");
  result->query = deserializer.ReadProperty<vector<float>>(102, "query");
  result->k = deserializer.ReadProperty<uint64_t>(103, "k");
  result->nprobes = deserializer.ReadProperty<uint64_t>(104, "nprobes");
  result->refine_factor =
      deserializer.ReadProperty<uint64_t>(105, "refine_factor");
  result->prefilter = deserializer.ReadProperty<bool>(106, "prefilter");
  result->use_index = deserializer.ReadProperty<bool>(107, "use_index");
  result->explain_verbose =
      deserializer.ReadProperty<bool>(108, "explain_verbose");
  result->namespace_backed =
      deserializer.ReadProperty<bool>(109, "namespace_backed");
  result->namespace_filter =
      deserializer.ReadProperty<string>(110, "namespace_filter");
  result->dataset_version =
      deserializer.ReadProperty<uint64_t>(111, "dataset_version");
  result->names = deserializer.ReadProperty<vector<string>>(112, "names");
  result->types = deserializer.ReadProperty<vector<LogicalType>>(113, "types");
  result->lance_pushed_filter_ir_parts =
      deserializer.ReadProperty<vector<string>>(114, "lance_filter_ir_parts");
  if (result->namespace_backed) {
    auto &cfg = result->namespace_config;
    cfg.kind = static_cast<LanceNamespaceKind>(
        deserializer.ReadProperty<uint8_t>(115, "namespace_kind"));
    cfg.root = deserializer.ReadProperty<string>(116, "namespace_root");
    cfg.endpoint = deserializer.ReadProperty<string>(117, "namespace_endpoint");
    cfg.table_id = deserializer.ReadProperty<string>(118, "namespace_table_id");
    cfg.delimiter =
        deserializer.ReadProperty<string>(119, "namespace_delimiter");
    cfg.display_uri =
        deserializer.ReadProperty<string>(120, "namespace_display_uri");
  }
  result->dataset_generation_id =
      deserializer.ReadProperty<string>(122, "dataset_generation_id");
  result->complex_filter_pushdown_failed =
      deserializer.ReadProperty<bool>(123, "complex_filter_pushdown_failed");
  result->namespace_config.requires_worker_auth =
      deserializer.ReadProperty<bool>(124, "namespace_requires_worker_auth");
  result->non_replayable_storage_options =
      deserializer.ReadProperty<bool>(125, "non_replayable_storage_options");
  ValidateSerializedQueryVector(result->query, "Lance vector search");
  ValidateSerializedSearchCString(result->file_path, "dataset path");
  ValidateSerializedSearchCString(result->vector_column, "vector column");
  ValidateSerializedSearchCString(result->namespace_filter, "filter");
  ValidateSerializedSearchCStrings(result->names, "output column");
  if (result->namespace_backed) {
    ValidateSerializedSearchCString(result->namespace_config.root,
                                    "namespace root");
    ValidateSerializedSearchCString(result->namespace_config.endpoint,
                                    "namespace endpoint");
    ValidateSerializedSearchCString(result->namespace_config.table_id,
                                    "namespace table id");
    ValidateSerializedSearchCString(result->namespace_config.delimiter,
                                    "namespace delimiter");
  }
  if (result->k == 0) {
    throw SerializationException(
        "Serialized Lance vector search requires k > 0");
  }
  if (result->names.empty() || result->names.size() != result->types.size()) {
    throw SerializationException(
        "Serialized Lance vector search has an invalid output schema");
  }
  if (result->namespace_config.requires_worker_auth) {
    throw SerializationException(
        "Serialized Lance namespace vector search requires credentials that "
        "were not transported to the worker");
  }
  if (result->non_replayable_storage_options) {
    throw SerializationException(
        "Serialized Lance vector search requires storage options that were "
        "not transported to the worker");
  }
  if (result->namespace_backed) {
    if (!result->namespace_config.IsRest()) {
      throw SerializationException(
          "Serialized Lance namespace vector search has an invalid namespace "
          "kind");
    }
    if (result->dataset_version == 0) {
      throw SerializationException("Serialized Lance namespace vector search "
                                   "is missing its fixed dataset version");
    }
  } else {
    if (result->dataset_version == 0) {
      throw SerializationException("Serialized Lance vector search is missing "
                                   "its fixed dataset version");
    }
    if (result->dataset_generation_id.empty()) {
      throw SerializationException("Serialized Lance vector search is missing "
                                   "its fixed dataset generation");
    }
    result->dataset_entry = LanceGetOrOpenDatasetEntryAtVersion(
        context, result->file_path, result->dataset_version,
        result->dataset_generation_id, &result->dataset_cache_hit);
    result->dataset =
        result->dataset_entry ? result->dataset_entry->Handle() : nullptr;
    if (!result->dataset ||
        lance_dataset_version(result->dataset) != result->dataset_version) {
      throw IOException(
          "Failed to reopen fixed Lance vector-search snapshot: " +
          result->file_path + LanceFormatErrorSuffix());
    }
  }
  PopulateSearchSchemaFromTypes(context, result->schema_root,
                                result->arrow_table, result->names,
                                result->types);
  return std::move(result);
}

static void RegisterLanceVectorSearch(ExtensionLoader &loader) {
  auto configure = [](TableFunction &fun) {
    fun.named_parameters["k"] = LogicalType::BIGINT;
    fun.named_parameters["nprobs"] = LogicalType::BIGINT;
    fun.named_parameters["refine_factor"] = LogicalType::BIGINT;
    fun.named_parameters["prefilter"] = LogicalType::BOOLEAN;
    fun.named_parameters["use_index"] = LogicalType::BOOLEAN;
    fun.named_parameters["explain_verbose"] = LogicalType::BOOLEAN;
    fun.named_parameters["filter"] = LogicalType::VARCHAR;
    fun.projection_pushdown = true;
    fun.filter_pushdown = true;
    fun.filter_prune = true;
    fun.pushdown_expression = LancePushdownExpression;
    fun.pushdown_complex_filter = LancePushdownComplexFilter;
    fun.to_string = LanceKnnToString;
    fun.dynamic_to_string = LanceKnnDynamicToString;
    fun.serialize = LanceKnnSerialize;
    fun.deserialize = LanceKnnDeserialize;
#ifdef LANCE_VANE_DISTRIBUTED
    fun.SetDistributedScanCallbacks(LanceKnnDistributedScanCallbacks());
#endif
  };

  TableFunction search_f32("lance_vector_search",
                           {LogicalType::VARCHAR, LogicalType::VARCHAR,
                            LogicalType::LIST(LogicalType::FLOAT)},
                           LanceKnnFunc, LanceSearchVectorBind,
                           LanceKnnInitGlobal, LanceKnnLocalInit);
  configure(search_f32);
  loader.RegisterFunction(search_f32);

  TableFunction search_f64("lance_vector_search",
                           {LogicalType::VARCHAR, LogicalType::VARCHAR,
                            LogicalType::LIST(LogicalType::DOUBLE)},
                           LanceKnnFunc, LanceSearchVectorBind,
                           LanceKnnInitGlobal, LanceKnnLocalInit);
  configure(search_f64);
  loader.RegisterFunction(search_f64);
}

// --- FTS / hybrid search ---

enum class LanceSearchMode : uint8_t { Fts = 0, Hybrid = 1 };

static LanceComputedSearchColumns
LanceSearchComputedColumns(LanceSearchMode mode) {
  return mode == LanceSearchMode::Fts ? LanceComputedSearchColumns::Score
                                      : LanceComputedSearchColumns::Hybrid;
}

struct LanceSearchBindData : public TableFunctionData,
                             public LanceGlobalSearchSplitState {
  LanceSearchMode mode = LanceSearchMode::Fts;

  string file_path;
  bool prefilter = false;
  bool namespace_backed = false;
  LanceNamespaceTableConfig namespace_config;
  string namespace_filter;

  // FTS mode
  string text_column;
  string query;

  // Hybrid mode
  string vector_column;
  vector<float> vector_query;
  string text_query;
  uint64_t nprobes = 0;
  uint64_t refine_factor = 0;
  bool use_index = true;
  float alpha = 0.5F;
  uint32_t oversample_factor = 4;

  uint64_t k = 10;

  shared_ptr<LanceDatasetCacheEntry> dataset_entry;
  void *dataset = nullptr;
  bool dataset_cache_hit = false;
  uint64_t dataset_version = 0;
  string dataset_generation_id;
  bool non_replayable_storage_options = false;
  ArrowSchemaWrapper schema_root;
  ArrowTableSchema arrow_table;
  vector<string> names;
  vector<LogicalType> types;
  vector<string> lance_pushed_filter_ir_parts;
  bool complex_filter_pushdown_failed = false;

  unique_ptr<FunctionData> Copy() const override {
    auto result = make_uniq<LanceSearchBindData>();
    result->column_ids = column_ids;
    result->mode = mode;
    result->file_path = file_path;
    result->prefilter = prefilter;
    result->namespace_backed = namespace_backed;
    result->namespace_config = namespace_config;
    result->namespace_filter = namespace_filter;
    result->text_column = text_column;
    result->query = query;
    result->vector_column = vector_column;
    result->vector_query = vector_query;
    result->text_query = text_query;
    result->nprobes = nprobes;
    result->refine_factor = refine_factor;
    result->use_index = use_index;
    result->alpha = alpha;
    result->oversample_factor = oversample_factor;
    result->k = k;
    result->dataset_entry = dataset_entry;
    result->dataset = dataset;
    result->dataset_cache_hit = dataset_cache_hit;
    result->dataset_version = dataset_version;
    result->dataset_generation_id = dataset_generation_id;
    result->non_replayable_storage_options = non_replayable_storage_options;
    result->arrow_table = arrow_table;
    result->names = names;
    result->types = types;
    result->lance_pushed_filter_ir_parts = lance_pushed_filter_ir_parts;
    result->complex_filter_pushdown_failed = complex_filter_pushdown_failed;
    result->assigned_empty_scan = assigned_empty_scan;
    return std::move(result);
  }
};

#ifdef LANCE_VANE_DISTRIBUTED
static unique_ptr<FunctionData> LanceCreateDistributedSearchWorkerBind(
    const TableFunctionDistributedScanInput &input) {
  auto &source = input.bind_data.Cast<LanceSearchBindData>();
  if (source.namespace_backed && source.namespace_config.requires_worker_auth) {
    throw NotImplementedException(
        "Distributed Lance REST namespace search does not transport bearer "
        "tokens, API keys, or custom headers to workers");
  }
  if (source.non_replayable_storage_options) {
    throw NotImplementedException(
        "Distributed Lance search cannot transport TYPE LANCE secrets or "
        "arbitrary storage options to workers; configure equivalent explicit "
        "s3_* connection settings or run locally");
  }
  auto result = make_uniq<LanceSearchBindData>();
  result->mode = source.mode;
  result->file_path = source.file_path;
  result->prefilter = source.prefilter;
  result->namespace_backed = source.namespace_backed;
  result->namespace_config =
      LanceCreateDistributedNamespaceConfig(source.namespace_config);
  result->namespace_filter = source.namespace_filter;
  result->text_column = source.text_column;
  result->query = source.query;
  result->vector_column = source.vector_column;
  result->vector_query = source.vector_query;
  result->text_query = source.text_query;
  result->nprobes = source.nprobes;
  result->refine_factor = source.refine_factor;
  result->use_index = source.use_index;
  result->alpha = source.alpha;
  result->oversample_factor = source.oversample_factor;
  result->k = source.k;
  result->dataset_version = source.dataset_version;
  result->dataset_generation_id = source.dataset_generation_id;
  result->names = source.names;
  result->types = source.types;
  result->lance_pushed_filter_ir_parts = source.lance_pushed_filter_ir_parts;
  result->complex_filter_pushdown_failed =
      source.complex_filter_pushdown_failed;
  return std::move(result);
}
#endif

static void
LanceSearchPushdownComplexFilter(ClientContext &, LogicalGet &get,
                                 FunctionData *bind_data,
                                 vector<unique_ptr<Expression>> &filters) {
  if (!bind_data || filters.empty()) {
    return;
  }
  auto &scan_bind = bind_data->Cast<LanceSearchBindData>();
  if (scan_bind.namespace_backed) {
    return;
  }

  for (auto &expr : filters) {
    if (!expr || expr->HasParameter() || expr->IsVolatile()) {
      scan_bind.complex_filter_pushdown_failed = true;
      continue;
    }
    if (expr->expression_class == ExpressionClass::BOUND_COMPARISON) {
      auto &cmp = expr->Cast<BoundComparisonExpression>();
      if (cmp.type == ExpressionType::COMPARE_DISTINCT_FROM ||
          cmp.type == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
        auto is_constant = [](const unique_ptr<Expression> &node) {
          return node &&
                 (node->expression_class == ExpressionClass::BOUND_CONSTANT ||
                  (node->expression_class == ExpressionClass::BOUND_CAST &&
                   !node->Cast<BoundCastExpression>().try_cast &&
                   node->Cast<BoundCastExpression>().child &&
                   node->Cast<BoundCastExpression>().child->expression_class ==
                       ExpressionClass::BOUND_CONSTANT));
        };
        auto is_column = [](const unique_ptr<Expression> &node) {
          return node &&
                 (node->expression_class == ExpressionClass::BOUND_COLUMN_REF ||
                  node->expression_class == ExpressionClass::BOUND_REF);
        };
        if ((is_column(cmp.left) && is_constant(cmp.right)) ||
            (is_column(cmp.right) && is_constant(cmp.left))) {
          continue;
        }
      }
    }
    string filter_ir;
    if (!TryBuildLanceExprFilterIR(get, scan_bind.names, scan_bind.types,
                                   LanceSearchComputedColumns(scan_bind.mode),
                                   *expr, filter_ir)) {
      scan_bind.complex_filter_pushdown_failed = true;
      continue;
    }
    scan_bind.lance_pushed_filter_ir_parts.push_back(std::move(filter_ir));
  }
}

#ifdef LANCE_VANE_DISTRIBUTED
static vector<DistributedScanSplit> LancePlanDistributedSearchSplits(
    const TableFunctionDistributedScanPlanningInput &input) {
  auto &bind_data = input.bind_data.Cast<LanceSearchBindData>();
  return {LanceGlobalSearchSplit(bind_data.k)};
}

static TableFunctionDistributedScanCallbacks
LanceSearchDistributedScanCallbacks() {
  TableFunctionDistributedScanCallbacks callbacks;
  callbacks.protocol_version = LANCE_SEARCH_PROTOCOL_VERSION;
  callbacks.split_codec = {LANCE_SEARCH_SPLIT_CODEC,
                           LANCE_SEARCH_SPLIT_CODEC_VERSION};
  callbacks.plan_splits = LancePlanDistributedSearchSplits;
  callbacks.create_worker_bind = LanceCreateDistributedSearchWorkerBind;
  callbacks.apply_splits = LanceApplyGlobalSearchSplits;
  return callbacks;
}
#endif

struct LanceSearchGlobalState : public GlobalTableFunctionState {
  unique_ptr<LanceLease> snapshot_lease;
  std::atomic<idx_t> lines_read{0};
  std::atomic<idx_t> record_batches{0};
  std::atomic<idx_t> record_batch_rows{0};
  string lance_filter_ir;
  bool filter_pushed_down = false;
  std::atomic<idx_t> filter_pushdown_fallbacks{0};

  vector<idx_t> projection_ids;
  vector<LogicalType> scanned_types;
  vector<string> namespace_columns;

  idx_t MaxThreads() const override { return 1; }
  bool CanRemoveFilterColumns() const { return !projection_ids.empty(); }
};

struct LanceSearchLocalState : public ArrowScanLocalState {
  explicit LanceSearchLocalState(unique_ptr<ArrowArrayWrapper> current_chunk,
                                 ClientContext &context)
      : ArrowScanLocalState(std::move(current_chunk), context),
        filter_sel(STANDARD_VECTOR_SIZE) {}

  void *stream = nullptr;
  LanceSearchGlobalState *global_state = nullptr;
  bool filter_pushed_down = false;
  SelectionVector filter_sel;

  ~LanceSearchLocalState() override {
    if (stream) {
      lance_close_stream(stream);
    }
  }
};

static bool LanceSearchLoadNextBatch(ClientContext &context,
                                     LanceSearchLocalState &local_state,
                                     const LanceSearchBindData &bind_data,
                                     LanceSearchGlobalState &global) {
  if (!local_state.stream) {
    if (bind_data.namespace_backed) {
      vector<string> resolved_option_keys;
      vector<string> resolved_option_values;
      vector<const char *> option_key_ptrs;
      vector<const char *> option_value_ptrs;
      vector<const char *> column_ptrs;
      string bearer_token;
      string api_key;
      string headers_tsv;
      LanceNamespaceQueryConfig config;
      FillLanceNamespaceQueryConfig(
          context, bind_data.namespace_config, bind_data.dataset_version,
          bind_data.k, bind_data.prefilter, bind_data.namespace_filter,
          global.namespace_columns, resolved_option_keys,
          resolved_option_values, option_key_ptrs, option_value_ptrs,
          column_ptrs, bearer_token, api_key, headers_tsv, config);
      vector<const char *> expected_column_ptrs;
      expected_column_ptrs.reserve(bind_data.names.size());
      for (auto &name : bind_data.names) {
        expected_column_ptrs.push_back(name.c_str());
      }
      config.expected_columns = expected_column_ptrs.data();
      config.expected_columns_len = expected_column_ptrs.size();
      LanceNamespaceFtsSearchOptions options;
      options.text_column = bind_data.text_column.c_str();
      options.query = bind_data.query.c_str();
      local_state.stream =
          lance_create_namespace_fts_search_stream(&config, &options);
      if (!local_state.stream) {
        throw IOException("Failed to create Lance namespace FTS stream" +
                          LanceFormatErrorSuffix());
      }
    } else {
      const uint8_t *filter_ir = global.lance_filter_ir.empty()
                                     ? nullptr
                                     : reinterpret_cast<const uint8_t *>(
                                           global.lance_filter_ir.data());
      auto filter_ir_len = NumericCast<idx_t>(global.lance_filter_ir.size());

      auto create_stream = [&](const uint8_t *ir, idx_t ir_len) -> void * {
        if (bind_data.mode == LanceSearchMode::Fts) {
          const char *filter_sql = bind_data.namespace_filter.empty()
                                       ? nullptr
                                       : bind_data.namespace_filter.c_str();
          return lance_create_fts_stream_ir(
              bind_data.dataset, bind_data.text_column.c_str(),
              bind_data.query.c_str(), bind_data.k, ir,
              NumericCast<size_t>(ir_len), filter_sql,
              bind_data.prefilter ? 1 : 0);
        }
        return lance_create_hybrid_stream_ir(
            bind_data.dataset, bind_data.vector_column.c_str(),
            bind_data.vector_query.data(), bind_data.vector_query.size(),
            bind_data.text_column.c_str(), bind_data.text_query.c_str(),
            bind_data.k, bind_data.nprobes, bind_data.refine_factor, ir,
            NumericCast<size_t>(ir_len), bind_data.prefilter ? 1 : 0,
            bind_data.use_index ? 1 : 0, bind_data.alpha,
            bind_data.oversample_factor);
      };

      local_state.stream = create_stream(filter_ir, filter_ir_len);
      if (!local_state.stream && filter_ir && !bind_data.prefilter) {
        // Best-effort: if filter pushdown failed, retry without it and rely on
        // DuckDB-side filter execution for correctness.
        global.filter_pushdown_fallbacks.fetch_add(1);
        global.filter_pushed_down = false;
        local_state.filter_pushed_down = false;
        local_state.stream = create_stream(nullptr, 0);
      }
      if (!local_state.stream) {
        throw IOException("Failed to create Lance search stream" +
                          LanceFormatErrorSuffix());
      }
    }
  }

  void *batch = nullptr;
  auto rc = lance_stream_next(local_state.stream, &batch);
  if (rc == 1) {
    lance_close_stream(local_state.stream);
    local_state.stream = nullptr;
    return false;
  }
  if (rc != 0) {
    throw IOException("Failed to read next Lance RecordBatch" +
                      LanceFormatErrorSuffix());
  }

  auto new_chunk = make_shared_ptr<ArrowArrayWrapper>();
  memset(&new_chunk->arrow_array, 0, sizeof(new_chunk->arrow_array));
  ArrowSchemaWrapper tmp_schema;
  memset(&tmp_schema.arrow_schema, 0, sizeof(tmp_schema.arrow_schema));

  if (lance_batch_to_arrow(batch, &new_chunk->arrow_array,
                           &tmp_schema.arrow_schema) != 0) {
    lance_free_batch(batch);
    throw IOException(
        "Failed to export Lance RecordBatch to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_batch(batch);

  LanceValidateAndReorderArrowBatch(context, tmp_schema.arrow_schema,
                                    new_chunk->arrow_array, bind_data.names,
                                    bind_data.types, "Lance search");

  local_state.global_state->record_batches.fetch_add(1);
  auto rows = NumericCast<idx_t>(new_chunk->arrow_array.length);
  local_state.global_state->record_batch_rows.fetch_add(rows);

  local_state.chunk = std::move(new_chunk);
  local_state.Reset();
  return true;
}

static unique_ptr<FunctionData> LanceFtsBind(ClientContext &context,
                                             TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types,
                                             vector<string> &names) {
  if (input.inputs.size() < 3) {
    throw InvalidInputException(
        "lance_fts requires (path, text_column, query)");
  }
  if (input.inputs[0].IsNull()) {
    throw InvalidInputException("lance_fts requires a dataset root path");
  }
  if (input.inputs[1].IsNull()) {
    throw InvalidInputException("lance_fts requires a non-null text_column");
  }
  if (input.inputs[2].IsNull()) {
    throw InvalidInputException("lance_fts requires a non-null query");
  }

  auto result = make_uniq<LanceSearchBindData>();
  result->mode = LanceSearchMode::Fts;
  result->text_column = input.inputs[1].GetValue<string>();
  result->query = input.inputs[2].GetValue<string>();
  result->namespace_filter = ParseOptionalNamedString(input, "filter");
  ValidateLanceCString(result->text_column, "Lance text column");
  ValidateLanceCString(result->query, "Lance text query");
  ValidateLanceCString(result->namespace_filter, "Lance search filter");

  int64_t k_val = 10;
  auto k_named = input.named_parameters.find("k");
  if (k_named != input.named_parameters.end() && !k_named->second.IsNull()) {
    k_val =
        k_named->second.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
  }
  if (k_val <= 0) {
    throw InvalidInputException("lance_fts requires k > 0");
  }
  result->k = NumericCast<uint64_t>(k_val);

  auto prefilter_named = input.named_parameters.find("prefilter");
  if (prefilter_named != input.named_parameters.end() &&
      !prefilter_named->second.IsNull()) {
    result->prefilter =
        prefilter_named->second.DefaultCastAs(LogicalType::BOOLEAN)
            .GetValue<bool>();
  }

  auto *namespace_table =
      TryResolveNamespaceBackedSearchTable(context, input.inputs[0]);
  if (namespace_table && namespace_table->NamespaceConfig().IsRest()) {
    auto *table = namespace_table;
    result->namespace_backed = true;
    result->namespace_config = table->NamespaceConfig();
    result->file_path = table->DatasetUri();
    result->dataset_version =
        ResolveLanceNamespaceTableVersion(context, result->namespace_config);
    result->text_column = RequireNamespaceSearchColumn(
        *table, result->text_column, "lance_fts", "text_column");
    if (result->prefilter && result->namespace_filter.empty()) {
      throw InvalidInputException(
          "lance_fts requires explicit filter when prefilter=true on "
          "namespace-backed tables");
    }
    PopulateNamespaceSearchSchema(
        context, *table, "_score", result->schema_root, result->arrow_table,
        result->names, result->types, names, return_types);
    return std::move(result);
  }

  if (!result->namespace_filter.empty() &&
      (!namespace_table || !namespace_table->NamespaceConfig().IsDirectory())) {
    throw InvalidInputException(
        "lance_fts filter parameter is only supported for namespace-backed "
        "tables");
  }

  result->file_path.clear();
  result->dataset_entry =
      OpenSearchDatasetEntry(context, input.inputs[0], "lance_fts",
                             result->file_path, &result->dataset_cache_hit);
  result->dataset =
      result->dataset_entry ? result->dataset_entry->Handle() : nullptr;

  if (!result->dataset) {
    throw IOException("Failed to open Lance dataset: " + result->file_path +
                      LanceFormatErrorSuffix());
  }
  result->non_replayable_storage_options =
      !SearchStorageOptionsAreWorkerReplayable(context, input.inputs[0],
                                               result->file_path);
  RejectSearchComputedColumnCollisions(
      result->dataset, LanceComputedSearchColumns::Score, "lance_fts");
  result->dataset_version = lance_dataset_version(result->dataset);
  if (result->dataset_version == 0) {
    throw IOException("Failed to capture Lance dataset version: " +
                      result->file_path + LanceFormatErrorSuffix());
  }
  result->dataset_generation_id = result->dataset_entry->GenerationId();

  auto *schema_handle = lance_get_fts_schema(
      result->dataset, result->text_column.c_str(), result->query.c_str(),
      result->k, result->prefilter ? 1 : 0);
  if (!schema_handle) {
    throw IOException("Failed to get Lance FTS schema: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  memset(&result->schema_root.arrow_schema, 0,
         sizeof(result->schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &result->schema_root.arrow_schema) !=
      0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance FTS schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(schema_handle);
  LanceCoerceArrowSchemaForDuckDB(&result->schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(
      context, result->arrow_table, result->schema_root.arrow_schema);
  result->names = result->arrow_table.GetNames();
  result->types = result->arrow_table.GetTypes();
  names = result->names;
  return_types = result->types;
  return std::move(result);
}

static unique_ptr<FunctionData>
LanceHybridBind(ClientContext &context, TableFunctionBindInput &input,
                vector<LogicalType> &return_types, vector<string> &names) {
  if (input.inputs.size() < 5) {
    throw InvalidInputException("lance_hybrid_search requires (path, "
                                "vector_column, vector, text_column, text)");
  }
  if (input.inputs[0].IsNull()) {
    throw InvalidInputException(
        "lance_hybrid_search requires a dataset root path");
  }
  if (input.inputs[1].IsNull()) {
    throw InvalidInputException(
        "lance_hybrid_search requires a non-null vector_column");
  }
  if (input.inputs[2].IsNull()) {
    throw InvalidInputException(
        "lance_hybrid_search requires a non-null query vector");
  }
  if (input.inputs[3].IsNull()) {
    throw InvalidInputException(
        "lance_hybrid_search requires a non-null text_column");
  }
  if (input.inputs[4].IsNull()) {
    throw InvalidInputException(
        "lance_hybrid_search requires a non-null query");
  }
  auto *table = TryResolveNamespaceBackedSearchTable(context, input.inputs[0]);
  if (table && table->NamespaceConfig().IsRest()) {
    throw NotImplementedException(
        "Lance hybrid search is not supported for REST namespace-backed "
        "tables; use vector or FTS search until "
        "namespace hybrid-query support is available");
  }

  auto result = make_uniq<LanceSearchBindData>();
  result->mode = LanceSearchMode::Hybrid;
  result->file_path.clear();
  result->dataset_entry =
      OpenSearchDatasetEntry(context, input.inputs[0], "lance_hybrid_search",
                             result->file_path, &result->dataset_cache_hit);
  result->dataset =
      result->dataset_entry ? result->dataset_entry->Handle() : nullptr;
  result->vector_column = input.inputs[1].GetValue<string>();
  result->vector_query =
      ParseQueryVector(input.inputs[2], "lance_hybrid_search");
  result->text_column = input.inputs[3].GetValue<string>();
  result->text_query = input.inputs[4].GetValue<string>();
  ValidateLanceCString(result->vector_column, "Lance vector column");
  ValidateLanceCString(result->text_column, "Lance text column");
  ValidateLanceCString(result->text_query, "Lance text query");

  int64_t k_val = 10;
  auto k_named = input.named_parameters.find("k");
  if (k_named != input.named_parameters.end() && !k_named->second.IsNull()) {
    k_val =
        k_named->second.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
  }
  if (k_val <= 0) {
    throw InvalidInputException("lance_hybrid_search requires k > 0");
  }
  result->k = NumericCast<uint64_t>(k_val);

  bool has_nprobes = false;
  int64_t nprobes_val = 0;
  auto nprobes_named = input.named_parameters.find("nprobs");
  if (nprobes_named != input.named_parameters.end() &&
      !nprobes_named->second.IsNull()) {
    has_nprobes = true;
    nprobes_val = nprobes_named->second.DefaultCastAs(LogicalType::BIGINT)
                      .GetValue<int64_t>();
  }
  if (has_nprobes && nprobes_val <= 0) {
    throw InvalidInputException("lance_hybrid_search requires nprobs > 0");
  }
  result->nprobes = has_nprobes ? NumericCast<uint64_t>(nprobes_val) : 0;

  bool has_refine_factor = false;
  int64_t refine_factor_val = 0;
  auto refine_factor_named = input.named_parameters.find("refine_factor");
  if (refine_factor_named != input.named_parameters.end() &&
      !refine_factor_named->second.IsNull()) {
    has_refine_factor = true;
    refine_factor_val =
        refine_factor_named->second.DefaultCastAs(LogicalType::BIGINT)
            .GetValue<int64_t>();
  }
  if (has_refine_factor && refine_factor_val <= 0) {
    throw InvalidInputException(
        "lance_hybrid_search requires refine_factor > 0");
  }
  result->refine_factor =
      has_refine_factor ? NumericCast<uint64_t>(refine_factor_val) : 0;

  auto prefilter_named = input.named_parameters.find("prefilter");
  if (prefilter_named != input.named_parameters.end() &&
      !prefilter_named->second.IsNull()) {
    result->prefilter =
        prefilter_named->second.DefaultCastAs(LogicalType::BOOLEAN)
            .GetValue<bool>();
  }
  auto use_index_named = input.named_parameters.find("use_index");
  if (use_index_named != input.named_parameters.end() &&
      !use_index_named->second.IsNull()) {
    result->use_index =
        use_index_named->second.DefaultCastAs(LogicalType::BOOLEAN)
            .GetValue<bool>();
  }

  auto alpha_named = input.named_parameters.find("alpha");
  if (alpha_named != input.named_parameters.end() &&
      !alpha_named->second.IsNull()) {
    result->alpha =
        alpha_named->second.DefaultCastAs(LogicalType::FLOAT).GetValue<float>();
  }
  if (!std::isfinite(result->alpha) || result->alpha < 0.0F ||
      result->alpha > 1.0F) {
    throw InvalidInputException(
        "lance_hybrid_search requires alpha between 0 and 1");
  }
  auto oversample_named = input.named_parameters.find("oversample_factor");
  if (oversample_named != input.named_parameters.end() &&
      !oversample_named->second.IsNull()) {
    auto v = oversample_named->second.DefaultCastAs(LogicalType::INTEGER)
                 .GetValue<int32_t>();
    if (v <= 0) {
      throw InvalidInputException(
          "lance_hybrid_search requires oversample_factor > 0");
    }
    result->oversample_factor = NumericCast<uint32_t>(v);
  }

  if (!result->dataset) {
    throw IOException("Failed to open Lance dataset: " + result->file_path +
                      LanceFormatErrorSuffix());
  }
  result->non_replayable_storage_options =
      !SearchStorageOptionsAreWorkerReplayable(context, input.inputs[0],
                                               result->file_path);
  RejectSearchComputedColumnCollisions(result->dataset,
                                       LanceComputedSearchColumns::Hybrid,
                                       "lance_hybrid_search");
  result->dataset_version = lance_dataset_version(result->dataset);
  if (result->dataset_version == 0) {
    throw IOException("Failed to capture Lance dataset version: " +
                      result->file_path + LanceFormatErrorSuffix());
  }
  result->dataset_generation_id = result->dataset_entry->GenerationId();

  auto *schema_handle = lance_get_hybrid_schema(result->dataset);
  if (!schema_handle) {
    throw IOException("Failed to get Lance hybrid schema: " +
                      result->file_path + LanceFormatErrorSuffix());
  }

  memset(&result->schema_root.arrow_schema, 0,
         sizeof(result->schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &result->schema_root.arrow_schema) !=
      0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance hybrid schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(schema_handle);
  LanceCoerceArrowSchemaForDuckDB(&result->schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(
      context, result->arrow_table, result->schema_root.arrow_schema);
  result->names = result->arrow_table.GetNames();
  result->types = result->arrow_table.GetTypes();
  names = result->names;
  return_types = result->types;
  return std::move(result);
}

static unique_ptr<GlobalTableFunctionState>
LanceSearchInitGlobal(ClientContext &, TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceSearchBindData>();
  auto state =
      make_uniq_base<GlobalTableFunctionState, LanceSearchGlobalState>();
  auto &global = state->Cast<LanceSearchGlobalState>();

  global.projection_ids = input.projection_ids;
  if (!input.projection_ids.empty()) {
    global.scanned_types.reserve(input.column_ids.size());
    for (auto col_id : input.column_ids) {
      if (col_id >= bind_data.types.size()) {
        throw IOException("Invalid column id in projection");
      }
      global.scanned_types.push_back(bind_data.types[col_id]);
    }
  }

  if (bind_data.namespace_backed) {
    if (bind_data.names.empty()) {
      throw InternalException("Lance namespace search has no output columns");
    }
    global.namespace_columns.assign(bind_data.names.begin(),
                                    bind_data.names.end() - 1);
    return state;
  }

  if (bind_data.dataset) {
    auto operation_id = "text-search:" + bind_data.file_path + ":" +
                        to_string(bind_data.dataset_version) + ":" +
                        bind_data.dataset_generation_id;
    global.snapshot_lease = LanceLease::AcquireForDataset(
        bind_data.dataset, LanceLeaseKind::Snapshot, operation_id);
  }

  auto table_filters =
      BuildLanceTableFilterIRParts(bind_data.names, bind_data.types, input,
                                   LanceSearchComputedColumns(bind_data.mode));
  if (bind_data.prefilter && bind_data.complex_filter_pushdown_failed) {
    auto function_name = bind_data.mode == LanceSearchMode::Fts
                             ? "lance_fts"
                             : "lance_hybrid_search";
    throw InvalidInputException(string(function_name) +
                                " cannot apply every complex predicate before "
                                "top-k when prefilter=true");
  }
  if (bind_data.prefilter && !table_filters.all_prefilterable_filters_pushed) {
    auto function_name = bind_data.mode == LanceSearchMode::Fts
                             ? "lance_fts"
                             : "lance_hybrid_search";
    throw InvalidInputException(string(function_name) +
                                " requires filter pushdown for prefilterable "
                                "columns when prefilter=true");
  }

  bool has_table_filter_parts = !table_filters.parts.empty();
  auto filter_parts = std::move(table_filters.parts);
  if (!bind_data.lance_pushed_filter_ir_parts.empty()) {
    filter_parts.reserve(filter_parts.size() +
                         bind_data.lance_pushed_filter_ir_parts.size());
    for (auto &part : bind_data.lance_pushed_filter_ir_parts) {
      filter_parts.push_back(part);
    }
  }
  string filter_ir_msg;
  if (!filter_parts.empty()) {
    if (!TryEncodeLanceFilterIRMessage(filter_parts, filter_ir_msg)) {
      filter_ir_msg.clear();
    }
    global.lance_filter_ir = std::move(filter_ir_msg);
  }
  if (bind_data.prefilter && has_table_filter_parts &&
      global.lance_filter_ir.empty()) {
    throw IOException("Failed to encode Lance filter IR");
  }
  global.filter_pushed_down =
      table_filters.all_filters_pushed && !global.lance_filter_ir.empty();
  return state;
}

static unique_ptr<LocalTableFunctionState>
LanceSearchLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                     GlobalTableFunctionState *global_state) {
  auto &global = global_state->Cast<LanceSearchGlobalState>();

  auto chunk = make_uniq<ArrowArrayWrapper>();
  auto result =
      make_uniq<LanceSearchLocalState>(std::move(chunk), context.client);
  result->column_ids = input.column_ids;
  result->filters = input.filters.get();
  result->global_state = &global;
  result->filter_pushed_down = global.filter_pushed_down;
  if (global.CanRemoveFilterColumns()) {
    result->all_columns.Initialize(context.client, global.scanned_types);
  }
  return std::move(result);
}

static void LanceSearchFunc(ClientContext &context, TableFunctionInput &data,
                            DataChunk &output) {
  if (!data.local_state) {
    return;
  }

  auto &bind_data = data.bind_data->Cast<LanceSearchBindData>();
  if (bind_data.assigned_empty_scan) {
    return;
  }
  auto &global_state = data.global_state->Cast<LanceSearchGlobalState>();
  auto &local_state = data.local_state->Cast<LanceSearchLocalState>();

  while (true) {
    if (local_state.chunk_offset >=
        NumericCast<idx_t>(local_state.chunk->arrow_array.length)) {
      if (!LanceSearchLoadNextBatch(context, local_state, bind_data,
                                    global_state)) {
        return;
      }
    }

    auto remaining = NumericCast<idx_t>(local_state.chunk->arrow_array.length) -
                     local_state.chunk_offset;
    auto output_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, remaining);
    global_state.lines_read.fetch_add(output_size);

    if (global_state.CanRemoveFilterColumns()) {
      local_state.all_columns.Reset();
      local_state.all_columns.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(local_state,
                                        bind_data.arrow_table.GetColumns(),
                                        local_state.all_columns, false);
      local_state.chunk_offset += output_size;
      if (local_state.filters && !local_state.filter_pushed_down) {
        ApplyDuckDBFilters(context, *local_state.filters,
                           local_state.all_columns, local_state.filter_sel);
      }
      output.ReferenceColumns(local_state.all_columns,
                              global_state.projection_ids);
      output.SetCardinality(local_state.all_columns);
    } else {
      output.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(
          local_state, bind_data.arrow_table.GetColumns(), output, false);
      local_state.chunk_offset += output_size;
      if (local_state.filters && !local_state.filter_pushed_down) {
        ApplyDuckDBFilters(context, *local_state.filters, output,
                           local_state.filter_sel);
      }
    }

    if (output.size() == 0) {
      continue;
    }
    output.Verify();
    return;
  }
}

static InsertionOrderPreservingMap<string>
LanceSearchBindToString(const LanceSearchBindData &bind_data) {
  InsertionOrderPreservingMap<string> result;
  result["Lance Path"] = bind_data.file_path;
  result["Lance Search Backend"] =
      bind_data.namespace_backed ? "namespace_query_table" : "dataset_scan";
  result["Lance Search Mode"] =
      bind_data.mode == LanceSearchMode::Fts ? "fts" : "hybrid";
  result["Lance K"] = to_string(bind_data.k);
  result["Lance Prefilter"] = bind_data.prefilter ? "true" : "false";
  result["Lance Dataset Cache Hit"] =
      bind_data.dataset_cache_hit ? "true" : "false";
  if (!bind_data.namespace_filter.empty()) {
    result["Lance Namespace Filter"] = bind_data.namespace_filter;
  }

  if (bind_data.mode == LanceSearchMode::Fts) {
    result["Lance Text Column"] = bind_data.text_column;
    result["Lance Query"] = bind_data.query;
  } else {
    result["Lance Vector Column"] = bind_data.vector_column;
    result["Lance Text Column"] = bind_data.text_column;
    result["Lance Vector Query Dim"] = to_string(bind_data.vector_query.size());
    result["Lance Text Query"] = bind_data.text_query;
    result["Lance Nprobes"] = to_string(bind_data.nprobes);
    result["Lance Refine Factor"] = to_string(bind_data.refine_factor);
    result["Lance Use Index"] = bind_data.use_index ? "true" : "false";
    result["Lance Alpha"] = to_string(bind_data.alpha);
    result["Lance Oversample Factor"] = to_string(bind_data.oversample_factor);
  }

  return result;
}

static InsertionOrderPreservingMap<string>
LanceSearchToString(TableFunctionToStringInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceSearchBindData>();
  return LanceSearchBindToString(bind_data);
}

static InsertionOrderPreservingMap<string>
LanceSearchDynamicToString(TableFunctionDynamicToStringInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceSearchBindData>();
  auto result = LanceSearchBindToString(bind_data);
  auto &global_state = input.global_state->Cast<LanceSearchGlobalState>();

  result["Lance Filter Pushed Down"] =
      global_state.filter_pushed_down ? "true" : "false";
  result["Lance Filter Pushdown Fallbacks"] =
      to_string(global_state.filter_pushdown_fallbacks.load());
  result["Lance Filter IR Bytes"] =
      to_string(global_state.lance_filter_ir.size());
  result["Lance Record Batches"] =
      to_string(global_state.record_batches.load());
  result["Lance Record Batch Rows"] =
      to_string(global_state.record_batch_rows.load());
  result["Lance Rows Out"] = to_string(global_state.lines_read.load());

  return result;
}

static void LanceSearchSerialize(Serializer &serializer,
                                 const optional_ptr<FunctionData> bind_data_p,
                                 const TableFunction &) {
  auto &bind_data = bind_data_p->Cast<LanceSearchBindData>();
  serializer.WriteProperty(100, "mode", static_cast<uint8_t>(bind_data.mode));
  serializer.WriteProperty(101, "file_path", bind_data.file_path);
  serializer.WriteProperty(102, "prefilter", bind_data.prefilter);
  serializer.WriteProperty(103, "namespace_backed", bind_data.namespace_backed);
  serializer.WriteProperty(104, "namespace_filter", bind_data.namespace_filter);
  serializer.WriteProperty(105, "text_column", bind_data.text_column);
  serializer.WriteProperty(106, "query", bind_data.query);
  serializer.WriteProperty(107, "vector_column", bind_data.vector_column);
  serializer.WriteProperty(108, "vector_query", bind_data.vector_query);
  serializer.WriteProperty(109, "text_query", bind_data.text_query);
  serializer.WriteProperty(110, "nprobes", bind_data.nprobes);
  serializer.WriteProperty(111, "refine_factor", bind_data.refine_factor);
  serializer.WriteProperty(112, "use_index", bind_data.use_index);
  serializer.WriteProperty(113, "alpha", bind_data.alpha);
  serializer.WriteProperty(114, "oversample_factor",
                           bind_data.oversample_factor);
  serializer.WriteProperty(115, "k", bind_data.k);
  serializer.WriteProperty(116, "dataset_version", bind_data.dataset_version);
  serializer.WriteProperty(117, "names", bind_data.names);
  serializer.WriteProperty(118, "types", bind_data.types);
  if (bind_data.namespace_backed) {
    auto &cfg = bind_data.namespace_config;
    serializer.WriteProperty(119, "namespace_kind",
                             static_cast<uint8_t>(cfg.kind));
    serializer.WriteProperty(120, "namespace_root", cfg.root);
    serializer.WriteProperty(121, "namespace_endpoint", cfg.endpoint);
    serializer.WriteProperty(122, "namespace_table_id", cfg.table_id);
    serializer.WriteProperty(123, "namespace_delimiter", cfg.delimiter);
    serializer.WriteProperty(124, "namespace_display_uri", cfg.display_uri);
  }
  serializer.WriteProperty(126, "dataset_generation_id",
                           bind_data.dataset_generation_id);
  serializer.WriteProperty(127, "lance_filter_ir_parts",
                           bind_data.lance_pushed_filter_ir_parts);
  serializer.WriteProperty(128, "complex_filter_pushdown_failed",
                           bind_data.complex_filter_pushdown_failed);
  serializer.WriteProperty(129, "namespace_requires_worker_auth",
                           bind_data.namespace_backed &&
                               bind_data.namespace_config.requires_worker_auth);
  serializer.WriteProperty(130, "non_replayable_storage_options",
                           bind_data.non_replayable_storage_options);
}

static unique_ptr<FunctionData>
LanceSearchDeserialize(Deserializer &deserializer, TableFunction &) {
  auto result = make_uniq<LanceSearchBindData>();
  auto &context = deserializer.Get<ClientContext &>();
  auto mode = deserializer.ReadProperty<uint8_t>(100, "mode");
  if (mode > static_cast<uint8_t>(LanceSearchMode::Hybrid)) {
    throw SerializationException(
        "Serialized Lance search has an invalid search mode");
  }
  result->mode = static_cast<LanceSearchMode>(mode);
  result->file_path = deserializer.ReadProperty<string>(101, "file_path");
  result->prefilter = deserializer.ReadProperty<bool>(102, "prefilter");
  result->namespace_backed =
      deserializer.ReadProperty<bool>(103, "namespace_backed");
  result->namespace_filter =
      deserializer.ReadProperty<string>(104, "namespace_filter");
  result->text_column = deserializer.ReadProperty<string>(105, "text_column");
  result->query = deserializer.ReadProperty<string>(106, "query");
  result->vector_column =
      deserializer.ReadProperty<string>(107, "vector_column");
  result->vector_query =
      deserializer.ReadProperty<vector<float>>(108, "vector_query");
  result->text_query = deserializer.ReadProperty<string>(109, "text_query");
  result->nprobes = deserializer.ReadProperty<uint64_t>(110, "nprobes");
  result->refine_factor =
      deserializer.ReadProperty<uint64_t>(111, "refine_factor");
  result->use_index = deserializer.ReadProperty<bool>(112, "use_index");
  result->alpha = deserializer.ReadProperty<float>(113, "alpha");
  result->oversample_factor =
      deserializer.ReadProperty<uint32_t>(114, "oversample_factor");
  result->k = deserializer.ReadProperty<uint64_t>(115, "k");
  result->dataset_version =
      deserializer.ReadProperty<uint64_t>(116, "dataset_version");
  result->names = deserializer.ReadProperty<vector<string>>(117, "names");
  result->types = deserializer.ReadProperty<vector<LogicalType>>(118, "types");
  if (result->namespace_backed) {
    auto &cfg = result->namespace_config;
    cfg.kind = static_cast<LanceNamespaceKind>(
        deserializer.ReadProperty<uint8_t>(119, "namespace_kind"));
    cfg.root = deserializer.ReadProperty<string>(120, "namespace_root");
    cfg.endpoint = deserializer.ReadProperty<string>(121, "namespace_endpoint");
    cfg.table_id = deserializer.ReadProperty<string>(122, "namespace_table_id");
    cfg.delimiter =
        deserializer.ReadProperty<string>(123, "namespace_delimiter");
    cfg.display_uri =
        deserializer.ReadProperty<string>(124, "namespace_display_uri");
  }
  result->dataset_generation_id =
      deserializer.ReadProperty<string>(126, "dataset_generation_id");
  result->lance_pushed_filter_ir_parts =
      deserializer.ReadProperty<vector<string>>(127, "lance_filter_ir_parts");
  result->complex_filter_pushdown_failed =
      deserializer.ReadProperty<bool>(128, "complex_filter_pushdown_failed");
  result->namespace_config.requires_worker_auth =
      deserializer.ReadProperty<bool>(129, "namespace_requires_worker_auth");
  result->non_replayable_storage_options =
      deserializer.ReadProperty<bool>(130, "non_replayable_storage_options");
  if (result->k == 0) {
    throw SerializationException("Serialized Lance search requires k > 0");
  }
  ValidateSerializedSearchCString(result->file_path, "dataset path");
  ValidateSerializedSearchCString(result->namespace_filter, "filter");
  ValidateSerializedSearchCString(result->text_column, "text column");
  ValidateSerializedSearchCString(result->query, "text query");
  ValidateSerializedSearchCString(result->vector_column, "vector column");
  ValidateSerializedSearchCString(result->text_query, "hybrid text query");
  ValidateSerializedSearchCStrings(result->names, "output column");
  if (result->namespace_backed) {
    ValidateSerializedSearchCString(result->namespace_config.root,
                                    "namespace root");
    ValidateSerializedSearchCString(result->namespace_config.endpoint,
                                    "namespace endpoint");
    ValidateSerializedSearchCString(result->namespace_config.table_id,
                                    "namespace table id");
    ValidateSerializedSearchCString(result->namespace_config.delimiter,
                                    "namespace delimiter");
  }
  if (result->names.empty() || result->names.size() != result->types.size()) {
    throw SerializationException(
        "Serialized Lance search has an invalid output schema");
  }
  if (result->namespace_config.requires_worker_auth) {
    throw SerializationException(
        "Serialized Lance namespace search requires credentials that were not "
        "transported to the worker");
  }
  if (result->non_replayable_storage_options) {
    throw SerializationException(
        "Serialized Lance search requires storage options that were not "
        "transported to the worker");
  }
  if (result->mode == LanceSearchMode::Hybrid) {
    ValidateSerializedQueryVector(result->vector_query, "Lance hybrid search");
    if (!std::isfinite(result->alpha) || result->alpha < 0.0F ||
        result->alpha > 1.0F) {
      throw SerializationException(
          "Serialized Lance hybrid search requires alpha between 0 and 1");
    }
    if (result->oversample_factor == 0) {
      throw SerializationException(
          "Serialized Lance hybrid search requires oversample_factor > 0");
    }
  }
  if (result->namespace_backed) {
    if (!result->namespace_config.IsRest() ||
        result->mode != LanceSearchMode::Fts) {
      throw SerializationException(
          "Serialized Lance namespace search has an invalid backend or mode");
    }
    if (result->dataset_version == 0) {
      throw SerializationException("Serialized Lance namespace search is "
                                   "missing its fixed dataset version");
    }
  } else {
    if (result->dataset_version == 0) {
      throw SerializationException(
          "Serialized Lance search is missing its fixed dataset version");
    }
    if (result->dataset_generation_id.empty()) {
      throw SerializationException(
          "Serialized Lance search is missing its fixed dataset generation");
    }
    result->dataset_entry = LanceGetOrOpenDatasetEntryAtVersion(
        context, result->file_path, result->dataset_version,
        result->dataset_generation_id, &result->dataset_cache_hit);
    result->dataset =
        result->dataset_entry ? result->dataset_entry->Handle() : nullptr;
    if (!result->dataset ||
        lance_dataset_version(result->dataset) != result->dataset_version) {
      throw IOException("Failed to reopen fixed Lance search snapshot: " +
                        result->file_path + LanceFormatErrorSuffix());
    }
  }
  PopulateSearchSchemaFromTypes(context, result->schema_root,
                                result->arrow_table, result->names,
                                result->types);
  return std::move(result);
}

static void RegisterLanceFtsSearch(ExtensionLoader &loader) {
  TableFunction fts(
      "lance_fts",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceSearchFunc, LanceFtsBind, LanceSearchInitGlobal,
      LanceSearchLocalInit);
  fts.named_parameters["k"] = LogicalType::BIGINT;
  fts.named_parameters["prefilter"] = LogicalType::BOOLEAN;
  fts.named_parameters["filter"] = LogicalType::VARCHAR;
  fts.projection_pushdown = true;
  fts.filter_pushdown = true;
  fts.filter_prune = true;
  fts.pushdown_expression = LancePushdownExpression;
  fts.pushdown_complex_filter = LanceSearchPushdownComplexFilter;
  fts.to_string = LanceSearchToString;
  fts.dynamic_to_string = LanceSearchDynamicToString;
  fts.serialize = LanceSearchSerialize;
  fts.deserialize = LanceSearchDeserialize;
#ifdef LANCE_VANE_DISTRIBUTED
  fts.SetDistributedScanCallbacks(LanceSearchDistributedScanCallbacks());
#endif
  loader.RegisterFunction(fts);
}

static void RegisterLanceHybridSearch(ExtensionLoader &loader) {
  auto configure = [](TableFunction &fun) {
    fun.named_parameters["k"] = LogicalType::BIGINT;
    fun.named_parameters["nprobs"] = LogicalType::BIGINT;
    fun.named_parameters["refine_factor"] = LogicalType::BIGINT;
    fun.named_parameters["prefilter"] = LogicalType::BOOLEAN;
    fun.named_parameters["use_index"] = LogicalType::BOOLEAN;
    fun.named_parameters["alpha"] = LogicalType::FLOAT;
    fun.named_parameters["oversample_factor"] = LogicalType::INTEGER;
    fun.projection_pushdown = true;
    fun.filter_pushdown = true;
    fun.filter_prune = true;
    fun.pushdown_expression = LancePushdownExpression;
    fun.pushdown_complex_filter = LanceSearchPushdownComplexFilter;
    fun.to_string = LanceSearchToString;
    fun.dynamic_to_string = LanceSearchDynamicToString;
    fun.serialize = LanceSearchSerialize;
    fun.deserialize = LanceSearchDeserialize;
#ifdef LANCE_VANE_DISTRIBUTED
    fun.SetDistributedScanCallbacks(LanceSearchDistributedScanCallbacks());
#endif
  };

  TableFunction hybrid_f32("lance_hybrid_search",
                           {LogicalType::VARCHAR, LogicalType::VARCHAR,
                            LogicalType::LIST(LogicalType::FLOAT),
                            LogicalType::VARCHAR, LogicalType::VARCHAR},
                           LanceSearchFunc, LanceHybridBind,
                           LanceSearchInitGlobal, LanceSearchLocalInit);
  configure(hybrid_f32);
  loader.RegisterFunction(hybrid_f32);

  TableFunction hybrid_f64("lance_hybrid_search",
                           {LogicalType::VARCHAR, LogicalType::VARCHAR,
                            LogicalType::LIST(LogicalType::DOUBLE),
                            LogicalType::VARCHAR, LogicalType::VARCHAR},
                           LanceSearchFunc, LanceHybridBind,
                           LanceSearchInitGlobal, LanceSearchLocalInit);
  configure(hybrid_f64);
  loader.RegisterFunction(hybrid_f64);
}

void RegisterLanceSearch(ExtensionLoader &loader) {
  RegisterLanceVectorSearch(loader);
  RegisterLanceFtsSearch(loader);
  RegisterLanceHybridSearch(loader);
}

} // namespace duckdb
