#include "duckdb/storage/storage_extension.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/copy_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_schema_entry.hpp"
#include "duckdb/catalog/catalog_entry/view_catalog_entry.hpp"
#include "duckdb/catalog/catalog_transaction.hpp"
#include "duckdb/catalog/default/default_generator.hpp"
#include "duckdb/catalog/default/default_schemas.hpp"
#include "duckdb/catalog/duck_catalog.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/arrow/schema_metadata.hpp"
#include "duckdb/common/exception_format_value.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/operator/persistent/physical_batch_copy_to_file.hpp"
#include "duckdb/execution/operator/persistent/physical_copy_to_file.hpp"
#include "duckdb/execution/operator/scan/physical_empty_result.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/parsed_data/alter_table_info.hpp"
#include "duckdb/parser/parsed_data/attach_info.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"
#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/parser/parsed_data/drop_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_merge_into.hpp"
#include "duckdb/planner/operator/logical_update.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"
#include "duckdb/transaction/transaction.hpp"

#include "lance_arrow_compat.hpp"
#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_delete.hpp"
#include "lance_ffi.hpp"
#include "lance_insert.hpp"
#include "lance_lease.hpp"
#include "lance_merge.hpp"
#include "lance_session_state.hpp"
#include "lance_table_entry.hpp"
#include "lance_update.hpp"
#include "lance_write.hpp"

#include <cctype>
#include <cstring>
#include <exception>

#include <algorithm>

namespace duckdb {

struct LanceDirectoryNamespaceConfig {
  string root;
  vector<string> option_keys;
  vector<string> option_values;
};

struct LanceRestNamespaceConfig {
  string endpoint;
  string namespace_id;
  string delimiter;
  string bearer_token_override;
  string api_key_override;
  string headers_tsv; // Tab-separated key\tvalue pairs for custom headers
};

static void ReleaseLanceStorageLease(unique_ptr<LanceLease> &lease,
                                     const string &operation) {
  if (!lease) {
    return;
  }
  try {
    lease->Release();
    lease.reset();
  } catch (...) {
    lease->RetainForReconciliation();
    throw IOException(operation +
                      " committed, but mutation lease release failed; do "
                      "not retry automatically (code=55)");
  }
}

static string GetLanceNamespaceEndpoint(const AttachInfo &info) {
  for (auto &kv : info.options) {
    if (!StringUtil::CIEquals(kv.first, "endpoint") || kv.second.IsNull()) {
      continue;
    }
    auto endpoint =
        kv.second.DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
    if (endpoint.empty()) {
      throw InvalidInputException(
          "Invalid Lance ENDPOINT option: endpoint must not be empty");
    }
    ValidateLanceCString(endpoint, "Lance ENDPOINT option");
    return endpoint;
  }
  return "";
}

static string GetLanceNamespaceDelimiter(const AttachInfo &info) {
  for (auto &kv : info.options) {
    if (!StringUtil::CIEquals(kv.first, "delimiter") || kv.second.IsNull()) {
      continue;
    }
    auto delimiter =
        kv.second.DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
    if (delimiter.empty()) {
      throw InvalidInputException(
          "Invalid Lance DELIMITER option: delimiter must not be empty");
    }
    ValidateLanceCString(delimiter, "Lance DELIMITER option");
    return delimiter;
  }
  return "";
}

// Parse HEADER options from ATTACH command
// Options like HEADER 'x-lancedb-database=lance_ns;x-api-key=sk_123' are parsed
// Multiple headers can be separated by semicolons within a single HEADER option
// Returns a TSV string with key\tvalue pairs separated by newlines
static string GetLanceNamespaceHeaders(const AttachInfo &info) {
  string headers_tsv;
  for (auto &kv : info.options) {
    // Handle 'HEADER' option with 'key=value' format
    if (StringUtil::CIEquals(kv.first, "header") && !kv.second.IsNull()) {
      auto header_str =
          kv.second.DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
      if (header_str.empty()) {
        throw InvalidInputException(
            "Invalid Lance HEADER option: expected at least one non-empty "
            "header name followed by '='");
      }
      ValidateLanceCString(header_str, "Lance HEADER option");
      // Split by semicolon to support multiple headers
      vector<string> header_parts;
      idx_t pos = 0;
      auto header_str_size = NumericCast<idx_t>(header_str.size());
      while (pos < header_str_size) {
        auto next_semi =
            header_str.find(';', NumericCast<string::size_type>(pos));
        if (next_semi == string::npos) {
          header_parts.push_back(
              header_str.substr(NumericCast<string::size_type>(pos)));
          break;
        }
        auto next_semi_idx = NumericCast<idx_t>(next_semi);
        header_parts.push_back(header_str.substr(
            NumericCast<string::size_type>(pos),
            NumericCast<string::size_type>(next_semi_idx - pos)));
        pos = next_semi_idx + 1;
      }
      for (auto &part : header_parts) {
        // Trim whitespace
        while (!part.empty() &&
               std::isspace(static_cast<unsigned char>(part.front()))) {
          part.erase(part.begin());
        }
        while (!part.empty() &&
               std::isspace(static_cast<unsigned char>(part.back()))) {
          part.pop_back();
        }
        auto eq_pos = part.find('=');
        if (eq_pos == string::npos || eq_pos == 0) {
          throw InvalidInputException(
              "Invalid Lance HEADER option '%s': expected a non-empty "
              "header name followed by '='",
              part);
        }
        auto key = part.substr(0, eq_pos);
        auto value = part.substr(eq_pos + 1);
        if (!headers_tsv.empty()) {
          headers_tsv += "\n";
        }
        headers_tsv += key + "\t" + value;
      }
    }
  }
  return headers_tsv;
}

static void PopulateColumnsFromArrowSchema(ClientContext &context,
                                           ArrowSchema &arrow_schema,
                                           ColumnList &out_columns) {
  ArrowTableSchema arrow_table;
  ArrowTableFunction::PopulateArrowTableSchema(context, arrow_table,
                                               arrow_schema);
  const auto names = arrow_table.GetNames();
  const auto types = arrow_table.GetTypes();
  if (names.size() != types.size()) {
    throw InternalException(
        "Arrow table schema returned mismatched names/types sizes");
  }
  for (idx_t i = 0; i < names.size(); i++) {
    ColumnDefinition column(names[i], types[i]);
    if (arrow_schema.children && arrow_schema.children[i]) {
      ArrowSchemaMetadata metadata(arrow_schema.children[i]->metadata);
      auto default_expression = metadata.GetOption("duckdb_default_expr");
      if (!default_expression.empty()) {
        auto expressions = Parser::ParseExpressionList(
            default_expression, context.GetParserOptions());
        if (expressions.size() != 1 || !expressions[0]) {
          throw IOException("Lance field '%s' has invalid persisted DuckDB "
                            "default metadata",
                            names[i]);
        }
        column.SetDefaultValue(std::move(expressions[0]));
      }
    }
    out_columns.AddColumn(std::move(column));
  }
}

static void PopulateLanceTableColumnsFromDataset(
    ClientContext &context, void *dataset, ColumnList &out_columns,
    vector<string> *out_coerced_columns = nullptr) {
  auto *schema_handle = lance_get_schema(dataset);
  if (!schema_handle) {
    throw IOException("Failed to get schema from Lance dataset" +
                      LanceFormatErrorSuffix());
  }

  ArrowSchemaWrapper schema_root;
  memset(&schema_root.arrow_schema, 0, sizeof(schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &schema_root.arrow_schema) != 0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(schema_handle);
  auto coerced = LanceCoerceArrowSchemaForDuckDB(&schema_root.arrow_schema);
  if (out_coerced_columns) {
    *out_coerced_columns = std::move(coerced);
  }
  PopulateColumnsFromArrowSchema(context, schema_root.arrow_schema,
                                 out_columns);
}

static string JoinNamespacePath(const string &root, const string &child) {
  if (root.empty()) {
    return child;
  }
  if (root.back() == '/' || root.back() == '\\') {
    return root + child;
  }
  return root + "/" + child;
}

static string GetDatasetDirName(const string &table_name);
static bool IsSafeDatasetTableName(const string &name);

static vector<string>
ListDirectoryNamespaceTables(const LanceDirectoryNamespaceConfig &ns) {
  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  BuildStorageOptionPointerArrays(ns.option_keys, ns.option_values, key_ptrs,
                                  value_ptrs);

  auto *ptr = lance_dir_namespace_list_tables(
      ns.root.c_str(), key_ptrs.empty() ? nullptr : key_ptrs.data(),
      value_ptrs.empty() ? nullptr : value_ptrs.data(), ns.option_keys.size());
  if (!ptr) {
    throw IOException("Failed to list tables from Lance directory namespace: " +
                      ns.root + LanceFormatErrorSuffix());
  }
  string joined = ptr;
  lance_free_string(ptr);

  vector<string> out;
  for (auto &p : StringUtil::Split(joined, '\n')) {
    if (!p.empty()) {
      out.push_back(std::move(p));
    }
  }
  return out;
}

static vector<string>
ListRestNamespaceTables(const string &endpoint, const string &namespace_id,
                        const string &bearer_token, const string &api_key,
                        const string &delimiter, const string &headers_tsv) {
  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();
  const char *headers_ptr = headers_tsv.empty() ? nullptr : headers_tsv.c_str();

  auto *ptr = lance_namespace_list_tables(
      endpoint.c_str(), namespace_id.c_str(), bearer_ptr, api_key_ptr,
      delimiter_ptr, headers_ptr);
  if (!ptr) {
    throw IOException("Failed to list tables from Lance namespace: " +
                      endpoint + "/" + namespace_id + LanceFormatErrorSuffix());
  }
  string joined = ptr;
  lance_free_string(ptr);

  vector<string> out;
  for (auto &p : StringUtil::Split(joined, '\n')) {
    if (!p.empty()) {
      out.push_back(std::move(p));
    }
  }
  return out;
}

static bool FindUniqueCaseInsensitiveName(const vector<string> &names,
                                          const string &requested,
                                          const string &source,
                                          string &resolved) {
  bool found = false;
  for (auto &name : names) {
    if (!StringUtil::CIEquals(name, requested)) {
      continue;
    }
    if (found && name != resolved) {
      throw IOException("Ambiguous case-insensitive Lance table name '" +
                        requested + "' in " + source + ": '" + resolved +
                        "' and '" + name + "'");
    }
    resolved = name;
    found = true;
  }
  return found;
}

static void ValidateUniqueCaseInsensitiveNames(const vector<string> &names,
                                               const string &source) {
  for (idx_t i = 0; i < names.size(); i++) {
    for (idx_t j = i + 1; j < names.size(); j++) {
      if (StringUtil::CIEquals(names[i], names[j])) {
        throw IOException("Ambiguous case-insensitive Lance table names in " +
                          source + ": '" + names[i] + "' and '" + names[j] +
                          "'");
      }
    }
  }
}

static bool
ResolveDirectoryNamespaceTableName(const LanceDirectoryNamespaceConfig &ns,
                                   const string &table_name, string &resolved) {
  auto tables = ListDirectoryNamespaceTables(ns);
  return FindUniqueCaseInsensitiveName(
      tables, table_name, "directory namespace '" + ns.root + "'", resolved);
}

static string RestNamespacePrefix(const string &namespace_id,
                                  const string &delimiter) {
  if (namespace_id.empty()) {
    return "";
  }
  return namespace_id + (delimiter.empty() ? "$" : delimiter);
}

static string RestNamespaceLeafName(const string &namespace_id,
                                    const string &delimiter,
                                    const string &table_id) {
  auto prefix = RestNamespacePrefix(namespace_id, delimiter);
  if (!prefix.empty() && StringUtil::CIStartsWith(table_id, prefix)) {
    return table_id.substr(prefix.size());
  }
  return table_id;
}

static string BuildRestNamespaceTableId(const string &namespace_id,
                                        const string &delimiter,
                                        const string &table_name) {
  auto prefix = RestNamespacePrefix(namespace_id, delimiter);
  if (prefix.empty()) {
    return table_name;
  }
  return prefix + RestNamespaceLeafName(namespace_id, delimiter, table_name);
}

static bool ResolveRestNamespaceTableId(const vector<string> &discovered,
                                        const string &namespace_id,
                                        const string &delimiter,
                                        const string &requested,
                                        string &resolved) {
  auto prefix = RestNamespacePrefix(namespace_id, delimiter);
  auto leaf = RestNamespaceLeafName(namespace_id, delimiter, requested);
  auto qualified =
      BuildRestNamespaceTableId(namespace_id, delimiter, requested);
  bool found = false;
  for (auto &candidate : discovered) {
    string candidate_id;
    if (StringUtil::CIEquals(candidate, qualified)) {
      candidate_id = candidate;
    } else if (StringUtil::CIEquals(candidate, leaf)) {
      // list_tables(namespace_id) is allowed to return a relative leaf name,
      // while describe/declare/drop consume the fully segmented object id.
      candidate_id = prefix + candidate;
    } else {
      continue;
    }
    if (found && candidate_id != resolved) {
      throw IOException("Ambiguous case-insensitive Lance table id '" +
                        requested + "': '" + resolved + "' and '" +
                        candidate_id + "'");
    }
    resolved = std::move(candidate_id);
    found = true;
  }
  return found;
}

class LanceDirectoryDefaultGenerator : public DefaultGenerator {
public:
  LanceDirectoryDefaultGenerator(Catalog &catalog, SchemaCatalogEntry &schema,
                                 shared_ptr<LanceDirectoryNamespaceConfig> ns)
      : DefaultGenerator(catalog), schema(schema), ns(std::move(ns)) {}

  unique_ptr<CatalogEntry>
  CreateDefaultEntry(ClientContext &context,
                     const string &entry_name) override {
    if (!ns) {
      throw InternalException("Lance directory namespace config is missing");
    }
    if (!IsSafeDatasetTableName(entry_name)) {
      throw InvalidInputException(
          "Unsafe Lance dataset name for directory namespace: " + entry_name);
    }

    string resolved_name;
    if (!ResolveDirectoryNamespaceTableName(*ns, entry_name, resolved_name)) {
      return nullptr;
    }

    vector<const char *> key_ptrs;
    vector<const char *> value_ptrs;
    BuildStorageOptionPointerArrays(ns->option_keys, ns->option_values,
                                    key_ptrs, value_ptrs);

    const char *uri_ptr = nullptr;
    auto *dataset = lance_open_dataset_in_dir_namespace(
        ns->root.c_str(), resolved_name.c_str(),
        key_ptrs.empty() ? nullptr : key_ptrs.data(),
        value_ptrs.empty() ? nullptr : value_ptrs.data(),
        ns->option_keys.size(), &uri_ptr);
    string dataset_uri;
    if (uri_ptr) {
      dataset_uri = uri_ptr;
      lance_free_string(uri_ptr);
    }
    if (!dataset) {
      return nullptr;
    }

    CreateTableInfo info(schema, resolved_name);
    info.internal = true;
    info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
    vector<string> coerced;
    try {
      PopulateLanceTableColumnsFromDataset(context, dataset, info.columns,
                                           &coerced);
      auto identity_uri =
          dataset_uri.empty()
              ? JoinNamespacePath(ns->root, GetDatasetDirName(resolved_name))
              : dataset_uri;
#ifdef LANCE_DUCKDB_HAS_LOGICAL_WRITE_TARGET
      info.logical_write_target_identity =
          LanceBuildLogicalWriteTargetIdentity(identity_uri, dataset);
#endif
    } catch (...) {
      lance_close_dataset(dataset);
      return nullptr;
    }
    lance_close_dataset(dataset);

    if (dataset_uri.empty()) {
      dataset_uri =
          JoinNamespacePath(ns->root, GetDatasetDirName(resolved_name));
    }
    LanceNamespaceTableConfig cfg;
    cfg.kind = LanceNamespaceKind::Directory;
    cfg.root = ns->root;
    cfg.table_id = resolved_name;
    cfg.option_keys = ns->option_keys;
    cfg.option_values = ns->option_values;
    cfg.display_uri = std::move(dataset_uri);
    auto entry =
        make_uniq<LanceTableEntry>(catalog, schema, info, std::move(cfg));
    entry->SetCoercedColumnNames(std::move(coerced));
    return unique_ptr_cast<LanceTableEntry, CatalogEntry>(std::move(entry));
  }

  vector<string> GetDefaultEntries() override {
    if (!ns) {
      return {};
    }
    auto all = ListDirectoryNamespaceTables(*ns);
    ValidateUniqueCaseInsensitiveNames(all, "directory namespace '" + ns->root +
                                                "'");
    // Filter out tables whose datasets cannot be opened (e.g. corrupt
    // manifests). CreateDefaultEntries requires every entry to produce
    // a non-null CatalogEntry, but CreateDefaultEntry must return
    // nullptr for datasets that are not yet written (CTAS planning
    // phase). Filtering here avoids the conflict.
    vector<string> valid;
    vector<const char *> key_ptrs;
    vector<const char *> value_ptrs;
    BuildStorageOptionPointerArrays(ns->option_keys, ns->option_values,
                                    key_ptrs, value_ptrs);
    for (auto &name : all) {
      const char *uri_ptr = nullptr;
      auto *ds = lance_open_dataset_in_dir_namespace(
          ns->root.c_str(), name.c_str(),
          key_ptrs.empty() ? nullptr : key_ptrs.data(),
          value_ptrs.empty() ? nullptr : value_ptrs.data(),
          ns->option_keys.size(), &uri_ptr);
      if (uri_ptr) {
        lance_free_string(uri_ptr);
      }
      if (ds) {
        lance_close_dataset(ds);
        valid.push_back(std::move(name));
      }
    }
    return valid;
  }

private:
  SchemaCatalogEntry &schema;
  shared_ptr<LanceDirectoryNamespaceConfig> ns;
};

static void PopulateLanceTableColumnsFromJsonSchema(
    ClientContext &context, const string &schema_json, ColumnList &out_columns,
    vector<string> *out_coerced_columns = nullptr) {
  ArrowSchemaWrapper schema_root;
  memset(&schema_root.arrow_schema, 0, sizeof(schema_root.arrow_schema));
  if (lance_json_arrow_schema_to_c(schema_json.c_str(),
                                   &schema_root.arrow_schema) != 0) {
    throw IOException("Failed to convert JSON Arrow schema to C Data "
                      "Interface" +
                      LanceFormatErrorSuffix());
  }
  auto coerced = LanceCoerceArrowSchemaForDuckDB(&schema_root.arrow_schema);
  if (out_coerced_columns) {
    *out_coerced_columns = std::move(coerced);
  }
  PopulateColumnsFromArrowSchema(context, schema_root.arrow_schema,
                                 out_columns);
}

class LanceRestNamespaceDefaultGenerator : public DefaultGenerator {
public:
  LanceRestNamespaceDefaultGenerator(
      Catalog &catalog, SchemaCatalogEntry &schema, string endpoint,
      string namespace_id, string bearer_token, string api_key,
      string delimiter, string bearer_token_override, string api_key_override,
      string headers_tsv)
      : DefaultGenerator(catalog), schema(schema),
        endpoint(std::move(endpoint)), namespace_id(std::move(namespace_id)),
        bearer_token(std::move(bearer_token)), api_key(std::move(api_key)),
        delimiter(std::move(delimiter)),
        bearer_token_override(std::move(bearer_token_override)),
        api_key_override(std::move(api_key_override)),
        headers_tsv(std::move(headers_tsv)) {}

  unique_ptr<CatalogEntry>
  CreateDefaultEntry(ClientContext &context,
                     const string &entry_name) override {
    unordered_map<string, Value> overrides;
    if (!bearer_token_override.empty()) {
      overrides["bearer_token"] = Value(bearer_token_override);
    }
    if (!api_key_override.empty()) {
      overrides["api_key"] = Value(api_key_override);
    }

    string resolved_bearer;
    string resolved_api_key;
    ResolveLanceNamespaceAuth(context, endpoint, overrides, resolved_bearer,
                              resolved_api_key);
    // Backward-compatible fallback to credentials resolved during ATTACH.
    if (resolved_bearer.empty() && resolved_api_key.empty() &&
        (!bearer_token.empty() || !api_key.empty())) {
      resolved_bearer = bearer_token;
      resolved_api_key = api_key;
    }
    const auto requires_worker_auth = !resolved_bearer.empty() ||
                                      !resolved_api_key.empty() ||
                                      !headers_tsv.empty();

    auto discovered =
        ListRestNamespaceTables(endpoint, namespace_id, resolved_bearer,
                                resolved_api_key, delimiter, headers_tsv);
    string table_id;
    if (!ResolveRestNamespaceTableId(discovered, namespace_id, delimiter,
                                     entry_name, table_id)) {
      // DuckDB probes the active catalog for system names (for example,
      // duckdb_tables while SHOW TABLES runs under USE <lance_catalog>).
      // Only names returned by list_tables belong to this catalog.
      return nullptr;
    }

    // Fast path: describe_table with schema from REST API (skips S3 open).
    {
      string schema_json;
      if (TryDescribeTableWithSchema(table_id, resolved_bearer,
                                     resolved_api_key, schema_json)) {
        auto table_name =
            RestNamespaceLeafName(namespace_id, delimiter, table_id);
        CreateTableInfo info(schema, table_name);
        info.internal = true;
        info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
        vector<string> coerced;
        try {
          PopulateLanceTableColumnsFromJsonSchema(context, schema_json,
                                                  info.columns, &coerced);
          // The schema endpoint does not expose the Lance generation. Open
          // the dataset once to establish the stable logical-write identity;
          // if that fails, fall through to the full dataset/schema path.
          string table_uri;
          auto *dataset = LanceOpenDatasetInNamespace(
              context, endpoint, table_id, resolved_bearer, resolved_api_key,
              delimiter, headers_tsv, table_uri);
          if (dataset) {
            try {
              auto identity_uri =
                  table_uri.empty() ? endpoint + "/" + table_id : table_uri;
#ifdef LANCE_DUCKDB_HAS_LOGICAL_WRITE_TARGET
              info.logical_write_target_identity =
                  LanceBuildLogicalWriteTargetIdentity(identity_uri, dataset);
#endif
              lance_close_dataset(dataset);
              dataset = nullptr;
              return MakeNamespaceEntry(table_id, std::move(info),
                                        std::move(coerced),
                                        requires_worker_auth);
            } catch (...) {
              lance_close_dataset(dataset);
              dataset = nullptr;
              throw;
            }
          }
        } catch (...) {
          // Fall through to opening the dataset. Some namespace services expose
          // an incomplete schema representation but a usable table location.
        }
      }
    }

    // Slow fallback: open dataset from S3.
    {
      string table_uri;
      void *dataset = nullptr;
      try {
        dataset = LanceOpenDatasetInNamespace(
            context, endpoint, table_id, resolved_bearer, resolved_api_key,
            delimiter, headers_tsv, table_uri);
      } catch (...) {
        dataset = nullptr;
      }
      if (dataset) {
        auto table_name =
            RestNamespaceLeafName(namespace_id, delimiter, table_id);
        CreateTableInfo info(schema, table_name);
        info.internal = true;
        info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
        vector<string> coerced;
        try {
          PopulateLanceTableColumnsFromDataset(context, dataset, info.columns,
                                               &coerced);
          auto identity_uri =
              table_uri.empty() ? endpoint + "/" + table_id : table_uri;
#ifdef LANCE_DUCKDB_HAS_LOGICAL_WRITE_TARGET
          info.logical_write_target_identity =
              LanceBuildLogicalWriteTargetIdentity(identity_uri, dataset);
#endif
        } catch (...) {
          lance_close_dataset(dataset);
          dataset = nullptr;
        }
        if (dataset) {
          lance_close_dataset(dataset);
          return MakeNamespaceEntry(table_id, std::move(info),
                                    std::move(coerced), requires_worker_auth);
        }
      }
    }

    // Keep an entry for a table returned by list_tables even if its schema is
    // temporarily unavailable. This lets SHOW TABLES complete without making
    // a different, case-folded table id addressable.
    auto resolved_entry_name =
        RestNamespaceLeafName(namespace_id, delimiter, table_id);
    CreateTableInfo info(schema, resolved_entry_name);
    info.internal = true;
    info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
    return MakeNamespaceEntry(table_id, std::move(info), {},
                              requires_worker_auth);
  }

  vector<string> GetDefaultEntries() override {
    auto tables = ListRestNamespaceTables(endpoint, namespace_id, bearer_token,
                                          api_key, delimiter, headers_tsv);
    for (auto &t : tables) {
      t = RestNamespaceLeafName(namespace_id, delimiter, t);
    }
    ValidateUniqueCaseInsensitiveNames(tables,
                                       "REST namespace '" + namespace_id + "'");
    return tables;
  }

private:
  unique_ptr<CatalogEntry> MakeNamespaceEntry(const string &table_id,
                                              CreateTableInfo info,
                                              vector<string> coerced_columns,
                                              bool requires_worker_auth) {
    LanceNamespaceTableConfig cfg;
    cfg.kind = LanceNamespaceKind::Rest;
    cfg.endpoint = endpoint;
    cfg.table_id = table_id;
    cfg.delimiter = delimiter;
    cfg.bearer_token_override = bearer_token_override;
    cfg.api_key_override = api_key_override;
    cfg.headers_tsv = headers_tsv;
    cfg.requires_worker_auth = requires_worker_auth;
    auto entry =
        make_uniq<LanceTableEntry>(catalog, schema, info, std::move(cfg));
    entry->SetCoercedColumnNames(std::move(coerced_columns));
    return unique_ptr_cast<LanceTableEntry, CatalogEntry>(std::move(entry));
  }

  bool TryDescribeTableWithSchema(const string &table_id,
                                  const string &resolved_bearer,
                                  const string &resolved_api_key,
                                  string &out_schema_json) {
    const char *bearer_ptr =
        resolved_bearer.empty() ? nullptr : resolved_bearer.c_str();
    const char *api_key_ptr =
        resolved_api_key.empty() ? nullptr : resolved_api_key.c_str();
    const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();
    const char *headers_ptr =
        headers_tsv.empty() ? nullptr : headers_tsv.c_str();
    const char *schema_ptr = nullptr;

    auto rc = lance_namespace_describe_table_with_schema(
        endpoint.c_str(), table_id.c_str(), bearer_ptr, api_key_ptr,
        delimiter_ptr, headers_ptr, &schema_ptr);
    if (rc != 0 || !schema_ptr) {
      if (schema_ptr) {
        lance_free_string(schema_ptr);
      }
      return false;
    }
    try {
      out_schema_json = schema_ptr;
    } catch (...) {
      lance_free_string(schema_ptr);
      throw;
    }
    lance_free_string(schema_ptr);
    return !out_schema_json.empty();
  }

  SchemaCatalogEntry &schema;
  string endpoint;
  string namespace_id;
  string bearer_token;
  string api_key;
  string delimiter;
  string bearer_token_override;
  string api_key_override;
  string headers_tsv;
};

static string GetDatasetDirName(const string &table_name) {
  return table_name + ".lance";
}

static bool IsSafeDatasetTableName(const string &name) {
  if (name.empty()) {
    return false;
  }
  if (name == "." || name == "..") {
    return false;
  }
  if (name.find('/') != string::npos || name.find('\\') != string::npos ||
      name.find('\0') != string::npos || name.find('\n') != string::npos) {
    return false;
  }
  return true;
}

static string CreateTableModeFromConflict(OnCreateConflict on_conflict) {
  switch (on_conflict) {
  case OnCreateConflict::ERROR_ON_CONFLICT:
  case OnCreateConflict::IGNORE_ON_CONFLICT:
    return "create";
  case OnCreateConflict::REPLACE_ON_CONFLICT:
    return "overwrite";
  case OnCreateConflict::ALTER_ON_CONFLICT:
    break;
  default:
    break;
  }
  return "overwrite";
}

static string
GetCreateTableDataStorageVersionOption(const CreateTableInfo &create_info) {
  auto it = create_info.options.find("data_storage_version");
  if (it == create_info.options.end()) {
    return LANCE_DEFAULT_DATA_STORAGE_VERSION;
  }
  if (!it->second) {
    throw BinderException("data_storage_version option cannot be NULL");
  }
  auto &expr = *it->second;
  if (expr.expression_class != ExpressionClass::CONSTANT) {
    throw BinderException(
        "data_storage_version option must be a constant string");
  }
  auto &constant = expr.Cast<ConstantExpression>();
  if (constant.value.IsNull()) {
    throw BinderException("data_storage_version option cannot be NULL");
  }
  auto value =
      constant.value.DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
  StringUtil::Trim(value);
  if (value.empty()) {
    throw BinderException("data_storage_version option cannot be empty");
  }
  ValidateLanceCString(value, "Lance data_storage_version option");
  return value;
}

class ScopedLanceDefaultMetadata final {
public:
  explicit ScopedLanceDefaultMetadata(ArrowSchema &schema_p)
      : schema(schema_p) {
    if (schema.n_children < 0 || (schema.n_children > 0 && !schema.children)) {
      throw InternalException("Invalid Arrow schema for Lance defaults");
    }
    originals.reserve(NumericCast<idx_t>(schema.n_children));
    metadata.reserve(NumericCast<idx_t>(schema.n_children));
    for (idx_t i = 0; i < NumericCast<idx_t>(schema.n_children); i++) {
      if (!schema.children[i]) {
        throw InternalException("Null Arrow child schema for Lance defaults");
      }
      originals.push_back(schema.children[i]->metadata);
    }
  }

  void Apply(const ColumnList &columns) {
    if (columns.LogicalColumnCount() != originals.size()) {
      throw InternalException(
          "Lance default column count does not match Arrow schema");
    }
    metadata.resize(originals.size());
    idx_t index = 0;
    for (auto &column : columns.Logical()) {
      if (column.HasDefaultValue()) {
        ArrowSchemaMetadata field_metadata(originals[index]);
        field_metadata.AddOption("duckdb_default_expr",
                                 column.DefaultValue().ToString());
        metadata[index] = field_metadata.SerializeMetadata();
        schema.children[index]->metadata = metadata[index].get();
      }
      index++;
    }
  }

  ~ScopedLanceDefaultMetadata() {
    for (idx_t i = 0; i < originals.size(); i++) {
      schema.children[i]->metadata = originals[i];
    }
  }

private:
  ArrowSchema &schema;
  vector<const char *> originals;
  vector<unsafe_unique_array<char>> metadata;
};

class LanceSchemaEntry final : public DuckSchemaEntry {
public:
  LanceSchemaEntry(Catalog &catalog, CreateSchemaInfo &info,
                   shared_ptr<LanceDirectoryNamespaceConfig> directory_ns,
                   shared_ptr<LanceRestNamespaceConfig> rest_ns)
      : DuckSchemaEntry(catalog, info), directory_ns(std::move(directory_ns)),
        rest_ns(std::move(rest_ns)) {}

  void SetTableDefaultGenerator(DefaultGenerator *generator) {
    table_default_generator = generator;
  }

  void Alter(CatalogTransaction transaction, AlterInfo &info) override {
    auto &set = GetCatalogSet(info.GetCatalogType());
    auto entry = set.GetEntry(transaction, info.name);
    auto *lance_entry =
        entry ? dynamic_cast<LanceTableEntry *>(entry.get()) : nullptr;

    if (!lance_entry) {
      DuckSchemaEntry::Alter(transaction, info);
      return;
    }

    auto &context = transaction.GetContext();
    if (!context.transaction.IsAutoCommit()) {
      throw NotImplementedException(
          "Lance DDL does not support explicit transactions yet");
    }
    RequireLanceTableWritable(*lance_entry, "ALTER TABLE");

    // Allow altering internal entries for attached Lance catalogs.
    info.allow_internal = true;
    bool lance_mutation_committed = false;
    unique_ptr<LanceLease> mutation_lease;

    if (info.type == AlterType::SET_COMMENT) {
      auto &comment = info.Cast<SetCommentInfo>();
      const char *comment_ptr = nullptr;
      string comment_str;
      if (!comment.comment_value.IsNull()) {
        comment_str = comment.comment_value.DefaultCastAs(LogicalType::VARCHAR)
                          .GetValue<string>();
        ValidateLanceCString(comment_str, "Lance table comment");
        comment_ptr = comment_str.c_str();
      }

      auto cache_key = LanceBuildDatasetCacheKeyForTable(context, *lance_entry);
      string display_uri;
      void *dataset =
          LanceOpenDatasetForTable(context, *lance_entry, display_uri);
      if (!dataset) {
        throw IOException("Failed to open Lance dataset: " + display_uri +
                          LanceFormatErrorSuffix());
      }
      try {
        mutation_lease = LanceLease::AcquireForDataset(
            dataset, LanceLeaseKind::Mutation, "alter-comment:" + display_uri);
      } catch (...) {
        lance_close_dataset(dataset);
        throw;
      }
      auto rc =
          lance_dataset_update_table_metadata(dataset, "comment", comment_ptr);
      lance_close_dataset(dataset);
      if (rc != 0) {
        throw IOException("Failed to update table comment in Lance dataset: " +
                          display_uri + LanceFormatErrorSuffix());
      }
      lance_mutation_committed = true;
      LanceInvalidateDatasetCache(context, cache_key);
      ReleaseLanceStorageLease(mutation_lease, "Lance table comment update");
    }

    auto system_tx =
        CatalogTransaction::GetSystemTransaction(catalog.GetDatabase());
    system_tx.context = &context;

    if (info.type == AlterType::CHANGE_OWNERSHIP) {
      if (!set.AlterOwnership(system_tx, info.Cast<ChangeOwnershipInfo>())) {
        throw CatalogException("Couldn't change ownership!");
      }
      return;
    }

    try {
      if (!set.AlterEntry(system_tx, info.name, info)) {
        throw CatalogException::MissingEntry(info.GetCatalogType(), info.name,
                                             string());
      }
    } catch (const std::exception &error) {
      if (!lance_mutation_committed) {
        throw;
      }
      throw IOException(
          "Lance table comment committed, but updating the DuckDB catalog "
          "entry failed: " +
          string(error.what()) + " (code=55)");
    } catch (...) {
      if (!lance_mutation_committed) {
        throw;
      }
      throw IOException(
          "Lance table comment committed, but updating the DuckDB catalog "
          "entry failed with an unknown error (code=55)");
    }
  }

  void DropEntry(ClientContext &context, DropInfo &info) override {
    if (info.type != CatalogType::TABLE_ENTRY) {
      DuckSchemaEntry::DropEntry(context, info);
      return;
    }

    // DuckDB stores TABLE and VIEW entries in the same catalog set (see
    // DuckSchemaEntry::GetCatalogSet), so we need to resolve the existing entry
    // type explicitly before attempting to drop anything.
    auto transaction = GetCatalogTransaction(context);
    auto &set = GetCatalogSet(info.type);
    auto existing_entry = set.GetEntry(transaction, info.name);
    if (!existing_entry) {
      throw InternalException(
          "Failed to drop entry \"%s\" - entry could not be found", info.name);
    }
    auto *lance_entry = dynamic_cast<LanceTableEntry *>(existing_entry.get());
    if (!lance_entry) {
      DuckSchemaEntry::DropEntry(context, info);
      return;
    }
    if (!context.transaction.IsAutoCommit()) {
      throw NotImplementedException(
          "Lance DDL does not support explicit transactions yet");
    }
    RequireLanceTableWritable(*lance_entry, "DROP TABLE");
    auto cache_key = LanceBuildDatasetCacheKeyForTable(context, *lance_entry);
    auto existing_type = existing_entry->type;
    auto &table_config = lance_entry->NamespaceConfig();
    unique_ptr<LanceLease> vacuum_lease;

    if (rest_ns) {
      if (!table_config.IsRest() || table_config.table_id.empty()) {
        throw InternalException(
            "REST-backed Lance table is missing its namespace table id");
      }
      unordered_map<string, Value> overrides;
      if (!rest_ns->bearer_token_override.empty()) {
        overrides["bearer_token"] = Value(rest_ns->bearer_token_override);
      }
      if (!rest_ns->api_key_override.empty()) {
        overrides["api_key"] = Value(rest_ns->api_key_override);
      }

      string bearer_token;
      string api_key;
      ResolveLanceNamespaceAuth(context, rest_ns->endpoint, overrides,
                                bearer_token, api_key);

      // Resolve the physical dataset before deleting its namespace entry so
      // DROP TABLE participates in the same storage-backed lease protocol as
      // scans and maintenance.  A missing/unauthorized dataset is a hard
      // failure: deleting only the catalog entry would leave the ownership
      // protocol split from the physical data.
      string dataset_display_uri;
      auto *dataset =
          LanceOpenDatasetForTable(context, *lance_entry, dataset_display_uri);
      if (!dataset) {
        throw IOException("Failed to open Lance dataset for DROP TABLE: " +
                          dataset_display_uri + LanceFormatErrorSuffix());
      }
      try {
        vacuum_lease =
            LanceLease::AcquireForDataset(dataset, LanceLeaseKind::Vacuum,
                                          "drop-table:" + dataset_display_uri);
      } catch (...) {
        lance_close_dataset(dataset);
        throw;
      }
      lance_close_dataset(dataset);

      string drop_error;
      bool namespace_mutated = false;
      if (!TryLanceNamespaceDropTable(
              context, rest_ns->endpoint, table_config.table_id, bearer_token,
              api_key, rest_ns->delimiter, rest_ns->headers_tsv, drop_error,
              namespace_mutated)) {
        throw IOException("Failed to drop Lance table via namespace: " +
                          (drop_error.empty() ? "unknown error" : drop_error));
      }
    } else {
      if (!table_config.IsDirectory() || table_config.table_id.empty()) {
        throw InternalException(
            "Directory-backed Lance table is missing its namespace table id");
      }
      if (!IsSafeDatasetTableName(table_config.table_id)) {
        throw InvalidInputException(
            "Unsafe Lance dataset name for DROP TABLE: " +
            table_config.table_id);
      }

      if (table_config.root.empty()) {
        throw InternalException("Lance directory namespace root is empty");
      }
      auto &root = table_config.root;
      auto &option_keys = table_config.option_keys;
      auto &option_values = table_config.option_values;

      auto dataset_path =
          JoinNamespacePath(root, GetDatasetDirName(table_config.table_id));
      vacuum_lease = LanceLease::Acquire(
          context, dataset_path, LanceLeaseKind::Vacuum,
          "drop-table:" + dataset_path, option_keys, option_values);

      vector<const char *> key_ptrs;
      vector<const char *> value_ptrs;
      BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                      value_ptrs);

      auto rc = lance_dir_namespace_drop_table(
          root.c_str(), table_config.table_id.c_str(),
          key_ptrs.empty() ? nullptr : key_ptrs.data(),
          value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size());
      if (rc != 0) {
        throw IOException("Failed to drop Lance dataset: " + root + "/" +
                          GetDatasetDirName(table_config.table_id) +
                          LanceFormatErrorSuffix());
      }
    }
    LanceInvalidateDatasetCache(context, cache_key);
    ReleaseLanceStorageLease(vacuum_lease, "Lance table drop");

    // Drop the DuckDB catalog entry after the dataset has been deleted
    // successfully.
    //
    // Note: TABLE_ENTRY and VIEW_ENTRY share the same underlying catalog set.
    // DuckDB's transactional DROP path assumes TABLE_ENTRY is always a
    // DuckTableEntry (and will fail for extension-backed TableCatalogEntry
    // implementations). To avoid that, perform the catalog drop using a system
    // (non-transactional) CatalogTransaction.
    try {
      if (existing_type != CatalogType::TABLE_ENTRY &&
          existing_type != CatalogType::VIEW_ENTRY) {
        throw InternalException(
            "Unexpected catalog entry type for DROP TABLE: %s",
            CatalogTypeToString(existing_type));
      }
      auto system_transaction =
          CatalogTransaction::GetSystemTransaction(catalog.GetDatabase());
      if (!set.DropEntry(system_transaction, info.name, info.cascade, true)) {
        throw InternalException(
            "Could not drop element because of an internal error");
      }

      // DropEntry with a system (committed) CatalogTransaction leaves a
      // committed tombstone behind. This blocks subsequent lazy discovery of a
      // recreated dataset with the same name, because CatalogSet finds the
      // tombstone and never consults the default generator. Attached Lance
      // catalogs are ephemeral, so clean up the entry chain eagerly.
      set.CleanupEntry(*existing_entry);
      InvalidateTableDefaults();
    } catch (const std::exception &error) {
      throw IOException("Lance table drop committed, but updating the DuckDB "
                        "catalog failed: " +
                        string(error.what()) + " (code=56)");
    } catch (...) {
      throw IOException(
          "Lance table drop committed, but updating the DuckDB catalog failed "
          "with an unknown error (code=56)");
    }
  }

  optional_ptr<CatalogEntry> CreateTable(CatalogTransaction transaction,
                                         BoundCreateTableInfo &info) override {
    auto &create_info = info.Base();
    if (create_info.temporary) {
      throw NotImplementedException(
          "Lance ATTACH TYPE LANCE does not support TEMPORARY tables");
    }
    if (!info.constraints.empty() || !create_info.constraints.empty()) {
      throw NotImplementedException(
          "Lance CREATE TABLE does not support constraints");
    }
    auto &context = transaction.GetContext();
    if (!context.transaction.IsAutoCommit()) {
      throw NotImplementedException(
          "Lance DDL does not support explicit transactions yet");
    }
    if (catalog.GetAttached().IsReadOnly()) {
      throw InvalidInputException(
          "CREATE TABLE cannot modify a Lance attachment in read-only mode");
    }
    auto data_storage_version =
        GetCreateTableDataStorageVersionOption(create_info);
    string dataset_path;
    vector<string> option_keys;
    vector<string> option_values;
    string cache_key;
    bool namespace_mutated = false;
    unique_ptr<LanceLease> mutation_lease;

    // Validate and materialize the complete Arrow schema before a REST
    // namespace declaration.  Schema/default failures are deterministic and
    // must not leave behind an externally visible empty table.
    vector<string> names;
    vector<LogicalType> types;
    names.reserve(create_info.columns.LogicalColumnCount());
    types.reserve(create_info.columns.LogicalColumnCount());
    for (auto &col : create_info.columns.Logical()) {
      names.push_back(col.Name());
      types.push_back(col.Type());
    }

    ArrowSchemaWrapper schema_root;
    memset(&schema_root.arrow_schema, 0, sizeof(schema_root.arrow_schema));
    auto props = context.GetClientProperties();
    ArrowConverter::ToArrowSchema(&schema_root.arrow_schema, types, names,
                                  props);
    ScopedLanceDefaultMetadata default_metadata(schema_root.arrow_schema);
    default_metadata.Apply(create_info.columns);

    try {

      if (rest_ns) {
        unordered_map<string, Value> overrides;
        if (!rest_ns->bearer_token_override.empty()) {
          overrides["bearer_token"] = Value(rest_ns->bearer_token_override);
        }
        if (!rest_ns->api_key_override.empty()) {
          overrides["api_key"] = Value(rest_ns->api_key_override);
        }

        string bearer_token;
        string api_key;
        ResolveLanceNamespaceAuth(context, rest_ns->endpoint, overrides,
                                  bearer_token, api_key);

        vector<string> discovered;
        string list_error;
        if (!TryLanceNamespaceListTables(
                context, rest_ns->endpoint, rest_ns->namespace_id, bearer_token,
                api_key, rest_ns->delimiter, rest_ns->headers_tsv, discovered,
                list_error)) {
          throw IOException(
              "Failed to list tables from Lance namespace: " +
              (list_error.empty() ? "unknown error" : list_error));
        }
        string existing_id;
        auto exists = ResolveRestNamespaceTableId(
            discovered, rest_ns->namespace_id, rest_ns->delimiter,
            create_info.table, existing_id);
        auto table_id_for_ops =
            exists ? existing_id
                   : BuildRestNamespaceTableId(rest_ns->namespace_id,
                                               rest_ns->delimiter,
                                               create_info.table);
        cache_key = LanceBuildNamespaceDatasetCacheKey(
            rest_ns->endpoint, table_id_for_ops, bearer_token, api_key,
            rest_ns->delimiter, rest_ns->headers_tsv);
        if (create_info.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT &&
            exists) {
          InvalidateTableDefaults();
          return nullptr;
        }
        if (create_info.on_conflict == OnCreateConflict::ERROR_ON_CONFLICT &&
            exists) {
          throw IOException("Lance table already exists: " + existing_id);
        }
        if (create_info.on_conflict == OnCreateConflict::REPLACE_ON_CONFLICT &&
            exists) {
          string drop_error;
          if (!TryLanceNamespaceDropTable(
                  context, rest_ns->endpoint, table_id_for_ops, bearer_token,
                  api_key, rest_ns->delimiter, rest_ns->headers_tsv, drop_error,
                  namespace_mutated)) {
            throw IOException(
                "Failed to drop Lance table via namespace: " +
                (drop_error.empty() ? "unknown error" : drop_error));
          }
          namespace_mutated = true;
          LanceInvalidateDatasetCache(context, cache_key);
        }

        string create_error;
        if (!TryLanceNamespaceCreateEmptyTable(
                context, rest_ns->endpoint, table_id_for_ops, bearer_token,
                api_key, rest_ns->delimiter, rest_ns->headers_tsv, dataset_path,
                option_keys, option_values, create_error, namespace_mutated)) {
          // declare_table is non-idempotent. Never probe a second table id
          // after a failed request: the first request may have reached the
          // service, and a fallback could create a different leaf table.
          throw IOException(
              "Failed to create Lance table via namespace: " +
              (create_error.empty() ? "unknown error" : create_error));
        }
        LanceInvalidateDatasetCache(context, cache_key);
        if (dataset_path.empty()) {
          throw IOException(
              "Failed to create Lance table via namespace: empty location");
        }
        dataset_path = LanceNormalizeS3Scheme(dataset_path);
      } else {
        if (!IsSafeDatasetTableName(create_info.table)) {
          throw InvalidInputException(
              "Unsafe Lance dataset name for CREATE TABLE: " +
              create_info.table);
        }
        if (!directory_ns || directory_ns->root.empty()) {
          throw InternalException("Lance directory namespace root is empty");
        }

        string existing_name;
        auto exists = ResolveDirectoryNamespaceTableName(
            *directory_ns, create_info.table, existing_name);
        auto table_name_for_ops = exists ? existing_name : create_info.table;
        dataset_path = JoinNamespacePath(directory_ns->root,
                                         GetDatasetDirName(table_name_for_ops));
        if (create_info.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT &&
            exists) {
          InvalidateTableDefaults();
          return nullptr;
        }
        if (create_info.on_conflict == OnCreateConflict::ERROR_ON_CONFLICT &&
            exists) {
          throw IOException("Lance dataset already exists: " + dataset_path);
        }

        option_keys = directory_ns->option_keys;
        option_values = directory_ns->option_values;
        cache_key = LanceBuildResolvedPathDatasetCacheKey(
            dataset_path, option_keys, option_values);
      }

      mutation_lease = LanceLease::Acquire(
          context, dataset_path, LanceLeaseKind::Mutation,
          "create-table:" + dataset_path, option_keys, option_values);

      auto mode = CreateTableModeFromConflict(create_info.on_conflict);
      vector<const char *> key_ptrs;
      vector<const char *> value_ptrs;
      BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                      value_ptrs);
      const char *data_storage_version_ptr =
          data_storage_version.empty() ? nullptr : data_storage_version.c_str();

      auto *writer = lance_open_writer_with_storage_options(
          dataset_path.c_str(), mode.c_str(),
          key_ptrs.empty() ? nullptr : key_ptrs.data(),
          value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size(),
          LANCE_DEFAULT_MAX_ROWS_PER_FILE, LANCE_DEFAULT_MAX_ROWS_PER_GROUP,
          LANCE_DEFAULT_MAX_BYTES_PER_FILE, data_storage_version_ptr, nullptr,
          1, LanceGetSessionHandle(context), &schema_root.arrow_schema);
      if (!writer) {
        throw IOException("Failed to open Lance writer: " + dataset_path +
                          LanceFormatErrorSuffix());
      }
      auto rc = lance_writer_finish(writer);
      lance_close_writer(writer);
      if (rc != 0) {
        throw IOException("Failed to finalize Lance dataset write" +
                          LanceFormatErrorSuffix());
      }
    } catch (const std::exception &error) {
      if (mutation_lease) {
        mutation_lease->RetainForReconciliation();
      }
      if (!namespace_mutated) {
        throw;
      }
      LanceInvalidateDatasetCache(context, cache_key);
      throw IOException(
          "Lance namespace table creation changed external state, but the "
          "dataset creation did not finish cleanly: " +
          string(error.what()) + " (code=56)");
    } catch (...) {
      if (mutation_lease) {
        mutation_lease->RetainForReconciliation();
      }
      if (!namespace_mutated) {
        throw;
      }
      LanceInvalidateDatasetCache(context, cache_key);
      throw IOException(
          "Lance namespace table creation changed external state, but the "
          "dataset creation failed with an unknown error (code=56)");
    }

    ReleaseLanceStorageLease(mutation_lease, "Lance table creation");
    LanceInvalidateDatasetCache(context, cache_key);
    InvalidateTableDefaults();
    return nullptr;
  }

public:
  void InvalidateTableDefaults() {
    if (!table_default_generator) {
      return;
    }
    table_default_generator->created_all_entries = false;
  }

private:
  shared_ptr<LanceDirectoryNamespaceConfig> directory_ns;
  shared_ptr<LanceRestNamespaceConfig> rest_ns;
  DefaultGenerator *table_default_generator = nullptr;
};

class LanceDuckCatalog final : public DuckCatalog {
public:
  using DuckCatalog::PlanDelete;
  using DuckCatalog::PlanMergeInto;

  LanceDuckCatalog(AttachedDatabase &db,
                   shared_ptr<LanceDirectoryNamespaceConfig> directory_ns,
                   shared_ptr<LanceRestNamespaceConfig> rest_ns)
      : DuckCatalog(db), directory_ns(std::move(directory_ns)),
        rest_ns(std::move(rest_ns)) {}

  using DuckCatalog::PlanUpdate;

  ErrorData SupportsCreateTable(BoundCreateTableInfo &info) override {
    auto &base = info.Base().Cast<CreateTableInfo>();
    if (!base.partition_keys.empty()) {
      return ErrorData(ExceptionType::CATALOG,
                       StringUtil::Format("PARTITIONED BY is not supported "
                                          "for tables in a %s catalog",
                                          GetCatalogType()));
    }
    if (!base.sort_keys.empty()) {
      return ErrorData(
          ExceptionType::CATALOG,
          StringUtil::Format("SORTED BY is not supported for tables in a %s "
                             "catalog",
                             GetCatalogType()));
    }
    for (auto &entry : base.options) {
      if (!StringUtil::CIEquals(entry.first, "data_storage_version")) {
        return ErrorData(ExceptionType::CATALOG,
                         "Only data_storage_version is supported in WITH "
                         "clause for Lance tables");
      }
    }
    return ErrorData();
  }

  PhysicalOperator &PlanUpdate(ClientContext &context,
                               PhysicalPlanGenerator &planner,
                               LogicalUpdate &op) override {
    if (dynamic_cast<LanceTableEntry *>(&op.table)) {
      return PlanLanceUpdateOverwrite(context, planner, op);
    }
    return Catalog::PlanUpdate(context, planner, op);
  }

  PhysicalOperator &PlanInsert(ClientContext &context,
                               PhysicalPlanGenerator &planner,
                               LogicalInsert &op,
                               optional_ptr<PhysicalOperator> plan) override {
    if (dynamic_cast<LanceTableEntry *>(&op.table)) {
      return PlanLanceInsertAppend(context, planner, op, plan);
    }
    return DuckCatalog::PlanInsert(context, planner, op, plan);
  }

  PhysicalOperator &PlanDelete(ClientContext &context,
                               PhysicalPlanGenerator &planner,
                               LogicalDelete &op,
                               PhysicalOperator &plan) override;

  PhysicalOperator &PlanMergeInto(ClientContext &context,
                                  PhysicalPlanGenerator &planner,
                                  LogicalMergeInto &op,
                                  PhysicalOperator &plan) override {
    if (dynamic_cast<LanceTableEntry *>(&op.table)) {
      return PlanLanceMergeInto(context, planner, op, plan);
    }
    return DuckCatalog::PlanMergeInto(context, planner, op, plan);
  }

  PhysicalOperator &PlanCreateTableAs(ClientContext &context,
                                      PhysicalPlanGenerator &planner,
                                      LogicalCreateTable &op,
                                      PhysicalOperator &plan) override {
    if (!context.transaction.IsAutoCommit()) {
      throw NotImplementedException(
          "Lance DDL does not support explicit transactions yet");
    }
    if (GetAttached().IsReadOnly()) {
      throw InvalidInputException(
          "CREATE TABLE AS cannot modify a Lance attachment in read-only "
          "mode");
    }
    auto &create_info = op.info->Base();
    auto data_storage_version =
        GetCreateTableDataStorageVersionOption(create_info);
    if (create_info.temporary) {
      throw NotImplementedException(
          "Lance ATTACH TYPE LANCE does not support TEMPORARY tables");
    }
    // CTAS bypasses LanceSchemaEntry::CreateTable, so explicitly refresh the
    // lazy namespace catalog after the statement. Invalidating at planning
    // time is safe: a failed write simply causes the next lookup to re-list an
    // unchanged namespace.
    op.schema.Cast<LanceSchemaEntry>().InvalidateTableDefaults();
    if (rest_ns) {
      class PhysicalLanceCreateTableAs final : public PhysicalOperator {
      public:
        PhysicalLanceCreateTableAs(
            PhysicalPlan &physical_plan, vector<LogicalType> types_p,
            string endpoint, string namespace_id, string delimiter,
            string bearer_token_override, string api_key_override,
            string headers_tsv, string table_id, string writer_mode,
            string data_storage_version, vector<string> column_names_p,
            vector<LogicalType> column_types_p, idx_t estimated_cardinality)
            : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                               std::move(types_p), estimated_cardinality),
              endpoint(std::move(endpoint)),
              namespace_id(std::move(namespace_id)),
              delimiter(std::move(delimiter)),
              bearer_token_override(std::move(bearer_token_override)),
              api_key_override(std::move(api_key_override)),
              headers_tsv(std::move(headers_tsv)),
              table_id(std::move(table_id)),
              writer_mode(std::move(writer_mode)),
              data_storage_version(std::move(data_storage_version)),
              column_names(std::move(column_names_p)),
              column_types(std::move(column_types_p)) {}

        bool IsSink() const override { return true; }
        bool IsSource() const override { return true; }
        bool ParallelSink() const override { return false; }
        bool SinkOrderDependent() const override { return false; }

        struct GlobalState final : public GlobalSinkState {
          mutex lock;

          string endpoint;
          string namespace_id;
          string delimiter;
          string bearer_token_override;
          string api_key_override;
          string headers_tsv;
          string table_id;
          string writer_mode;
          string data_storage_version;

          string open_path;
          vector<string> option_keys;
          vector<string> option_values;

          vector<string> column_names;
          vector<LogicalType> column_types;
          ColumnDataCollection buffered_rows;

          int64_t insert_count = 0;
          bool namespace_mutated = false;
          bool write_committed = false;
          void *writer = nullptr;
          unique_ptr<LanceLease> mutation_lease;
          ArrowSchemaWrapper schema_root;

          explicit GlobalState(ClientContext &context, string endpoint_p,
                               string namespace_id_p, string delimiter_p,
                               string bearer_override_p, string api_override_p,
                               string headers_tsv_p, string table_id_p,
                               string writer_mode_p,
                               string data_storage_version_p,
                               vector<string> col_names_p,
                               vector<LogicalType> col_types_p)
              : endpoint(std::move(endpoint_p)),
                namespace_id(std::move(namespace_id_p)),
                delimiter(std::move(delimiter_p)),
                bearer_token_override(std::move(bearer_override_p)),
                api_key_override(std::move(api_override_p)),
                headers_tsv(std::move(headers_tsv_p)),
                table_id(std::move(table_id_p)),
                writer_mode(std::move(writer_mode_p)),
                data_storage_version(std::move(data_storage_version_p)),
                column_names(std::move(col_names_p)),
                column_types(std::move(col_types_p)),
                buffered_rows(context, column_types) {}

          ~GlobalState() override {
            if (writer) {
              lance_close_writer(writer);
              writer = nullptr;
            }
          }
        };

        unique_ptr<GlobalSinkState>
        GetGlobalSinkState(ClientContext &context) const override {
          auto state = make_uniq<GlobalState>(
              context, endpoint, namespace_id, delimiter, bearer_token_override,
              api_key_override, headers_tsv, table_id, writer_mode,
              data_storage_version, column_names, column_types);

          auto props = context.GetClientProperties();
          memset(&state->schema_root.arrow_schema, 0,
                 sizeof(state->schema_root.arrow_schema));
          ArrowConverter::ToArrowSchema(&state->schema_root.arrow_schema,
                                        state->column_types,
                                        state->column_names, props);

          return std::move(state);
        }

        struct LocalState final : public LocalSinkState {};
        unique_ptr<LocalSinkState>
        GetLocalSinkState(ExecutionContext &) const override {
          return make_uniq<LocalState>();
        }

        SinkResultType Sink(ExecutionContext &, DataChunk &chunk,
                            OperatorSinkInput &input) const override {
          if (chunk.size() == 0) {
            return SinkResultType::NEED_MORE_INPUT;
          }

          auto &gstate = input.global_state.Cast<GlobalState>();
          lock_guard<mutex> guard(gstate.lock);
          // A REST namespace declaration is externally visible and cannot be
          // rolled back with the DuckDB pipeline. Buffer the complete child
          // result first so a child/Sink failure has no namespace side effect.
          // ColumnDataCollection uses DuckDB's buffer manager and can spill.
          gstate.buffered_rows.Append(chunk);
          return SinkResultType::NEED_MORE_INPUT;
        }

        SinkCombineResultType
        Combine(ExecutionContext &, OperatorSinkCombineInput &) const override {
          return SinkCombineResultType::FINISHED;
        }

        SinkFinalizeType
        Finalize(Pipeline &, Event &, ClientContext &context,
                 OperatorSinkFinalizeInput &input) const override {
          auto &gstate = input.global_state.Cast<GlobalState>();
          lock_guard<mutex> guard(gstate.lock);

          // The result count conversion and every potentially failing cache-key
          // dependency must be resolved before the first namespace mutation.
          gstate.insert_count =
              NumericCast<int64_t>(gstate.buffered_rows.Count());

          unordered_map<string, Value> overrides;
          if (!gstate.bearer_token_override.empty()) {
            overrides["bearer_token"] = Value(gstate.bearer_token_override);
          }
          if (!gstate.api_key_override.empty()) {
            overrides["api_key"] = Value(gstate.api_key_override);
          }

          string bearer_token;
          string api_key;
          ResolveLanceNamespaceAuth(context, gstate.endpoint, overrides,
                                    bearer_token, api_key);

          vector<string> discovered;
          string list_error;
          if (!TryLanceNamespaceListTables(
                  context, gstate.endpoint, gstate.namespace_id, bearer_token,
                  api_key, gstate.delimiter, gstate.headers_tsv, discovered,
                  list_error)) {
            throw IOException(
                "Failed to list tables from Lance namespace: " +
                (list_error.empty() ? "unknown error" : list_error));
          }

          string existing_id;
          auto exists = ResolveRestNamespaceTableId(
              discovered, gstate.namespace_id, gstate.delimiter,
              gstate.table_id, existing_id);
          if (exists) {
            gstate.table_id = std::move(existing_id);
          }
          auto cache_key = LanceBuildNamespaceDatasetCacheKey(
              gstate.endpoint, gstate.table_id, bearer_token, api_key,
              gstate.delimiter, gstate.headers_tsv);

          try {
            if (gstate.writer_mode == "overwrite" && exists) {
              string drop_error;
              if (!TryLanceNamespaceDropTable(
                      context, gstate.endpoint, gstate.table_id, bearer_token,
                      api_key, gstate.delimiter, gstate.headers_tsv, drop_error,
                      gstate.namespace_mutated)) {
                throw IOException(
                    "Failed to drop Lance table via namespace: " +
                    (drop_error.empty() ? "unknown error" : drop_error));
              }
              gstate.namespace_mutated = true;
              LanceInvalidateDatasetCache(context, cache_key);
            }

            string create_error;
            if (!TryLanceNamespaceCreateEmptyTable(
                    context, gstate.endpoint, gstate.table_id, bearer_token,
                    api_key, gstate.delimiter, gstate.headers_tsv,
                    gstate.open_path, gstate.option_keys, gstate.option_values,
                    create_error, gstate.namespace_mutated)) {
              // The namespace specification uses the fully segmented table id.
              // A second request with only the leaf id is ambiguous and unsafe
              // after a non-idempotent declare failure.
              throw IOException(
                  "Failed to create Lance table via namespace: " +
                  (create_error.empty() ? "unknown error" : create_error));
            }
            LanceInvalidateDatasetCache(context, cache_key);
            if (gstate.open_path.empty()) {
              throw IOException(
                  "Failed to create Lance table via namespace: empty location");
            }
            gstate.open_path = LanceNormalizeS3Scheme(gstate.open_path);

            gstate.mutation_lease = LanceLease::Acquire(
                context, gstate.open_path, LanceLeaseKind::Mutation,
                "ctas:" + gstate.open_path, gstate.option_keys,
                gstate.option_values);

            vector<const char *> key_ptrs;
            vector<const char *> value_ptrs;
            BuildStorageOptionPointerArrays(
                gstate.option_keys, gstate.option_values, key_ptrs, value_ptrs);

            const char *data_storage_version_ptr =
                gstate.data_storage_version.empty()
                    ? nullptr
                    : gstate.data_storage_version.c_str();
            gstate.writer = lance_open_writer_with_storage_options(
                gstate.open_path.c_str(), gstate.writer_mode.c_str(),
                key_ptrs.empty() ? nullptr : key_ptrs.data(),
                value_ptrs.empty() ? nullptr : value_ptrs.data(),
                gstate.option_keys.size(), LANCE_DEFAULT_MAX_ROWS_PER_FILE,
                LANCE_DEFAULT_MAX_ROWS_PER_GROUP,
                LANCE_DEFAULT_MAX_BYTES_PER_FILE, data_storage_version_ptr,
                nullptr, 1, LanceGetSessionHandle(context),
                &gstate.schema_root.arrow_schema);
            if (!gstate.writer) {
              throw IOException("Failed to open Lance writer: " +
                                gstate.open_path + LanceFormatErrorSuffix());
            }

            ColumnDataScanState scan_state;
            gstate.buffered_rows.InitializeScan(scan_state);
            DataChunk chunk;
            gstate.buffered_rows.InitializeScanChunk(scan_state, chunk);
            auto props = context.GetClientProperties();
            unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>>
                extension_type_cast;
            while (gstate.buffered_rows.Scan(scan_state, chunk)) {
              ArrowArrayWrapper array;
              ArrowConverter::ToArrowArray(chunk, &array.arrow_array, props,
                                           extension_type_cast);
              if (lance_writer_write_batch(gstate.writer, &array.arrow_array) !=
                  0) {
                throw IOException("Failed to write to Lance dataset" +
                                  LanceFormatErrorSuffix());
              }
            }

            auto rc = lance_writer_finish(gstate.writer);
            auto finish_error = rc == 0 ? string() : LanceFormatErrorSuffix();
            lance_close_writer(gstate.writer);
            gstate.writer = nullptr;
            if (rc != 0) {
              throw IOException("Failed to finalize Lance CTAS write" +
                                finish_error);
            }
            gstate.write_committed = true;
            LanceInvalidateDatasetCache(context, cache_key);
          } catch (const std::exception &error) {
            if (gstate.writer) {
              lance_close_writer(gstate.writer);
              gstate.writer = nullptr;
            }
            if (gstate.mutation_lease) {
              gstate.mutation_lease->RetainForReconciliation();
            }
            if (!gstate.namespace_mutated) {
              throw;
            }
            LanceInvalidateDatasetCache(context, cache_key);
            throw IOException(
                "Lance namespace CTAS changed external state, but did not "
                "finish cleanly: " +
                string(error.what()) + " (code=56)");
          } catch (...) {
            if (gstate.writer) {
              lance_close_writer(gstate.writer);
              gstate.writer = nullptr;
            }
            if (gstate.mutation_lease) {
              gstate.mutation_lease->RetainForReconciliation();
            }
            if (!gstate.namespace_mutated) {
              throw;
            }
            LanceInvalidateDatasetCache(context, cache_key);
            throw IOException(
                "Lance namespace CTAS changed external state, but failed with "
                "an unknown error (code=56)");
          }

          ReleaseLanceStorageLease(gstate.mutation_lease,
                                   "Lance namespace CTAS");

          return SinkFinalizeType::READY;
        }

        class SourceState : public GlobalSourceState {
        public:
          bool emitted = false;
        };

        unique_ptr<GlobalSourceState>
        GetGlobalSourceState(ClientContext &) const override {
          return make_uniq<SourceState>();
        }

        SourceResultType
        GetDataInternal(ExecutionContext &, DataChunk &chunk,
                        OperatorSourceInput &input) const override {
          auto &state = input.global_state.Cast<SourceState>();
          if (state.emitted) {
            return SourceResultType::FINISHED;
          }
          state.emitted = true;

          auto &gstate = sink_state->Cast<GlobalState>();
          try {
            chunk.SetCardinality(1);
            chunk.SetValue(0, 0, Value::BIGINT(gstate.insert_count));
          } catch (const std::exception &error) {
            if (!gstate.write_committed) {
              throw;
            }
            throw IOException(
                "Lance namespace CTAS committed, but constructing its SQL "
                "result failed: " +
                string(error.what()) + " (code=56)");
          } catch (...) {
            if (!gstate.write_committed) {
              throw;
            }
            throw IOException(
                "Lance namespace CTAS committed, but constructing its SQL "
                "result failed with an unknown error (code=56)");
          }
          return SourceResultType::FINISHED;
        }

        string GetName() const override { return "LanceCreateTableAs"; }

      private:
        string endpoint;
        string namespace_id;
        string delimiter;
        string bearer_token_override;
        string api_key_override;
        string headers_tsv;
        string table_id;
        string writer_mode;
        string data_storage_version;
        vector<string> column_names;
        vector<LogicalType> column_types;
      };

      // Use LIST TABLES to implement conflict behavior in a side-effect-free
      // way.
      unordered_map<string, Value> overrides;
      if (!rest_ns->bearer_token_override.empty()) {
        overrides["bearer_token"] = Value(rest_ns->bearer_token_override);
      }
      if (!rest_ns->api_key_override.empty()) {
        overrides["api_key"] = Value(rest_ns->api_key_override);
      }
      string bearer_token;
      string api_key;
      ResolveLanceNamespaceAuth(context, rest_ns->endpoint, overrides,
                                bearer_token, api_key);

      vector<string> discovered;
      string list_error;
      if (!TryLanceNamespaceListTables(
              context, rest_ns->endpoint, rest_ns->namespace_id, bearer_token,
              api_key, rest_ns->delimiter, rest_ns->headers_tsv, discovered,
              list_error)) {
        throw IOException("Failed to list tables from Lance namespace: " +
                          (list_error.empty() ? "unknown error" : list_error));
      }

      string existing_id;
      auto exists = ResolveRestNamespaceTableId(
          discovered, rest_ns->namespace_id, rest_ns->delimiter,
          create_info.table, existing_id);

      if (create_info.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT &&
          exists) {
        return planner.Make<PhysicalEmptyResult>(op.types,
                                                 op.estimated_cardinality);
      }
      if (create_info.on_conflict == OnCreateConflict::ERROR_ON_CONFLICT &&
          exists) {
        throw IOException("Lance table already exists: " + existing_id);
      }

      auto names = create_info.columns.GetColumnNames();
      auto types = create_info.columns.GetColumnTypes();
      string mode = CreateTableModeFromConflict(create_info.on_conflict);
      auto table_id_for_ops =
          exists ? existing_id
                 : BuildRestNamespaceTableId(rest_ns->namespace_id,
                                             rest_ns->delimiter,
                                             create_info.table);
      auto &create_as = planner.Make<PhysicalLanceCreateTableAs>(
          op.types, rest_ns->endpoint, rest_ns->namespace_id,
          rest_ns->delimiter, rest_ns->bearer_token_override,
          rest_ns->api_key_override, rest_ns->headers_tsv, table_id_for_ops,
          mode, data_storage_version, std::move(names), std::move(types),
          op.estimated_cardinality);
      create_as.children.push_back(plan);
      return create_as;
    }

    if (!IsSafeDatasetTableName(create_info.table)) {
      throw InvalidInputException(
          "Unsafe Lance dataset name for CREATE TABLE: " + create_info.table);
    }
    if (!directory_ns || directory_ns->root.empty()) {
      throw InternalException("Lance directory namespace root is empty");
    }

    string existing_name;
    auto exists = ResolveDirectoryNamespaceTableName(
        *directory_ns, create_info.table, existing_name);
    auto table_name_for_ops = exists ? existing_name : create_info.table;
    auto dataset_path = JoinNamespacePath(
        directory_ns->root, GetDatasetDirName(table_name_for_ops));

    if (create_info.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT &&
        exists) {
      return planner.Make<PhysicalEmptyResult>(op.types,
                                               op.estimated_cardinality);
    }
    if (create_info.on_conflict == OnCreateConflict::ERROR_ON_CONFLICT &&
        exists) {
      throw IOException("Lance dataset already exists: " + dataset_path);
    }

    auto mode = CreateTableModeFromConflict(create_info.on_conflict);

    CopyInfo copy_info;
    copy_info.is_from = false;
    copy_info.format = "lance";
    copy_info.file_path = dataset_path;
    copy_info.options["mode"] = {Value(mode)};
    if (!data_storage_version.empty()) {
      copy_info.options["data_storage_version"] = {Value(data_storage_version)};
    }

    auto &system_catalog = Catalog::GetSystemCatalog(context);
    auto entry = system_catalog.GetEntry(
        context, CatalogType::COPY_FUNCTION_ENTRY, DEFAULT_SCHEMA, "lance",
        OnEntryNotFound::THROW_EXCEPTION);
    auto &copy_function = entry->Cast<CopyFunctionCatalogEntry>().function;

    if (!copy_function.copy_to_bind) {
      throw NotImplementedException(
          "COPY TO is not supported for FORMAT \"lance\"");
    }

    auto names = create_info.columns.GetColumnNames();
    auto types = create_info.columns.GetColumnTypes();
    CopyFunctionBindInput bind_input(copy_info);
    auto bind_data =
        copy_function.copy_to_bind(context, bind_input, names, types);

    // CTAS uses the same Lance writer root as COPY ... FORMAT LANCE.  This is
    // important for Vane: the root exposes the callback provider in the
    // distributed build, while official DuckDB simply executes its native
    // sink in-process.  Keeping the target and bound data in Lance's own
    // operator avoids turning CTAS into a generic file-artifact write (which
    // cannot represent a Lance transaction).
    return PlanLanceWriteFromBoundData(planner, plan, std::move(dataset_path),
                                       std::move(bind_data), op.types,
                                       op.estimated_cardinality);
  }

  PhysicalOperator &PlanDelete(ClientContext &context,
                               PhysicalPlanGenerator &planner,
                               LogicalDelete &op) override {
    if (dynamic_cast<LanceTableEntry *>(&op.table)) {
      return PlanLanceDelete(context, planner, op);
    }
    return Catalog::PlanDelete(context, planner, op);
  }

  void ReplaceDefaultSchemaWithLanceSchema(CatalogTransaction transaction) {
    auto &schemas = GetSchemaCatalogSet();
    (void)schemas.DropEntry(transaction, DEFAULT_SCHEMA, true, true);

    CreateSchemaInfo info;
    info.schema = DEFAULT_SCHEMA;
    info.internal = true;
    info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;

    LogicalDependencyList dependencies;
    auto entry =
        make_uniq<LanceSchemaEntry>(*this, info, directory_ns, rest_ns);
    if (!schemas.CreateEntry(transaction, info.schema, std::move(entry),
                             dependencies)) {
      throw InternalException("Failed to replace Lance schema entry");
    }
  }

private:
  shared_ptr<LanceDirectoryNamespaceConfig> directory_ns;
  shared_ptr<LanceRestNamespaceConfig> rest_ns;
};

PhysicalOperator &LanceDuckCatalog::PlanDelete(ClientContext &context,
                                               PhysicalPlanGenerator &planner,
                                               LogicalDelete &op,
                                               PhysicalOperator &plan) {
  auto *lance_table = dynamic_cast<LanceTableEntry *>(&op.table);
  if (!lance_table) {
    return DuckCatalog::PlanDelete(context, planner, op, plan);
  }
  (void)plan;
  return PlanLanceDelete(context, planner, op);
}

static unique_ptr<Catalog>
LanceStorageAttach(optional_ptr<StorageExtensionInfo>, ClientContext &context,
                   AttachedDatabase &db, const string &name, AttachInfo &info,
                   AttachOptions &attach_options) {
  // AttachedDatabase records the requested access mode before this callback.
  // Its Lance catalog is backed by an in-memory DuckDB storage manager, which
  // cannot itself start read-only. Keep the database entry read-only while
  // allowing only that internal backing store to initialize read-write.
  if (attach_options.access_mode == AccessMode::READ_ONLY) {
    attach_options.access_mode = AccessMode::READ_WRITE;
  }

  // Consume Lance-specific options from attach_options.options so that
  // DuckDB doesn't complain about unrecognized options when creating storage.
  attach_options.options.erase("endpoint");
  attach_options.options.erase("delimiter");
  attach_options.options.erase("header");
  attach_options.options.erase("bearer_token");
  attach_options.options.erase("api_key");

  auto attach_path = info.path;
  auto endpoint = GetLanceNamespaceEndpoint(info);
  auto delimiter = GetLanceNamespaceDelimiter(info);
  auto headers_tsv = GetLanceNamespaceHeaders(info);

  unique_ptr<DefaultGenerator> generator;
  shared_ptr<LanceDirectoryNamespaceConfig> directory_ns;
  shared_ptr<LanceRestNamespaceConfig> rest_ns;

  auto is_rest_namespace = !endpoint.empty();
  string namespace_id;
  string bearer_token;
  string api_key;
  string bearer_token_override;
  string api_key_override;

  if (!is_rest_namespace) {
    auto root = FileSystem::GetFileSystem(context).ExpandPath(attach_path);
    vector<string> option_keys;
    vector<string> option_values;
    string open_root;
    ResolveLanceStorageOptions(context, root, open_root, option_keys,
                               option_values);

    string list_error;
    vector<string> discovered_tables;
    // Validate the namespace during ATTACH.
    if (!TryLanceDirNamespaceListTables(context, open_root, discovered_tables,
                                        list_error)) {
      throw IOException(
          "Failed to list tables from Lance directory namespace: " +
          list_error);
    }
    ValidateUniqueCaseInsensitiveNames(
        discovered_tables, "directory namespace '" + open_root + "'");
    directory_ns = make_shared_ptr<LanceDirectoryNamespaceConfig>();
    directory_ns->root = std::move(open_root);
    directory_ns->option_keys = std::move(option_keys);
    directory_ns->option_values = std::move(option_values);
  } else {
    namespace_id = attach_path;
    if (namespace_id.empty()) {
      throw InvalidInputException(
          "ATTACH TYPE LANCE with ENDPOINT requires a non-empty namespace id");
    }
    ResolveLanceNamespaceAuth(context, endpoint, info.options, bearer_token,
                              api_key);
    ResolveLanceNamespaceAuthOverrides(info.options, bearer_token_override,
                                       api_key_override);
    string list_error;
    vector<string> discovered_tables;
    // Validate the namespace during ATTACH.
    if (!TryLanceNamespaceListTables(
            context, endpoint, namespace_id, bearer_token, api_key, delimiter,
            headers_tsv, discovered_tables, list_error)) {
      throw IOException("Failed to list tables from Lance namespace: " +
                        list_error);
    }
    for (auto &table : discovered_tables) {
      table = RestNamespaceLeafName(namespace_id, delimiter, table);
    }
    ValidateUniqueCaseInsensitiveNames(discovered_tables,
                                       "REST namespace '" + namespace_id + "'");

    rest_ns = make_shared_ptr<LanceRestNamespaceConfig>();
    rest_ns->endpoint = endpoint;
    rest_ns->namespace_id = namespace_id;
    rest_ns->delimiter = delimiter;
    rest_ns->bearer_token_override = bearer_token_override;
    rest_ns->api_key_override = api_key_override;
    rest_ns->headers_tsv = headers_tsv;
  }

  // Back the attached catalog by an in-memory DuckCatalog that lazily
  // materializes per-table entries mapping to internal scan / namespace scan,
  // scan, and supports CREATE TABLE for directory namespaces.
  info.path = ":memory:";
  auto catalog = make_uniq<LanceDuckCatalog>(db, directory_ns, rest_ns);
  catalog->Initialize(false);

  auto system_transaction =
      CatalogTransaction::GetSystemTransaction(db.GetDatabase());
  catalog->ReplaceDefaultSchemaWithLanceSchema(system_transaction);
  auto &schema = catalog->GetSchema(system_transaction, DEFAULT_SCHEMA);

  auto &lance_schema = schema.Cast<LanceSchemaEntry>();
  auto &duck_schema = schema.Cast<DuckSchemaEntry>();
  auto &catalog_set = duck_schema.GetCatalogSet(CatalogType::TABLE_ENTRY);

  if (!is_rest_namespace) {
    generator = make_uniq<LanceDirectoryDefaultGenerator>(*catalog, schema,
                                                          directory_ns);
  } else {
    generator = make_uniq<LanceRestNamespaceDefaultGenerator>(
        *catalog, schema, endpoint, namespace_id, std::move(bearer_token),
        std::move(api_key), delimiter, bearer_token_override, api_key_override,
        headers_tsv);
  }
  auto *generator_ptr = generator.get();
  catalog_set.SetDefaultGenerator(std::move(generator));
  lance_schema.SetTableDefaultGenerator(generator_ptr);

  (void)name;
  return std::move(catalog);
}

struct LancePendingAppend {
  LancePendingAppend() = default;
  LancePendingAppend(const LancePendingAppend &) = delete;
  LancePendingAppend &operator=(const LancePendingAppend &) = delete;

  LancePendingAppend(LancePendingAppend &&other) noexcept
      : path(std::move(other.path)), option_keys(std::move(other.option_keys)),
        option_values(std::move(other.option_values)),
        cache_key(std::move(other.cache_key)),
        mutation_lease(std::move(other.mutation_lease)),
        transaction(other.transaction) {
    other.transaction = nullptr;
  }

  LancePendingAppend &operator=(LancePendingAppend &&other) = delete;

  ~LancePendingAppend() {
    if (transaction) {
      lance_free_transaction(transaction);
    }
  }

  string path;
  vector<string> option_keys;
  vector<string> option_values;
  string cache_key;
  unique_ptr<LanceLease> mutation_lease;
  void *transaction = nullptr;

  //! Preserve a remote lease when the native commit outcome is unknown. The
  //! record must be reconciled by an owner that can prove the operation's
  //! final state; an RAII destructor must not turn this into an apparent
  //! successful release.
  void RetainMutationLeaseForReconciliation() noexcept {
    if (mutation_lease) {
      mutation_lease->RetainForReconciliation();
    }
  }
};

static string ReleaseLanceMutationLease(LancePendingAppend &pending) noexcept {
  if (!pending.mutation_lease) {
    return "";
  }
  try {
    pending.mutation_lease->Release();
    pending.mutation_lease.reset();
    return "";
  } catch (const std::exception &error) {
    pending.RetainMutationLeaseForReconciliation();
    try {
      return "mutation lease release failed: " + string(error.what());
    } catch (...) {
      return "mutation lease release failed";
    }
  } catch (...) {
    pending.RetainMutationLeaseForReconciliation();
    return "mutation lease release failed: unknown error";
  }
}

static string AbortLancePendingAppend(ClientContext *context,
                                      LancePendingAppend &pending) {
  if (!pending.transaction) {
    // A zero-row operation can still have acquired a lease before discovering
    // that no native transaction is needed. Release it on the definitive
    // no-op path.
    return ReleaseLanceMutationLease(pending);
  }
  try {
    vector<const char *> key_ptrs;
    vector<const char *> value_ptrs;
    BuildStorageOptionPointerArrays(pending.option_keys, pending.option_values,
                                    key_ptrs, value_ptrs);
    auto rc = lance_abort_transaction_with_storage_options(
        pending.path.c_str(), key_ptrs.empty() ? nullptr : key_ptrs.data(),
        value_ptrs.empty() ? nullptr : value_ptrs.data(),
        pending.option_keys.size(),
        context ? LanceGetSessionHandle(*context) : nullptr,
        pending.transaction);
    pending.transaction = nullptr;
    if (rc == 0) {
      return ReleaseLanceMutationLease(pending);
    }
    pending.RetainMutationLeaseForReconciliation();
    return "'" + pending.path + "'" + LanceFormatErrorSuffix() +
           "; mutation lease retained for reconciliation";
  } catch (const std::exception &error) {
    lance_free_transaction(pending.transaction);
    pending.transaction = nullptr;
    pending.RetainMutationLeaseForReconciliation();
    auto result = "'" + pending.path + "' (could not prepare orphan cleanup: " +
                  string(error.what()) + ")";
    // The transaction was not cleanly aborted, so keep the lease record. It
    // must not be released merely because this local handle is being dropped.
    return result;
  } catch (...) {
    lance_free_transaction(pending.transaction);
    pending.transaction = nullptr;
    pending.RetainMutationLeaseForReconciliation();
    return "'" + pending.path +
           "' (could not prepare orphan cleanup: unknown error)";
  }
}

// The commit/rollback hooks are called from DuckDB's transaction machinery,
// where an exception escaping the hook can leave the native Lance transaction
// ownership and the DuckDB transaction state out of sync.  Keep the diagnostic
// helpers below non-throwing so an allocation or FFI error while reporting one
// failure cannot bypass orphan cleanup for the remaining pending appends.
static string LanceCurrentExceptionMessage() noexcept {
  auto exception = std::current_exception();
  if (!exception) {
    return "unknown error";
  }
  try {
    std::rethrow_exception(exception);
  } catch (const std::exception &error) {
    try {
      return error.what();
    } catch (...) {
      return "unknown error";
    }
  } catch (...) {
    return "unknown error";
  }
}

static void AppendLanceFailureText(string &message,
                                   const string &suffix) noexcept {
  if (suffix.empty()) {
    return;
  }
  try {
    if (!message.empty()) {
      message += "; ";
    }
    message += suffix;
  } catch (...) {
    // The error still carries the transaction-outcome marker at its call
    // site.  Do not let a diagnostic allocation failure skip cleanup.
  }
}

static void AppendLanceExceptionText(string &message,
                                     const char *prefix) noexcept {
  try {
    auto detail = LanceCurrentExceptionMessage();
    auto suffix = string(prefix) + detail;
    AppendLanceFailureText(message, suffix);
  } catch (...) {
    AppendLanceFailureText(message, prefix);
  }
}

static string CollectLanceAbortErrors(ClientContext *context,
                                      vector<LancePendingAppend> &appends,
                                      idx_t first) noexcept {
  string result;
  bool unreported_error = false;
  for (idx_t idx = first; idx < appends.size(); idx++) {
    try {
      auto error = AbortLancePendingAppend(context, appends[idx]);
      if (!error.empty()) {
        AppendLanceFailureText(result, error);
      }
    } catch (...) {
      unreported_error = true;
    }
  }
  if (unreported_error) {
    AppendLanceFailureText(result, "one or more Lance orphan cleanups failed");
  }
  return result;
}

static LanceLastError ConsumeLanceCommitErrorNoThrow() noexcept {
  try {
    return LanceConsumeLastErrorDetail();
  } catch (...) {
    LanceLastError result;
    // If the native call returned an error but its diagnostic could not be
    // copied, the commit outcome must be treated as unknown.  This value is
    // intentionally stronger than a fabricated definitive error.
    result.code = 55;
    return result;
  }
}

// Error codes are part of the Rust/C++ FFI contract.  A non-zero return with
// an absent, stale, or namespace-only code cannot safely be interpreted as a
// definitive dataset rejection: the native commit may already have reached
// the object store.  Keep this allow-list deliberately closed so a newly
// introduced code is fail-closed until its commit semantics are reviewed.
static bool IsKnownLanceDatasetCommitErrorCode(int32_t code) noexcept {
  switch (code) {
  case 1:  // InvalidArgument
  case 2:  // Utf8
  case 3:  // Runtime before the commit request is issued
  case 25: // DatasetCommitTransaction
  case 55: // DatasetCommitOutcomeUnknown
    return true;
  default:
    return false;
  }
}

class LanceTransactionManager final : public DuckTransactionManager {
public:
  explicit LanceTransactionManager(AttachedDatabase &db)
      : DuckTransactionManager(db) {}

  void RegisterPendingAppend(ClientContext &context, Transaction &transaction_p,
                             LancePendingAppend pending) {
    try {
      auto &transaction = transaction_p.Cast<DuckTransaction>();
      {
        lock_guard<mutex> guard(pending_lock);
        auto &appends = pending_appends[transaction.transaction_id];
        if (appends.empty()) {
          appends.push_back(std::move(pending));
          return;
        }
      }
    } catch (...) {
      auto registration_error = std::current_exception();
      auto cleanup_error = AbortLancePendingAppend(&context, pending);
      if (!cleanup_error.empty()) {
        throw IOException(
            "Failed to register a pending Lance mutation and its cleanup "
            "also failed for " +
            cleanup_error + "; cleanup is incomplete (code=55)");
      }
      std::rethrow_exception(registration_error);
    }

    // RequireLanceMutationSlot rejects this before any files are written in all
    // normal plans. Keep this registration check as a race/invariant guard, but
    // abort rather than merely freeing the transaction so an unexpected second
    // writer cannot leak its unreferenced data files.
    auto cleanup_error = AbortLancePendingAppend(&context, pending);
    string message =
        "A DuckDB transaction may contain at most one Lance mutation; commit "
        "or roll back the current mutation before starting another";
    if (!cleanup_error.empty()) {
      message += "; rejected mutation cleanup failed for " + cleanup_error +
                 "; cleanup is incomplete (code=55)";
    }
    throw TransactionException(message);
  }

  bool HasPendingAppend(Transaction &transaction_p) {
    auto &transaction = transaction_p.Cast<DuckTransaction>();
    lock_guard<mutex> guard(pending_lock);
    auto it = pending_appends.find(transaction.transaction_id);
    return it != pending_appends.end() && !it->second.empty();
  }

  ErrorData CommitTransaction(ClientContext &context,
                              Transaction &transaction_p) override {
    auto &transaction = transaction_p.Cast<DuckTransaction>();
    vector<LancePendingAppend> appends;
    {
      lock_guard<mutex> guard(pending_lock);
      auto it = pending_appends.find(transaction.transaction_id);
      if (it != pending_appends.end()) {
        appends = std::move(it->second);
        pending_appends.erase(it);
      }
    }

    // Keep the normal no-Lance path identical to DuckDB.  In particular, do
    // not convert a DuckDB-only commit exception into a Lance outcome marker.
    if (appends.empty()) {
      return DuckTransactionManager::CommitTransaction(context, transaction_p);
    }

    bool duck_rollback_failed = false;
    auto rollback_duck_transaction = [&](string &message) noexcept {
      try {
        DuckTransactionManager::RollbackTransaction(transaction_p);
      } catch (...) {
        duck_rollback_failed = true;
        AppendLanceExceptionText(message, "DuckDB rollback failed: ");
      }
    };

    for (idx_t append_idx = 0; append_idx < appends.size(); append_idx++) {
      auto &pending = appends[append_idx];
      bool native_commit_returned = false;
      try {
        vector<const char *> key_ptrs;
        vector<const char *> value_ptrs;
        BuildStorageOptionPointerArrays(
            pending.option_keys, pending.option_values, key_ptrs, value_ptrs);

        auto *session = LanceGetSessionHandle(context);
        // Ownership transfers to the C ABI at the call boundary, including
        // error/exception paths.  Clear our owner before invoking it so an
        // unexpected C++ exception cannot make the cleanup path double-free a
        // consumed transaction.
        auto *native_transaction = pending.transaction;
        pending.transaction = nullptr;
        native_commit_returned = true;
        auto rc = lance_commit_transaction_with_storage_options(
            pending.path.c_str(), key_ptrs.empty() ? nullptr : key_ptrs.data(),
            value_ptrs.empty() ? nullptr : value_ptrs.data(),
            pending.option_keys.size(), session, native_transaction);

        if (rc != 0) {
          // The native transaction handle was consumed by the commit call.
          // Even a typed error can be returned after the write request reached
          // the object store, so keep the remote mutation lease until an
          // operator/coordinator reconciles the outcome.
          pending.RetainMutationLeaseForReconciliation();
          // Capture the primary commit error before any cleanup FFI call can
          // overwrite the thread-local last-error slot.
          auto commit_error = ConsumeLanceCommitErrorNoThrow();
          if (!IsKnownLanceDatasetCommitErrorCode(commit_error.code)) {
            auto raw_code = commit_error.code;
            commit_error.code = 55;
            AppendLanceFailureText(
                commit_error.message,
                "native Lance commit returned an unrecognized error code " +
                    to_string(raw_code) +
                    "; treating the commit outcome as unknown");
          }
          string commit_error_text;
          try {
            commit_error_text = commit_error.ToString();
          } catch (...) {
            commit_error.code = 55;
          }

          string message = "Failed to commit Lance append transaction for '" +
                           pending.path + "'";
          if (!commit_error_text.empty()) {
            try {
              AppendLanceFailureText(message,
                                     "Lance error: " + commit_error_text);
            } catch (...) {
              AppendLanceFailureText(message, "Lance error unavailable");
              commit_error.code = 55;
            }
          }

          // The attempted transaction was consumed by the commit call. Abort
          // every transaction that was not attempted yet so its unreferenced
          // data and deletion files do not accumulate forever.
          auto abort_errors =
              CollectLanceAbortErrors(&context, appends, append_idx + 1);
          if (!abort_errors.empty()) {
            AppendLanceFailureText(message,
                                   "abort cleanup failed: " + abort_errors);
          }

          // Earlier native commits in this DuckDB transaction are already
          // durable. Retain every lease that covers an attempted mutation;
          // dropping it here would make an outcome-unknown retry unsafe.
          for (idx_t retained_idx = 0; retained_idx <= append_idx;
               retained_idx++) {
            appends[retained_idx].RetainMutationLeaseForReconciliation();
          }

          bool cache_invalidation_failed = false;
          try {
            LanceInvalidateDatasetCache(context, pending.cache_key);
          } catch (...) {
            cache_invalidation_failed = true;
            AppendLanceExceptionText(message,
                                     "dataset cache invalidation failed: ");
          }
          rollback_duck_transaction(message);

          if (commit_error.code == 55 || append_idx > 0 ||
              !abort_errors.empty() || cache_invalidation_failed ||
              duck_rollback_failed) {
            AppendLanceFailureText(
                message, "commit outcome or cleanup is unresolved (code=55)");
          }
          return ErrorData(ExceptionType::TRANSACTION, message);
        }

        // A successful native commit followed by a cache failure is still an
        // ambiguous SQL outcome: the dataset is durable, but a retry could
        // duplicate the mutation while this connection holds stale state.
        try {
          LanceInvalidateDatasetCache(context, pending.cache_key);
        } catch (...) {
          for (idx_t retained_idx = 0; retained_idx <= append_idx;
               retained_idx++) {
            appends[retained_idx].RetainMutationLeaseForReconciliation();
          }
          string message = "Lance append transaction for '" + pending.path +
                           "' committed, but dataset cache invalidation "
                           "failed: ";
          AppendLanceExceptionText(message, "");
          auto abort_errors =
              CollectLanceAbortErrors(&context, appends, append_idx + 1);
          if (!abort_errors.empty()) {
            AppendLanceFailureText(message,
                                   "abort cleanup failed: " + abort_errors);
          }
          rollback_duck_transaction(message);
          AppendLanceFailureText(
              message, "dataset commit outcome is unresolved; do not retry "
                       "automatically (code=55)");
          return ErrorData(ExceptionType::TRANSACTION, message);
        }

      } catch (...) {
        // This covers C++ preparation/session exceptions before the FFI call,
        // and also protects the hook if a future FFI wrapper unexpectedly
        // throws.  If the call returned, the pointer was cleared above and the
        // native outcome is already accounted for; otherwise abort the current
        // and all later pending transactions.
        string message = "Failed to prepare Lance append transaction for '" +
                         pending.path + "': ";
        AppendLanceExceptionText(message, "");
        auto cleanup_start =
            native_commit_returned ? append_idx + 1 : append_idx;
        if (native_commit_returned) {
          for (idx_t retained_idx = 0; retained_idx <= append_idx;
               retained_idx++) {
            appends[retained_idx].RetainMutationLeaseForReconciliation();
          }
        }
        auto abort_errors =
            CollectLanceAbortErrors(&context, appends, cleanup_start);
        if (!abort_errors.empty()) {
          AppendLanceFailureText(message,
                                 "abort cleanup failed: " + abort_errors);
        }
        rollback_duck_transaction(message);
        // Any earlier Lance mutation in this DuckDB transaction is already
        // durable, even when preparation of this later mutation failed before
        // its FFI call.  Report the whole SQL outcome as ambiguous so callers
        // cannot retry and duplicate the earlier commit.
        if (append_idx > 0 || native_commit_returned || !abort_errors.empty()) {
          AppendLanceFailureText(
              message, "commit outcome or cleanup is unresolved (code=55)");
        }
        return ErrorData(ExceptionType::TRANSACTION, message);
      }
    }

    try {
      auto result =
          DuckTransactionManager::CommitTransaction(context, transaction_p);
      if (result.HasError()) {
        // The dataset mutation succeeded, but the SQL transaction did not
        // finish cleanly.  Surface the same terminal marker as an ambiguous
        // native commit so callers retain coordination leases and never retry
        // the write automatically.
        for (auto &pending : appends) {
          pending.RetainMutationLeaseForReconciliation();
        }
        return ErrorData(
            result.Type(),
            result.RawMessage() +
                "; Lance dataset commit succeeded before DuckDB transaction "
                "finalization failed; do not retry automatically (code=55)");
      }
      string lease_errors;
      for (auto &pending : appends) {
        AppendLanceFailureText(lease_errors,
                               ReleaseLanceMutationLease(pending));
      }
      if (!lease_errors.empty()) {
        return ErrorData(
            ExceptionType::TRANSACTION,
            "DuckDB transaction finalized after Lance dataset commit, but " +
                lease_errors +
                "; lease records were retained for reconciliation; do not "
                "retry automatically (code=55)");
      }
      return result;
    } catch (...) {
      // DuckDB may throw after writing the WAL or removing the transaction
      // from its active set (for example during checkpoint/cleanup).  At this
      // point all Lance mutations above are already durable and cannot be
      // rolled back, so returning an explicit unknown marker is safer than
      // propagating a retryable exception.
      string message = "Lance dataset commit succeeded, but DuckDB transaction "
                       "finalization raised an exception: ";
      for (auto &pending : appends) {
        pending.RetainMutationLeaseForReconciliation();
      }
      AppendLanceExceptionText(message, "");
      AppendLanceFailureText(message, "do not retry automatically (code=55)");
      return ErrorData(ExceptionType::TRANSACTION, message);
    }
  }

  void RollbackTransaction(Transaction &transaction_p) override {
    auto &transaction = transaction_p.Cast<DuckTransaction>();
    vector<LancePendingAppend> appends;
    {
      lock_guard<mutex> guard(pending_lock);
      auto it = pending_appends.find(transaction.transaction_id);
      if (it != pending_appends.end()) {
        appends = std::move(it->second);
        pending_appends.erase(it);
      }
    }
    vector<string> abort_errors;
    for (auto &pending : appends) {
      auto cleanup_error = AbortLancePendingAppend(nullptr, pending);
      if (!cleanup_error.empty()) {
        abort_errors.push_back(std::move(cleanup_error));
      }
    }
    DuckTransactionManager::RollbackTransaction(transaction_p);
    if (!abort_errors.empty()) {
      throw IOException(
          "Failed to clean up rolled-back Lance transaction(s): " +
          StringUtil::Join(abort_errors, "; ") +
          "; cleanup is incomplete (code=55)");
    }
  }

private:
  mutex pending_lock;
  unordered_map<transaction_t, vector<LancePendingAppend>> pending_appends;
};

static unique_ptr<TransactionManager>
LanceStorageTransactionManager(optional_ptr<StorageExtensionInfo>,
                               AttachedDatabase &db, Catalog &) {
  return make_uniq<LanceTransactionManager>(db);
}

void RegisterLanceStorage(DBConfig &config) {
  auto ext = make_uniq<StorageExtension>();
  ext->attach = LanceStorageAttach;
  ext->create_transaction_manager = LanceStorageTransactionManager;
  StorageExtension::Register(config, "lance", std::move(ext));
}

void RegisterLancePendingAppend(ClientContext &context, Catalog &catalog,
                                string dataset_uri, vector<string> option_keys,
                                vector<string> option_values, string cache_key,
                                void *lance_transaction,
                                unique_ptr<LanceLease> mutation_lease) {
  LancePendingAppend pending;
  pending.path = std::move(dataset_uri);
  pending.option_keys = std::move(option_keys);
  pending.option_values = std::move(option_values);
  pending.cache_key = std::move(cache_key);
  pending.mutation_lease = std::move(mutation_lease);
  pending.transaction = lance_transaction;

  try {
    auto &txn = Transaction::Get(context, catalog);
    auto *tm = dynamic_cast<LanceTransactionManager *>(&txn.manager);
    if (!tm) {
      throw InternalException(
          "RegisterLancePendingAppend requires LanceTransactionManager");
    }
    tm->RegisterPendingAppend(context, txn, std::move(pending));
  } catch (...) {
    auto registration_error = std::current_exception();
    auto cleanup_error = AbortLancePendingAppend(&context, pending);
    if (!cleanup_error.empty()) {
      throw IOException(
          "Failed to register a pending Lance mutation and its cleanup also "
          "failed for " +
          cleanup_error + "; cleanup is incomplete (code=55)");
    }
    std::rethrow_exception(registration_error);
  }
}

void RequireLanceMutationSlot(ClientContext &context, Catalog &catalog) {
  if (context.transaction.IsAutoCommit()) {
    return;
  }
  auto &txn = Transaction::Get(context, catalog);
  auto *tm = dynamic_cast<LanceTransactionManager *>(&txn.manager);
  if (!tm) {
    throw InternalException(
        "RequireLanceMutationSlot requires LanceTransactionManager");
  }
  if (tm->HasPendingAppend(txn)) {
    throw TransactionException(
        "A DuckDB transaction may contain at most one Lance mutation; commit "
        "or roll back the current mutation before starting another");
  }
}

} // namespace duckdb
