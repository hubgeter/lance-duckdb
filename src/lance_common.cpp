#include "lance_common.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_transaction.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "lance_arrow_compat.hpp"
#include "lance_ffi.hpp"
#include "lance_session_state.hpp"
#include "lance_table_entry.hpp"

#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace duckdb {

namespace {
class ScopedLanceString {
public:
  explicit ScopedLanceString(const char *value_p) : value(value_p) {}
  ScopedLanceString(const ScopedLanceString &) = delete;
  ScopedLanceString &operator=(const ScopedLanceString &) = delete;

  ~ScopedLanceString() {
    if (value) {
      lance_free_string(value);
    }
  }

  const char *Get() const { return value; }
  explicit operator bool() const { return value != nullptr; }

private:
  const char *value;
};
} // namespace

// Namespace mutation wrappers must distinguish deterministic validation/API
// failures from a lost response. Keep this allow-list closed: an unexpected
// native code may mean the request reached the catalog, so it must be treated
// as an outcome-unknown mutation rather than as a safe retry.
static bool IsKnownLanceNamespaceMutationErrorCode(int32_t code) {
  switch (code) {
  case 1:  // InvalidArgument
  case 2:  // Utf8
  case 3:  // Runtime (before the request was issued)
  case 50: // NamespaceCreateEmptyTable
  case 51: // NamespaceDropTable
  case 56: // NamespaceMutationOutcomeUnknown
    return true;
  default:
    return false;
  }
}

string LanceLastError::ToString() const {
  if (code == 0 && message.empty()) {
    return "";
  }
  if (message.empty()) {
    return "code=" + to_string(code);
  }
  if (code == 0) {
    return message;
  }
  return message + " (code=" + to_string(code) + ")";
}

string LanceBuildLogicalWriteTargetIdentity(const string &dataset_identity,
                                            void *dataset) {
  if (dataset_identity.empty()) {
    throw InvalidInputException(
        "Lance logical write target identity requires a dataset identity");
  }
  if (!dataset) {
    throw InvalidInputException(
        "Lance logical write target identity requires an open dataset");
  }
  auto *generation_ptr = lance_dataset_generation_id(dataset);
  if (!generation_ptr) {
    throw IOException("Failed to identify Lance dataset generation" +
                      LanceFormatErrorSuffix());
  }
  string generation;
  try {
    generation = generation_ptr;
  } catch (...) {
    lance_free_string(generation_ptr);
    throw;
  }
  lance_free_string(generation_ptr);
  if (generation.empty()) {
    throw IOException("Lance dataset generation identity is empty");
  }
  return "lance-table:v1:" + dataset_identity + ":generation:" + generation;
}

LanceLastError LanceConsumeLastErrorDetail() {
  LanceLastError result;
  result.code = lance_last_error_code();
  ScopedLanceString message(lance_last_error_message());
  if (message) {
    result.message = message.Get();
  }
  return result;
}

string LanceConsumeLastError() {
  return LanceConsumeLastErrorDetail().ToString();
}

string LanceFormatErrorSuffix() {
  auto err = LanceConsumeLastError();
  if (err.empty()) {
    return "";
  }
  return " (Lance error: " + err + ")";
}

bool IsComputedSearchColumn(const string &name) {
  return IsComputedSearchColumn(name, LanceComputedSearchColumns::Hybrid);
}

bool IsComputedSearchColumn(const string &name,
                            LanceComputedSearchColumns columns) {
  auto mask = static_cast<uint8_t>(columns);
  if (StringUtil::CIEquals(name, "_distance")) {
    return (mask &
            static_cast<uint8_t>(LanceComputedSearchColumns::Distance)) != 0;
  }
  if (StringUtil::CIEquals(name, "_score")) {
    return (mask & static_cast<uint8_t>(LanceComputedSearchColumns::Score)) !=
           0;
  }
  if (StringUtil::CIEquals(name, "_hybrid_score")) {
    return (mask &
            static_cast<uint8_t>(LanceComputedSearchColumns::HybridScore)) != 0;
  }
  return false;
}

static string NormalizeLanceArrowColumnName(const string &name) {
  if (name.size() < 2 || name.front() != '`' || name.back() != '`') {
    return name;
  }

  string result;
  result.reserve(name.size() - 2);
  for (idx_t i = 1; i + 1 < name.size(); i++) {
    if (name[i] != '`') {
      result.push_back(name[i]);
      continue;
    }
    if (i + 2 < name.size() && name[i + 1] == '`') {
      result.push_back('`');
      i++;
      continue;
    }
    // Only a single quoted top-level field is normalized. Nested field paths
    // such as `parent`.`child` retain their full path identity.
    return name;
  }
  return result;
}

void LanceValidateAndReorderArrowBatch(
    ClientContext &context, ArrowSchema &schema, ArrowArray &array,
    const vector<string> &expected_names,
    const vector<LogicalType> &expected_types, const string &source_name) {
  if (expected_names.size() != expected_types.size()) {
    throw InternalException(
        "Lance expected Arrow column names and types have different sizes");
  }
  if (schema.n_children < 0 || array.n_children < 0 ||
      schema.n_children != array.n_children) {
    throw IOException(source_name +
                      " returned inconsistent Arrow schema and array shapes");
  }

  auto child_count = NumericCast<idx_t>(schema.n_children);
  if (child_count != expected_names.size()) {
    throw IOException(source_name + " returned " + to_string(child_count) +
                      " columns, but " + to_string(expected_names.size()) +
                      " were expected");
  }
  if (child_count > 0 && (!schema.children || !array.children)) {
    throw IOException(source_name + " returned null Arrow child pointers");
  }

  unordered_set<string> unique_expected_names;
  unordered_set<string> unique_actual_names;
  unique_expected_names.reserve(child_count);
  unique_actual_names.reserve(child_count);
  for (idx_t i = 0; i < child_count; i++) {
    if (!schema.children[i] || !schema.children[i]->name ||
        !array.children[i]) {
      throw IOException(source_name +
                        " returned a null Arrow child schema or array");
    }
    if (!unique_expected_names.insert(expected_names[i]).second) {
      throw InternalException("Lance bound duplicate expected column: " +
                              expected_names[i]);
    }
    auto actual_name =
        NormalizeLanceArrowColumnName(string(schema.children[i]->name));
    if (!unique_actual_names.insert(actual_name).second) {
      throw IOException(source_name +
                        " returned a duplicate Arrow column: " + actual_name);
    }
  }

  // The array buffers must be widened while the schema still contains the
  // original Arrow format strings. The schema is then widened independently
  // so DuckDB can derive the actual logical types for validation.
  LanceCoerceArrowArrayForDuckDB(&schema, &array);
  LanceCoerceArrowSchemaForDuckDB(&schema);

  ArrowTableSchema actual_schema;
  ArrowTableFunction::PopulateArrowTableSchema(context, actual_schema, schema);
  auto &actual_names = actual_schema.GetNames();
  auto &actual_types = actual_schema.GetTypes();
  if (actual_names.size() != child_count ||
      actual_types.size() != child_count) {
    throw IOException(source_name +
                      " returned an invalid Arrow schema column count");
  }

  unordered_map<string, idx_t> index_by_name;
  index_by_name.reserve(child_count);
  for (idx_t i = 0; i < child_count; i++) {
    auto actual_name = NormalizeLanceArrowColumnName(actual_names[i]);
    if (!index_by_name.emplace(actual_name, i).second) {
      throw IOException(source_name +
                        " returned a duplicate Arrow column: " + actual_name);
    }
  }

  vector<ArrowArray *> reordered_children;
  vector<ArrowSchema *> reordered_schema_children;
  reordered_children.reserve(child_count);
  reordered_schema_children.reserve(child_count);
  for (idx_t i = 0; i < child_count; i++) {
    auto entry = index_by_name.find(expected_names[i]);
    if (entry == index_by_name.end()) {
      throw IOException(source_name + " did not return expected column: " +
                        expected_names[i]);
    }
    auto source_index = entry->second;
    if (actual_types[source_index] != expected_types[i]) {
      throw IOException(source_name + " returned column '" + expected_names[i] +
                        "' as " + actual_types[source_index].ToString() +
                        ", but DuckDB bound it as " +
                        expected_types[i].ToString());
    }
    reordered_schema_children.push_back(schema.children[source_index]);
    reordered_children.push_back(array.children[source_index]);
  }
  for (idx_t i = 0; i < child_count; i++) {
    schema.children[i] = reordered_schema_children[i];
    array.children[i] = reordered_children[i];
  }
}

static void BuildStringPointerArray(const vector<string> &values,
                                    vector<const char *> &out_ptrs) {
  out_ptrs.clear();
  out_ptrs.reserve(values.size());
  for (auto &value : values) {
    ValidateLanceCString(value, "Lance column name");
    out_ptrs.push_back(value.c_str());
  }
}

void FillLanceNamespaceQueryConfig(
    ClientContext &context, const LanceNamespaceTableConfig &cfg,
    uint64_t dataset_version, uint64_t k, bool prefilter, const string &filter,
    const vector<string> &columns, vector<string> &resolved_option_keys,
    vector<string> &resolved_option_values,
    vector<const char *> &option_key_ptrs,
    vector<const char *> &option_value_ptrs, vector<const char *> &column_ptrs,
    string &bearer_token, string &api_key, string &headers_tsv,
    LanceNamespaceQueryConfig &out_config) {
  static constexpr uint8_t NAMESPACE_KIND_DIRECTORY = 0;
  static constexpr uint8_t NAMESPACE_KIND_REST = 1;

  out_config = {};
  ValidateLanceCString(cfg.table_id, "Lance namespace table id");
  ValidateLanceCString(filter, "Lance namespace filter");
  out_config.table_id = cfg.table_id.c_str();
  out_config.dataset_version = dataset_version;
  out_config.k = k;
  out_config.prefilter = prefilter ? 1 : 0;
  out_config.filter = filter.empty() ? nullptr : filter.c_str();

  BuildStringPointerArray(columns, column_ptrs);
  out_config.columns = column_ptrs.empty() ? nullptr : column_ptrs.data();
  out_config.columns_len = column_ptrs.size();
  out_config.expected_columns = out_config.columns;
  out_config.expected_columns_len = out_config.columns_len;

  if (cfg.IsDirectory()) {
    ValidateLanceCString(cfg.root, "Lance directory namespace root");
    out_config.namespace_kind = NAMESPACE_KIND_DIRECTORY;
    out_config.root = cfg.root.c_str();
    resolved_option_keys = cfg.option_keys;
    resolved_option_values = cfg.option_values;
    if (resolved_option_keys.empty()) {
      LanceFillStorageOptions(context, LanceDirectoryNamespaceDatasetUri(cfg),
                              resolved_option_keys, resolved_option_values);
    }
    BuildStorageOptionPointerArrays(resolved_option_keys,
                                    resolved_option_values, option_key_ptrs,
                                    option_value_ptrs);
    out_config.option_keys =
        option_key_ptrs.empty() ? nullptr : option_key_ptrs.data();
    out_config.option_values =
        option_value_ptrs.empty() ? nullptr : option_value_ptrs.data();
    out_config.options_len = option_key_ptrs.size();
    return;
  }

  out_config.namespace_kind = NAMESPACE_KIND_REST;
  ValidateLanceCString(cfg.endpoint, "Lance namespace endpoint");
  ValidateLanceCString(cfg.delimiter, "Lance namespace delimiter");
  out_config.endpoint = cfg.endpoint.c_str();
  out_config.delimiter =
      cfg.delimiter.empty() ? nullptr : cfg.delimiter.c_str();
  ResolveLanceNamespaceTableAuth(context, cfg, bearer_token, api_key,
                                 headers_tsv);
  ValidateLanceCString(bearer_token, "Lance namespace bearer token");
  ValidateLanceCString(api_key, "Lance namespace API key");
  ValidateLanceCString(headers_tsv, "Lance namespace headers");
  out_config.headers_tsv = headers_tsv.empty() ? nullptr : headers_tsv.c_str();
  out_config.bearer_token =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  out_config.api_key = api_key.empty() ? nullptr : api_key.c_str();
}

string LanceNormalizeS3Scheme(const string &path) {
  if (StringUtil::StartsWith(path, "s3a://")) {
    return "s3://" + path.substr(6);
  }
  if (StringUtil::StartsWith(path, "s3n://")) {
    return "s3://" + path.substr(6);
  }
  return path;
}

static uint8_t DecodeLanceHexDigit(char value, const string &description) {
  if (value >= '0' && value <= '9') {
    return NumericCast<uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return NumericCast<uint8_t>(value - 'a' + 10);
  }
  if (value >= 'A' && value <= 'F') {
    return NumericCast<uint8_t>(value - 'A' + 10);
  }
  throw IOException("Invalid hexadecimal row returned for " + description);
}

static string DecodeLanceHexField(const string &value,
                                  const string &description) {
  if (value.size() % 2 != 0) {
    throw IOException("Odd-length hexadecimal row returned for " + description);
  }
  string result;
  result.reserve(value.size() / 2);
  for (idx_t i = 0; i < value.size(); i += 2) {
    auto high = DecodeLanceHexDigit(value[i], description);
    auto low = DecodeLanceHexDigit(value[i + 1], description);
    result.push_back(static_cast<char>((high << 4U) | low));
  }
  return result;
}

vector<pair<string, string>> ParseLanceKeyValueRows(const char *ptr,
                                                    const string &description) {
  if (!ptr) {
    throw IOException("Failed to read " + description +
                      LanceFormatErrorSuffix());
  }

  ScopedLanceString owned_ptr(ptr);
  string joined = owned_ptr.Get();

  vector<pair<string, string>> result;
  for (auto &line : StringUtil::Split(joined, '\n')) {
    if (line.empty()) {
      continue;
    }
    auto separator = line.find('\t');
    if (separator == string::npos ||
        line.find('\t', separator + 1) != string::npos) {
      throw IOException("Malformed hexadecimal row returned for " +
                        description);
    }
    result.emplace_back(
        DecodeLanceHexField(line.substr(0, separator), description),
        DecodeLanceHexField(line.substr(separator + 1), description));
  }
  return result;
}

void ValidateLanceCString(const string &value, const string &label) {
  if (value.find('\0') != string::npos) {
    throw InvalidInputException(label + " must not contain a NUL byte");
  }
}

string LanceNormalizeDatasetPath(ClientContext &context, const string &path) {
  auto result = LanceNormalizeS3Scheme(path);
#ifdef LANCE_VANE_DISTRIBUTED
  auto &fs = FileSystem::GetFileSystem(context);
  // DuckDB accepts local file URIs in every path-taking API, while the
  // coordinator deliberately canonicalizes them to the corresponding local
  // path.  Expand the URI before canonicalization so the native cache key,
  // Lance object-store path, and Python lease identity all agree.  Calling
  // ExpandPath for remote schemes is harmless (it returns them unchanged).
  result = fs.ExpandPath(result);
  if (result.find("://") == string::npos) {
    result = fs.CanonicalizePath(result);
  }
#else
  // Keep the official DuckDB display/open path semantics unchanged.  Vane's
  // distributed adapter needs a canonical identity for worker replay; the
  // native extension does not and should retain the caller's relative URI in
  // plans and diagnostics.
  (void)context;
#endif
  return result;
}

string LanceDirectoryNamespaceDatasetUri(const LanceNamespaceTableConfig &cfg) {
  if (!cfg.display_uri.empty()) {
    return LanceNormalizeS3Scheme(cfg.display_uri);
  }

  auto child = cfg.table_id;
  if (!StringUtil::EndsWith(child, ".lance")) {
    child += ".lance";
  }
  string uri;
  if (cfg.root.empty()) {
    uri = std::move(child);
  } else if (cfg.root.back() == '/' || cfg.root.back() == '\\') {
    uri = cfg.root + child;
  } else {
    uri = cfg.root + "/" + child;
  }
  return LanceNormalizeS3Scheme(uri);
}

static string SecretValueToString(const Value &value) {
  if (value.IsNull()) {
    return "";
  }
  return value.ToString();
}

static bool TryGetLanceS3Setting(ClientContext &context, const string &name,
                                 string &out_value) {
  Value value;
  if (!context.TryGetCurrentSetting(name, value) || value.IsNull()) {
    return false;
  }
  out_value = value.DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
  return !out_value.empty();
}

static void AppendLanceStorageOption(const string &key, string value,
                                     vector<string> &out_keys,
                                     vector<string> &out_values) {
  out_keys.push_back(key);
  out_values.push_back(std::move(value));
}

static void FillLanceStorageOptionsFromS3Settings(ClientContext &context,
                                                  const string &path,
                                                  vector<string> &out_keys,
                                                  vector<string> &out_values) {
  if (!StringUtil::StartsWith(path, "s3://")) {
    return;
  }

  string access_key_id;
  string secret_access_key;
  string session_token;
  auto has_access_key =
      TryGetLanceS3Setting(context, "s3_access_key_id", access_key_id);
  auto has_secret_key =
      TryGetLanceS3Setting(context, "s3_secret_access_key", secret_access_key);
  auto has_session_token =
      TryGetLanceS3Setting(context, "s3_session_token", session_token);
  if (has_access_key != has_secret_key ||
      (has_session_token && !has_access_key)) {
    throw InvalidInputException(
        "Lance S3 settings require s3_access_key_id and "
        "s3_secret_access_key together, and s3_session_token requires that "
        "key pair");
  }
  if (has_access_key) {
    AppendLanceStorageOption("access_key_id", std::move(access_key_id),
                             out_keys, out_values);
    AppendLanceStorageOption("secret_access_key", std::move(secret_access_key),
                             out_keys, out_values);
    if (has_session_token) {
      AppendLanceStorageOption("session_token", std::move(session_token),
                               out_keys, out_values);
    }
  }

  string region;
  if (TryGetLanceS3Setting(context, "s3_region", region)) {
    AppendLanceStorageOption("region", std::move(region), out_keys, out_values);
  }

  string endpoint;
  if (TryGetLanceS3Setting(context, "s3_endpoint", endpoint)) {
    Value use_ssl_value;
    auto use_ssl = true;
    if (context.TryGetCurrentSetting("s3_use_ssl", use_ssl_value) &&
        !use_ssl_value.IsNull()) {
      use_ssl = use_ssl_value.GetValue<bool>();
    }
    auto normalized_endpoint = StringUtil::Lower(endpoint);
    auto endpoint_is_http =
        StringUtil::StartsWith(normalized_endpoint, "http://");
    auto endpoint_is_https =
        StringUtil::StartsWith(normalized_endpoint, "https://");
    if (!endpoint_is_http && !endpoint_is_https) {
      endpoint = string(use_ssl ? "https://" : "http://") + endpoint;
      endpoint_is_http = !use_ssl;
    }
    AppendLanceStorageOption("endpoint", std::move(endpoint), out_keys,
                             out_values);
    if (endpoint_is_http) {
      AppendLanceStorageOption("allow_http", "true", out_keys, out_values);
    }
  }

  string url_style;
  if (TryGetLanceS3Setting(context, "s3_url_style", url_style)) {
    url_style = StringUtil::Lower(url_style);
    if (url_style != "path" && url_style != "vhost") {
      throw InvalidInputException(
          "Lance S3 setting s3_url_style must be 'path' or 'vhost'");
    }
    AppendLanceStorageOption("virtual_hosted_style_request",
                             url_style == "vhost" ? "true" : "false", out_keys,
                             out_values);
  }
}

static bool LanceStorageOptionsEqual(const vector<string> &left_keys,
                                     const vector<string> &left_values,
                                     const vector<string> &right_keys,
                                     const vector<string> &right_values) {
  if (left_keys.size() != left_values.size() ||
      right_keys.size() != right_values.size() ||
      left_keys.size() != right_keys.size()) {
    return false;
  }

  unordered_map<string, string> left;
  left.reserve(left_keys.size());
  for (idx_t i = 0; i < left_keys.size(); i++) {
    auto key = StringUtil::Lower(left_keys[i]);
    if (!left.emplace(std::move(key), left_values[i]).second) {
      return false;
    }
  }
  unordered_map<string, string> right;
  right.reserve(right_keys.size());
  for (idx_t i = 0; i < right_keys.size(); i++) {
    auto key = StringUtil::Lower(right_keys[i]);
    if (!right.emplace(std::move(key), right_values[i]).second) {
      return false;
    }
  }
  return left == right;
}

bool LanceStorageOptionsAreWorkerReplayable(
    ClientContext &context, const string &path,
    const vector<string> &resolved_keys,
    const vector<string> &resolved_values) {
  vector<string> setting_keys;
  vector<string> setting_values;
  FillLanceStorageOptionsFromS3Settings(context, LanceNormalizeS3Scheme(path),
                                        setting_keys, setting_values);
  return LanceStorageOptionsEqual(resolved_keys, resolved_values, setting_keys,
                                  setting_values);
}

bool LanceStorageOptionsAreWorkerReplayable(ClientContext &context,
                                            const string &path) {
  string open_path;
  vector<string> resolved_keys;
  vector<string> resolved_values;
  ResolveLanceStorageOptions(context, path, open_path, resolved_keys,
                             resolved_values);
  return LanceStorageOptionsAreWorkerReplayable(context, open_path,
                                                resolved_keys, resolved_values);
}

void LanceFillStorageOptions(ClientContext &context, const string &path,
                             vector<string> &out_keys,
                             vector<string> &out_values) {
  auto &secret_manager = SecretManager::Get(context);
  auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);
  auto secret_match = secret_manager.LookupSecret(transaction, path, "lance");
  if (!secret_match.HasMatch() || !secret_match.secret_entry ||
      !secret_match.secret_entry->secret) {
    FillLanceStorageOptionsFromS3Settings(context, path, out_keys, out_values);
    return;
  }

  auto *kv_secret = dynamic_cast<const KeyValueSecret *>(
      secret_match.secret_entry->secret.get());
  if (!kv_secret) {
    return;
  }

  for (auto &kv : kv_secret->secret_map) {
    if (kv.second.IsNull()) {
      continue;
    }
    out_keys.push_back(kv.first);
    out_values.push_back(kv.second.ToString());
  }
}

static void FillLanceNamespaceAuthFromSecrets(ClientContext &context,
                                              const string &endpoint,
                                              string &out_bearer_token,
                                              string &out_api_key) {
  out_bearer_token.clear();
  out_api_key.clear();

  auto &secret_manager = SecretManager::Get(context);
  auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);
  auto secret_match =
      secret_manager.LookupSecret(transaction, endpoint, "lance_namespace");
  if (!secret_match.HasMatch() || !secret_match.secret_entry ||
      !secret_match.secret_entry->secret) {
    return;
  }

  auto *kv_secret = dynamic_cast<const KeyValueSecret *>(
      secret_match.secret_entry->secret.get());
  if (!kv_secret) {
    return;
  }

  out_bearer_token = SecretValueToString(kv_secret->TryGetValue("token"));
  if (out_bearer_token.empty()) {
    out_bearer_token =
        SecretValueToString(kv_secret->TryGetValue("bearer_token"));
  }
  out_api_key = SecretValueToString(kv_secret->TryGetValue("api_key"));
}

static void ApplyAuthOverrideValue(const Value &value, string &out) {
  if (value.IsNull()) {
    return;
  }
  auto s = value.DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
  if (!s.empty()) {
    out = std::move(s);
  }
}

void ResolveLanceNamespaceAuth(ClientContext &context, const string &endpoint,
                               const unordered_map<string, Value> &options,
                               string &out_bearer_token, string &out_api_key) {
  FillLanceNamespaceAuthFromSecrets(context, endpoint, out_bearer_token,
                                    out_api_key);
  ResolveLanceNamespaceAuthOverrides(options, out_bearer_token, out_api_key);
}

void ResolveLanceNamespaceAuth(ClientContext &context, const string &endpoint,
                               const named_parameter_map_t &options,
                               string &out_bearer_token, string &out_api_key) {
  FillLanceNamespaceAuthFromSecrets(context, endpoint, out_bearer_token,
                                    out_api_key);
  auto token_it = options.find("token");
  if (token_it != options.end()) {
    ApplyAuthOverrideValue(token_it->second, out_bearer_token);
  }
  auto bearer_it = options.find("bearer_token");
  if (bearer_it != options.end()) {
    ApplyAuthOverrideValue(bearer_it->second, out_bearer_token);
  }
  auto key_it = options.find("api_key");
  if (key_it != options.end()) {
    ApplyAuthOverrideValue(key_it->second, out_api_key);
  }
}

void ResolveLanceNamespaceAuthOverrides(
    const unordered_map<string, Value> &options, string &out_bearer_token,
    string &out_api_key) {
  auto apply_unique = [&](const string &name, string &out) {
    const Value *match = nullptr;
    for (auto &kv : options) {
      if (!StringUtil::CIEquals(kv.first, name)) {
        continue;
      }
      if (match) {
        throw InvalidInputException("Duplicate Lance namespace option: " +
                                    name);
      }
      match = &kv.second;
    }
    if (match) {
      ApplyAuthOverrideValue(*match, out);
    }
  };

  // bearer_token is the canonical spelling and intentionally takes
  // precedence over the legacy token alias, independent of unordered_map
  // iteration order.
  apply_unique("token", out_bearer_token);
  apply_unique("bearer_token", out_bearer_token);
  apply_unique("api_key", out_api_key);
}

void ResolveLanceNamespaceTableAuth(ClientContext &context,
                                    const LanceNamespaceTableConfig &cfg,
                                    string &out_bearer_token,
                                    string &out_api_key,
                                    string &out_headers_tsv) {
  out_headers_tsv = cfg.headers_tsv;
  unordered_map<string, Value> overrides;
  if (!cfg.bearer_token_override.empty()) {
    overrides["bearer_token"] = Value(cfg.bearer_token_override);
  }
  if (!cfg.api_key_override.empty()) {
    overrides["api_key"] = Value(cfg.api_key_override);
  }
  ResolveLanceNamespaceAuth(context, cfg.endpoint, overrides, out_bearer_token,
                            out_api_key);
}

uint64_t ResolveLanceNamespaceTableVersion(ClientContext &context,
                                           LanceNamespaceTableConfig &cfg) {
  if (!cfg.IsRest()) {
    throw InternalException("Lance namespace table versions are resolved "
                            "through the REST namespace");
  }
  string bearer_token;
  string api_key;
  string headers_tsv;
  ResolveLanceNamespaceTableAuth(context, cfg, bearer_token, api_key,
                                 headers_tsv);
  ValidateLanceCString(cfg.endpoint, "Lance namespace endpoint");
  ValidateLanceCString(cfg.table_id, "Lance namespace table id");
  ValidateLanceCString(cfg.delimiter, "Lance namespace delimiter");
  ValidateLanceCString(bearer_token, "Lance namespace bearer token");
  ValidateLanceCString(api_key, "Lance namespace API key");
  ValidateLanceCString(headers_tsv, "Lance namespace headers");
  cfg.requires_worker_auth =
      !bearer_token.empty() || !api_key.empty() || !headers_tsv.empty();
  auto version = lance_namespace_get_table_version(
      cfg.endpoint.c_str(), cfg.table_id.c_str(),
      bearer_token.empty() ? nullptr : bearer_token.c_str(),
      api_key.empty() ? nullptr : api_key.c_str(),
      cfg.delimiter.empty() ? nullptr : cfg.delimiter.c_str(),
      headers_tsv.empty() ? nullptr : headers_tsv.c_str());
  if (version == 0) {
    throw IOException(
        "Failed to resolve a fixed Lance namespace table version: " +
        cfg.endpoint + "/" + cfg.table_id + LanceFormatErrorSuffix());
  }
  return version;
}

void ResolveLanceStorageOptions(ClientContext &context, const string &path,
                                string &out_open_path, vector<string> &out_keys,
                                vector<string> &out_values) {
  ValidateLanceCString(path, "Lance dataset path");
  out_open_path = path;
  out_keys.clear();
  out_values.clear();

  out_open_path = LanceNormalizeDatasetPath(context, out_open_path);
  LanceFillStorageOptions(context, out_open_path, out_keys, out_values);
}

void BuildStorageOptionPointerArrays(const vector<string> &option_keys,
                                     const vector<string> &option_values,
                                     vector<const char *> &out_key_ptrs,
                                     vector<const char *> &out_value_ptrs) {
  if (option_keys.size() != option_values.size()) {
    throw InternalException(
        "Storage option keys/values size mismatch for Lance");
  }
  out_key_ptrs.clear();
  out_value_ptrs.clear();
  out_key_ptrs.reserve(option_keys.size());
  out_value_ptrs.reserve(option_values.size());
  for (idx_t i = 0; i < option_keys.size(); i++) {
    if (option_keys[i].empty()) {
      throw InvalidInputException("Lance storage option key must not be empty");
    }
    ValidateLanceCString(option_keys[i], "Lance storage option key");
    ValidateLanceCString(option_values[i], "Lance storage option value");
    out_key_ptrs.push_back(option_keys[i].c_str());
    out_value_ptrs.push_back(option_values[i].c_str());
  }
}

bool TryLanceNamespaceListTables(
    ClientContext &context, const string &endpoint, const string &namespace_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, vector<string> &out_tables, string &out_error) {
  out_tables.clear();
  out_error.clear();

  ValidateLanceCString(endpoint, "Lance namespace endpoint");
  ValidateLanceCString(namespace_id, "Lance namespace id");
  ValidateLanceCString(bearer_token, "Lance namespace bearer token");
  ValidateLanceCString(api_key, "Lance namespace API key");
  ValidateLanceCString(delimiter, "Lance namespace delimiter");
  ValidateLanceCString(headers_tsv, "Lance namespace headers");

  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();
  const char *headers_ptr = headers_tsv.empty() ? nullptr : headers_tsv.c_str();

  auto *ptr = lance_namespace_list_tables(
      endpoint.c_str(), namespace_id.c_str(), bearer_ptr, api_key_ptr,
      delimiter_ptr, headers_ptr);
  if (!ptr) {
    out_error = LanceConsumeLastError();
    if (out_error.empty()) {
      out_error = "unknown error";
    }
    return false;
  }
  ScopedLanceString owned_ptr(ptr);
  string joined = owned_ptr.Get();

  vector<string> parts = StringUtil::Split(joined, '\n');
  for (auto &p : parts) {
    if (!p.empty()) {
      out_tables.push_back(std::move(p));
    }
  }
  return true;
}

static void ParseStorageOptionsTsv(const char *ptr, vector<string> &out_keys,
                                   vector<string> &out_values) {
  out_keys.clear();
  out_values.clear();
  if (!ptr) {
    return;
  }
  string joined = ptr;

  for (auto &line : StringUtil::Split(joined, '\n')) {
    if (line.empty()) {
      continue;
    }
    auto separator = line.find('\t');
    if (separator == string::npos || separator == 0 ||
        line.find('\t', separator + 1) != string::npos) {
      throw IOException("Lance namespace returned malformed storage options");
    }
    out_keys.push_back(line.substr(0, separator));
    out_values.push_back(line.substr(separator + 1));
  }
}

bool TryLanceNamespaceDescribeTable(
    ClientContext &context, const string &endpoint, const string &table_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, string &out_location,
    vector<string> &out_option_keys, vector<string> &out_option_values,
    string &out_error) {
  (void)context;
  out_location.clear();
  out_option_keys.clear();
  out_option_values.clear();
  out_error.clear();

  ValidateLanceCString(endpoint, "Lance namespace endpoint");
  ValidateLanceCString(table_id, "Lance namespace table id");
  ValidateLanceCString(bearer_token, "Lance namespace bearer token");
  ValidateLanceCString(api_key, "Lance namespace API key");
  ValidateLanceCString(delimiter, "Lance namespace delimiter");
  ValidateLanceCString(headers_tsv, "Lance namespace headers");

  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();
  const char *headers_ptr = headers_tsv.empty() ? nullptr : headers_tsv.c_str();

  const char *location_ptr = nullptr;
  const char *options_ptr = nullptr;
  auto rc = lance_namespace_describe_table(
      endpoint.c_str(), table_id.c_str(), bearer_ptr, api_key_ptr,
      delimiter_ptr, headers_ptr, &location_ptr, &options_ptr);
  ScopedLanceString location(location_ptr);
  ScopedLanceString options(options_ptr);
  if (rc != 0) {
    out_error = LanceConsumeLastError();
    if (out_error.empty()) {
      out_error = "unknown error";
    }
    return false;
  }
  if (location) {
    out_location = location.Get();
  }
  ParseStorageOptionsTsv(options.Get(), out_option_keys, out_option_values);
  return true;
}

bool TryLanceNamespaceCreateEmptyTable(
    ClientContext &context, const string &endpoint, const string &table_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, string &out_location,
    vector<string> &out_option_keys, vector<string> &out_option_values,
    string &out_error, bool &inout_namespace_mutated) {
  (void)context;
  out_location.clear();
  out_option_keys.clear();
  out_option_values.clear();
  out_error.clear();

  ValidateLanceCString(endpoint, "Lance namespace endpoint");
  ValidateLanceCString(table_id, "Lance namespace table id");
  ValidateLanceCString(bearer_token, "Lance namespace bearer token");
  ValidateLanceCString(api_key, "Lance namespace API key");
  ValidateLanceCString(delimiter, "Lance namespace delimiter");
  ValidateLanceCString(headers_tsv, "Lance namespace headers");

  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();
  const char *headers_ptr = headers_tsv.empty() ? nullptr : headers_tsv.c_str();

  const char *location_ptr = nullptr;
  const char *options_ptr = nullptr;
  auto rc = lance_namespace_create_empty_table(
      endpoint.c_str(), table_id.c_str(), bearer_ptr, api_key_ptr,
      delimiter_ptr, headers_ptr, &location_ptr, &options_ptr);
  ScopedLanceString location(location_ptr);
  ScopedLanceString options(options_ptr);
  if (rc != 0) {
    auto error = LanceConsumeLastErrorDetail();
    // A timeout or transport failure can happen after declare_table was
    // accepted. Preserve that fact so callers do not retry or compensate.
    if (!IsKnownLanceNamespaceMutationErrorCode(error.code)) {
      auto raw_code = error.code;
      error.code = 56;
      inout_namespace_mutated = true;
      try {
        error.message +=
            "; native Lance namespace mutation returned an unrecognized "
            "error code " +
            to_string(raw_code) + "; treating its outcome as unknown";
      } catch (...) {
        // Preserve the machine-readable outcome code if diagnostic allocation
        // fails while handling the original failure.
      }
    } else if (error.code == 56) {
      inout_namespace_mutated = true;
    }
    out_error = error.ToString();
    if (out_error.empty()) {
      out_error = "unknown error";
    }
    return false;
  }
  // A successful return from declare_table means the namespace changed even
  // if converting its response into C++ strings fails below. Tell the caller
  // before doing any allocation or validation that could throw.
  inout_namespace_mutated = true;
  if (location) {
    out_location = location.Get();
  }
  ParseStorageOptionsTsv(options.Get(), out_option_keys, out_option_values);
  return true;
}

bool TryLanceNamespaceDropTable(ClientContext &context, const string &endpoint,
                                const string &table_id,
                                const string &bearer_token,
                                const string &api_key, const string &delimiter,
                                const string &headers_tsv, string &out_error,
                                bool &inout_namespace_mutated) {
  (void)context;
  out_error.clear();

  ValidateLanceCString(endpoint, "Lance namespace endpoint");
  ValidateLanceCString(table_id, "Lance namespace table id");
  ValidateLanceCString(bearer_token, "Lance namespace bearer token");
  ValidateLanceCString(api_key, "Lance namespace API key");
  ValidateLanceCString(delimiter, "Lance namespace delimiter");
  ValidateLanceCString(headers_tsv, "Lance namespace headers");

  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();
  const char *headers_ptr = headers_tsv.empty() ? nullptr : headers_tsv.c_str();

  auto rc =
      lance_namespace_drop_table(endpoint.c_str(), table_id.c_str(), bearer_ptr,
                                 api_key_ptr, delimiter_ptr, headers_ptr);
  if (rc != 0) {
    auto error = LanceConsumeLastErrorDetail();
    // DROP is non-idempotent from the caller's perspective: a lost response
    // does not prove that the table remained present. Keep the mutation fence
    // raised for the outcome-unknown error code.
    if (!IsKnownLanceNamespaceMutationErrorCode(error.code)) {
      auto raw_code = error.code;
      error.code = 56;
      inout_namespace_mutated = true;
      try {
        error.message +=
            "; native Lance namespace mutation returned an unrecognized "
            "error code " +
            to_string(raw_code) + "; treating its outcome as unknown";
      } catch (...) {
      }
    } else if (error.code == 56) {
      inout_namespace_mutated = true;
    }
    out_error = error.ToString();
    if (out_error.empty()) {
      out_error = "unknown error";
    }
    return false;
  }
  inout_namespace_mutated = true;
  return true;
}

bool TryLanceDirNamespaceListTables(ClientContext &context, const string &root,
                                    vector<string> &out_tables,
                                    string &out_error) {
  out_tables.clear();
  out_error.clear();

  string open_root;
  vector<string> option_keys;
  vector<string> option_values;
  ResolveLanceStorageOptions(context, root, open_root, option_keys,
                             option_values);

  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                  value_ptrs);

  auto *ptr = lance_dir_namespace_list_tables(
      open_root.c_str(), key_ptrs.empty() ? nullptr : key_ptrs.data(),
      value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size());
  if (!ptr) {
    out_error = LanceConsumeLastError();
    if (out_error.empty()) {
      out_error = "unknown error";
    }
    return false;
  }

  ScopedLanceString owned_ptr(ptr);
  string joined = owned_ptr.Get();

  vector<string> parts = StringUtil::Split(joined, '\n');
  for (auto &p : parts) {
    if (!p.empty()) {
      out_tables.push_back(std::move(p));
    }
  }
  return true;
}

void *
LanceOpenDatasetInNamespace(ClientContext &context, const string &endpoint,
                            const string &table_id, const string &bearer_token,
                            const string &api_key, const string &delimiter,
                            const string &headers_tsv, string &out_table_uri) {
  out_table_uri.clear();
  ValidateLanceCString(endpoint, "Lance namespace endpoint");
  ValidateLanceCString(table_id, "Lance namespace table id");
  ValidateLanceCString(bearer_token, "Lance namespace bearer token");
  ValidateLanceCString(api_key, "Lance namespace API key");
  ValidateLanceCString(delimiter, "Lance namespace delimiter");
  ValidateLanceCString(headers_tsv, "Lance namespace headers");
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
  ScopedLanceString uri(uri_ptr);
  if (uri) {
    out_table_uri = uri.Get();
  }
  return dataset;
}

void *LanceOpenDataset(ClientContext &context, const string &path) {
  string open_path;
  vector<string> option_keys;
  vector<string> option_values;
  ResolveLanceStorageOptions(context, path, open_path, option_keys,
                             option_values);
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

static unordered_map<string, Value>
BuildNamespaceAuthOverrideOptions(const string &bearer_token_override,
                                  const string &api_key_override) {
  unordered_map<string, Value> options;
  if (!bearer_token_override.empty()) {
    options["bearer_token"] = Value(bearer_token_override);
  }
  if (!api_key_override.empty()) {
    options["api_key"] = Value(api_key_override);
  }
  return options;
}

LanceTableEntry *TryResolveLanceTableEntry(ClientContext &context,
                                           const string &input) {
  auto candidate = input;
  bool force_table = false;
  if (candidate.rfind("path:", 0) == 0) {
    return nullptr;
  }
  if (candidate.rfind("table:", 0) == 0) {
    candidate = candidate.substr(6);
    force_table = true;
  }
  // Fast-path bail-out: obvious filesystem / URL literals can never match a
  // qualified catalog identifier.  Avoids parsing attempts and potential
  // secondary lookups that would just end up throwing ParserException.
  if (candidate.empty() || candidate.find('/') != string::npos ||
      candidate.find('\\') != string::npos ||
      candidate.find("://") != string::npos ||
      (!force_table && candidate.size() >= 6 &&
       StringUtil::CIEquals(candidate.substr(candidate.size() - 6),
                            ".lance"))) {
    return nullptr;
  }

  QualifiedName qname;
  try {
    qname = QualifiedName::Parse(candidate);
  } catch (ParserException &) {
    return nullptr;
  }

  EntryLookupInfo lookup_info(CatalogType::TABLE_ENTRY, qname.name);
  auto entry = Catalog::GetEntry(context, qname.catalog, qname.schema,
                                 lookup_info, OnEntryNotFound::RETURN_NULL);
  if (!entry) {
    return nullptr;
  }
  auto *table_entry = dynamic_cast<TableCatalogEntry *>(entry.get());
  if (!table_entry) {
    return nullptr;
  }
  return dynamic_cast<LanceTableEntry *>(table_entry);
}

void RequireLanceTableWritable(const LanceTableEntry &table,
                               const string &operation) {
  if (table.ParentCatalog().GetAttached().IsReadOnly()) {
    throw InvalidInputException(
        operation +
        " cannot modify a Lance table through an attachment in read-only mode");
  }
}

void *LanceOpenDatasetForTable(ClientContext &context,
                               const LanceTableEntry &table,
                               string &out_display_uri) {
  out_display_uri = table.DatasetUri();
  if (!table.IsNamespaceBacked()) {
    return LanceOpenDataset(context, table.DatasetUri());
  }

  auto &cfg = table.NamespaceConfig();
  if (cfg.IsDirectory()) {
    vector<const char *> key_ptrs;
    vector<const char *> value_ptrs;
    BuildStorageOptionPointerArrays(cfg.option_keys, cfg.option_values,
                                    key_ptrs, value_ptrs);

    const char *uri_ptr = nullptr;
    auto *dataset = lance_open_dataset_in_dir_namespace_with_session(
        cfg.root.c_str(), cfg.table_id.c_str(),
        key_ptrs.empty() ? nullptr : key_ptrs.data(),
        value_ptrs.empty() ? nullptr : value_ptrs.data(),
        cfg.option_keys.size(), LanceGetSessionHandle(context), &uri_ptr);
    ScopedLanceString uri(uri_ptr);
    if (uri) {
      out_display_uri = uri.Get();
    } else if (!cfg.display_uri.empty()) {
      out_display_uri = cfg.display_uri;
    } else {
      out_display_uri = LanceDirectoryNamespaceDatasetUri(cfg);
    }
    return dataset;
  }

  unordered_map<string, Value> overrides = BuildNamespaceAuthOverrideOptions(
      cfg.bearer_token_override, cfg.api_key_override);
  string bearer_token;
  string api_key;
  ResolveLanceNamespaceAuth(context, cfg.endpoint, overrides, bearer_token,
                            api_key);

  string table_uri;
  auto *dataset = LanceOpenDatasetInNamespace(
      context, cfg.endpoint, cfg.table_id, bearer_token, api_key, cfg.delimiter,
      cfg.headers_tsv, table_uri);
  if (!table_uri.empty()) {
    out_display_uri = table_uri;
  } else {
    out_display_uri = cfg.endpoint + "/" + cfg.table_id;
  }
  return dataset;
}

void ResolveLanceStorageOptionsForTable(ClientContext &context,
                                        const LanceTableEntry &table,
                                        string &out_open_path,
                                        vector<string> &out_option_keys,
                                        vector<string> &out_option_values,
                                        string &out_display_uri) {
  out_display_uri = table.DatasetUri();
  if (!table.IsNamespaceBacked()) {
    ResolveLanceStorageOptions(context, table.DatasetUri(), out_open_path,
                               out_option_keys, out_option_values);
    return;
  }

  auto &cfg = table.NamespaceConfig();
  if (cfg.IsDirectory()) {
    out_display_uri = LanceDirectoryNamespaceDatasetUri(cfg);
    out_open_path = out_display_uri;
    out_option_keys = cfg.option_keys;
    out_option_values = cfg.option_values;
    return;
  }

  unordered_map<string, Value> overrides = BuildNamespaceAuthOverrideOptions(
      cfg.bearer_token_override, cfg.api_key_override);
  string bearer_token;
  string api_key;
  ResolveLanceNamespaceAuth(context, cfg.endpoint, overrides, bearer_token,
                            api_key);

  string location;
  string error;
  vector<string> option_keys;
  vector<string> option_values;
  if (!TryLanceNamespaceDescribeTable(context, cfg.endpoint, cfg.table_id,
                                      bearer_token, api_key, cfg.delimiter,
                                      cfg.headers_tsv, location, option_keys,
                                      option_values, error)) {
    throw IOException("Failed to describe Lance table via namespace: " +
                      (error.empty() ? "unknown error" : error));
  }

  out_display_uri =
      location.empty() ? (cfg.endpoint + "/" + cfg.table_id) : location;

  if (!option_keys.empty()) {
    out_open_path = LanceNormalizeS3Scheme(location);
    out_option_keys = std::move(option_keys);
    out_option_values = std::move(option_values);
    return;
  }

  ResolveLanceStorageOptions(context, location, out_open_path, out_option_keys,
                             out_option_values);
}

int64_t LanceTruncateDatasetWithStorageOptions(
    ClientContext &context, const string &open_path,
    const vector<string> &option_keys, const vector<string> &option_values,
    const string &display_uri) {
  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                  value_ptrs);

  void *dataset = nullptr;
  if (option_keys.empty()) {
    dataset = lance_open_dataset(open_path.c_str());
  } else {
    dataset = lance_open_dataset_with_storage_options(
        open_path.c_str(), key_ptrs.data(), value_ptrs.data(),
        option_keys.size());
  }
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + display_uri +
                      LanceFormatErrorSuffix());
  }

  auto row_count = lance_dataset_count_rows(dataset);
  if (row_count < 0) {
    lance_close_dataset(dataset);
    throw IOException("Failed to count rows from Lance dataset: " +
                      display_uri + LanceFormatErrorSuffix());
  }

  auto *schema_handle = lance_get_schema(dataset);
  if (!schema_handle) {
    lance_close_dataset(dataset);
    throw IOException("Failed to get schema from Lance dataset: " +
                      display_uri + LanceFormatErrorSuffix());
  }

  ArrowSchemaWrapper schema_root;
  memset(&schema_root.arrow_schema, 0, sizeof(schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &schema_root.arrow_schema) != 0) {
    lance_free_schema(schema_handle);
    lance_close_dataset(dataset);
    throw IOException(
        "Failed to export Lance schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }

  lance_free_schema(schema_handle);
  lance_close_dataset(dataset);

  auto *writer = lance_open_writer_with_storage_options(
      open_path.c_str(), "overwrite",
      key_ptrs.empty() ? nullptr : key_ptrs.data(),
      value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size(),
      LANCE_DEFAULT_MAX_ROWS_PER_FILE, LANCE_DEFAULT_MAX_ROWS_PER_GROUP,
      LANCE_DEFAULT_MAX_BYTES_PER_FILE, nullptr, nullptr, 1,
      LanceGetSessionHandle(context), &schema_root.arrow_schema);
  if (!writer) {
    throw IOException("Failed to open Lance writer: " + open_path +
                      LanceFormatErrorSuffix());
  }
  auto rc = lance_writer_finish(writer);
  lance_close_writer(writer);
  if (rc != 0) {
    throw IOException("Failed to finalize Lance dataset write" +
                      LanceFormatErrorSuffix());
  }

  return row_count;
}

int64_t LanceTruncateDataset(ClientContext &context,
                             const string &dataset_uri) {
  string open_path;
  vector<string> option_keys;
  vector<string> option_values;
  ResolveLanceStorageOptions(context, dataset_uri, open_path, option_keys,
                             option_values);
  return LanceTruncateDatasetWithStorageOptions(context, open_path, option_keys,
                                                option_values, dataset_uri);
}

void ApplyDuckDBFilters(ClientContext &context, TableFilterSet &filters,
                        DataChunk &chunk, SelectionVector &sel) {
  if (chunk.size() == 0) {
    return;
  }
  unique_ptr<Expression> combined;
  for (auto &it : filters.filters) {
    auto scan_col_idx = it.first;
    if (scan_col_idx >= chunk.ColumnCount()) {
      continue;
    }
    BoundReferenceExpression col_expr(chunk.data[scan_col_idx].GetType(),
                                      NumericCast<storage_t>(scan_col_idx));
    auto expr = it.second->ToExpression(col_expr);
    if (!combined) {
      combined = std::move(expr);
    } else {
      auto conj = make_uniq<BoundConjunctionExpression>(
          ExpressionType::CONJUNCTION_AND);
      conj->children.push_back(std::move(combined));
      conj->children.push_back(std::move(expr));
      combined = std::move(conj);
    }
  }
  if (!combined) {
    return;
  }
  ExpressionExecutor executor(context, *combined);
  auto selected = executor.SelectExpression(chunk, sel);
  if (selected != chunk.size()) {
    chunk.Slice(sel, selected);
  }
}

} // namespace duckdb
