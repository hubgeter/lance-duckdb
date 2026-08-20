#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/distributed/extension_write_task_provider.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/function/distributed_write.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/operator/logical_copy_to_file.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator_extension.hpp"

#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_ffi.hpp"
#include "lance_session_state.hpp"

#include <cstdint>
#include <limits>

namespace duckdb {

static constexpr const char *LANCE_WRITE_LOGICAL_EXTENSION = "lance_write";
static constexpr const char *LANCE_DISTRIBUTED_WRITE_OPERATOR = "lance_write";
static constexpr idx_t LANCE_DISTRIBUTED_WRITE_PROTOCOL_VERSION = 2;
static constexpr const char *LANCE_TRANSACTION_CODEC = "lance.transaction";
static constexpr idx_t LANCE_TRANSACTION_CODEC_VERSION = 1;
static constexpr const char *LANCE_STAGING_ARTIFACT_CODEC =
    "lance.staging-dataset";
static constexpr idx_t LANCE_STAGING_ARTIFACT_CODEC_VERSION = 1;
static constexpr int32_t LANCE_DATASET_COMMIT_OUTCOME_UNKNOWN = 55;

struct LanceWriteBindData : public FunctionData {
  string mode = "create";
  string data_storage_version = LANCE_DEFAULT_DATA_STORAGE_VERSION;
  uint64_t max_rows_per_file = LANCE_DEFAULT_MAX_ROWS_PER_FILE;
  uint64_t max_rows_per_group = LANCE_DEFAULT_MAX_ROWS_PER_GROUP;
  uint64_t max_bytes_per_file = LANCE_DEFAULT_MAX_BYTES_PER_FILE;
  string vector_dims;
  bool infer_vector_dims = true;

  vector<string> names;
  vector<LogicalType> types;

  unique_ptr<LanceWriteBindData> CopyTyped() const {
    auto result = make_uniq<LanceWriteBindData>();
    result->mode = mode;
    result->data_storage_version = data_storage_version;
    result->max_rows_per_file = max_rows_per_file;
    result->max_rows_per_group = max_rows_per_group;
    result->max_bytes_per_file = max_bytes_per_file;
    result->vector_dims = vector_dims;
    result->infer_vector_dims = infer_vector_dims;
    result->names = names;
    result->types = types;
    return std::move(result);
  }

  unique_ptr<FunctionData> Copy() const override { return CopyTyped(); }

  bool Equals(const FunctionData &other_p) const override {
    auto &other = other_p.Cast<LanceWriteBindData>();
    return mode == other.mode &&
           data_storage_version == other.data_storage_version &&
           max_rows_per_file == other.max_rows_per_file &&
           max_rows_per_group == other.max_rows_per_group &&
           max_bytes_per_file == other.max_bytes_per_file &&
           vector_dims == other.vector_dims &&
           infer_vector_dims == other.infer_vector_dims &&
           names == other.names && types == other.types;
  }
};

static string HexLanceWriteIdentity(const string &value) {
  static constexpr const char *HEX = "0123456789abcdef";
  string result;
  result.reserve(value.size() * 2);
  for (auto byte : value) {
    auto unsigned_byte = static_cast<uint8_t>(byte);
    result.push_back(HEX[unsigned_byte >> 4]);
    result.push_back(HEX[unsigned_byte & 0x0f]);
  }
  return result;
}

static string JoinLanceWriteUri(const string &base, const string &suffix) {
  auto end = base.size();
  while (end > 0 && (base[end - 1] == '/' || base[end - 1] == '\\')) {
    end--;
  }
  if (end == 0) {
    return suffix;
  }
  return base.substr(0, end) + "/" + suffix;
}

static string LanceStagingUri(const string &target, const string &operation_id,
                              const string &task_attempt_id) {
  return JoinLanceWriteUri(
      target, "_vane_staging/" + HexLanceWriteIdentity(operation_id) + "/" +
                  HexLanceWriteIdentity(task_attempt_id));
}

static string
SerializeLanceDistributedWriteBind(const string &target,
                                   const LanceWriteBindData &bind_data) {
  MemoryStream stream(Allocator::DefaultAllocator());
  BinarySerializer serializer(stream);
  serializer.Begin();
  serializer.WriteProperty(1, "target", target);
  serializer.WriteProperty(2, "mode", bind_data.mode);
  serializer.WriteProperty(3, "data_storage_version",
                           bind_data.data_storage_version);
  serializer.WriteProperty(4, "max_rows_per_file", bind_data.max_rows_per_file);
  serializer.WriteProperty(5, "max_rows_per_group",
                           bind_data.max_rows_per_group);
  serializer.WriteProperty(6, "max_bytes_per_file",
                           bind_data.max_bytes_per_file);
  serializer.WriteProperty(7, "names", bind_data.names);
  serializer.WriteProperty(8, "types", bind_data.types);
  serializer.WriteProperty(9, "vector_dims", bind_data.vector_dims);
  serializer.WriteProperty(10, "infer_vector_dims",
                           bind_data.infer_vector_dims);
  serializer.End();
  return string(reinterpret_cast<const char *>(stream.GetData()),
                stream.GetPosition());
}

struct LanceDistributedWriteBind {
  string target;
  LanceWriteBindData bind_data;
};

static LanceDistributedWriteBind
DeserializeLanceDistributedWriteBind(const string &bytes) {
  if (bytes.empty()) {
    throw SerializationException("distributed Lance write bind data is empty");
  }
  auto *data = reinterpret_cast<data_ptr_t>(const_cast<char *>(bytes.data()));
  MemoryStream stream(data, bytes.size());
  BinaryDeserializer deserializer(stream);
  deserializer.Begin();
  LanceDistributedWriteBind result;
  result.target = deserializer.ReadProperty<string>(1, "target");
  result.bind_data.mode = deserializer.ReadProperty<string>(2, "mode");
  result.bind_data.data_storage_version =
      deserializer.ReadProperty<string>(3, "data_storage_version");
  result.bind_data.max_rows_per_file =
      deserializer.ReadProperty<uint64_t>(4, "max_rows_per_file");
  result.bind_data.max_rows_per_group =
      deserializer.ReadProperty<uint64_t>(5, "max_rows_per_group");
  result.bind_data.max_bytes_per_file =
      deserializer.ReadProperty<uint64_t>(6, "max_bytes_per_file");
  result.bind_data.names =
      deserializer.ReadProperty<vector<string>>(7, "names");
  result.bind_data.types =
      deserializer.ReadProperty<vector<LogicalType>>(8, "types");
  result.bind_data.vector_dims =
      deserializer.ReadProperty<string>(9, "vector_dims");
  result.bind_data.infer_vector_dims =
      deserializer.ReadProperty<bool>(10, "infer_vector_dims");
  deserializer.End();
  if (result.target.empty() ||
      result.bind_data.names.size() != result.bind_data.types.size()) {
    throw SerializationException(
        "distributed Lance write bind data is invalid");
  }
  return result;
}

struct LanceWriteGlobalState : public GlobalFunctionData {
  explicit LanceWriteGlobalState() = default;

  string cache_key;
  void *writer = nullptr;
  ArrowSchemaWrapper schema_root;

  ~LanceWriteGlobalState() override {
    if (writer) {
      lance_close_writer(writer);
      writer = nullptr;
    }
  }
};

struct LanceWriteLocalState : public LocalFunctionData {};

static void WriteLanceDataChunk(ExecutionContext &context, void *writer,
                                DataChunk &input) {
  if (input.size() == 0) {
    return;
  }

  auto props = context.client.GetClientProperties();
  unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>>
      extension_type_cast;

  ArrowArray array;
  memset(&array, 0, sizeof(array));
  ArrowConverter::ToArrowArray(input, &array, props, extension_type_cast);

  auto rc = lance_writer_write_batch(writer, &array);
  if (array.release) {
    array.release(&array);
  }
  if (rc != 0) {
    throw IOException("Failed to write to Lance dataset" +
                      LanceFormatErrorSuffix());
  }
}

static void LanceWriteCopyOptions(ClientContext &, CopyOptionsInput &input) {
  auto &options = input.options;
  options["mode"] =
      CopyOption(LogicalType::VARCHAR, CopyOptionMode::WRITE_ONLY);
  options["data_storage_version"] =
      CopyOption(LogicalType::VARCHAR, CopyOptionMode::WRITE_ONLY);
  options["max_rows_per_file"] =
      CopyOption(LogicalType::UBIGINT, CopyOptionMode::WRITE_ONLY);
  options["max_rows_per_group"] =
      CopyOption(LogicalType::UBIGINT, CopyOptionMode::WRITE_ONLY);
  options["max_bytes_per_file"] =
      CopyOption(LogicalType::UBIGINT, CopyOptionMode::WRITE_ONLY);
  options["vector_dims"] =
      CopyOption(LogicalType::VARCHAR, CopyOptionMode::WRITE_ONLY);
  options["infer_vector_dims"] =
      CopyOption(LogicalType::BOOLEAN, CopyOptionMode::WRITE_ONLY);
}

static unique_ptr<FunctionData>
LanceWriteBind(ClientContext &, CopyFunctionBindInput &input,
               const vector<string> &names,
               const vector<LogicalType> &sql_types) {
  auto bind_data = make_uniq<LanceWriteBindData>();
  bind_data->names = names;
  bind_data->types = sql_types;

  for (auto &option : input.info.options) {
    const auto key = StringUtil::Lower(option.first);
    if (option.second.size() != 1) {
      throw BinderException("%s requires exactly one argument",
                            StringUtil::Upper(key));
    }
    auto &value = option.second[0];

    if (key == "mode") {
      if (value.IsNull()) {
        throw BinderException("mode cannot be NULL");
      }
      auto mode = StringUtil::Lower(value.ToString());
      if (mode != "create" && mode != "append" && mode != "overwrite") {
        throw BinderException(
            "mode must be one of [create, append, overwrite]");
      }
      bind_data->mode = std::move(mode);
    } else if (key == "data_storage_version") {
      if (value.IsNull()) {
        throw BinderException("data_storage_version cannot be NULL");
      }
      bind_data->data_storage_version = value.GetValue<string>();
      if (bind_data->data_storage_version.empty()) {
        throw BinderException("data_storage_version cannot be empty");
      }
    } else if (key == "max_rows_per_file") {
      bind_data->max_rows_per_file = value.GetValue<uint64_t>();
    } else if (key == "max_rows_per_group") {
      bind_data->max_rows_per_group = value.GetValue<uint64_t>();
    } else if (key == "max_bytes_per_file") {
      bind_data->max_bytes_per_file = value.GetValue<uint64_t>();
    } else if (key == "vector_dims") {
      if (value.IsNull()) {
        throw BinderException("vector_dims cannot be NULL");
      }
      bind_data->vector_dims = value.GetValue<string>();
      if (bind_data->vector_dims.empty()) {
        throw BinderException("vector_dims cannot be empty");
      }
    } else if (key == "infer_vector_dims") {
      if (value.IsNull()) {
        throw BinderException("infer_vector_dims cannot be NULL");
      }
      bind_data->infer_vector_dims = value.GetValue<bool>();
    }
  }
  if (bind_data->max_rows_per_file == 0 || bind_data->max_rows_per_group == 0 ||
      bind_data->max_bytes_per_file == 0) {
    throw BinderException(
        "Lance write row and byte limits must be greater than zero");
  }

  return std::move(bind_data);
}

static void LanceWriteSerialize(Serializer &serializer,
                                const FunctionData &bind_data_p,
                                const CopyFunction &) {
  auto &bind_data = bind_data_p.Cast<LanceWriteBindData>();
  serializer.WriteProperty(100, "mode", bind_data.mode);
  serializer.WriteProperty(101, "data_storage_version",
                           bind_data.data_storage_version);
  serializer.WriteProperty(102, "max_rows_per_file",
                           bind_data.max_rows_per_file);
  serializer.WriteProperty(103, "max_rows_per_group",
                           bind_data.max_rows_per_group);
  serializer.WriteProperty(104, "max_bytes_per_file",
                           bind_data.max_bytes_per_file);
  serializer.WriteProperty(105, "names", bind_data.names);
  serializer.WriteProperty(106, "types", bind_data.types);
  serializer.WriteProperty(107, "vector_dims", bind_data.vector_dims);
  serializer.WriteProperty(108, "infer_vector_dims",
                           bind_data.infer_vector_dims);
}

static unique_ptr<FunctionData>
LanceWriteDeserialize(Deserializer &deserializer, CopyFunction &) {
  auto result = make_uniq<LanceWriteBindData>();
  result->mode = deserializer.ReadProperty<string>(100, "mode");
  result->data_storage_version =
      deserializer.ReadProperty<string>(101, "data_storage_version");
  result->max_rows_per_file =
      deserializer.ReadProperty<uint64_t>(102, "max_rows_per_file");
  result->max_rows_per_group =
      deserializer.ReadProperty<uint64_t>(103, "max_rows_per_group");
  result->max_bytes_per_file =
      deserializer.ReadProperty<uint64_t>(104, "max_bytes_per_file");
  result->names = deserializer.ReadProperty<vector<string>>(105, "names");
  result->types = deserializer.ReadProperty<vector<LogicalType>>(106, "types");
  result->vector_dims = deserializer.ReadProperty<string>(107, "vector_dims");
  result->infer_vector_dims =
      deserializer.ReadProperty<bool>(108, "infer_vector_dims");
  return std::move(result);
}

static unique_ptr<GlobalFunctionData>
LanceWriteInitGlobal(ClientContext &context, FunctionData &bind_data_p,
                     const string &file_path) {
  auto &bind_data = bind_data_p.Cast<LanceWriteBindData>();
  auto state = make_uniq<LanceWriteGlobalState>();

  auto props = context.GetClientProperties();
  memset(&state->schema_root.arrow_schema, 0,
         sizeof(state->schema_root.arrow_schema));
  ArrowConverter::ToArrowSchema(&state->schema_root.arrow_schema,
                                bind_data.types, bind_data.names, props);

  vector<string> option_keys;
  vector<string> option_values;
  string open_path;
  ResolveLanceStorageOptions(context, file_path, open_path, option_keys,
                             option_values);
  state->cache_key = LanceBuildResolvedPathDatasetCacheKey(
      open_path, option_keys, option_values);

  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                  value_ptrs);

  const char *data_storage_version_ptr =
      bind_data.data_storage_version.empty()
          ? nullptr
          : bind_data.data_storage_version.c_str();
  const char *vector_dims_ptr =
      bind_data.vector_dims.empty() ? nullptr : bind_data.vector_dims.c_str();
  auto *session = LanceGetSessionHandle(context);
  state->writer = lance_open_writer_with_storage_options(
      open_path.c_str(), bind_data.mode.c_str(),
      key_ptrs.empty() ? nullptr : key_ptrs.data(),
      value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size(),
      bind_data.max_rows_per_file, bind_data.max_rows_per_group,
      bind_data.max_bytes_per_file, data_storage_version_ptr, vector_dims_ptr,
      bind_data.infer_vector_dims ? 1 : 0, session,
      &state->schema_root.arrow_schema);
  if (!state->writer) {
    throw IOException("Failed to open Lance writer: " + open_path +
                      LanceFormatErrorSuffix());
  }

  return std::move(state);
}

static unique_ptr<LocalFunctionData> LanceWriteInitLocal(ExecutionContext &,
                                                         FunctionData &) {
  return make_uniq<LanceWriteLocalState>();
}

static void LanceWriteSink(ExecutionContext &context, FunctionData &,
                           GlobalFunctionData &gstate_p, LocalFunctionData &,
                           DataChunk &input) {
  auto &gstate = gstate_p.Cast<LanceWriteGlobalState>();
  WriteLanceDataChunk(context, gstate.writer, input);
}

static void LanceWriteFinalize(ClientContext &context, FunctionData &,
                               GlobalFunctionData &gstate_p) {
  auto &gstate = gstate_p.Cast<LanceWriteGlobalState>();
  if (!gstate.writer) {
    return;
  }
  auto rc = lance_writer_finish(gstate.writer);
  lance_close_writer(gstate.writer);
  gstate.writer = nullptr;
  if (rc != 0) {
    throw IOException("Failed to finalize Lance dataset write" +
                      LanceFormatErrorSuffix());
  }
  LanceInvalidateDatasetCache(context, gstate.cache_key);
}

class LanceDistributedWriteGlobalState final
    : public DistributedWriteGlobalState {
public:
  mutex lock;
  LanceDistributedWriteBind bind;
  string open_target;
  string staging_uri;
  vector<string> option_keys;
  vector<string> option_values;
  void *writer = nullptr;
  ArrowSchemaWrapper schema_root;
  idx_t row_count = 0;

  ~LanceDistributedWriteGlobalState() override {
    if (writer) {
      lance_close_writer(writer);
      writer = nullptr;
    }
  }
};

class LanceDistributedWriteLocalState final
    : public DistributedWriteLocalState {};

static unique_ptr<DistributedWriteGlobalState>
LanceDistributedWriteInitializeGlobal(ClientContext &context,
                                      const DistributedExtensionWriteInfo &info,
                                      const DistributedWriteTaskContext &task) {
  auto state = make_uniq<LanceDistributedWriteGlobalState>();
  state->bind = DeserializeLanceDistributedWriteBind(info.worker_bind_data);
  ResolveLanceStorageOptions(context, state->bind.target, state->open_target,
                             state->option_keys, state->option_values);
  state->staging_uri = LanceStagingUri(state->open_target, task.operation_id,
                                       task.task_attempt_id);

  auto props = context.GetClientProperties();
  memset(&state->schema_root.arrow_schema, 0,
         sizeof(state->schema_root.arrow_schema));
  ArrowConverter::ToArrowSchema(&state->schema_root.arrow_schema,
                                state->bind.bind_data.types,
                                state->bind.bind_data.names, props);

  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  BuildStorageOptionPointerArrays(state->option_keys, state->option_values,
                                  key_ptrs, value_ptrs);
  const auto &bind_data = state->bind.bind_data;
  const char *data_storage_version =
      bind_data.data_storage_version.empty()
          ? nullptr
          : bind_data.data_storage_version.c_str();
  const char *vector_dims =
      bind_data.vector_dims.empty() ? nullptr : bind_data.vector_dims.c_str();
  state->writer = lance_open_uncommitted_writer_with_storage_options(
      state->staging_uri.c_str(), "create",
      key_ptrs.empty() ? nullptr : key_ptrs.data(),
      value_ptrs.empty() ? nullptr : value_ptrs.data(),
      state->option_keys.size(), bind_data.max_rows_per_file,
      bind_data.max_rows_per_group, bind_data.max_bytes_per_file,
      data_storage_version, vector_dims, bind_data.infer_vector_dims ? 1 : 0,
      LanceGetSessionHandle(context), &state->schema_root.arrow_schema);
  if (!state->writer) {
    throw IOException("Failed to open distributed Lance staging writer: " +
                      state->staging_uri + LanceFormatErrorSuffix());
  }
  return std::move(state);
}

static unique_ptr<DistributedWriteLocalState>
LanceDistributedWriteInitializeLocal(ExecutionContext &,
                                     const DistributedExtensionWriteInfo &,
                                     const DistributedWriteTaskContext &,
                                     DistributedWriteGlobalState &) {
  return make_uniq<LanceDistributedWriteLocalState>();
}

static void LanceDistributedWriteSink(ExecutionContext &context,
                                      const DistributedExtensionWriteInfo &,
                                      const DistributedWriteTaskContext &,
                                      DistributedWriteGlobalState &global_p,
                                      DistributedWriteLocalState &,
                                      DataChunk &input) {
  auto &global = global_p.Cast<LanceDistributedWriteGlobalState>();
  lock_guard<mutex> guard(global.lock);
  WriteLanceDataChunk(context, global.writer, input);
  if (input.size() > std::numeric_limits<idx_t>::max() - global.row_count) {
    throw OutOfRangeException("distributed Lance write row count overflow");
  }
  global.row_count += input.size();
}

static void LanceDistributedWriteCombine(ExecutionContext &,
                                         const DistributedExtensionWriteInfo &,
                                         const DistributedWriteTaskContext &,
                                         DistributedWriteGlobalState &,
                                         DistributedWriteLocalState &) {}

static vector<DistributedWriteFragment>
LanceDistributedWriteFinalize(ClientContext &,
                              const DistributedExtensionWriteInfo &,
                              const DistributedWriteTaskContext &task,
                              DistributedWriteGlobalState &global_p) {
  auto &global = global_p.Cast<LanceDistributedWriteGlobalState>();
  void *transaction = nullptr;
  uint8_t *serialized = nullptr;
  size_t serialized_len = 0;
  {
    lock_guard<mutex> guard(global.lock);
    if (!global.writer) {
      throw InternalException(
          "distributed Lance staging writer was already finalized");
    }
    auto rc = lance_writer_finish_uncommitted(global.writer, &transaction);
    lance_close_writer(global.writer);
    global.writer = nullptr;
    if (rc != 0 || !transaction) {
      if (transaction) {
        lance_free_transaction(transaction);
      }
      throw IOException(
          "Failed to finalize distributed Lance staging transaction" +
          LanceFormatErrorSuffix());
    }
    rc = lance_serialize_transaction(transaction, &serialized, &serialized_len);
    lance_free_transaction(transaction);
    transaction = nullptr;
    if (rc != 0 || !serialized || serialized_len == 0) {
      if (serialized) {
        lance_free_bytes(serialized, serialized_len);
      }
      throw IOException("Failed to serialize distributed Lance transaction" +
                        LanceFormatErrorSuffix());
    }
  }

  DistributedWriteArtifact artifact;
  artifact.artifact_id = task.task_attempt_id;
  artifact.uri = global.staging_uri;
  artifact.codec.name = LANCE_STAGING_ARTIFACT_CODEC;
  artifact.codec.version = LANCE_STAGING_ARTIFACT_CODEC_VERSION;

  DistributedWriteFragment fragment;
  fragment.fragment_id = task.task_attempt_id;
  fragment.payload.assign(reinterpret_cast<const char *>(serialized),
                          serialized_len);
  fragment.artifacts.push_back(std::move(artifact));
  fragment.row_count = global.row_count;
  fragment.byte_count = 0;
  lance_free_bytes(serialized, serialized_len);

  vector<DistributedWriteFragment> result;
  result.push_back(std::move(fragment));
  return result;
}

struct LanceResolvedWriteTarget {
  string open_path;
  vector<string> option_keys;
  vector<string> option_values;
  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  string cache_key;
};

static LanceResolvedWriteTarget ResolveLanceWriteTarget(ClientContext &context,
                                                        const string &target) {
  LanceResolvedWriteTarget result;
  ResolveLanceStorageOptions(context, target, result.open_path,
                             result.option_keys, result.option_values);
  BuildStorageOptionPointerArrays(result.option_keys, result.option_values,
                                  result.key_ptrs, result.value_ptrs);
  result.cache_key = LanceBuildResolvedPathDatasetCacheKey(
      result.open_path, result.option_keys, result.option_values);
  return result;
}

class LancePhysicalWriteGlobalState final : public GlobalSinkState {
public:
  unique_ptr<FunctionData> bind_data;
  unique_ptr<GlobalFunctionData> function_state;
  idx_t row_count = 0;
  bool finalized = false;
};

class LancePhysicalWriteLocalState final : public LocalSinkState {
public:
  explicit LancePhysicalWriteLocalState(
      unique_ptr<LocalFunctionData> function_state_p)
      : function_state(std::move(function_state_p)) {}

  unique_ptr<LocalFunctionData> function_state;
};

class LancePhysicalWriteSourceState final : public GlobalSourceState {
public:
  bool emitted = false;
};

class PhysicalLanceWrite final
    : public PhysicalOperator,
      public distributed::ExtensionWriteTaskProvider {
public:
  static constexpr const PhysicalOperatorType TYPE =
      PhysicalOperatorType::EXTENSION;

  PhysicalLanceWrite(PhysicalPlan &physical_plan, string target_p,
                     unique_ptr<LanceWriteBindData> bind_data_p,
                     idx_t estimated_cardinality)
      : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                         {LogicalType::BIGINT}, estimated_cardinality),
        target(std::move(target_p)), bind_data(std::move(bind_data_p)) {
    if (!bind_data) {
      throw InternalException("PhysicalLanceWrite requires bind data");
    }
    write_plan.extension_name = "lance";
    write_plan.operator_name = LANCE_DISTRIBUTED_WRITE_OPERATOR;
    write_plan.worker_bind_data =
        SerializeLanceDistributedWriteBind(target, *bind_data);
  }

  bool IsSink() const override { return true; }
  bool IsSource() const override { return true; }
  bool ParallelSink() const override { return false; }
  bool SinkOrderDependent() const override { return true; }

  unique_ptr<GlobalSinkState>
  GetGlobalSinkState(ClientContext &context) const override {
    auto result = make_uniq<LancePhysicalWriteGlobalState>();
    result->bind_data = bind_data->Copy();
    result->function_state =
        LanceWriteInitGlobal(context, *result->bind_data, target);
    return std::move(result);
  }

  unique_ptr<LocalSinkState>
  GetLocalSinkState(ExecutionContext &context) const override {
    if (!sink_state) {
      throw InternalException("Lance write has no global sink state");
    }
    auto &global = sink_state->Cast<LancePhysicalWriteGlobalState>();
    return make_uniq<LancePhysicalWriteLocalState>(
        LanceWriteInitLocal(context, *global.bind_data));
  }

  SinkResultType Sink(ExecutionContext &context, DataChunk &chunk,
                      OperatorSinkInput &input) const override {
    auto &global = input.global_state.Cast<LancePhysicalWriteGlobalState>();
    auto &local = input.local_state.Cast<LancePhysicalWriteLocalState>();
    LanceWriteSink(context, *global.bind_data, *global.function_state,
                   *local.function_state, chunk);
    if (chunk.size() > std::numeric_limits<idx_t>::max() - global.row_count) {
      throw OutOfRangeException("Lance write row count overflow");
    }
    global.row_count += chunk.size();
    return SinkResultType::NEED_MORE_INPUT;
  }

  SinkCombineResultType Combine(ExecutionContext &,
                                OperatorSinkCombineInput &) const override {
    return SinkCombineResultType::FINISHED;
  }

  SinkFinalizeType Finalize(Pipeline &, Event &, ClientContext &context,
                            OperatorSinkFinalizeInput &input) const override {
    auto &global = input.global_state.Cast<LancePhysicalWriteGlobalState>();
    if (global.finalized) {
      throw InternalException("Lance write finalized more than once");
    }
    LanceWriteFinalize(context, *global.bind_data, *global.function_state);
    global.finalized = true;
    return SinkFinalizeType::READY;
  }

  unique_ptr<GlobalSourceState>
  GetGlobalSourceState(ClientContext &) const override {
    return make_uniq<LancePhysicalWriteSourceState>();
  }

  SourceResultType GetDataInternal(ExecutionContext &, DataChunk &chunk,
                                   OperatorSourceInput &input) const override {
    auto &source = input.global_state.Cast<LancePhysicalWriteSourceState>();
    if (source.emitted) {
      return SourceResultType::FINISHED;
    }
    if (!sink_state) {
      throw InternalException("Lance write source has no sink state");
    }
    auto &global = sink_state->Cast<LancePhysicalWriteGlobalState>();
    if (!global.finalized) {
      throw InternalException("Lance write source ran before finalization");
    }
    chunk.SetValue(0, 0, Value::BIGINT(NumericCast<int64_t>(global.row_count)));
    chunk.SetCardinality(1);
    source.emitted = true;
    return SourceResultType::FINISHED;
  }

  optional_ptr<distributed::ExtensionWriteTaskProvider>
  GetExtensionWriteTaskProvider() override {
    return this;
  }

  const distributed::DistributedExtensionWritePlan &WritePlan() const override {
    return write_plan;
  }

  void
  ValidateDistributedWrite(ClientContext &context,
                           const distributed::DistributedWriteOperationContext
                               &operation) const override {
    auto resolved = ResolveLanceWriteTarget(context, target);
    auto rc = lance_distributed_write_validate(
        resolved.open_path.c_str(), bind_data->mode.c_str(),
        resolved.key_ptrs.empty() ? nullptr : resolved.key_ptrs.data(),
        resolved.value_ptrs.empty() ? nullptr : resolved.value_ptrs.data(),
        resolved.option_keys.size(), LanceGetSessionHandle(context),
        operation.operation_id.c_str());
    if (rc != 0) {
      throw IOException("Distributed Lance write validation failed: " +
                        resolved.open_path + LanceFormatErrorSuffix());
    }
  }

  idx_t FinalizeDistributedWrite(
      ClientContext &context,
      const distributed::DistributedWriteOperationContext &operation,
      const vector<DistributedWriteTaskResult> &results) const override {
    auto resolved = ResolveLanceWriteTarget(context, target);
    vector<const char *> task_ids;
    vector<const uint8_t *> transaction_data;
    vector<size_t> transaction_lens;
    set<string> selected_tasks;
    idx_t selected_rows = 0;
    task_ids.reserve(results.size());
    transaction_data.reserve(results.size());
    transaction_lens.reserve(results.size());

    for (const auto &result : results) {
      if (result.operation_id != operation.operation_id) {
        throw SerializationException(
            "Distributed Lance result operation identity mismatch");
      }
      if (!selected_tasks.insert(result.task_attempt_id).second) {
        throw SerializationException(
            "Distributed Lance write selected task '%s' more than once",
            result.task_attempt_id);
      }
      if (result.fragments.size() != 1) {
        throw SerializationException(
            "Distributed Lance task '%s' returned %llu fragments instead of "
            "one",
            result.task_attempt_id,
            static_cast<unsigned long long>(result.fragments.size()));
      }
      const auto &fragment = result.fragments[0];
      if (fragment.fragment_id != result.task_attempt_id ||
          fragment.payload.empty()) {
        throw SerializationException("Distributed Lance task '%s' returned an "
                                     "invalid transaction fragment",
                                     result.task_attempt_id);
      }
      if (fragment.artifacts.size() != 1) {
        throw SerializationException("Distributed Lance task '%s' returned an "
                                     "invalid staging artifact count",
                                     result.task_attempt_id);
      }
      const auto &artifact = fragment.artifacts[0];
      DistributedPayloadCodec expected_artifact_codec;
      expected_artifact_codec.name = LANCE_STAGING_ARTIFACT_CODEC;
      expected_artifact_codec.version = LANCE_STAGING_ARTIFACT_CODEC_VERSION;
      auto expected_uri = LanceStagingUri(
          resolved.open_path, operation.operation_id, result.task_attempt_id);
      if (artifact.artifact_id != result.task_attempt_id ||
          artifact.codec != expected_artifact_codec ||
          artifact.uri != expected_uri || !artifact.payload.empty()) {
        throw SerializationException(
            "Distributed Lance task '%s' returned an invalid staging artifact",
            result.task_attempt_id);
      }
      if (fragment.row_count >
          std::numeric_limits<idx_t>::max() - selected_rows) {
        throw OutOfRangeException("Distributed Lance write row count overflow");
      }
      selected_rows += fragment.row_count;
      task_ids.push_back(result.task_attempt_id.c_str());
      transaction_data.push_back(
          reinterpret_cast<const uint8_t *>(fragment.payload.data()));
      transaction_lens.push_back(fragment.payload.size());
    }

    uint64_t committed_rows = 0;
    auto rc = lance_distributed_write_commit(
        resolved.open_path.c_str(), bind_data->mode.c_str(),
        resolved.key_ptrs.empty() ? nullptr : resolved.key_ptrs.data(),
        resolved.value_ptrs.empty() ? nullptr : resolved.value_ptrs.data(),
        resolved.option_keys.size(), LanceGetSessionHandle(context),
        operation.operation_id.c_str(),
        task_ids.empty() ? nullptr : task_ids.data(),
        transaction_data.empty() ? nullptr : transaction_data.data(),
        transaction_lens.empty() ? nullptr : transaction_lens.data(),
        task_ids.size(), NumericCast<uint64_t>(selected_rows), &committed_rows);
    if (rc != 0) {
      auto error = LanceConsumeLastErrorDetail();
      auto suffix = error.ToString();
      if (!suffix.empty()) {
        suffix = " (Lance error: " + suffix + ")";
      }
      if (error.code == LANCE_DATASET_COMMIT_OUTCOME_UNKNOWN) {
        throw distributed::DistributedWriteOutcomeUnknownException(
            "Distributed Lance commit outcome is unknown for " +
            resolved.open_path + suffix);
      }
      throw IOException(
          "Distributed Lance commit failed: " + resolved.open_path + suffix);
    }
    if (committed_rows != NumericCast<uint64_t>(selected_rows)) {
      throw distributed::DistributedWriteOutcomeUnknownException(
          "Distributed Lance commit returned after committing an unexpected "
          "row count: expected %llu, got %llu; "
          "the operation must not be retried",
          static_cast<unsigned long long>(selected_rows),
          static_cast<unsigned long long>(committed_rows));
    }
    LanceInvalidateDatasetCache(context, resolved.cache_key);
    return NumericCast<idx_t>(committed_rows);
  }

  void AbortDistributedWrite(
      ClientContext &context,
      const distributed::DistributedWriteOperationContext &operation,
      const vector<DistributedWriteTaskResult> &) const override {
    auto resolved = ResolveLanceWriteTarget(context, target);
    auto rc = lance_distributed_write_abort(
        resolved.open_path.c_str(),
        resolved.key_ptrs.empty() ? nullptr : resolved.key_ptrs.data(),
        resolved.value_ptrs.empty() ? nullptr : resolved.value_ptrs.data(),
        resolved.option_keys.size(), LanceGetSessionHandle(context),
        operation.operation_id.c_str());
    if (rc != 0) {
      throw IOException("Distributed Lance abort failed: " +
                        resolved.open_path + LanceFormatErrorSuffix());
    }
  }

  string GetName() const override { return "LANCE_WRITE"; }

protected:
  void SerializeOperatorData(Serializer &) const override {
    throw NotImplementedException(
        "Coordinator-only Lance write roots are not serializable");
  }

private:
  string target;
  unique_ptr<LanceWriteBindData> bind_data;
  distributed::DistributedExtensionWritePlan write_plan;
};

class LogicalLanceWrite final : public LogicalExtensionOperator {
public:
  LogicalLanceWrite(string target_p, unique_ptr<LanceWriteBindData> bind_data_p)
      : target(std::move(target_p)), bind_data(std::move(bind_data_p)) {
    if (!bind_data) {
      throw InternalException("LogicalLanceWrite requires bind data");
    }
  }

  PhysicalOperator &CreatePlan(ClientContext &,
                               PhysicalPlanGenerator &planner) override {
    if (children.size() != 1) {
      throw InternalException("LogicalLanceWrite requires exactly one child");
    }
    auto &child = planner.CreatePlan(*children[0]);
    auto &result = planner.Make<PhysicalLanceWrite>(
        target, bind_data->CopyTyped(), estimated_cardinality);
    result.children.push_back(child);
    return result;
  }

  vector<ColumnBinding> GetColumnBindings() override {
    return {ColumnBinding(0, 0)};
  }

  idx_t EstimateCardinality(ClientContext &) override { return 1; }

  string GetName() const override { return "LANCE_WRITE"; }

  string GetExtensionName() const override {
    return LANCE_WRITE_LOGICAL_EXTENSION;
  }

  void Serialize(Serializer &serializer) const override {
    LogicalExtensionOperator::Serialize(serializer);
    serializer.WriteProperty(201, "target", target);
    serializer.WriteProperty(202, "mode", bind_data->mode);
    serializer.WriteProperty(203, "data_storage_version",
                             bind_data->data_storage_version);
    serializer.WriteProperty(204, "max_rows_per_file",
                             bind_data->max_rows_per_file);
    serializer.WriteProperty(205, "max_rows_per_group",
                             bind_data->max_rows_per_group);
    serializer.WriteProperty(206, "max_bytes_per_file",
                             bind_data->max_bytes_per_file);
    serializer.WriteProperty(207, "names", bind_data->names);
    serializer.WriteProperty(208, "types", bind_data->types);
    serializer.WriteProperty(209, "vector_dims", bind_data->vector_dims);
    serializer.WriteProperty(210, "infer_vector_dims",
                             bind_data->infer_vector_dims);
  }

protected:
  void ResolveTypes() override { types = {LogicalType::BIGINT}; }

private:
  string target;
  unique_ptr<LanceWriteBindData> bind_data;
};

static BoundStatement LanceWriteFallbackBind(ClientContext &, Binder &,
                                             OperatorExtensionInfo *,
                                             SQLStatement &) {
  return BoundStatement();
}

class LanceWriteOperatorExtension final : public OperatorExtension {
public:
  LanceWriteOperatorExtension() { Bind = LanceWriteFallbackBind; }

  string GetName() override { return LANCE_WRITE_LOGICAL_EXTENSION; }

  unique_ptr<LogicalExtensionOperator>
  Deserialize(Deserializer &deserializer) override {
    auto target = deserializer.ReadProperty<string>(201, "target");
    auto bind_data = make_uniq<LanceWriteBindData>();
    bind_data->mode = deserializer.ReadProperty<string>(202, "mode");
    bind_data->data_storage_version =
        deserializer.ReadProperty<string>(203, "data_storage_version");
    bind_data->max_rows_per_file =
        deserializer.ReadProperty<uint64_t>(204, "max_rows_per_file");
    bind_data->max_rows_per_group =
        deserializer.ReadProperty<uint64_t>(205, "max_rows_per_group");
    bind_data->max_bytes_per_file =
        deserializer.ReadProperty<uint64_t>(206, "max_bytes_per_file");
    bind_data->names = deserializer.ReadProperty<vector<string>>(207, "names");
    bind_data->types =
        deserializer.ReadProperty<vector<LogicalType>>(208, "types");
    bind_data->vector_dims =
        deserializer.ReadProperty<string>(209, "vector_dims");
    bind_data->infer_vector_dims =
        deserializer.ReadProperty<bool>(210, "infer_vector_dims");
    return make_uniq<LogicalLanceWrite>(std::move(target),
                                        std::move(bind_data));
  }
};

static BoundStatement LanceWritePlan(Binder &binder, CopyStatement &stmt);

static CopyFunction MakeLanceWriteFunction(bool with_plan) {
  CopyFunction function("lance");
  function.extension = "lance";
  function.plan = with_plan ? LanceWritePlan : nullptr;
  function.copy_options = LanceWriteCopyOptions;
  function.copy_to_bind = LanceWriteBind;
  function.copy_to_initialize_global = LanceWriteInitGlobal;
  function.copy_to_initialize_local = LanceWriteInitLocal;
  function.copy_to_sink = LanceWriteSink;
  function.copy_to_finalize = LanceWriteFinalize;
  function.serialize = LanceWriteSerialize;
  function.deserialize = LanceWriteDeserialize;
  return function;
}

static void ValidateLanceWriteTargetIsPortable(const string &target) {
  if (target.empty()) {
    throw BinderException("Lance COPY target cannot be empty");
  }
  if (target.find('?') != string::npos || target.find('#') != string::npos) {
    throw BinderException("Lance COPY targets cannot contain URI query "
                          "parameters or fragments; use CREATE SECRET "
                          "for credentials and storage options");
  }
  auto scheme = target.find("://");
  if (scheme == string::npos) {
    return;
  }
  auto authority_begin = scheme + 3;
  auto authority_end = target.find('/', authority_begin);
  auto authority =
      target.substr(authority_begin, authority_end - authority_begin);
  if (authority.find('@') != string::npos) {
    throw BinderException("Lance COPY targets cannot contain URI user "
                          "information; use CREATE SECRET for "
                          "credentials and storage options");
  }
}

static BoundStatement LanceWritePlan(Binder &binder, CopyStatement &stmt) {
  static const set<string> supported_options = {"mode",
                                                "data_storage_version",
                                                "max_rows_per_file",
                                                "max_rows_per_group",
                                                "max_bytes_per_file",
                                                "vector_dims",
                                                "infer_vector_dims"};
  for (const auto &option : stmt.info->options) {
    auto key = StringUtil::Lower(option.first);
    if (supported_options.find(key) == supported_options.end()) {
      throw NotImplementedException(
          "COPY TO FORMAT LANCE does not support option '%s'", option.first);
    }
  }
  ValidateLanceWriteTargetIsPortable(stmt.info->file_path);

  BoundStatement select;
  if (stmt.info->select_relation) {
    select = stmt.info->select_relation->Bind(binder);
  } else {
    auto query = stmt.info->select_statement->Copy();
    select = binder.Bind(*query);
  }
  if (!select.plan || select.names.size() != select.types.size()) {
    throw InternalException(
        "Lance COPY source bind produced an invalid logical plan");
  }
  auto names = select.names;
  QueryResult::DeduplicateColumns(names);
  CopyFunctionBindInput bind_input(*stmt.info);
  bind_input.file_extension = "lance";
  auto function_data =
      LanceWriteBind(binder.context, bind_input, names, select.types);
  auto bind_data = unique_ptr<LanceWriteBindData>(
      static_cast<LanceWriteBindData *>(function_data.release()));

  auto target = LanceNormalizeDatasetPath(binder.context, stmt.info->file_path);
  auto logical =
      make_uniq<LogicalLanceWrite>(std::move(target), std::move(bind_data));
  logical->AddChild(std::move(select.plan));
  logical->ResolveOperatorTypes();

  BoundStatement result;
  result.plan = std::move(logical);
  result.names = {"Count"};
  result.types = {LogicalType::BIGINT};
  binder.GetStatementProperties().return_type =
      StatementReturnType::CHANGED_ROWS;
  return result;
}

void RegisterLanceWrite(ExtensionLoader &loader) {
  loader.RegisterFunction(MakeLanceWriteFunction(true));

  auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());
  OperatorExtension::Register(config,
                              make_shared_ptr<LanceWriteOperatorExtension>());

  DistributedWriteOperatorExtension distributed_write;
  distributed_write.name = LANCE_DISTRIBUTED_WRITE_OPERATOR;
  distributed_write.protocol_version = LANCE_DISTRIBUTED_WRITE_PROTOCOL_VERSION;
  distributed_write.mode = DistributedWriteMode::CALLBACK;
  distributed_write.fragment_codec.name = LANCE_TRANSACTION_CODEC;
  distributed_write.fragment_codec.version = LANCE_TRANSACTION_CODEC_VERSION;
  distributed_write.callbacks.initialize_global =
      LanceDistributedWriteInitializeGlobal;
  distributed_write.callbacks.initialize_local =
      LanceDistributedWriteInitializeLocal;
  distributed_write.callbacks.sink = LanceDistributedWriteSink;
  distributed_write.callbacks.combine = LanceDistributedWriteCombine;
  distributed_write.callbacks.finalize = LanceDistributedWriteFinalize;
  DistributedWriteOperatorExtension::Register(loader,
                                              std::move(distributed_write));
}

} // namespace duckdb
