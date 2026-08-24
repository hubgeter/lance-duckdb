#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"

#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_ffi.hpp"
#include "lance_insert.hpp"
#include "lance_lease.hpp"
#include "lance_session_state.hpp"
#include "lance_table_entry.hpp"
#include "lance_write.hpp"

#include <cstdint>
#include <limits>

namespace duckdb {

struct LanceInsertGlobalState : public GlobalSinkState {
  explicit LanceInsertGlobalState(const LanceTableEntry &table_p,
                                  vector<string> column_names_p,
                                  vector<LogicalType> column_types_p)
      : table(&table_p), column_names(std::move(column_names_p)),
        column_types(std::move(column_types_p)) {}

  mutex lock;

  const LanceTableEntry *table = nullptr;
  string display_uri;
  string open_path;
  string cache_key;
  vector<string> option_keys;
  vector<string> option_values;
  unique_ptr<LanceLease> mutation_lease;

  vector<string> column_names;
  vector<LogicalType> column_types;

  idx_t insert_count = 0;

  void *writer = nullptr;
  ArrowSchemaWrapper schema_root;

  ~LanceInsertGlobalState() override {
    if (writer) {
      lance_close_writer(writer);
      writer = nullptr;
    }
  }
};

struct LanceInsertLocalState : public LocalSinkState {};

class PhysicalLanceInsert final : public PhysicalOperator {
public:
  static constexpr const PhysicalOperatorType TYPE =
      PhysicalOperatorType::EXTENSION;

  PhysicalLanceInsert(
      PhysicalPlan &physical_plan, vector<LogicalType> types_p,
      LanceTableEntry &table_p, vector<string> column_names_p,
      vector<LogicalType> column_types_p, idx_t estimated_cardinality
#ifdef LANCE_VANE_DISTRIBUTED
      ,
      unique_ptr<distributed::ExtensionWriteTaskProvider> distributed_provider_p
#endif
      )
      : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                         std::move(types_p), estimated_cardinality),
        table(table_p), column_names(std::move(column_names_p)),
        column_types(std::move(column_types_p))
#ifdef LANCE_VANE_DISTRIBUTED
        ,
        distributed_provider(std::move(distributed_provider_p))
#endif
  {
  }

  bool IsSink() const override { return true; }
  bool IsSource() const override { return true; }
  bool ParallelSink() const override { return false; }
  bool SinkOrderDependent() const override { return false; }

  unique_ptr<GlobalSinkState>
  GetGlobalSinkState(ClientContext &context) const override {
    RequireLanceMutationSlot(context, table.catalog);
    return make_uniq<LanceInsertGlobalState>(table, column_names, column_types);
  }

  unique_ptr<LocalSinkState>
  GetLocalSinkState(ExecutionContext &) const override {
    return make_uniq<LanceInsertLocalState>();
  }

  SinkResultType Sink(ExecutionContext &context, DataChunk &chunk,
                      OperatorSinkInput &input) const override {
    if (chunk.size() == 0) {
      return SinkResultType::NEED_MORE_INPUT;
    }

    auto &gstate = input.global_state.Cast<LanceInsertGlobalState>();
    lock_guard<mutex> guard(gstate.lock);
    if (chunk.size() >
        std::numeric_limits<idx_t>::max() - gstate.insert_count) {
      throw OutOfRangeException("Lance INSERT row count overflow");
    }

    if (!gstate.writer) {
      auto props = context.client.GetClientProperties();
      memset(&gstate.schema_root.arrow_schema, 0,
             sizeof(gstate.schema_root.arrow_schema));
      ArrowConverter::ToArrowSchema(&gstate.schema_root.arrow_schema,
                                    gstate.column_types, gstate.column_names,
                                    props);

      if (!gstate.table) {
        throw InternalException("Lance INSERT missing table reference");
      }
      ResolveLanceStorageOptionsForTable(
          context.client, *gstate.table, gstate.open_path, gstate.option_keys,
          gstate.option_values, gstate.display_uri);
      gstate.cache_key =
          LanceBuildDatasetCacheKeyForTable(context.client, *gstate.table);

      auto operation_id = "insert:" + gstate.display_uri;
      gstate.mutation_lease = LanceLease::Acquire(
          context.client, gstate.open_path, LanceLeaseKind::Mutation,
          operation_id, gstate.option_keys, gstate.option_values);

      vector<const char *> key_ptrs;
      vector<const char *> value_ptrs;
      BuildStorageOptionPointerArrays(gstate.option_keys, gstate.option_values,
                                      key_ptrs, value_ptrs);

      gstate.writer = lance_open_uncommitted_writer_with_storage_options(
          gstate.open_path.c_str(), "append",
          key_ptrs.empty() ? nullptr : key_ptrs.data(),
          value_ptrs.empty() ? nullptr : value_ptrs.data(),
          gstate.option_keys.size(), LANCE_DEFAULT_MAX_ROWS_PER_FILE,
          LANCE_DEFAULT_MAX_ROWS_PER_GROUP, LANCE_DEFAULT_MAX_BYTES_PER_FILE,
          nullptr, nullptr, 1, LanceGetSessionHandle(context.client),
          &gstate.schema_root.arrow_schema);
      if (!gstate.writer) {
        throw IOException("Failed to open Lance writer: " + gstate.open_path +
                          LanceFormatErrorSuffix());
      }
    }

    unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>>
        extension_type_cast;
    auto props = context.client.GetClientProperties();

    ArrowArrayWrapper array;
    ArrowConverter::ToArrowArray(chunk, &array.arrow_array, props,
                                 extension_type_cast);

    auto rc = lance_writer_write_batch(gstate.writer, &array.arrow_array);
    if (rc != 0) {
      throw IOException("Failed to write to Lance dataset" +
                        LanceFormatErrorSuffix());
    }
    gstate.insert_count += chunk.size();
    return SinkResultType::NEED_MORE_INPUT;
  }

  SinkCombineResultType Combine(ExecutionContext &,
                                OperatorSinkCombineInput &) const override {
    return SinkCombineResultType::FINISHED;
  }

  SinkFinalizeType Finalize(Pipeline &, Event &, ClientContext &context,
                            OperatorSinkFinalizeInput &input) const override {
    auto &gstate = input.global_state.Cast<LanceInsertGlobalState>();
    void *txn = nullptr;

    {
      lock_guard<mutex> guard(gstate.lock);
      if (!gstate.writer) {
        return SinkFinalizeType::READY;
      }
      auto rc = lance_writer_finish_uncommitted(gstate.writer, &txn);
      lance_close_writer(gstate.writer);
      gstate.writer = nullptr;
      if (rc != 0 || !txn) {
        if (txn) {
          lance_free_transaction(txn);
        }
        throw IOException("Failed to finalize Lance append transaction" +
                          LanceFormatErrorSuffix());
      }
    }

    RegisterLancePendingAppend(
        context, table.catalog, std::move(gstate.open_path),
        std::move(gstate.option_keys), std::move(gstate.option_values),
        std::move(gstate.cache_key), txn, std::move(gstate.mutation_lease));
    return SinkFinalizeType::READY;
  }

  class LanceInsertSourceState : public GlobalSourceState {
  public:
    bool emitted = false;
  };

  unique_ptr<GlobalSourceState>
  GetGlobalSourceState(ClientContext &) const override {
    return make_uniq<LanceInsertSourceState>();
  }

  SourceResultType GetDataInternal(ExecutionContext &, DataChunk &chunk,
                                   OperatorSourceInput &input) const override {
    auto &state = input.global_state.Cast<LanceInsertSourceState>();
    if (state.emitted) {
      return SourceResultType::FINISHED;
    }
    state.emitted = true;

    auto &gstate = sink_state->Cast<LanceInsertGlobalState>();
    chunk.SetCardinality(1);
    chunk.SetValue(0, 0,
                   Value::BIGINT(NumericCast<int64_t>(gstate.insert_count)));
    return SourceResultType::FINISHED;
  }

  string GetName() const override { return "LanceInsert"; }

#ifdef LANCE_VANE_DISTRIBUTED
  optional_ptr<distributed::ExtensionWriteTaskProvider>
  GetExtensionWriteTaskProvider() override {
    return distributed_provider.get();
  }
#endif

private:
  LanceTableEntry &table;
  vector<string> column_names;
  vector<LogicalType> column_types;
#ifdef LANCE_VANE_DISTRIBUTED
  unique_ptr<distributed::ExtensionWriteTaskProvider> distributed_provider;
#endif
};

PhysicalOperator &PlanLanceInsertAppend(ClientContext &context,
                                        PhysicalPlanGenerator &planner,
                                        LogicalInsert &op,
                                        optional_ptr<PhysicalOperator> plan) {
  if (op.return_chunk) {
    throw NotImplementedException(
        "Lance INSERT does not support RETURNING yet");
  }
  if (op.on_conflict_info.action_type != OnConflictAction::THROW) {
    throw NotImplementedException("Lance INSERT does not support ON CONFLICT");
  }
  if (!plan) {
    throw InternalException("Lance INSERT requires a child plan");
  }

  if (!op.column_index_map.empty()) {
    plan = planner.ResolveDefaultsProjection(op, *plan);
  }

  auto *lance_table = dynamic_cast<LanceTableEntry *>(&op.table);
  if (!lance_table) {
    throw InternalException("PlanLanceInsertAppend called for non-Lance table");
  }
  if (lance_table->HasCoercedColumns()) {
    throw NotImplementedException(
        "INSERT into Lance table '" + lance_table->name +
        "' is not supported: column(s) [" +
        StringUtil::Join(lance_table->CoercedColumnNames(), ", ") +
        "] have Arrow types DuckDB cannot represent natively, so the "
        "catalog exposes a coerced type. Writing in the coerced type would "
        "silently change the on-disk storage.");
  }

  vector<string> column_names;
  vector<LogicalType> column_types;
  for (auto &col : op.table.GetColumns().Physical()) {
    column_names.push_back(col.Name());
    column_types.push_back(col.Type());
  }

#ifdef LANCE_VANE_DISTRIBUTED
  string distributed_target;
  vector<string> distributed_option_keys;
  vector<string> distributed_option_values;
  string distributed_display_uri;
  ResolveLanceStorageOptionsForTable(
      context, *lance_table, distributed_target, distributed_option_keys,
      distributed_option_values, distributed_display_uri);
  LanceDistributedWriteSpec distributed_spec;
  distributed_spec.target = distributed_target;
  distributed_spec.mode = "append";
  distributed_spec.names = column_names;
  distributed_spec.types = column_types;
  distributed_spec.option_keys = distributed_option_keys;
  distributed_spec.option_values = distributed_option_values;
  auto distributed_provider =
      MakeLanceDistributedWriteProvider(std::move(distributed_spec));
#endif

  auto &insert = planner.Make<PhysicalLanceInsert>(
      op.types, *lance_table, std::move(column_names), std::move(column_types),
      op.estimated_cardinality
#ifdef LANCE_VANE_DISTRIBUTED
      ,
      std::move(distributed_provider)
#endif
  );
  insert.children.push_back(*plan);
  return insert;
}

} // namespace duckdb
