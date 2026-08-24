#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/types.hpp"

#include <cstdint>

#ifdef LANCE_VANE_DISTRIBUTED
#include "duckdb/execution/distributed/extension_write_task_provider.hpp"
#endif

namespace duckdb {

class FunctionData;
class PhysicalOperator;
class PhysicalPlanGenerator;

#ifdef LANCE_VANE_DISTRIBUTED

//! Immutable coordinator-side description shared by Lance's direct COPY and
//! attached-table write roots.  The distributed provider serializes this
//! description into an opaque worker envelope; Vane never interprets Lance
//! schema, transaction, or storage details.
struct LanceDistributedWriteSpec {
  string target;
  string mode = "append";
  string data_storage_version = "2.2";
  uint64_t max_rows_per_file = 1024ULL * 1024ULL;
  uint64_t max_rows_per_group = 1024ULL;
  uint64_t max_bytes_per_file = 90ULL * 1024ULL * 1024ULL * 1024ULL;
  string vector_dims;
  bool infer_vector_dims = true;

  vector<string> names;
  vector<LogicalType> types;

  // These values are coordinator-only.  Worker replayability is validated
  // against the worker's own Lance session before any task starts; resolved
  // secret-backed values are never serialized into worker_bind_data.
  vector<string> option_keys;
  vector<string> option_values;
};

//! Build the callback provider used by a coordinator-only Lance physical root.
//! The returned object owns a fresh operation identity and remains valid for
//! the lifetime of the physical plan.
unique_ptr<distributed::ExtensionWriteTaskProvider>
MakeLanceDistributedWriteProvider(LanceDistributedWriteSpec spec);

#endif // LANCE_VANE_DISTRIBUTED

//! Plan a normal Lance writer from an already-bound COPY function.  This is
//! used by catalog CTAS planning so attached-table CTAS shares the same local
//! writer and (when enabled) distributed provider as COPY ... FORMAT LANCE.
PhysicalOperator &PlanLanceWriteFromBoundData(
    PhysicalPlanGenerator &planner, PhysicalOperator &child, string target,
    unique_ptr<FunctionData> bind_data, vector<LogicalType> result_types,
    idx_t estimated_cardinality);

} // namespace duckdb
