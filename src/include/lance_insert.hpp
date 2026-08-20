#pragma once

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {

class LogicalInsert;
class PhysicalPlanGenerator;

PhysicalOperator &PlanLanceInsertAppend(ClientContext &context,
                                        PhysicalPlanGenerator &planner,
                                        LogicalInsert &op,
                                        optional_ptr<PhysicalOperator> plan);

void RegisterLancePendingAppend(ClientContext &context, Catalog &catalog,
                                string dataset_uri, vector<string> option_keys,
                                vector<string> option_values, string cache_key,
                                void *lance_transaction);

//! Reject a second Lance mutation in the same explicit DuckDB transaction
//! before it can create uncommitted data files. Lance commits are external and
//! cannot be atomically rolled back across multiple pending operations.
void RequireLanceMutationSlot(ClientContext &context, Catalog &catalog);

} // namespace duckdb
