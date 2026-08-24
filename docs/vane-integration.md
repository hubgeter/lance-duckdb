# Vane Integration Architecture

## Goal

`lance-duckdb` is an out-of-tree DuckDB extension. Vane is one supported host,
not the owner or distributor of the extension. The integration follows the
same ownership direction as a normal DuckDB extension:

```text
lance-duckdb source repository
  ├── official-DuckDB loadable artifact
  ├── Vane-DuckDB loadable artifact
  ├── SQL and optional Python convenience APIs
  ├── native, Python, Vane, and service-backed tests
  ├── examples and user/architecture documentation
  └── release and CI workflows

Vane source repository
  └── format-neutral extension host and distributed execution contracts
```

Vane does not fetch this repository, pin a lance-duckdb commit, compile Lance
as an in-tree extension, link Lance into `vane._native`, copy Lance tests or
examples, or publish Lance artifacts.

The `[vane] revision` in `vane-extension.toml` points in the opposite
direction: this plugin pins the exact Vane ABI used by its own reproducible CI.
Updating that tested host revision is a lance-duckdb maintenance operation; it
does not make Lance part of a Vane release.

## Change inventory and necessity

### Vane repository

The host-side patch contains only format-neutral runtime contracts:

| Area | Files | Necessity |
| --- | --- | --- |
| Loadable extension identity and replay | `src/vane_py/ray/logical_plan_bindings.cpp`, `distributed_plan_bindings.cpp`, `ray_module.cpp`, `CMakeLists.txt`, and `vane/runners/ray/worker.py` | Required for any independently shipped extension used by Ray. The coordinator records mode/path/digest/contracts; workers load only the exact pre-provisioned artifact, and worker database cache identity includes those fields. |
| Original `ATTACH` identity | DuckDB `attached_database.hpp/.cpp` plus connection snapshot capture | Required for distributed attached catalogs. A storage extension may normalize its catalog path and may report a catalog implementation type different from the requested `TYPE`, so neither existing value is sufficient to reconstruct the original `ATTACH`. |
| External catalog refresh | DuckDB `catalog_set.hpp/.cpp` plus `src/vane_py/pyrelation.cpp` | Required for distributed CTAS visibility. The commit occurs through a replay/coordinator connection; resetting a generic lazy default-entry generator lets the caller discover the newly committed external table. It is a no-op for catalog sets without such a generator. |
| Contract documentation and regression tests | `DISTRIBUTED_EXTENSIONS.md` and the two connection-snapshot test modules | Required to make the security, compatibility, legacy-static, digest, and cache-key behavior reviewable and stable. These tests contain no Lance fixtures or package dependency. |

No Vane runtime code names Lance, stores a Lance dataset lease, or exposes a
Lance-specific API. A proposed generic `EXTENSION_RELATION` write
classification was also removed because the final integration uses DuckDB's
existing create, insert, and write-file Relation types and therefore has no
consumer for that extra branch.

### lance-duckdb repository

| Area | Files | Necessity |
| --- | --- | --- |
| Independent dual-ABI build | `CMakeLists.txt`, `extension_config_vane.cmake`, `vane-extension.toml`, and the CI-tools submodule | Required. The same implementation is conditionally compiled against official DuckDB or Vane headers, while the resulting binaries remain separate ABI products. |
| Current distributed scan contract | `src/lance_scan.cpp` and `src/lance_search.cpp` | Required for the pinned Vane host. Lance bind state is serialized once, the coordinator returns opaque `DistributedScanSplit` values, and workers apply only assigned splits. The official build excludes these callbacks. |
| Native write and lease ownership | Lance physical operators, C++ lease state, and Rust FFI/storage records | Required to retain distributed INSERT/CTAS/COPY, retry identity, snapshot safety, mutation serialization, and VACUUM coordination without a Vane Python lease coordinator. |
| Optional Python convenience package | `python/lance_duckdb` | Required only for the retained `LanceDataset`, namespace, index, and search convenience surface. It contains no native binary and uses standard SQL/Relation operations; native SQL remains usable without installing this package. Default direct creates use the generic `write_file` Relation boundary when a Vane runner is active. |
| Plugin-owned validation and delivery | `.github/workflows`, `python/tests`, `test/sql`, `examples/vane`, and helper scripts | Required repository ownership, though not runtime code. Official-DuckDB tests, Vane integration tests, examples, and artifacts are produced here instead of in Vane. |
| Host-version compatibility fixes | Lance scan/search/storage/write sources and focused SQL expectations | Required to keep existing behavior correct on the pinned official-DuckDB and Vane revisions: preserve persisted defaults, accept DuckDB's `write_empty_file=true` contract, reject unsafe sampled exec pushdown, and follow current SQL result semantics. These fixes are shared and are not Vane-only APIs. |
| Architecture and user documentation | `README.md` and `docs/` | Required for maintainability and release clarity, but not for execution. They describe artifact selection, deployment security, ownership, and operational limits. |

During coordinated development, the Vane host changes must be committed first.
Only then should `vane-extension.toml` be updated to that immutable commit and
the plugin CI rerun. The manifest must never point at an uncommitted worktree.

## Artifact matrix

The same source has two compile-time configurations and two native products:

| Product | DuckDB source | Configuration | Distributed adapter | Delivery |
| --- | --- | --- | --- | --- |
| Official DuckDB | `duckdb/` submodule | `extension_config.cmake` | Disabled | Loadable extension from lance-duckdb |
| Vane | pinned Vane `external/duckdb` | `extension_config_vane.cmake` | Enabled | Loadable extension from lance-duckdb |

The C++ ABIs are not interchangeable. A source revision match is insufficient:
an official-DuckDB artifact must not be loaded into Vane, and a Vane artifact
must not be loaded into official DuckDB.

`build_static_extension` remains in `CMakeLists.txt` only because DuckDB's
native `unittest` executable links the extension target while running SQLLogic
tests. The independently shipped product in both modes is
`lance.duckdb_extension`; no static Lance object is linked into a Vane package.

## Building

### Official DuckDB ABI

```bash
git submodule update --init --recursive
GEN=ninja make release -j 4
./build/release/test/unittest "test/*"
```

The loadable artifact is produced below
`build/release/extension/lance/lance.duckdb_extension`.

### Vane ABI

The Vane-specific CI tools submodule consumes `vane-extension.toml`, verifies
exact tool and host commits, builds against the selected Vane
`external/duckdb`, and runs the plugin-owned native smoke test:

```bash
export VCPKG_TOOLCHAIN_PATH=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
make vane_ci
```

The default artifact is produced below
`build/vane-native/extension/lance/lance.duckdb_extension`. The native test
binary is only a test harness; it is not the Vane deliverable.

## Loading and deployment

The optional `lance-duckdb` Python package contains convenience code, not the
native extension. `load_lance_extension` accepts an explicit artifact path or
reads `LANCE_DUCKDB_EXTENSION`, canonicalizes the local path, checks that it is
a file, and calls the host connection's normal `load_extension` method. It has
no `INSTALL`, repository lookup, network download, or ABI fallback.

```python
import vane
from lance_duckdb import load_lance_extension

connection = vane.connect()
load_lance_extension(
    connection,
    "/opt/vane/extensions/lance.duckdb_extension",
)
```

Development builds are normally unsigned. A local test may create its
coordinator connection with `allow_unsigned_extensions=true`, but production
must use a signed Vane-ABI artifact. Ray snapshot replay deliberately forces
all of these settings off:

- `allow_unsigned_extensions`;
- `autoinstall_known_extensions`;
- `autoload_known_extensions`.

Every Ray node must already contain the same signed bytes at the same canonical
absolute path, normally through a shared image or shared filesystem. Vane does
not distribute the binary at query time.

## Coordinator-to-worker extension identity

Vane's connection snapshot is format-neutral. For every loaded extension it
records:

```text
name
extension version
mode = STATICALLY_LINKED | LOADABLE
canonical local path     # loadable only
SHA-256 of artifact      # loadable only
```

It also records the DuckDB SourceID and the sorted distributed contract
identities registered by all extensions.

The Ray worker performs the following fail-closed sequence before plan
deserialization:

```text
receive connection snapshot
  -> compare DuckDB SourceID
  -> force extension security settings off
  -> locate only the recorded pre-provisioned path
  -> hash artifact
  -> load through DuckDB's normal ExtensionHelper
  -> hash artifact again
  -> compare all loaded extension identities exactly
  -> compare all distributed contract identities exactly
  -> replay attached catalogs and deserialize the logical plan
```

The digest detects a different or concurrently replaced artifact. The second
hash closes the load-time replacement window when a binary is first loaded.
After a successful full verification, Vane stores the exact extension identity
in that DatabaseInstance's generic object cache. Later task cursors still
compare the loaded name, version, mode, canonical path, and distributed
contracts, but do not re-read a large immutable binary on every replay. A later
snapshot may add an extension, but it cannot change or omit an already verified
binary. The worker database-cache key also includes mode, path, and digest so
two extension artifacts cannot share a cached worker database identity
accidentally.

This connection snapshot is not a Lance dataset snapshot. It fixes executable
code and host contracts. The Lance scan bind state separately fixes the
dataset version/generation used by one query.

## Distributed reads

When `LANCE_VANE_DISTRIBUTED=ON`, the extension registers Vane's generic
`TableFunctionDistributedScanCallbacks` on Lance scan and search table
functions. The normal bind callback resolves the dataset and serializes the
immutable bind state. The coordinator then enumerates opaque fragment splits.
Workers deserialize that bind state and apply only their selected splits; they
do not repeat catalog discovery or choose a newer dataset version.

The native physical/table-function global state owns the snapshot lease for
the lifetime of the scan. This keeps ownership next to the resource it guards
and works for SQL, Python relations, and Ray workers without a second Python
lease coordinator.

## Distributed writes

There is no `ExecuteWriteRelation` hook and no Lance-specific write method in
Vane. The entry points are DuckDB's ordinary SQL and Relation operations:

```sql
COPY (SELECT ...) TO 'dataset.lance' (FORMAT lance, MODE 'append');

ATTACH '/datasets' AS lake (TYPE LANCE);
CREATE TABLE lake.main.items AS SELECT ...;
INSERT INTO lake.main.items SELECT ...;
```

For Vane distributed execution, direct-path COPY is represented by the
existing format-neutral Relation API:

```python
source.write_file("/datasets/items.lance", format="lance")
```

`LanceDataset.write(source)` uses that generic path for a default create when a
Vane Ray/local-fast runner is active. Plain `connection.execute("COPY ...")`,
direct append/overwrite, and option-bearing direct writes remain driver-local;
the current generic `WriteFileRelation` does not carry those Lance writer
options. Attached-table `create()` and `insert_into()` are the standard
distributed paths for CTAS and append.

The optional Python facade otherwise generates standard statements or calls
the standard Relation `create`/`insert_into` methods. It does not add
`relation.write_lance()`, `relation.to_lance()`, or `vane.read_lance()`.

For attached tables, `LanceDuckCatalog::PlanInsert` and
`PlanCreateTableAs` select Lance physical operators. Under the Vane build those
operators expose the generic distributed write provider. Direct
`COPY ... FORMAT lance` represented by `WriteFileRelation` selects the same
provider. The execution flow is:

```text
ordinary DuckDB write relation
  -> Lance PlanInsert / PlanCreateTableAs / COPY operator
  -> ExtensionWriteTaskProvider creates one operation identity
  -> Vane schedules input partitions on Ray workers
  -> workers write retry-scoped staging artifacts and transaction fragments
  -> scheduler selects one successful attempt per task
  -> coordinator validates selected fragments
  -> coordinator performs exactly one Lance commit
  -> coordinator releases the mutation lease
```

Vane owns scheduling, retry selection, transport, and the coordinator
transaction boundary. lance-duckdb owns target validation, worker artifact
format, commit, abort, cache invalidation, and outcome-unknown behavior.

## Snapshot, mutation, and vacuum leases

Lease implementation and ownership live in lance-duckdb. The C++ operator
states hold `LanceLease` objects; the Rust FFI stores lease records in the
dataset's backing object store. Independent Vane processes and Ray workers
using the same dataset identity and credentials therefore observe the same
records.

The three lease kinds have different conflicts:

- a snapshot lease protects the fixed dataset version used by an active read;
- a mutation lease serializes participating writers and destructive metadata
  changes;
- a vacuum lease waits for participating snapshots and excludes participating
  mutations while obsolete files may be deleted.

A normal commit releases its mutation lease. When commit success or lease
release is uncertain, the exact token is retained for operator-assisted
reconciliation and callers are told not to retry automatically.

These records coordinate only clients that implement the lance-duckdb lease
protocol. An external Lance writer or vacuum tool that ignores `_vane_leases`
is not controlled by Vane. Deployments must therefore enforce one of the
following policies:

- route destructive maintenance through cooperating lance-duckdb instances;
- configure an external retention window that outlives all possible readers;
- or integrate the external writer/maintenance system with the same storage
  lease protocol.

VACUUM is not required to read or write Lance data. It is an explicit
maintenance operation used to reclaim obsolete files. The vacuum lease is
required only when this plugin performs destructive cleanup; it cannot make an
uncooperative external vacuum safe.

## Optional Python surface

The `python/lance_duckdb` package owns these convenience APIs:

- `load_lance_extension`;
- `LanceDataset`;
- `LanceNamespace` and attached-table helpers;
- index, vector search, full-text search, and hybrid-search helpers;
- typed outcome-unknown errors and URI identity helpers.

The package works without Vane for official DuckDB-local use. Vane imports are
optional and limited to distributed runner dispatch and richer Vane error
types when that host is installed. The facade does not require Vane's former
private `_shares_connection` or `_is_auto_commit` methods. Native Lance
operators remain authoritative for transaction and lease validation.

## Required Vane changes

The integration requires a small set of host capabilities. Each is
format-neutral and can be used by another out-of-tree extension:

| Vane area | Change | Why it is required |
| --- | --- | --- |
| Ray connection snapshot | Represent loadable extension mode, canonical path, SHA-256, and exact contract identity; replay it with downloads disabled | Without this, an independently shipped extension is absent on workers and distributed plans cannot deserialize safely |
| Attached database metadata | Preserve the original `ATTACH` path and requested storage type | A replay connection must reconstruct any external catalog; `Catalog::GetCatalogType()` and the normalized backing path can lose that information |
| External catalog visibility | Invalidate lazy default-entry enumeration after a distributed external write | The commit occurs through a coordinator/replay connection; the caller's catalog cache must re-list externally created tables |
| Distributed extension documentation and tests | Cover loadable identity, security, replay, and cache keys | These are Vane host invariants and therefore remain in Vane |

## Changes deliberately absent from Vane

The following mechanisms are not needed and must not be reintroduced for
Lance:

- a lance-duckdb source URL, commit, CMake fetch, or package pin;
- `DUCKDB_LANCE_DIRECTORY` or a Lance static-extension build entry;
- Lance binaries in Vane wheels and source distributions;
- Lance tests, examples, Python modules, or architecture documents in Vane;
- `AttachExternalDependency` or a Python relation dependency graph;
- `ExecuteWriteRelation` or another extension-specific relation executor;
- `_shares_connection`, `_is_auto_commit`, `read_lance`, `write_lance`, or
  `to_lance` in the Vane API.

## CI ownership

This repository has three independent validation layers:

1. The official DuckDB workflow builds the official ABI artifact and runs the
   complete SQLLogicTest suite, including MinIO where configured.
2. The Vane workflow invokes the pinned Vane extension CI tools, builds the
   Vane ABI loadable artifact, and runs the plugin-owned native dual-build
   smoke test. It then installs that exact Vane host and the lance-duckdb
   Python package non-editably and runs `python/tests/test_vane.py` in unsigned
   local mode. It uploads only the independent extension artifact, never a
   Lance-bearing Vane wheel.
3. The Python workflow installs this package non-editably and runs pure helper
   tests without requiring Vane or a native artifact.

`python/tests/test_vane.py` also owns the signed-artifact Ray integration
matrix. Service-backed REST, S3/MinIO, signed-artifact Ray, and multi-node tests
remain plugin-owned lanes. A local unsigned build cannot prove the production
Ray loading path because workers intentionally reject unsigned extensions.

## Known operational limits

- Every distributed worker needs the same canonical absolute artifact path.
- Production distributed execution needs a Vane-ABI artifact signed by a key
  trusted by that Vane build.
- Extension identity pinning prevents code drift; it does not prevent an
  external Lance client from committing a newer dataset version.
- Lance bind state and native snapshot leases protect the version selected by
  a participating query. They do not govern external clients that ignore the
  lease protocol.
- Outcome-unknown commits are never safe to retry automatically solely because
  a task failed or timed out.
