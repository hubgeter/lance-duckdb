# AGENTS.md

## Project Overview

This repository contains the Lance extension source for querying Lance format datasets (including scan, vector search, and full-text search). It can be compiled either against official DuckDB or against Vane's DuckDB fork. The DuckDB integration is implemented in C++ (under `src/`) and links a Rust static library (`lance_duckdb_ffi`) that uses the Lance Rust crate and exports data via the Arrow C Data Interface.

## Documentation Language

All documentation in this repository (including `README.md` and files under `docs/`) must be written in English.

## Constraint Hierarchy (Highest Priority)

For any planning, implementation, review, or refactor work in this repository:

- The hard constraints are Vane's DuckDB fork contracts and upstream Lance requirements.
- If any repository-local guidance conflicts with Vane, DuckDB, or Lance behavior/contracts, follow those upstream contracts.
- Any decision made inside `lance-duckdb` (APIs/ABIs, FFI boundaries, internal formats, module boundaries, naming, testing conventions, or implementation strategies) is mutable and may be changed when a better design is found.
- Treat repository-local rules in this document as default guidance, not immutable policy.

## Deliverable & Compatibility Policy

This project produces two independent loadable artifacts: one for official
DuckDB and one for Vane's DuckDB fork. They must never be mixed because their
C++ ABIs differ. The artifacts, Python helpers, tests, examples, and
release workflows are owned by this repository; Vane must not fetch, pin,
statically link, package, or test Lance as an in-tree component.
Low-level APIs/ABIs/formats in this repository (C++/Rust FFI boundaries,
internal encodings, file/IPC formats, etc.) are strictly internal
implementation details:

- They are not intended to be directly consumed by end users.
- There are no external downstream users that depend on them.
- Prioritize first principles and the most direct correct design; do not optimize for migrations or compatibility unless explicitly requested.
- Compatibility commitments are driven by DuckDB/Lance behavior, not by historical repository-local decisions.
- User-facing contracts should be defined at the SQL surface.

## Public Surface Policy (SQL-First)

Expose user-facing capabilities primarily through SQL, not internal helper functions.

- Default: expose features via DuckDB SQL mechanisms (replacement scan, `ATTACH ... (TYPE LANCE)`, standard DDL/DML, and `COPY ... (FORMAT lance)`).
- Internal functions may exist for implementation/composition, but should not be the primary user-facing entry points.
- Narrow exceptions are allowed when a dedicated function is the clearest SQL contract (for example, `lance_fts` and similar search entry points).

When in doubt, prefer SQL surface area that matches DuckDB idioms over extension-specific internal entry points.

## Reuse Upstream Mechanisms First

Prefer DuckDB/Lance native mechanisms over extension-specific replacements.

- Before adding a new mechanism, first check whether DuckDB or Lance already provides the required hook, lifecycle, cache, profiling surface, or introspection path.
- If an upstream mechanism can satisfy the requirement with reasonable adaptation, use it instead of inventing a parallel extension-specific mechanism.
- Prefer integration with DuckDB-native observability surfaces such as profiling, `EXPLAIN`, and catalog/state hooks over custom debug-only SQL functions.
- Introduce new extension-specific mechanisms only when upstream surfaces cannot express the requirement cleanly, and document why reuse was insufficient.

## Essential Commands

### Building against Vane

```bash
# Run from this checkout. The manifest pins the Vane ABI tested by the plugin.
export VCPKG_TOOLCHAIN_PATH=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
make vane_ci
```

### Building with official DuckDB

```bash
git submodule update --init --recursive
GEN=ninja make release -j 4
./build/release/test/unittest "test/*"
```

Run Rust-only checks from this checkout without a full Vane/CMake build:

```bash
cd /path/to/lance-duckdb
cargo check --manifest-path Cargo.toml
cargo clippy --manifest-path Cargo.toml --all-targets
```

### Formatting

Run Rust formatting from this checkout:

```bash
cargo fmt --all --check
```

Format C++ with DuckDB's pinned clang-format version and CMake with cmake-format before submitting changes.

### Commit & PR Conventions

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type[(scope)]: short description
```

- **Common types**: `feat`, `fix`, `refactor`, `docs`, `ci`, `test`, `chore`.
- **Scope** is optional. When present, use the affected module (e.g. `search`, `resolver`, `scan`).
- Use `!` suffix for breaking changes (e.g. `refactor!: ...`).
- PR numbers are appended automatically by GitHub on merge (e.g. `(#153)`).

### Testing

The official-DuckDB and Vane native lanes are both owned by this repository.
Run the pure Python tests from a non-editable installation. Vane integration
tests require a separately installed Vane package and an explicitly supplied
matching artifact.

```bash
python -m pip install . pytest
pytest python/tests/test_identity.py python/tests/test_loading.py

export LANCE_DUCKDB_EXTENSION=/absolute/path/to/vane/lance.duckdb_extension
pytest python/tests/test_vane.py
```

## Architecture & Key Design Decisions

### Data Flow & FFI Boundary

C++ and Rust communicate through the Arrow C Data Interface:

```
DuckDB SQL → C++ Extension (src/) → FFI calls → Rust (rust/) → Lance crate
                                   ← Arrow C Data Interface (schema + batches) ←
```

See [docs/rust_guidelines.md](docs/rust_guidelines.md) for the FFI function patterns, error propagation, and memory ownership conventions.

### Key Concepts: Filter IR & Exec IR

The extension uses two custom binary wire formats to push work from C++ into Rust:

- **Filter IR** (magic `LFT1`): Serializes DuckDB bound filter expressions into a compact binary format that Rust deserializes into DataFusion `Expr`. This avoids passing expression trees through C function signatures.
- **Exec IR** (magic `LEX1`): Encodes aggregate/projection plans so that Rust can execute them entirely inside Lance/DataFusion, returning only pre-aggregated results to C++.

When modifying filter or aggregate pushdown, changes typically span both sides: C++ serialization (`lance_filter_ir.cpp` / `lance_exec_ir.cpp`) and Rust deserialization (`rust/filter_ir.rs` / `rust/exec_ir.rs`). The magic bytes and version numbers must stay in sync.

### Namespace & Catalog Architecture

`ATTACH ... (TYPE LANCE)` supports two namespace backends (directory for local, REST for remote LanceDB). Both lazily discover tables and auto-create catalog entries on first access.

## Testing Conventions

Tests use DuckDB's `sqllogictest` format in `test/sql/`. S3/MinIO tests
(`s3_*.test`) are gated by `LANCE_TEST_S3=1` and require a running MinIO
instance. Vane integration tests and examples live under `python/tests/` and
`examples/vane/`, never in the Vane repository.

## Common Gotchas

- **Extension availability**: Load only a pre-provisioned artifact compiled for
  the exact host ABI. Do not load an official-DuckDB artifact in Vane (or the
  reverse). Production Ray workers require the same signed bytes at the same
  absolute path; runtime install or download is forbidden.
- **Debug vs release**: `D_ASSERT` checks are stripped in release builds. A test that passes in one configuration but fails in the other usually indicates an assertion violation or uninitialized-variable issue.
- **Rust build diagnostics**: When Cargo fails inside the CMake build, error output is hard to read. Run `cargo check --manifest-path Cargo.toml` in isolation for clearer diagnostics.

## Implementation Guidelines

- **C++**: See [docs/cpp_guidelines.md](docs/cpp_guidelines.md) for coding style, error handling, and naming conventions.
- **Rust**: See [docs/rust_guidelines.md](docs/rust_guidelines.md) for unsafe/FFI conventions and error handling.
