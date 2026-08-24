# SPDX-FileCopyrightText: 2026 lance-duckdb contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from lance_duckdb._identity import canonicalize_dataset_uri, normalize_dataset_uri

try:
    from vane.runners.copy_outcome import CopyOutcomeUnknownError
except ImportError:

    class CopyOutcomeUnknownError(RuntimeError):
        """Fallback used when the optional Vane runtime is not installed."""

        def __init__(
            self,
            operation_id: str = "",
            base_path: str = "",
            run_id: str = "",
            manifest_path: str = "",
            committed_marker_path: str = "",
            detail: str = "",
            cleanup_warnings: tuple[str, ...] = (),
        ) -> None:
            self.operation_id = operation_id
            self.base_path = base_path
            self.run_id = run_id
            self.manifest_path = manifest_path
            self.committed_marker_path = committed_marker_path
            self.detail = detail
            self.cleanup_warnings = cleanup_warnings
            super().__init__(detail or "Lance commit outcome is unknown")


LANCE_EXTENSION_PATH_ENV = "LANCE_DUCKDB_EXTENSION"


def load_lance_extension(
    connection: Any, extension_path: str | os.PathLike[str] | None = None
) -> Any:
    """Load a prebuilt Lance artifact into a compatible DuckDB or Vane connection.

    The helper never installs or downloads an extension. ``extension_path`` or
    ``LANCE_DUCKDB_EXTENSION`` must identify an artifact built against the
    connection's exact DuckDB ABI.
    """
    configured_path = (
        extension_path
        if extension_path is not None
        else os.environ.get(LANCE_EXTENSION_PATH_ENV)
    )
    if configured_path is None:
        raise ValueError(
            f"pass extension_path or set {LANCE_EXTENSION_PATH_ENV} to a pre-provisioned lance.duckdb_extension"
        )
    path = Path(os.fspath(configured_path)).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Lance extension path is not a file: {path}")
    connection.load_extension(str(path))
    return connection


class LanceCommitOutcomeUnknownError(CopyOutcomeUnknownError):
    """A Lance writer may have committed; repeating it is unsafe."""

    def __reduce__(
        self,
    ) -> tuple[Any, tuple[str, str, str, str, str, str, tuple[str, ...]]]:
        return type(self), (
            self.operation_id,
            self.base_path,
            self.run_id,
            self.manifest_path,
            self.committed_marker_path,
            self.detail,
            self.cleanup_warnings,
        )


class LanceMutationOutcomeUnknownError(RuntimeError):
    """A Lance namespace mutation may have completed; repeating it is unsafe."""

    safe_to_retry = False

    def __init__(
        self,
        identity: str,
        operation: str,
        lease_kind: str,
        lease_token: str,
        detail: str,
    ) -> None:
        self.identity = identity
        self.operation = operation
        self.lease_kind = lease_kind
        self.lease_token = lease_token
        self.detail = detail
        super().__init__(
            f"{operation} outcome is unknown for {identity!r}; do not retry automatically. "
            f"The native {lease_kind} lease {lease_token!r} may remain held; inspect the "
            "storage-backed `_vane_leases` records with deployment-specific reconciliation tooling and release "
            "only that exact token after reconciliation. Python state is not authoritative. "
            f"Native error: {detail}"
        )

    def __reduce__(self) -> tuple[Any, tuple[str, str, str, str, str]]:
        return type(self), (
            self.identity,
            self.operation,
            self.lease_kind,
            self.lease_token,
            self.detail,
        )


def _is_lance_mutation_outcome_unknown(exc: BaseException | str) -> bool:
    # Native wrappers may add context around an FFI failure. Only the final
    # formatted marker is authoritative; earlier markers can be untrusted
    # service error text embedded inside an outer error with another code.
    markers = re.findall(r"\(code=(\d+)\)", str(exc))
    return bool(markers) and markers[-1] in {"55", "56"}


def _retained_lease_detail(
    exc: CopyOutcomeUnknownError, identity: str, lease_kind: str = "mutation"
) -> str:
    suffix = (
        f"The native Lance {lease_kind} lease for {identity!r} remains held while the commit is reconciled; "
        "inspect the storage-backed lease records and release only the exact token after proving the writer "
        "stopped."
    )
    return f"{exc.detail} {suffix}".strip()


def _lance_commit_outcome_unknown(
    exc: CopyOutcomeUnknownError, identity: str, lease_kind: str = "mutation"
) -> LanceCommitOutcomeUnknownError:
    return LanceCommitOutcomeUnknownError(
        exc.operation_id,
        exc.base_path,
        exc.run_id,
        exc.manifest_path,
        exc.committed_marker_path,
        _retained_lease_detail(exc, identity, lease_kind),
        exc.cleanup_warnings,
    )


def _connection(connection: Any | None) -> Any:
    if connection is not None:
        return connection
    try:
        import vane

        return vane.default_connection()
    except ImportError:
        import duckdb

        return duckdb.default_connection()


def _quote_identifier_part(value: str) -> str:
    if not value:
        raise ValueError("SQL identifier cannot be empty")
    return '"' + value.replace('"', '""') + '"'


def _quote_lance_top_level_column(value: str) -> str:
    if "." in value:
        raise ValueError(f"Lance top-level column names cannot contain '.': {value!r}")
    return _quote_identifier_part(value)


def _lance_field_path(value: str) -> str:
    if not value:
        raise ValueError("Lance field path cannot be empty")
    segments = value.split(".")
    if any(not segment for segment in segments):
        raise ValueError(f"invalid Lance field path: {value!r}")

    rendered: list[str] = []
    for segment in segments:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", segment):
            rendered.append(segment)
        else:
            rendered.append("`" + segment.replace("`", "``") + "`")
    return ".".join(rendered)


def _bare_identifier(
    value: str,
    *,
    label: str = "SQL identifier",
    allow_qualified: bool = True,
) -> str:
    parts = value.split(".")
    if (
        not parts
        or (not allow_qualified and len(parts) != 1)
        or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) for part in parts)
    ):
        raise ValueError(f"invalid {label}: {value!r}")
    return ".".join(parts)


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _render_option_sql(options: Mapping[str, Any], assignment: str) -> str:
    rendered: list[str] = []
    seen_keys: set[str] = set()
    for key, value in options.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Lance option names must be strings, got {type(key).__name__}"
            )
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid Lance option name: {key!r}")
        normalized_key = key.casefold()
        if normalized_key in seen_keys:
            raise ValueError(f"duplicate case-insensitive Lance option name: {key!r}")
        seen_keys.add(normalized_key)
        if isinstance(value, bool):
            literal = "true" if value else "false"
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"Lance option {key!r} must be finite")
            literal = str(value)
        elif isinstance(value, int):
            literal = str(value)
        elif isinstance(value, str):
            literal = _quote_literal(value)
        else:
            raise TypeError(
                f"Lance option {key!r} must be a string, boolean, integer, or finite float"
            )
        rendered.append(f"{key}{assignment}{literal}")
    return ", ".join(rendered)


def _option_sql(options: Mapping[str, Any]) -> str:
    return _render_option_sql(options, " = ")


def _copy_option_sql(options: Mapping[str, Any]) -> str:
    """Render DuckDB COPY options (COPY uses ``key value`` syntax)."""
    return _render_option_sql(options, " ")


def _search_vector_sql(query: Sequence[float]) -> str:
    values: list[str] = []
    for value in query:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("Lance search vectors must contain numbers") from exc
        if not math.isfinite(numeric):
            raise ValueError("Lance search vectors must contain finite numbers")
        values.append(repr(numeric))
    return "[" + ", ".join(values) + "]::DOUBLE[]"


def _search_relation(
    connection: Any,
    function_name: str,
    positional_sql: Sequence[str],
    named_parameters: Mapping[str, Any],
) -> Any:
    arguments = list(positional_sql)
    if named_parameters:
        arguments.append(_option_sql(named_parameters))
    return connection.sql(f"SELECT * FROM {function_name}({', '.join(arguments)})")


def _configured_runner_type() -> str:
    from vane.runners import get_or_infer_runner_type

    runner_type = get_or_infer_runner_type().strip().lower()
    if runner_type not in {"local", "local-fast", "ray"}:
        raise RuntimeError(
            f"Unsupported Vane runner type for Lance relation dispatch: {runner_type!r}"
        )
    return runner_type


def _relation_dispatch_available(relation: Any, method_name: str) -> bool:
    """Whether a generic Relation write method can use the Vane runner.

    DuckDB's upstream Relation methods are the public compatibility boundary, but
    the local (non-FTE) runner intentionally rejects distributed relation
    dispatch.  Driver-local SQL remains the compatibility fallback in that
    mode, and test doubles that do not expose the method naturally use the
    same fallback.
    """
    method = getattr(relation, method_name, None)
    if not callable(method):
        return False
    try:
        runner_type = _configured_runner_type()
    except Exception:
        runner_type = os.environ.get("VANE_RUNNER", "").strip().lower()
    return runner_type in {"ray", "local-fast"}


def _validate_relation_connection(source: Any, connection: Any, operation: str) -> None:
    del connection
    if not callable(getattr(source, "sql_query", None)):
        raise TypeError(
            f"{operation} source must be a DuckDB relation or a SELECT SQL string"
        )


def _relation_sql(source: Any, connection: Any, operation: str) -> str:
    if isinstance(source, str):
        if not source:
            raise ValueError(f"{operation} source SQL cannot be empty")
        return source
    _validate_relation_connection(source, connection, operation)
    source_sql = source.sql_query()
    if not source_sql:
        raise ValueError(
            f"{operation} source relation is not serializable; provide a SELECT SQL string"
        )
    return source_sql


class LanceDataset:
    def __init__(
        self, uri: str | os.PathLike[str], connection: Any | None = None
    ) -> None:
        self.uri = canonicalize_dataset_uri(uri)
        self.connection = _connection(connection)

    def scan(self) -> Any:
        return self._snapshot_relation(self._scan_without_lease)

    def _scan_without_lease(self) -> Any:
        return self.connection.table_function("__lance_scan", [self.uri])

    def _snapshot_relation(self, factory: Any) -> Any:
        # The table function / attached table owns its native snapshot lease
        # in DuckDB global operator state. Keeping a Python object alive here
        # would split ownership from workers and cannot coordinate Ray
        # processes.
        return factory()

    def _scan_reference(self) -> str:
        return self.uri

    def _sql_target(self) -> str:
        return _quote_literal(self.uri)

    def _lance_command_target(self) -> str:
        return self._sql_target()

    @contextmanager
    def _mutation(self, operation: str) -> Iterator[None]:
        try:
            yield
        except CopyOutcomeUnknownError as primary:
            if isinstance(primary, LanceCommitOutcomeUnknownError):
                raise
            raise _lance_commit_outcome_unknown(primary, self.uri) from primary
        except BaseException as primary:
            if _is_lance_mutation_outcome_unknown(primary):
                raise LanceMutationOutcomeUnknownError(
                    self.uri,
                    operation,
                    "mutation",
                    "native",
                    str(primary),
                ) from primary
            raise

    @contextmanager
    def _vacuum(self, operation: str) -> Iterator[None]:
        # Native maintenance/drop operators own the storage-backed vacuum
        # lease.  This context manager only keeps the common autocommit and
        # outcome translation contract for the Python convenience methods.
        try:
            yield
        except CopyOutcomeUnknownError as primary:
            if isinstance(primary, LanceCommitOutcomeUnknownError):
                raise
            raise _lance_commit_outcome_unknown(
                primary, self.uri, "vacuum"
            ) from primary
        except BaseException as primary:
            if _is_lance_mutation_outcome_unknown(primary):
                raise LanceMutationOutcomeUnknownError(
                    self.uri,
                    operation,
                    "vacuum",
                    "native",
                    str(primary),
                ) from primary
            raise

    @contextmanager
    def snapshot(self) -> Iterator[Any]:
        """Return a relation whose native scan state pins its Lance snapshot."""
        yield self._snapshot_relation(self._scan_without_lease)

    def vector_search(
        self,
        vector_column: str,
        query: Sequence[float],
        *,
        k: int = 10,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        prefilter: bool = False,
        use_index: bool = True,
        filter: str | None = None,
    ) -> Any:
        named: dict[str, Any] = {"k": k, "prefilter": prefilter, "use_index": use_index}
        if nprobes is not None:
            named["nprobs"] = nprobes
        if refine_factor is not None:
            named["refine_factor"] = refine_factor
        if filter is not None:
            named["filter"] = filter
        return self._snapshot_relation(
            lambda: _search_relation(
                self.connection,
                "lance_vector_search",
                [
                    _quote_literal(self._scan_reference()),
                    _quote_literal(vector_column),
                    _search_vector_sql(query),
                ],
                named,
            )
        )

    def fts(
        self,
        text_column: str,
        query: str,
        *,
        k: int = 10,
        prefilter: bool = False,
        filter: str | None = None,
    ) -> Any:
        named: dict[str, Any] = {"k": k, "prefilter": prefilter}
        if filter is not None:
            named["filter"] = filter
        return self._snapshot_relation(
            lambda: _search_relation(
                self.connection,
                "lance_fts",
                [
                    _quote_literal(self._scan_reference()),
                    _quote_literal(text_column),
                    _quote_literal(query),
                ],
                named,
            )
        )

    def hybrid_search(
        self,
        vector_column: str,
        vector_query: Sequence[float],
        text_column: str,
        text_query: str,
        *,
        k: int = 10,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        prefilter: bool = False,
        use_index: bool = True,
        alpha: float = 0.5,
        oversample_factor: int = 4,
    ) -> Any:
        named: dict[str, Any] = {
            "k": k,
            "prefilter": prefilter,
            "use_index": use_index,
            "alpha": alpha,
            "oversample_factor": oversample_factor,
        }
        if nprobes is not None:
            named["nprobs"] = nprobes
        if refine_factor is not None:
            named["refine_factor"] = refine_factor
        return self._snapshot_relation(
            lambda: _search_relation(
                self.connection,
                "lance_hybrid_search",
                [
                    _quote_literal(self._scan_reference()),
                    _quote_literal(vector_column),
                    _search_vector_sql(vector_query),
                    _quote_literal(text_column),
                    _quote_literal(text_query),
                ],
                named,
            )
        )

    def write(self, relation: Any, *, mode: str = "create", **options: Any) -> None:
        if isinstance(relation, str):
            raise TypeError(
                "Lance dataset writes require a DuckDB-compatible relation, not SQL text"
            )
        supported_options = {
            "data_storage_version",
            "max_bytes_per_file",
            "max_rows_per_file",
            "max_rows_per_group",
        }
        unsupported_options = set(options).difference(supported_options)
        if unsupported_options:
            rendered = ", ".join(sorted(unsupported_options))
            raise ValueError(
                f"Lance dataset writes do not support these writer options: {rendered}"
            )
        _validate_relation_connection(relation, self.connection, "Lance dataset write")
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"create", "append", "overwrite"}:
            raise ValueError(
                "Lance dataset write mode must be one of create, append, or overwrite"
            )
        option_sql = "FORMAT lance, mode " + _quote_literal(normalized_mode)
        if options:
            option_sql += ", " + _copy_option_sql(options)
        with self._mutation("Lance dataset write"):
            if (
                normalized_mode == "create"
                and not options
                and _relation_dispatch_available(relation, "write_file")
            ):
                # Vane's format-neutral WriteFileRelation is the distributed
                # direct-path boundary. Official DuckDB and option-bearing
                # writes keep using ordinary COPY SQL below.
                relation.write_file(self.uri, format="lance")
                return
            self.connection.execute(
                f"COPY ({_relation_sql(relation, self.connection, 'Lance dataset write')}) "
                f"TO {_quote_literal(self.uri)} ({option_sql})"
            )

    def create_index(
        self, name: str, column: str, *, index_type: str, **options: Any
    ) -> None:
        sql = (
            f"CREATE INDEX {_bare_identifier(name, label='Lance index name', allow_qualified=False)} "
            f"ON {self._lance_command_target()} "
            f"({_lance_field_path(column)}) "
            f"USING {_bare_identifier(index_type, label='Lance index type', allow_qualified=False)}"
        )
        if options:
            sql += " WITH (" + _option_sql(options) + ")"
        with self._mutation("Lance index creation"):
            self.connection.execute(sql)

    def show_indexes(self) -> Any:
        return self._snapshot_relation(
            lambda: self.connection.sql(
                f"SHOW INDEXES ON {self._lance_command_target()}"
            )
        )

    def drop_index(self, name: str) -> None:
        with self._mutation("Lance index removal"):
            self.connection.execute(
                f"DROP INDEX {_bare_identifier(name, label='Lance index name', allow_qualified=False)} "
                f"ON {self._lance_command_target()}"
            )

    def optimize(self, **options: Any) -> Any:
        sql = f"OPTIMIZE {self._lance_command_target()}"
        if options:
            sql += " WITH (" + _option_sql(options) + ")"
        with self._mutation("Lance optimization"):
            return self.connection.sql(sql).fetchall()

    def vacuum(self, **options: Any) -> Any:
        sql = f"VACUUM LANCE {self._lance_command_target()}"
        if options:
            sql += " WITH (" + _option_sql(options) + ")"
        with self._vacuum("Lance vacuum"):
            return self.connection.sql(sql).fetchall()


class LanceNamespace:
    def __init__(
        self,
        namespace_id: str | os.PathLike[str],
        alias: str,
        *,
        endpoint: str | None = None,
        read_only: bool = False,
        connection: Any | None = None,
        attach: bool = True,
    ) -> None:
        raw_namespace_id = os.fspath(namespace_id)
        if not isinstance(raw_namespace_id, str):
            raise TypeError("Lance namespace ID must be a string or text path")
        if endpoint is not None:
            if not isinstance(endpoint, str):
                raise TypeError("Lance namespace endpoint must be a string or None")
            endpoint = endpoint.strip()
            if not endpoint:
                raise ValueError("Lance namespace endpoint cannot be empty")
        self.namespace_id = (
            raw_namespace_id
            if endpoint is not None
            else canonicalize_dataset_uri(raw_namespace_id)
        )
        self.alias = alias
        self.endpoint = endpoint
        self.read_only = read_only
        self.connection = _connection(connection)
        if attach:
            options = ["TYPE LANCE", f"READ_ONLY {'true' if read_only else 'false'}"]
            if endpoint is not None:
                options.append("ENDPOINT " + _quote_literal(endpoint))
            self.connection.execute(
                f"ATTACH {_quote_literal(self.namespace_id)} AS {_quote_identifier_part(alias)} ({', '.join(options)})"
            )

    def table(self, name: str, schema: str = "main") -> LanceTable:
        if schema.casefold() != "main":
            raise ValueError("Lance namespaces expose only the 'main' schema")
        return LanceTable(self, name, schema=schema)

    def _attachment_is_read_only(self) -> bool:
        row = self.connection.execute(
            "SELECT readonly FROM duckdb_databases() WHERE lower(database_name) = lower(?)",
            [self.alias],
        ).fetchone()
        return bool(row[0]) if row is not None else self.read_only

    def create_table(
        self,
        name: str,
        source: Any,
        *,
        schema: str = "main",
        if_not_exists: bool = False,
    ) -> LanceTable:
        table = self.table(name, schema=schema)
        source_sql = _relation_sql(source, self.connection, "CREATE TABLE")
        guard = " IF NOT EXISTS" if if_not_exists else ""
        with table._mutation("Lance table creation"):
            if if_not_exists:
                # DuckDB's Relation API has no IF NOT EXISTS flag. Keep this
                # branch as an explicit catalog statement; the normal CTAS
                # path below is the Vane runner/provider entry point.
                self.connection.execute(
                    f"CREATE TABLE{guard} {table._sql_target()} AS {source_sql}"
                )
            else:
                source_relation = (
                    source
                    if not isinstance(source, str)
                    else self.connection.sql(source)
                )
                if _relation_dispatch_available(source_relation, "create"):
                    source_relation.create(table._sql_target())
                else:
                    self.connection.execute(
                        f"CREATE TABLE {table._sql_target()} AS {source_sql}"
                    )
        return table

    def drop_table(
        self,
        name: str,
        *,
        schema: str = "main",
        if_exists: bool = False,
    ) -> None:
        table = self.table(name, schema=schema)
        guard = " IF EXISTS" if if_exists else ""
        # A directory table drop removes files used by active fixed snapshots.
        # The vacuum lease waits for those readers and blocks new ones.
        with table._vacuum("Lance table removal"):
            self.connection.execute(f"DROP TABLE{guard} {table._sql_target()}")

    def detach(self, *, timeout: float | None = 30.0) -> None:
        """Detach the DuckDB catalog attachment.

        ``timeout`` is retained for source compatibility and input
        validation.  DETACH does not wait on a Python coordinator; native scan
        state and the DuckDB catalog own their respective lifetimes.
        """
        if timeout is not None:
            timeout_value = float(timeout)
            if not math.isfinite(timeout_value) or timeout_value < 0:
                raise ValueError(
                    "Lance namespace detach timeout must be a finite non-negative number"
                )
        self.connection.execute(f"DETACH {_quote_identifier_part(self.alias)}")


class LanceTable(LanceDataset):
    def __init__(
        self, namespace: LanceNamespace, name: str, *, schema: str = "main"
    ) -> None:
        self.namespace = namespace
        self.name = name
        self.schema = schema
        self.qualified_name = f"{namespace.alias}.{schema}.{name}"
        self._quoted_name = ".".join(
            _quote_identifier_part(part) for part in (namespace.alias, schema, name)
        )
        if namespace.endpoint is not None:
            identity = (
                "lance-rest://namespace/"
                + quote(normalize_dataset_uri(namespace.endpoint), safe="")
                + "/"
                + quote(namespace.namespace_id, safe="")
                + "/"
                + quote(schema.casefold(), safe="")
                + "/"
                + quote(name.casefold(), safe="")
            )
        else:
            # The extension exposes only DuckDB's ``main`` schema for a
            # directory namespace; its physical dataset is root/name.lance.
            # Use that exact identity so namespace and direct-path APIs share
            # one coordinator.
            physical_name = name
            execute = getattr(namespace.connection, "execute", None)
            if callable(execute):
                existing = execute(
                    """
                    SELECT table_name
                    FROM duckdb_tables()
                    WHERE lower(database_name) = lower(?)
                      AND lower(schema_name) = 'main'
                      AND lower(table_name) = lower(?)
                    LIMIT 1
                    """,
                    [namespace.alias, name],
                ).fetchone()
                if existing is not None:
                    physical_name = existing[0]
            identity = (
                normalize_dataset_uri(namespace.namespace_id).rstrip("/")
                + "/"
                + physical_name
                + ".lance"
            )
        super().__init__(identity, namespace.connection)

    def scan(self) -> Any:
        return self._snapshot_relation(self._scan_without_lease)

    def _scan_without_lease(self) -> Any:
        return self.connection.table(self._quoted_name)

    def _scan_reference(self) -> str:
        return self._quoted_name

    def _sql_target(self) -> str:
        return self._quoted_name

    def _lance_command_target(self) -> str:
        return self._quoted_name

    def _existing_schema(self) -> tuple[list[str], list[Any]] | None:
        exists = self.connection.execute(
            """
            SELECT 1
            FROM duckdb_tables()
            WHERE lower(database_name) = lower(?)
              AND lower(schema_name) = lower(?)
              AND lower(table_name) = lower(?)
            LIMIT 1
            """,
            [self.namespace.alias, self.schema, self.name],
        ).fetchone()
        if exists is None:
            return None
        relation = self.connection.table(self._quoted_name)
        return relation.columns, relation.types

    def _reject_unsupported_schema_overwrite(self, source_sql: str) -> None:
        existing_schema = self._existing_schema()
        if existing_schema is None:
            return
        source = self.connection.sql(source_sql)
        source_schema = (source.columns, source.types)
        if source_schema != existing_schema:
            raise NotImplementedError(
                "Lance attached-table overwrite does not support schema changes; "
                "drop and recreate the table through LanceNamespace instead"
            )

    def write(self, relation: Any, *, mode: str = "create", **options: Any) -> None:
        if self.namespace._attachment_is_read_only():
            raise PermissionError(
                f"cannot write Lance table {self.qualified_name!r}: attachment is read-only"
            )
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"create", "append", "overwrite"}:
            raise ValueError(
                "Lance table write mode must be one of create, append, or overwrite"
            )
        source_sql = _relation_sql(relation, self.connection, "Lance table write")
        if normalized_mode == "append" and options:
            raise ValueError("Lance table append does not support writer options")
        unsupported_options = set(options).difference({"data_storage_version"})
        if unsupported_options:
            rendered = ", ".join(sorted(unsupported_options))
            raise ValueError(
                f"Lance attached-table writes do not support these writer options: {rendered}"
            )

        if normalized_mode == "append":
            sql = f"INSERT INTO {self._sql_target()} {source_sql}"
        else:
            replace = " OR REPLACE" if normalized_mode == "overwrite" else ""
            option_sql = " WITH (" + _option_sql(options) + ")" if options else ""
            sql = f"CREATE{replace} TABLE {self._sql_target()}{option_sql} AS {source_sql}"
        with self._mutation("Lance table writes"):
            if normalized_mode == "overwrite":
                self._reject_unsupported_schema_overwrite(source_sql)
            if normalized_mode == "append" and not options:
                source_relation = (
                    relation
                    if not isinstance(relation, str)
                    else self.connection.sql(source_sql)
                )
                if _relation_dispatch_available(source_relation, "insert_into"):
                    source_relation.insert_into(self._sql_target())
                else:
                    self.connection.execute(sql)
            elif normalized_mode == "create" and not options:
                source_relation = (
                    relation
                    if not isinstance(relation, str)
                    else self.connection.sql(source_sql)
                )
                if _relation_dispatch_available(source_relation, "create"):
                    source_relation.create(self._sql_target())
                else:
                    self.connection.execute(sql)
            else:
                # Overwrite and writer-option forms do not yet have a generic
                # Relation contract. They remain explicit SQL/driver-local
                # operations until DuckDB exposes a replace/option-bearing
                # CTAS provider hook.
                self.connection.execute(sql)

    def insert(self, source: Any, *, columns: Sequence[str] | None = None) -> None:
        source_sql = _relation_sql(source, self.connection, "INSERT")
        column_sql = ""
        if columns:
            column_sql = (
                " ("
                + ", ".join(_quote_lance_top_level_column(column) for column in columns)
                + ")"
            )
        with self._mutation("Lance table inserts"):
            if columns:
                self.connection.execute(
                    f"INSERT INTO {self._sql_target()}{column_sql} {source_sql}"
                )
            else:
                source_relation = (
                    source
                    if not isinstance(source, str)
                    else self.connection.sql(source_sql)
                )
                if _relation_dispatch_available(source_relation, "insert_into"):
                    source_relation.insert_into(self._sql_target())
                else:
                    self.connection.execute(
                        f"INSERT INTO {self._sql_target()} {source_sql}"
                    )

    def update(
        self, assignments: Mapping[str, str], *, where: str | None = None
    ) -> None:
        if not assignments:
            raise ValueError("UPDATE requires at least one assignment")
        set_sql = ", ".join(
            f"{_quote_lance_top_level_column(column)} = {expression}"
            for column, expression in assignments.items()
        )
        where_sql = f" WHERE {where}" if where else ""
        with self._mutation("Lance table updates"):
            self.connection.execute(
                f"UPDATE {self._sql_target()} SET {set_sql}{where_sql}"
            )

    def delete(self, *, where: str | None = None) -> None:
        where_sql = f" WHERE {where}" if where else ""
        with self._mutation("Lance table deletes"):
            self.connection.execute(f"DELETE FROM {self._sql_target()}{where_sql}")

    def truncate(self) -> None:
        with self._mutation("Lance table truncation"):
            self.connection.execute(f"TRUNCATE TABLE {self._sql_target()}")

    def add_column(
        self, name: str, sql_type: str, *, default: str | None = None
    ) -> None:
        quoted_name = _quote_lance_top_level_column(name)
        default_sql = f" DEFAULT {default}" if default is not None else ""
        with self._mutation("Lance table schema changes"):
            self.connection.execute(
                f"ALTER TABLE {self._sql_target()} ADD COLUMN {quoted_name} {sql_type}{default_sql}"
            )

    def drop_column(self, name: str) -> None:
        quoted_name = _quote_lance_top_level_column(name)
        with self._mutation("Lance table schema changes"):
            self.connection.execute(
                f"ALTER TABLE {self._sql_target()} DROP COLUMN {quoted_name}"
            )

    def rename_column(self, old_name: str, new_name: str) -> None:
        quoted_old_name = _quote_lance_top_level_column(old_name)
        quoted_new_name = _quote_lance_top_level_column(new_name)
        with self._mutation("Lance table schema changes"):
            self.connection.execute(
                f"ALTER TABLE {self._sql_target()} RENAME COLUMN {quoted_old_name} TO {quoted_new_name}"
            )

    def merge(self, source: Any, on: str) -> LanceMergeBuilder:
        if not on.strip():
            raise ValueError("MERGE join condition cannot be empty")
        return LanceMergeBuilder(self, source, on)


@dataclass
class LanceMergeBuilder:
    table: LanceTable
    source: Any
    on: str
    _clauses: list[str] = field(default_factory=list)

    def when_matched_update(
        self, assignments: Mapping[str, str], *, condition: str | None = None
    ) -> LanceMergeBuilder:
        if not assignments:
            raise ValueError("MERGE UPDATE requires at least one assignment")
        prefix = "WHEN MATCHED" + (f" AND {condition}" if condition else "")
        values = ", ".join(
            f"{_quote_lance_top_level_column(key)} = {value}"
            for key, value in assignments.items()
        )
        self._clauses.append(f"{prefix} THEN UPDATE SET {values}")
        return self

    def when_matched_delete(self, *, condition: str | None = None) -> LanceMergeBuilder:
        prefix = "WHEN MATCHED" + (f" AND {condition}" if condition else "")
        self._clauses.append(f"{prefix} THEN DELETE")
        return self

    def when_not_matched_insert(
        self, values: Mapping[str, str], *, condition: str | None = None
    ) -> LanceMergeBuilder:
        if not values:
            raise ValueError("MERGE INSERT requires at least one value")
        prefix = "WHEN NOT MATCHED" + (f" AND {condition}" if condition else "")
        columns = ", ".join(_quote_lance_top_level_column(key) for key in values)
        expressions = ", ".join(values.values())
        self._clauses.append(f"{prefix} THEN INSERT ({columns}) VALUES ({expressions})")
        return self

    def execute(self) -> None:
        if not self._clauses:
            raise ValueError("MERGE requires at least one action")
        source_sql = _relation_sql(self.source, self.table.connection, "MERGE")
        sql = (
            f"MERGE INTO {self.table._sql_target()} AS target "
            f"USING ({source_sql}) AS source ON {self.on} " + " ".join(self._clauses)
        )
        with self.table._mutation("Lance table merges"):
            self.table.connection.execute(sql)


__all__ = [
    "LANCE_EXTENSION_PATH_ENV",
    "LanceCommitOutcomeUnknownError",
    "LanceMutationOutcomeUnknownError",
    "LanceDataset",
    "LanceMergeBuilder",
    "LanceNamespace",
    "LanceTable",
    "load_lance_extension",
    "normalize_dataset_uri",
]
