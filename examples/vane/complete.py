#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 lance-duckdb contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the complete local Lance example documented in docs/vane-integration.md.

The example intentionally uses assertions instead of illustrative-only output.
Every section fails immediately when the installed Vane package does not
provide the documented behavior.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import vane
from lance_duckdb import (
    LanceDataset,
    LanceMutationOutcomeUnknownError,
    LanceNamespace,
    load_lance_extension,
)
from vane.runners.copy_outcome import CopyOutcomeUnknownError


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_cleanup_steps(steps: list[tuple[str, Any]]) -> None:
    active_error = sys.exc_info()[1]
    failures: list[tuple[str, BaseException]] = []
    for label, cleanup in steps:
        try:
            cleanup()
        except BaseException as error:
            failures.append((label, error))
    if not failures:
        return

    detail = "; ".join(
        f"{label}: {type(error).__name__}: {error}" for label, error in failures
    )
    if active_error is not None:
        add_note = getattr(active_error, "add_note", None)
        if add_note is not None:
            add_note(f"complete Lance example cleanup also failed: {detail}")
        return
    first_error = failures[0][1]
    add_note = getattr(first_error, "add_note", None)
    if add_note is not None and len(failures) > 1:
        add_note(f"additional complete Lance example cleanup failures: {detail}")
    raise first_error


def _section(name: str) -> None:
    print(f"\n== {name} ==")


@contextmanager
def _example_root(requested_root: Path | None) -> Iterator[Path]:
    if requested_root is None:
        root = Path(tempfile.mkdtemp(prefix="vane-lance-complete-"))
        try:
            yield root
        except (CopyOutcomeUnknownError, LanceMutationOutcomeUnknownError) as error:
            add_note = getattr(error, "add_note", None)
            if add_note is not None:
                add_note(f"generated data was retained for reconciliation at {root}")
            raise
        else:
            shutil.rmtree(root)
        return

    root = requested_root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"--root must be absent or empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    yield root


def _create_search_dataset(connection: Any, dataset_path: Path) -> LanceDataset:
    connection.execute(
        """
        COPY (
          SELECT *
          FROM (VALUES
            (1::BIGINT, 'duck'::VARCHAR,
             'a puppy follows a yellow duck'::VARCHAR, 'animal'::VARCHAR,
             0.95::DOUBLE, [0.0, 0.0, 0.0, 0.0]::FLOAT[4]),
            (2::BIGINT, 'horse'::VARCHAR,
             'a horse runs beside the puppy'::VARCHAR, 'animal'::VARCHAR,
             0.88::DOUBLE, [1.0, 0.0, 0.0, 0.0]::FLOAT[4]),
            (3::BIGINT, 'dragon'::VARCHAR,
             'a dragon appears in a fantasy novel'::VARCHAR, 'fiction'::VARCHAR,
             0.72::DOUBLE, [2.0, 0.0, 0.0, 0.0]::FLOAT[4]),
            (4::BIGINT, 'database'::VARCHAR,
             'Lance stores vectors for machine learning'::VARCHAR, 'technology'::VARCHAR,
             0.91::DOUBLE, [3.0, 0.0, 0.0, 0.0]::FLOAT[4]),
            (5::BIGINT, 'lakehouse'::VARCHAR,
             'DuckDB queries a Lance lakehouse'::VARCHAR, 'technology'::VARCHAR,
             0.84::DOUBLE, [4.0, 0.0, 0.0, 0.0]::FLOAT[4])
          ) AS source(id, title, text, category, score, vec)
        ) TO """
        + _sql_literal(dataset_path)
        + """ (
          FORMAT LANCE,
          MODE 'create',
          DATA_STORAGE_VERSION '2.2',
          MAX_ROWS_PER_FILE 2,
          MAX_ROWS_PER_GROUP 2,
          MAX_BYTES_PER_FILE 1048576
        )
        """
    )

    connection.execute(
        """
        COPY (
          SELECT *
          FROM (VALUES
            (6::BIGINT, 'retrieval'::VARCHAR,
             'hybrid retrieval combines vector and text search'::VARCHAR,
             'technology'::VARCHAR, 0.89::DOUBLE,
             [5.0, 0.0, 0.0, 0.0]::FLOAT[4])
          ) AS source(id, title, text, category, score, vec)
        ) TO """
        + _sql_literal(dataset_path)
        + " (FORMAT LANCE, MODE 'append')"
    )
    return LanceDataset(dataset_path, connection)


def _verify_extension(connection: Any) -> None:
    _section("loadable extension")
    row = connection.execute(
        """
        SELECT loaded, install_mode, extension_version
        FROM duckdb_extensions()
        WHERE extension_name = 'lance'
        """
    ).fetchone()
    assert (
        row is not None and row[0] is True and row[1] == "NOT_INSTALLED" and row[2]
    ), row
    assert connection.execute(
        "SELECT current_setting('lance_deferred_materialization')"
    ).fetchone() == (True,)
    connection.execute("SET lance_deferred_materialization = false")
    assert connection.execute(
        "SELECT current_setting('lance_deferred_materialization')"
    ).fetchone() == (False,)
    connection.execute("SET lance_deferred_materialization = true")
    print(f"lance extension revision: {row[2]}")


def _verify_scan_and_write(connection: Any, root: Path) -> LanceDataset:
    _section("COPY create/append, replacement scan, Relation API, and snapshots")
    dataset_path = root / "documents.lance"
    dataset = _create_search_dataset(connection, dataset_path)

    replacement_rows = connection.execute(
        "SELECT id, title FROM " + _sql_literal(dataset_path) + " ORDER BY id"
    ).fetchall()
    assert replacement_rows == [
        (1, "duck"),
        (2, "horse"),
        (3, "dragon"),
        (4, "database"),
        (5, "lakehouse"),
        (6, "retrieval"),
    ]

    assert LanceDataset(dataset_path, connection).scan().filter(
        "category = 'technology' AND score >= 0.85"
    ).project("id, title").order("id").fetchall() == [
        (4, "database"),
        (6, "retrieval"),
    ]
    assert dataset.scan().aggregate("count(*), min(id), max(id)").fetchone() == (
        6,
        1,
        6,
    )

    with dataset.snapshot() as snapshot:
        assert snapshot.filter("id BETWEEN 2 AND 4").order("id").fetchall()[0][0] == 2

    explain = connection.execute(
        "EXPLAIN (FORMAT JSON) SELECT title FROM "
        + _sql_literal(dataset_path)
        + " WHERE category = 'technology' LIMIT 2"
    ).fetchone()[1]
    assert "Lance" in explain and "Filter" in explain, explain

    empty_path = root / "empty.lance"
    connection.execute(
        """
        COPY (
          SELECT * FROM (VALUES (1::BIGINT, 'unused'::VARCHAR)) AS source(id, label)
          WHERE false
        ) TO """
        + _sql_literal(empty_path)
        + " (FORMAT LANCE, MODE 'create')"
    )
    empty = LanceDataset(empty_path, connection).scan()
    assert empty.fetchall() == []
    assert empty.columns == ["id", "label"]

    relation_path = root / "relation-api.lance"
    relation = connection.sql(
        "SELECT * FROM (VALUES (10::BIGINT, 'ten'::VARCHAR), (11::BIGINT, 'eleven'::VARCHAR)) AS source(id, label)"
    )
    relation_dataset = LanceDataset(relation_path, connection)
    relation_dataset.write(
        relation,
        mode="create",
        max_rows_per_file=1,
    )
    relation_dataset.write(
        connection.sql("SELECT 12::BIGINT AS id, 'twelve'::VARCHAR AS label"),
        mode="append",
    )
    assert relation_dataset.scan().order("id").fetchall() == [
        (10, "ten"),
        (11, "eleven"),
        (12, "twelve"),
    ]
    relation_dataset.write(
        connection.sql("SELECT 20::BIGINT AS id, 'twenty'::VARCHAR AS label"),
        mode="overwrite",
    )
    assert relation_dataset.scan().fetchall() == [(20, "twenty")]

    alias_path = root / "to-lance-alias.lance"
    alias_dataset = LanceDataset(alias_path, connection)
    alias_dataset.write(connection.sql("SELECT 30::BIGINT AS id"))
    assert alias_dataset.scan().fetchall() == [(30,)]
    print(f"dataset: {dataset_path}")
    return dataset


def _verify_search_and_indexes(connection: Any, dataset: LanceDataset) -> None:
    _section("vector, full-text, hybrid search, and all index families")
    exact = dataset.vector_search("vec", [0.1, 0.0, 0.0, 0.0], k=3, use_index=False)
    assert exact.project("id").fetchall() == [(1,), (2,), (3,)]

    fts_rows = (
        dataset.fts("text", "puppy", k=10, prefilter=True)
        .filter("category = 'animal'")
        .project("id")
        .order("id")
        .fetchall()
    )
    assert fts_rows == [(1,), (2,)]

    hybrid_rows = (
        dataset.hybrid_search(
            "vec",
            [0.1, 0.0, 0.0, 0.0],
            "text",
            "puppy",
            k=2,
            use_index=False,
            alpha=0.6,
            oversample_factor=3,
        )
        .project("id, _hybrid_score, _distance, _score")
        .fetchall()
    )
    assert len(hybrid_rows) == 2

    dataset.create_index(
        "documents_vec_idx",
        "vec",
        index_type="IVF_FLAT",
        num_partitions=1,
        metric_type="l2",
    )
    dataset.create_index("documents_text_idx", "text", index_type="INVERTED")
    dataset.create_index("documents_category_idx", "category", index_type="BTREE")
    index_names = {row[0] for row in dataset.show_indexes().fetchall()}
    assert index_names == {
        "documents_category_idx",
        "documents_text_idx",
        "documents_vec_idx",
    }, index_names

    indexed = dataset.vector_search(
        "vec",
        [0.1, 0.0, 0.0, 0.0],
        k=2,
        nprobes=1,
        refine_factor=2,
        use_index=True,
    )
    assert indexed.project("id").fetchall() == [(1,), (2,)]
    indexed_hybrid = dataset.hybrid_search(
        "vec",
        [0.1, 0.0, 0.0, 0.0],
        "text",
        "puppy",
        k=2,
        nprobes=1,
        refine_factor=2,
        prefilter=True,
        use_index=True,
        alpha=0.5,
        oversample_factor=2,
    ).filter("category = 'animal'")
    assert [row[0] for row in indexed_hybrid.project("id").fetchall()] == [1, 2]

    verbose_plan = connection.execute(
        "EXPLAIN (FORMAT JSON) SELECT id FROM lance_vector_search("
        + _sql_literal(dataset.uri)
        + ", 'vec', [0.1, 0.0, 0.0, 0.0]::FLOAT[4], "
        "k = 1, use_index = false, explain_verbose = true)"
    ).fetchone()[1]
    assert '"Lance Explain Verbose": "true"' in verbose_plan, verbose_plan

    path = _sql_literal(dataset.uri)
    for statement in (
        f"ALTER INDEX documents_vec_idx ON {path} OPTIMIZE WITH (mode = 'append')",
        f"ALTER INDEX documents_vec_idx ON {path} OPTIMIZE WITH (mode = 'merge', num_indices_to_merge = 1)",
        f"ALTER INDEX documents_vec_idx ON {path} OPTIMIZE WITH (mode = 'retrain')",
    ):
        result = connection.execute(statement).fetchone()
        assert result is not None and result[0] == "optimize_index", result

    dataset.drop_index("documents_category_idx")
    dataset.drop_index("documents_text_idx")
    dataset.drop_index("documents_vec_idx")
    assert dataset.show_indexes().fetchall() == []
    print(
        "exact/vector-index/FTS/hybrid searches and IVF_FLAT/INVERTED/BTREE indexes passed"
    )


def _verify_maintenance(connection: Any, dataset: LanceDataset) -> None:
    _section("compaction, vacuum, and automatic cleanup policy")
    optimize_rows = dataset.optimize(
        target_rows_per_fragment=2,
        max_rows_per_group=2,
        max_bytes_per_file=1_048_576,
        materialize_deletions=True,
        materialize_deletions_threshold=0.1,
        num_threads=1,
        batch_size=2,
        defer_index_remap=False,
    )
    assert optimize_rows and optimize_rows[0][0] == "compact", optimize_rows
    json.loads(optimize_rows[0][2])

    path = _sql_literal(dataset.uri)
    set_row = connection.execute(
        f"ALTER TABLE {path} SET AUTO_CLEANUP WITH (interval = 1, older_than = '1h', retain_versions = 2)"
    ).fetchone()
    assert set_row is not None and set_row[0] == "set_auto_cleanup", set_row
    maintenance = dict(connection.execute(f"SHOW MAINTENANCE ON {path}").fetchall())
    assert maintenance == {
        "enabled": "true",
        "interval": "1",
        "older_than": "1h",
        "retain_versions": "2",
    }, maintenance
    unset_row = connection.execute(f"ALTER TABLE {path} UNSET AUTO_CLEANUP").fetchone()
    assert unset_row is not None and unset_row[0] == "unset_auto_cleanup", unset_row

    vacuum_rows = dataset.vacuum(
        older_than_seconds=0,
        delete_unverified=False,
        error_if_tagged_old_versions=True,
        retain_n_versions=1,
    )
    assert vacuum_rows and vacuum_rows[0][0] == "cleanup", vacuum_rows
    json.loads(vacuum_rows[0][2])
    print("OPTIMIZE, VACUUM LANCE, and SET/SHOW/UNSET AUTO_CLEANUP passed")


def _verify_namespace_and_dml(connection: Any, root: Path) -> None:
    _section("directory namespace, DML, MERGE, schema evolution, and DROP")
    namespace_root = root / "namespace"
    namespace = LanceNamespace(namespace_root, "demo_lance", connection=connection)
    try:
        items = namespace.create_table(
            "items",
            """
            SELECT * FROM (VALUES
              (1::BIGINT, 'one'::VARCHAR, 10::INTEGER),
              (2::BIGINT, 'two'::VARCHAR, 20::INTEGER),
              (3::BIGINT, 'three'::VARCHAR, 30::INTEGER)
            ) AS source(id, label, quantity)
            """,
        )
        assert items.scan().order("id").fetchall() == [
            (1, "one", 10),
            (2, "two", 20),
            (3, "three", 30),
        ]

        items.insert(
            "SELECT 4::BIGINT AS id, 'four'::VARCHAR AS label, 40::INTEGER AS quantity"
        )
        items.update({"label": "'TWO'", "quantity": "quantity + 1"}, where="id = 2")
        items.delete(where="id = 1")
        items.merge(
            """
            SELECT * FROM (VALUES
              (2::BIGINT, 'two-merged'::VARCHAR, 22::INTEGER),
              (5::BIGINT, 'five'::VARCHAR, 50::INTEGER)
            ) AS source(id, label, quantity)
            """,
            "target.id = source.id",
        ).when_matched_update(
            {"label": "source.label", "quantity": "source.quantity"}
        ).when_not_matched_insert(
            {"id": "source.id", "label": "source.label", "quantity": "source.quantity"}
        ).execute()
        assert items.scan().order("id").fetchall() == [
            (2, "two-merged", 22),
            (3, "three", 30),
            (4, "four", 40),
            (5, "five", 50),
        ]

        returned = connection.execute(
            """
            MERGE INTO demo_lance.main.items AS target
            USING (SELECT 5::BIGINT AS id) AS source
            ON target.id = source.id
            WHEN MATCHED THEN DELETE
            RETURNING merge_action, id, label
            """
        ).fetchall()
        assert returned == [("DELETE", 5, "five")], returned

        items.add_column("quantity_plus_one", "BIGINT")
        items.update({"quantity_plus_one": "quantity + 1"})
        items.add_column("constant_value", "INTEGER", default="42")
        assert items.scan().project("quantity_plus_one, constant_value").order(
            "quantity_plus_one"
        ).fetchall() == [
            (23, 42),
            (31, 42),
            (41, 42),
        ]
        connection.execute(
            "COMMENT ON TABLE demo_lance.main.items IS 'Lance example items'"
        )
        connection.execute(
            "COMMENT ON COLUMN demo_lance.main.items.quantity_plus_one IS 'derived quantity'"
        )
        items.rename_column("label", "display_label")
        connection.execute(
            "ALTER TABLE demo_lance.main.items ALTER COLUMN quantity TYPE BIGINT"
        )
        connection.execute(
            "ALTER TABLE demo_lance.main.items ALTER COLUMN quantity SET NOT NULL"
        )
        connection.execute(
            "ALTER TABLE demo_lance.main.items ALTER COLUMN quantity DROP NOT NULL"
        )
        items.drop_column("quantity_plus_one")
        items.drop_column("constant_value")
        assert items.scan().columns == ["id", "display_label", "quantity"]

        items.create_index("items_id_idx", "id", index_type="BTREE")
        assert [row[0] for row in items.show_indexes().fetchall()] == ["items_id_idx"]
        items.drop_index("items_id_idx")

        connection.execute(
            "CREATE TABLE demo_lance.main.schema_only (id BIGINT, label VARCHAR) WITH (data_storage_version = '2.2')"
        )
        assert namespace.table("schema_only").scan().fetchall() == []

        written = namespace.table("written_with_python")
        written.write(
            connection.sql("SELECT 100::BIGINT AS id, 'python'::VARCHAR AS label"),
            mode="create",
        )
        assert written.scan().fetchall() == [(100, "python")]

        namespace_search = namespace.create_table(
            "namespace_search",
            """
            SELECT * FROM (VALUES
              (1::BIGINT, 'a puppy follows a duck'::VARCHAR,
               'animal'::VARCHAR, [0.0, 0.0, 0.0, 0.0]::FLOAT[4]),
              (2::BIGINT, 'vector retrieval for technology'::VARCHAR,
               'technology'::VARCHAR, [1.0, 0.0, 0.0, 0.0]::FLOAT[4])
            ) AS source(id, text, category, vec)
            """,
        )
        assert namespace_search.vector_search(
            "vec",
            [0.1, 0.0, 0.0, 0.0],
            k=1,
            prefilter=True,
            use_index=False,
            filter="category = 'technology'",
        ).project("id").fetchall() == [(2,)]
        assert namespace_search.fts(
            "text",
            "retrieval",
            k=1,
            prefilter=True,
            filter="category = 'technology'",
        ).project("id").fetchall() == [(2,)]
        assert namespace_search.hybrid_search(
            "vec",
            [1.1, 0.0, 0.0, 0.0],
            "text",
            "retrieval",
            k=1,
            use_index=False,
        ).project("id").fetchall() == [(2,)]

        dynamic_path = namespace_root / "dynamic_after_attach.lance"
        connection.execute(
            "COPY (SELECT 200::BIGINT AS id) TO "
            + _sql_literal(dynamic_path)
            + " (FORMAT LANCE, MODE 'create')"
        )
        tables = {
            row[0]
            for row in connection.execute("SHOW TABLES FROM demo_lance.main").fetchall()
        }
        assert tables == {
            "dynamic_after_attach",
            "items",
            "namespace_search",
            "schema_only",
            "written_with_python",
        }, tables

        items.truncate()
        assert items.scan().aggregate("count(*)").fetchone() == (0,)
        namespace.drop_table("items")
        namespace.drop_table("namespace_search")
        namespace.drop_table("schema_only")
        namespace.drop_table("written_with_python")
        namespace.drop_table("dynamic_after_attach")
        assert connection.execute("SHOW TABLES FROM demo_lance.main").fetchall() == []
    finally:
        active_error = sys.exc_info()[1]
        retained_mutation = isinstance(
            active_error,
            (CopyOutcomeUnknownError, LanceMutationOutcomeUnknownError),
        )
        if not retained_mutation:
            _run_cleanup_steps([("detach writable namespace", namespace.detach)])
        elif active_error is not None:
            add_note = getattr(active_error, "add_note", None)
            if add_note is not None:
                add_note(
                    f"namespace cleanup for {namespace_root!s} was skipped because a mutation lease remains held"
                )

    read_only_seed = namespace_root / "read_only_seed.lance"
    connection.execute(
        "COPY (SELECT 1::BIGINT AS id) TO "
        + _sql_literal(read_only_seed)
        + " (FORMAT LANCE, MODE 'create')"
    )
    read_only = LanceNamespace(
        namespace_root,
        "read_only_lance",
        read_only=True,
        connection=connection,
    )
    try:
        assert read_only.table("read_only_seed").scan().fetchall() == [(1,)]
        try:
            read_only.table("read_only_seed").write(
                connection.sql("SELECT 2::BIGINT AS id"),
                mode="overwrite",
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("read-only Lance attachment accepted a write")
    finally:
        _run_cleanup_steps([("detach read-only namespace", read_only.detach)])
    print("directory namespace and complete table lifecycle passed")


def run(root: Path) -> None:
    vane.set_runner_local()
    try:
        connection = load_lance_extension(vane.connect())
    except BaseException:
        _run_cleanup_steps([("tear down runner", vane.teardown_runner)])
        raise
    try:
        _verify_extension(connection)
        dataset = _verify_scan_and_write(connection, root)
        _verify_search_and_indexes(connection, dataset)
        _verify_maintenance(connection, dataset)
        _verify_namespace_and_dml(connection, root)
    finally:
        _run_cleanup_steps(
            [
                ("close connection", connection.close),
                ("tear down runner", vane.teardown_runner),
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="keep generated datasets in this absent or empty directory",
    )
    args = parser.parse_args()
    with _example_root(args.root) as root:
        print(f"working directory: {root}")
        run(root)
        print("\nALL LOCAL LANCE EXAMPLES PASSED")


if __name__ == "__main__":
    main()
