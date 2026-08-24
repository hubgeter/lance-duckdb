#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 lance-duckdb contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Vane against a live Lance REST Namespace endpoint."""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from typing import Any

import vane
from lance_duckdb import (
    LanceMutationOutcomeUnknownError,
    LanceNamespace,
    load_lance_extension,
)
from vane.runners.copy_outcome import CopyOutcomeUnknownError


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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
            add_note(f"Lance REST example cleanup also failed: {detail}")
        return
    first_error = failures[0][1]
    add_note = getattr(first_error, "add_note", None)
    if add_note is not None and len(failures) > 1:
        add_note(f"additional Lance REST example cleanup failures: {detail}")
    raise first_error


def _attach(connection: Any, args: argparse.Namespace) -> LanceNamespace:
    options = [
        "TYPE LANCE",
        "READ_ONLY false",
        "ENDPOINT " + _sql_literal(args.endpoint),
        "DELIMITER " + _sql_literal(args.delimiter),
    ]
    if args.header:
        options.append("HEADER " + _sql_literal(args.header))
    if args.bearer_token:
        options.append("BEARER_TOKEN " + _sql_literal(args.bearer_token))
    if args.api_key:
        options.append("API_KEY " + _sql_literal(args.api_key))
    connection.execute(
        f"ATTACH {_sql_literal(args.namespace)} AS rest_lance ({', '.join(options)})"
    )
    return LanceNamespace(
        args.namespace,
        "rest_lance",
        endpoint=args.endpoint,
        connection=connection,
        attach=False,
    )


def run(args: argparse.Namespace) -> None:
    vane.set_runner_local()
    try:
        connection = load_lance_extension(vane.connect())
    except BaseException:
        _run_cleanup_steps([("tear down runner", vane.teardown_runner)])
        raise
    namespace: LanceNamespace | None = None
    table_name = "vane_example_" + uuid.uuid4().hex
    table_created = False

    try:
        namespace = _attach(connection, args)

        # Materialize the lazy catalog before CTAS. The new table must still
        # become visible immediately in this same connection.
        assert table_name not in {
            row[0]
            for row in connection.execute("SHOW TABLES FROM rest_lance.main").fetchall()
        }
        table = namespace.create_table(
            table_name,
            """
            SELECT * FROM (VALUES
              (1::BIGINT, 'duck'::VARCHAR,
               'a puppy follows a duck'::VARCHAR, 'animal'::VARCHAR,
               [0.0, 0.0, 0.0, 0.0]::FLOAT[4]),
              (2::BIGINT, 'horse'::VARCHAR,
               'a horse runs beside the puppy'::VARCHAR, 'animal'::VARCHAR,
               [1.0, 0.0, 0.0, 0.0]::FLOAT[4]),
              (3::BIGINT, 'dragon'::VARCHAR,
               'a fantasy dragon'::VARCHAR, 'fiction'::VARCHAR,
               [2.0, 0.0, 0.0, 0.0]::FLOAT[4]),
              (4::BIGINT, 'database'::VARCHAR,
               'Lance stores vectors'::VARCHAR, 'technology'::VARCHAR,
               [3.0, 0.0, 0.0, 0.0]::FLOAT[4])
            ) AS source(id, name, text, category, vec)
            """,
        )
        table_created = True
        assert table.scan().aggregate("count(*), min(id), max(id)").fetchone() == (
            4,
            1,
            4,
        )
        assert table_name in {
            row[0]
            for row in connection.execute("SHOW TABLES FROM rest_lance.main").fetchall()
        }

        qualified = f"rest_lance.main.{table_name}"
        rows = connection.execute(
            f"SELECT id, name FROM {qualified} WHERE id >= 1 LIMIT 2 OFFSET 1"
        ).fetchall()
        assert rows == [(2, "horse"), (3, "dragon")], rows
        plan = connection.execute(
            f"EXPLAIN (FORMAT JSON) SELECT name FROM {qualified} WHERE id >= 1 LIMIT 1 OFFSET 1"
        ).fetchone()[1]
        assert '"Lance Scan Backend": "namespace_query_table"' in plan, plan
        assert '"Lance Limit Offset Pushdown": "true"' in plan, plan
        assert '"Lance Pushed Filter Parts": "1"' in plan, plan

        table.insert(
            "SELECT 5::BIGINT AS id, 'retrieval'::VARCHAR AS name, "
            "'hybrid retrieval combines vector and text'::VARCHAR AS text, "
            "'technology'::VARCHAR AS category, "
            "[4.0, 0.0, 0.0, 0.0]::FLOAT[4] AS vec"
        )
        table.update({"name": "'HORSE'"}, where="id = 2")
        table.delete(where="id = 3")
        table.merge(
            """
            SELECT * FROM (VALUES
              (2::BIGINT, 'horse-merged'::VARCHAR,
               'the merged horse follows a puppy'::VARCHAR, 'animal'::VARCHAR,
               [1.0, 0.0, 0.0, 0.0]::FLOAT[4]),
              (6::BIGINT, 'temporary'::VARCHAR,
               'temporary row'::VARCHAR, 'temporary'::VARCHAR,
               [5.0, 0.0, 0.0, 0.0]::FLOAT[4])
            ) AS source(id, name, text, category, vec)
            """,
            "target.id = source.id",
        ).when_matched_update(
            {
                "name": "source.name",
                "text": "source.text",
                "category": "source.category",
                "vec": "source.vec",
            }
        ).when_not_matched_insert(
            {
                "id": "source.id",
                "name": "source.name",
                "text": "source.text",
                "category": "source.category",
                "vec": "source.vec",
            }
        ).execute()
        table.merge(
            "SELECT 6::BIGINT AS id",
            "target.id = source.id",
        ).when_matched_delete().execute()
        assert table.scan().project("id, name").order("id").fetchall() == [
            (1, "duck"),
            (2, "horse-merged"),
            (4, "database"),
            (5, "retrieval"),
        ]

        table.create_index(
            "rest_vec_idx",
            "vec",
            index_type="IVF_FLAT",
            num_partitions=1,
            metric_type="l2",
        )
        assert [row[0] for row in table.show_indexes().fetchall()] == ["rest_vec_idx"]
        vector_rows = (
            table.vector_search(
                "vec",
                [3.1, 0.0, 0.0, 0.0],
                k=2,
                nprobes=1,
                refine_factor=2,
                prefilter=True,
                use_index=True,
                filter="category = 'technology'",
            )
            .project("id, _distance")
            .fetchall()
        )
        assert [row[0] for row in vector_rows] == [4, 5], vector_rows

        fts_rows = (
            table.fts(
                "text",
                "retrieval",
                k=2,
                prefilter=True,
                filter="category = 'technology'",
            )
            .project("id, _score")
            .fetchall()
        )
        assert fts_rows and fts_rows[0][0] == 5, fts_rows

        try:
            table.hybrid_search(
                "vec",
                [3.1, 0.0, 0.0, 0.0],
                "text",
                "retrieval",
                k=2,
                use_index=False,
            ).fetchall()
        except vane.NotImplementedException as error:
            assert "not supported for REST namespace-backed tables" in str(error)
        else:
            raise AssertionError("REST namespace hybrid search unexpectedly succeeded")

        table.drop_index("rest_vec_idx")
        compact = table.optimize(target_rows_per_fragment=2, num_threads=1)
        assert compact and compact[0][0] == "compact", compact
        connection.execute(
            f"ALTER TABLE {qualified} SET AUTO_CLEANUP WITH (interval = 1, older_than = '1h', retain_versions = 2)"
        )
        assert (
            dict(connection.execute(f"SHOW MAINTENANCE ON {qualified}").fetchall())[
                "enabled"
            ]
            == "true"
        )
        connection.execute(f"ALTER TABLE {qualified} UNSET AUTO_CLEANUP")
        cleanup = table.vacuum(older_than_seconds=0, retain_n_versions=1)
        assert cleanup and cleanup[0][0] == "cleanup", cleanup

        namespace.drop_table(table_name)
        table_created = False
        assert table_name not in {
            row[0]
            for row in connection.execute("SHOW TABLES FROM rest_lance.main").fetchall()
        }

        print(f"REST endpoint: {args.endpoint}; namespace: {args.namespace}")
        print("same-connection CTAS discovery and query_table pushdown: passed")
        print("REST INSERT/UPDATE/DELETE/MERGE and DROP: passed")
        print("REST vector/FTS, index, optimize/vacuum/auto-cleanup: passed")
        print("REST hybrid-search unsupported contract: passed")
        print("ALL REST LANCE EXAMPLES PASSED")
    finally:
        active_error = sys.exc_info()[1]
        retained_mutation = isinstance(
            active_error,
            (CopyOutcomeUnknownError, LanceMutationOutcomeUnknownError),
        )
        cleanup_steps: list[tuple[str, Any]] = []
        if namespace is not None and not retained_mutation:
            if table_created:
                cleanup_steps.append(
                    (
                        "drop REST table",
                        lambda: namespace.drop_table(table_name, if_exists=True),
                    )
                )
            cleanup_steps.append(("detach namespace", namespace.detach))
        elif retained_mutation and active_error is not None:
            add_note = getattr(active_error, "add_note", None)
            if add_note is not None:
                add_note(
                    f"REST cleanup for table {table_name!r} was skipped because its mutation lease remains held"
                )
        cleanup_steps.append(("close connection", connection.close))
        cleanup_steps.append(("tear down runner", vane.teardown_runner))
        _run_cleanup_steps(cleanup_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=os.getenv("LANCE_REST_ENDPOINT", "http://127.0.0.1:12333"),
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("LANCE_REST_NAMESPACE", "vane_example"),
    )
    parser.add_argument(
        "--delimiter",
        default=os.getenv("LANCE_REST_DELIMITER", "$"),
    )
    parser.add_argument(
        "--header",
        default=os.getenv("LANCE_REST_HEADER", "x-vane-example=executed"),
    )
    parser.add_argument(
        "--bearer-token",
        default=os.getenv("LANCE_REST_BEARER_TOKEN", ""),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LANCE_REST_API_KEY", ""),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
