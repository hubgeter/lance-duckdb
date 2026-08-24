#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 lance-duckdb contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Vane's distributed Lance scan, search, and write contracts."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import ray

import vane
from vane import runners
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
            add_note(f"distributed Lance example cleanup also failed: {detail}")
        return
    first_error = failures[0][1]
    add_note = getattr(first_error, "add_note", None)
    if add_note is not None and len(failures) > 1:
        add_note(f"additional distributed Lance example cleanup failures: {detail}")
    raise first_error


@contextmanager
def _example_root(requested_root: Path | None) -> Iterator[Path]:
    if requested_root is None:
        root = Path(tempfile.mkdtemp(prefix="vane-lance-distributed-"))
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


def _write_parquet_parts(
    connection: Any,
    root: Path,
    *,
    start: int,
    count: int,
    file_count: int,
) -> None:
    root.mkdir()
    end = start + count
    for file_id in range(file_count):
        connection.execute(
            "COPY (SELECT i::BIGINT AS id, "
            "('token' || i::VARCHAR)::VARCHAR AS text, "
            "CASE WHEN i % 2 = 0 THEN 'even' ELSE 'odd' END::VARCHAR AS category, "
            "[i::FLOAT, 0.0::FLOAT, 0.0::FLOAT, 0.0::FLOAT]::FLOAT[4] AS vec "
            f"FROM range({start}, {end}) AS source(i) WHERE i % {file_count} = {file_id}) "
            f"TO {_sql_literal(root / f'part-{file_id}.parquet')} (FORMAT PARQUET)"
        )


def _collect(runner: Any, relation: Any) -> list[tuple[Any, ...]]:
    return [
        tuple(row.values())
        for table in runner.run_iter_tables(relation)
        for row in table.to_pylist()
    ]


def _transaction_count(dataset_path: Path) -> int:
    return sum(path.is_file() for path in (dataset_path / "_transactions").iterdir())


def _lance_data_files(dataset_path: Path) -> set[Path]:
    data_path = dataset_path / "data"
    if not data_path.exists():
        return set()
    return {
        path.relative_to(dataset_path)
        for path in data_path.rglob("*")
        if path.is_file()
    }


def _distributed_task_prefixes(files: set[Path]) -> set[str]:
    prefixes: set[str] = set()
    for path in files:
        parts = path.name.split("_", 3)
        if len(parts) == 4 and parts[0] == "vane":
            prefixes.add("_".join(parts[:3]))
    return prefixes


@contextmanager
def _owned_ray_runtime(root: Path) -> Iterator[Any]:
    if ray.is_initialized():
        raise RuntimeError("this example requires an uninitialized Ray runtime")

    environment = {
        "RAY_ADDRESS": None,
        "VANE_DISTRIBUTED_NODE_COUNT": "1",
        "VANE_DISTRIBUTED_WORKER_SLOTS": "4",
        "VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM": "4",
        "VANE_RAY_SCAN_TASK_SIZE_GROUPING": "0",
        "VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION": "1",
    }
    previous_environment = {name: os.environ.get(name) for name in environment}
    previous_cwd = Path.cwd()
    ray_started = False
    runner_started = False
    try:
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

        # Avoid source-tree shadowing when Ray imports the installed package in
        # fresh worker processes, and force a self-contained local cluster.
        os.chdir(root)
        ray.init(address="local", num_cpus=4, include_dashboard=False)
        ray_started = True
        vane.set_runner_ray()
        runner_started = True
        yield runners.get_or_create_runner()
    finally:

        def restore_process_state() -> None:
            os.chdir(previous_cwd)
            for name, value in previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        cleanup_steps: list[tuple[str, Any]] = []
        if runner_started:
            cleanup_steps.append(("tear down runner", vane.teardown_runner))
        if ray_started:
            cleanup_steps.append(("shut down Ray", ray.shutdown))
        cleanup_steps.append(("restore cwd and environment", restore_process_state))
        _run_cleanup_steps(cleanup_steps)


def run(root: Path) -> None:
    with _owned_ray_runtime(root) as runner:
        connection = load_lance_extension(vane.connect())
        namespace: LanceNamespace | None = None

        try:
            create_input = root / "create-input"
            append_input = root / "append-input"
            empty_input = root / "empty-input"
            dataset_path = root / "distributed.lance"
            # Keep the direct dataset URI identical to the attached table name
            # below.  A hyphen here would make the subsequent scan open a
            # different, non-existent dataset after the underscore-named CTAS.
            empty_path = root / "distributed_empty.lance"

            _write_parquet_parts(
                connection, create_input, start=0, count=32, file_count=8
            )
            _write_parquet_parts(
                connection, append_input, start=32, count=16, file_count=8
            )
            _write_parquet_parts(
                connection, empty_input, start=0, count=0, file_count=1
            )

            namespace = LanceNamespace(root, "distributed_lance", connection=connection)
            table = namespace.table("distributed")
            empty_table = namespace.table("distributed_empty")
            dataset = LanceDataset(dataset_path, connection)
            connection.read_parquet(str(create_input / "*.parquet")).create(
                table._sql_target()
            )
            create_files = _lance_data_files(dataset_path)
            assert len(_distributed_task_prefixes(create_files)) > 1
            assert _transaction_count(dataset_path) == 1

            rows = _collect(
                runner, dataset.scan().project("id, text").filter("id % 7 = 0")
            )
            assert sorted(rows) == [
                (0, "token0"),
                (7, "token7"),
                (14, "token14"),
                (21, "token21"),
                (28, "token28"),
            ]

            vector_rows = _collect(
                runner,
                dataset.vector_search(
                    "vec", [0.1, 0.0, 0.0, 0.0], k=3, use_index=False
                ).project("id, _distance"),
            )
            assert [row[0] for row in vector_rows] == [0, 1, 2], vector_rows

            fts_rows = _collect(
                runner, dataset.fts("text", "token17", k=3).project("id, _score")
            )
            assert fts_rows and fts_rows[0][0] == 17, fts_rows

            hybrid_rows = _collect(
                runner,
                dataset.hybrid_search(
                    "vec",
                    [17.1, 0.0, 0.0, 0.0],
                    "text",
                    "token17",
                    k=3,
                    use_index=False,
                    alpha=0.5,
                ).project("id, _hybrid_score"),
            )
            assert hybrid_rows and hybrid_rows[0][0] == 17, hybrid_rows

            connection.read_parquet(str(append_input / "*.parquet")).insert_into(
                table._sql_target()
            )
            append_files = _lance_data_files(dataset_path) - create_files
            assert len(_distributed_task_prefixes(append_files)) > 1
            assert _transaction_count(dataset_path) == 2
            assert dataset.scan().aggregate(
                "count(*), min(id), max(id)"
            ).fetchone() == (48, 0, 47)

            connection.read_parquet(str(empty_input / "*.parquet")).create(
                empty_table._sql_target()
            )
            empty_dataset = LanceDataset(empty_path, connection)
            empty = empty_dataset.scan()
            assert empty.fetchall() == []
            assert empty.columns == ["id", "text", "category", "vec"]
            assert _transaction_count(empty_path) == 1

            for path in (dataset_path, empty_path):
                staging = path / "_vane_staging"
                assert not staging.exists() or not any(staging.rglob("*"))

            print("distributed fragment scan: passed")
            print("global vector/FTS/hybrid ranking: passed")
            print(
                "multi-task, single-transaction attached INSERT/CTAS and empty schema: passed"
            )
            print("ALL DISTRIBUTED LANCE EXAMPLES PASSED")
        finally:
            cleanup_steps: list[tuple[str, Any]] = [
                ("close connection", connection.close)
            ]
            if namespace is not None:
                cleanup_steps.insert(0, ("detach namespace", namespace.detach))
            _run_cleanup_steps(cleanup_steps)


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


if __name__ == "__main__":
    main()
