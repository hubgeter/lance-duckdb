#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 lance-duckdb contributors
# SPDX-License-Identifier: Apache-2.0

"""Run a real distributed Lance round trip against S3 or MinIO."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import ray
from botocore.config import Config
from botocore.exceptions import ClientError
from botocore.session import Session

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
            add_note(f"Lance S3 example cleanup also failed: {detail}")
        return
    first_error = failures[0][1]
    add_note = getattr(first_error, "add_note", None)
    if add_note is not None and len(failures) > 1:
        add_note(f"additional Lance S3 example cleanup failures: {detail}")
    raise first_error


def _validate_endpoint(endpoint: str) -> tuple[str, str, bool]:
    parsed = urlparse(endpoint)
    try:
        parsed.port
    except ValueError as error:
        raise ValueError(f"--endpoint has an invalid port: {endpoint}") from error
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("--endpoint must be an absolute http:// or https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("--endpoint must not contain credentials")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("--endpoint must not contain a path, query, or fragment")
    normalized_endpoint = f"{parsed.scheme}://{parsed.netloc}"
    return normalized_endpoint, parsed.netloc, parsed.scheme == "http"


@contextmanager
def _s3_client(args: argparse.Namespace, endpoint: str) -> Iterator[Any]:
    client = Session().create_client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=args.access_key_id,
        aws_secret_access_key=args.secret_access_key,
        region_name=args.region,
        config=Config(s3={"addressing_style": "path"}),
    )
    try:
        yield client
    finally:
        client.close()


def _ensure_bucket(args: argparse.Namespace, endpoint: str) -> None:
    create_options: dict[str, Any] = {"Bucket": args.bucket}
    if args.region != "us-east-1":
        create_options["CreateBucketConfiguration"] = {
            "LocationConstraint": args.region
        }
    try:
        with _s3_client(args, endpoint) as client:
            client.create_bucket(**create_options)
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code != "BucketAlreadyOwnedByYou":
            raise


def _write_parquet_parts(
    connection: Any, root: Path, *, start: int, count: int
) -> None:
    root.mkdir()
    end = start + count
    for file_id in range(4):
        connection.execute(
            "COPY (SELECT i::BIGINT AS id, ('token' || i::VARCHAR)::VARCHAR AS text, "
            "[i::FLOAT, 0.0::FLOAT, 0.0::FLOAT, 0.0::FLOAT]::FLOAT[4] AS vec "
            f"FROM range({start}, {end}) AS source(i) WHERE i % 4 = {file_id}) "
            f"TO {_sql_literal(root / f'part-{file_id}.parquet')} (FORMAT PARQUET)"
        )


def _collect(runner: Any, relation: Any) -> list[tuple[Any, ...]]:
    return [
        tuple(row.values())
        for table in runner.run_iter_tables(relation)
        for row in table.to_pylist()
    ]


def _distributed_task_prefixes(keys: set[str]) -> set[str]:
    prefixes: set[str] = set()
    for key in keys:
        parts = Path(key).name.split("_", 3)
        if len(parts) == 4 and parts[0] == "vane":
            prefixes.add("_".join(parts[:3]))
    return prefixes


def _s3_data_keys(
    args: argparse.Namespace, endpoint: str, dataset_prefix: str
) -> set[str]:
    keys: set[str] = set()
    with _s3_client(args, endpoint) as client:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=args.bucket, Prefix=f"{dataset_prefix}/data/"
        ):
            keys.update(str(item["Key"]) for item in page.get("Contents", []))
    return keys


def _configure_distributed_s3_settings(
    connection: Any,
    args: argparse.Namespace,
    duckdb_endpoint: str,
    *,
    allow_http: bool,
) -> None:
    connection.execute(f"SET s3_access_key_id={_sql_literal(args.access_key_id)}")
    connection.execute(
        f"SET s3_secret_access_key={_sql_literal(args.secret_access_key)}"
    )
    connection.execute(f"SET s3_region={_sql_literal(args.region)}")
    connection.execute(f"SET s3_endpoint={_sql_literal(duckdb_endpoint)}")
    connection.execute(f"SET s3_use_ssl={'false' if allow_http else 'true'}")
    connection.execute("SET s3_url_style='path'")


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


def run(args: argparse.Namespace) -> None:
    endpoint, duckdb_endpoint, allow_http = _validate_endpoint(args.endpoint)
    _ensure_bucket(args, endpoint)
    run_id = "run_" + uuid.uuid4().hex
    table_name = "items"
    dataset_prefix = f"{run_id}/{table_name}.lance"
    namespace_uri = f"s3://{args.bucket}/{run_id}"
    dataset_uri = f"s3://{args.bucket}/{dataset_prefix}"
    secret_name = "lance_minio_" + uuid.uuid4().hex

    with tempfile.TemporaryDirectory(prefix="vane-lance-s3-") as temp_dir:
        root = Path(temp_dir)
        create_input = root / "create-input"
        append_input = root / "append-input"

        with _owned_ray_runtime(root) as runner:
            connection = load_lance_extension(vane.connect())
            namespace: LanceNamespace | None = None
            secret_created = False
            table_may_exist = False

            try:
                # Vane snapshots explicit connection settings for workers.
                # DuckDB secret objects remain driver-local by design.
                _configure_distributed_s3_settings(
                    connection,
                    args,
                    duckdb_endpoint,
                    allow_http=allow_http,
                )
                connection.execute(
                    f"""
                    CREATE SECRET {secret_name} (
                      TYPE LANCE,
                      PROVIDER config,
                      SCOPE {_sql_literal(f"s3://{args.bucket}/")},
                      ACCESS_KEY_ID {_sql_literal(args.access_key_id)},
                      SECRET_ACCESS_KEY {_sql_literal(args.secret_access_key)},
                      REGION {_sql_literal(args.region)},
                      ENDPOINT {_sql_literal(endpoint)},
                      VIRTUAL_HOSTED_STYLE_REQUEST false,
                      ALLOW_HTTP {"true" if allow_http else "false"}
                    )
                    """
                )
                secret_created = True
                secret = connection.execute(
                    "SELECT type, provider, scope FROM duckdb_secrets() WHERE name = "
                    + _sql_literal(secret_name)
                ).fetchone()
                assert secret == ("lance", "config", [f"s3://{args.bucket}/"]), secret

                # Attach before creating the dataset to exercise dynamic table
                # discovery on an already materialized directory namespace.
                namespace = LanceNamespace(
                    namespace_uri, "s3_lance", connection=connection
                )
                assert (
                    connection.execute("SHOW TABLES FROM s3_lance.main").fetchall()
                    == []
                )

                _write_parquet_parts(connection, create_input, start=0, count=24)
                _write_parquet_parts(connection, append_input, start=24, count=8)
                table_may_exist = True
                dataset = LanceDataset(dataset_uri, connection)
                table = namespace.table(table_name)
                connection.read_parquet(str(create_input / "*.parquet")).create(
                    table._sql_target()
                )
                create_keys = _s3_data_keys(args, endpoint, dataset_prefix)
                assert len(_distributed_task_prefixes(create_keys)) > 1
                connection.read_parquet(str(append_input / "*.parquet")).insert_into(
                    table._sql_target()
                )
                append_keys = (
                    _s3_data_keys(args, endpoint, dataset_prefix) - create_keys
                )
                assert len(_distributed_task_prefixes(append_keys)) > 1

                distributed_rows = _collect(
                    runner,
                    dataset.scan().filter("id % 9 = 0").project("id, text"),
                )
                assert sorted(distributed_rows) == [
                    (0, "token0"),
                    (9, "token9"),
                    (18, "token18"),
                    (27, "token27"),
                ]

                vector_rows = _collect(
                    runner,
                    dataset.vector_search(
                        "vec", [11.1, 0.0, 0.0, 0.0], k=3, use_index=False
                    ).project("id"),
                )
                assert [row[0] for row in vector_rows] == [11, 12, 10], vector_rows
                fts_rows = _collect(
                    runner, dataset.fts("text", "token11", k=3).project("id")
                )
                assert fts_rows == [(11,)], fts_rows
                hybrid_rows = _collect(
                    runner,
                    dataset.hybrid_search(
                        "vec",
                        [11.1, 0.0, 0.0, 0.0],
                        "text",
                        "token11",
                        k=3,
                        use_index=False,
                    ).project("id"),
                )
                assert hybrid_rows and hybrid_rows[0] == (11,), hybrid_rows

                for scheme in ("s3", "s3a", "s3n"):
                    alias_uri = dataset_uri.replace("s3://", f"{scheme}://", 1)
                    assert LanceDataset(alias_uri, connection).scan().aggregate(
                        "count(*)"
                    ).fetchone() == (32,)

                assert connection.execute(
                    "SHOW TABLES FROM s3_lance.main"
                ).fetchall() == [(table_name,)]
                assert namespace.table(table_name).scan().aggregate(
                    "min(id), max(id)"
                ).fetchone() == (0, 31)
                namespace.drop_table(table_name)
                table_may_exist = False
                assert (
                    connection.execute("SHOW TABLES FROM s3_lance.main").fetchall()
                    == []
                )

                print(f"S3 dataset: {dataset_uri}")
                print("TYPE LANCE secret plus distributed s3_* settings: passed")
                print("s3/s3a/s3n resolution: passed")
                print(
                    "multi-task distributed S3 attached CTAS/INSERT/scan/vector/FTS/hybrid: passed"
                )
                print("S3 directory namespace dynamic discovery and DROP TABLE: passed")
                print("ALL S3 LANCE EXAMPLES PASSED")
            finally:
                active_error = sys.exc_info()[1]
                retained_mutation = isinstance(
                    active_error,
                    (CopyOutcomeUnknownError, LanceMutationOutcomeUnknownError),
                )
                cleanup_steps: list[tuple[str, Any]] = []
                if namespace is not None and not retained_mutation:
                    if table_may_exist:
                        cleanup_steps.append(
                            (
                                "drop remote table",
                                lambda: namespace.drop_table(
                                    table_name, if_exists=True
                                ),
                            )
                        )
                    cleanup_steps.append(("detach namespace", namespace.detach))
                elif retained_mutation and active_error is not None:
                    add_note = getattr(active_error, "add_note", None)
                    if add_note is not None:
                        add_note(
                            f"remote cleanup for {dataset_uri!r} was skipped because its mutation lease remains held"
                        )
                if secret_created:
                    cleanup_steps.append(
                        (
                            "drop secret",
                            lambda: connection.execute(f"DROP SECRET {secret_name}"),
                        )
                    )
                cleanup_steps.append(("close connection", connection.close))
                _run_cleanup_steps(cleanup_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint", default=os.getenv("LANCE_S3_ENDPOINT", "http://127.0.0.1:19000")
    )
    parser.add_argument(
        "--bucket", default=os.getenv("LANCE_S3_BUCKET", "vane-lance-example")
    )
    parser.add_argument("--region", default=os.getenv("LANCE_S3_REGION", "us-east-1"))
    parser.add_argument(
        "--access-key-id", default=os.getenv("LANCE_S3_ACCESS_KEY_ID", "minioadmin")
    )
    parser.add_argument(
        "--secret-access-key",
        default=os.getenv("LANCE_S3_SECRET_ACCESS_KEY", "minioadmin"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
