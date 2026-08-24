#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 lance-duckdb contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate every public TYPE LANCE secret provider and object-store family."""

from __future__ import annotations

import sys
from typing import Any

import vane
from lance_duckdb import load_lance_extension

SECRETS = {
    "lance_example_s3_config": ("config", "s3://example-bucket/"),
    "lance_example_s3_chain": ("credential_chain", "s3a://example-bucket/"),
    "lance_example_s3_env": ("env", "s3n://example-bucket/"),
    "lance_example_gcs": ("config", "gs://example-bucket/"),
    "lance_example_azure": ("config", "az://example-container/"),
    "lance_example_oss": ("config", "oss://example-bucket/"),
    "lance_example_hf": ("config", "hf://datasets/example/repository/"),
}


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
            add_note(f"Lance secret example cleanup also failed: {detail}")
        return
    first_error = failures[0][1]
    add_note = getattr(first_error, "add_note", None)
    if add_note is not None and len(failures) > 1:
        add_note(f"additional Lance secret example cleanup failures: {detail}")
    raise first_error


def run() -> None:
    connection = load_lance_extension(vane.connect())
    created: list[str] = []
    try:
        statements = {
            "lance_example_s3_config": """
                CREATE SECRET lance_example_s3_config (
                  TYPE LANCE, PROVIDER config, SCOPE 's3://example-bucket/',
                  ACCESS_KEY_ID 'example-access', SECRET_ACCESS_KEY 'example-secret',
                  REGION 'us-east-1', ENDPOINT 'https://s3.example.invalid',
                  VIRTUAL_HOSTED_STYLE_REQUEST false, ALLOW_HTTP false,
                  CONNECT_TIMEOUT '5s', REQUEST_TIMEOUT '30s', CLIENT_MAX_RETRIES '3',
                  STORAGE_OPTIONS map(['custom_option'], ['custom-value'])
                )
            """,
            "lance_example_s3_chain": """
                CREATE SECRET lance_example_s3_chain (
                  TYPE LANCE, PROVIDER credential_chain,
                  SCOPE 's3a://example-bucket/', REGION 'us-east-1'
                )
            """,
            "lance_example_s3_env": """
                CREATE SECRET lance_example_s3_env (
                  TYPE LANCE, PROVIDER env,
                  SCOPE 's3n://example-bucket/', REGION 'us-east-1'
                )
            """,
            "lance_example_gcs": """
                CREATE SECRET lance_example_gcs (
                  TYPE LANCE, PROVIDER config, SCOPE 'gs://example-bucket/',
                  GOOGLE_STORAGE_TOKEN 'example-google-token', USE_OPENDAL false
                )
            """,
            "lance_example_azure": """
                CREATE SECRET lance_example_azure (
                  TYPE LANCE, PROVIDER config, SCOPE 'az://example-container/',
                  ACCOUNT_NAME 'example-account', ACCOUNT_KEY 'ZXhhbXBsZQ==',
                  USE_OPENDAL true
                )
            """,
            "lance_example_oss": """
                CREATE SECRET lance_example_oss (
                  TYPE LANCE, PROVIDER config, SCOPE 'oss://example-bucket/',
                  OSS_ENDPOINT 'https://oss-cn-hangzhou.aliyuncs.com',
                  OSS_ACCESS_KEY_ID 'example-oss-access',
                  OSS_SECRET_ACCESS_KEY 'example-oss-secret',
                  OSS_REGION 'cn-hangzhou'
                )
            """,
            "lance_example_hf": """
                CREATE SECRET lance_example_hf (
                  TYPE LANCE, PROVIDER config,
                  SCOPE 'hf://datasets/example/repository/',
                  HF_TOKEN 'example-hf-token', HF_REVISION 'main', HF_ROOT 'datasets'
                )
            """,
        }

        for name, statement in statements.items():
            connection.execute(statement)
            created.append(name)

        rows = connection.execute(
            "SELECT name, provider, scope, secret_string FROM duckdb_secrets() "
            "WHERE name LIKE 'lance_example_%' ORDER BY name"
        ).fetchall()
        assert len(rows) == len(SECRETS), rows
        for name, provider, scopes, secret_string in rows:
            expected_provider, expected_scope = SECRETS[name]
            assert provider == expected_provider
            assert scopes == [expected_scope]
            assert "example-secret" not in secret_string
            assert "example-google-token" not in secret_string
            assert "example-hf-token" not in secret_string
        s3_config = next(row[3] for row in rows if row[0] == "lance_example_s3_config")
        assert "secret_access_key=redacted" in s3_config
        assert "custom_option=custom-value" in s3_config

        print("config, credential_chain, and env providers: passed")
        print("S3/s3a/s3n, GCS, Azure, OSS, and Hugging Face scopes: passed")
        print("allowlisted options, STORAGE_OPTIONS fallback, and redaction: passed")
        print("ALL LANCE SECRET EXAMPLES PASSED")
    finally:
        cleanup_steps: list[tuple[str, Any]] = [
            (
                f"drop secret {name}",
                lambda name=name: connection.execute(f"DROP SECRET {name}"),
            )
            for name in reversed(created)
        ]
        cleanup_steps.append(("close connection", connection.close))
        _run_cleanup_steps(cleanup_steps)


if __name__ == "__main__":
    run()
