# SPDX-FileCopyrightText: 2026 lance-duckdb contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from lance_duckdb._identity import canonicalize_dataset_uri, normalize_dataset_uri


def test_normalize_dataset_uri_removes_credentials_and_query() -> None:
    assert (
        normalize_dataset_uri(
            "S3://access:secret@Bucket/path/data.lance/?token=hidden#fragment"
        )
        == "s3://bucket/path/data.lance"
    )


def test_normalize_dataset_uri_canonicalizes_s3_aliases() -> None:
    expected = "s3://bucket/path/data.lance"
    assert normalize_dataset_uri("s3://bucket/path/data.lance") == expected
    assert normalize_dataset_uri("s3a://bucket/path/data.lance") == expected
    assert normalize_dataset_uri("s3n://bucket/path/data.lance") == expected


def test_normalize_dataset_uri_preserves_ipv6_host_brackets() -> None:
    assert (
        normalize_dataset_uri(
            "HTTPS://user:secret@[2001:DB8::1]:8443/data/?token=hidden"
        )
        == "https://[2001:db8::1]:8443/data"
    )


def test_normalize_dataset_uri_removes_default_http_ports() -> None:
    assert (
        normalize_dataset_uri("http://example.test:80/catalog")
        == "http://example.test/catalog"
    )
    assert (
        normalize_dataset_uri("https://example.test:443/catalog")
        == "https://example.test/catalog"
    )
    assert (
        normalize_dataset_uri("https://example.test:8443/catalog")
        == "https://example.test:8443/catalog"
    )


def test_dataset_uri_helpers_canonicalize_local_file_uri(tmp_path) -> None:
    dataset_path = tmp_path / "space in name.lance"

    assert canonicalize_dataset_uri(dataset_path) == str(dataset_path.resolve())
    assert normalize_dataset_uri(dataset_path) == str(dataset_path.resolve())
    assert canonicalize_dataset_uri(dataset_path.as_uri()) == str(
        dataset_path.resolve()
    )
    assert normalize_dataset_uri(f"file://localhost{dataset_path.as_posix()}") == str(
        dataset_path.resolve()
    )


def test_dataset_uri_helpers_preserve_significant_path_whitespace(tmp_path) -> None:
    dataset_path = tmp_path / " dataset.lance "

    assert canonicalize_dataset_uri(dataset_path) == str(dataset_path.resolve())
    assert normalize_dataset_uri(dataset_path) == str(dataset_path.resolve())
    assert (
        canonicalize_dataset_uri("s3://bucket/ dataset.lance ")
        == "s3://bucket/ dataset.lance "
    )
    assert (
        normalize_dataset_uri("s3://bucket/ dataset.lance ")
        == "s3://bucket/ dataset.lance "
    )


def test_dataset_uri_helpers_reject_byte_paths() -> None:
    with pytest.raises(TypeError, match="text path"):
        canonicalize_dataset_uri(b"dataset.lance")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="text path"):
        normalize_dataset_uri(b"dataset.lance")  # type: ignore[arg-type]


def test_dataset_uri_helpers_reject_nul_bytes() -> None:
    with pytest.raises(ValueError, match="NUL"):
        canonicalize_dataset_uri("s3://bucket/path\x00dataset.lance")
    with pytest.raises(ValueError, match="NUL"):
        normalize_dataset_uri("/tmp/path\x00dataset.lance")
