# SPDX-FileCopyrightText: 2026 lance-duckdb contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from lance_duckdb import LANCE_EXTENSION_PATH_ENV, load_lance_extension


class RecordingConnection:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    def load_extension(self, path: str) -> None:
        self.loaded.append(path)


def test_load_lance_extension_uses_explicit_preprovisioned_path(tmp_path: Path) -> None:
    artifact = tmp_path / "lance.duckdb_extension"
    artifact.write_bytes(b"artifact")
    connection = RecordingConnection()

    assert load_lance_extension(connection, artifact) is connection
    assert connection.loaded == [str(artifact.resolve())]


def test_load_lance_extension_uses_environment_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "lance.duckdb_extension"
    artifact.write_bytes(b"artifact")
    monkeypatch.setenv(LANCE_EXTENSION_PATH_ENV, str(artifact))
    connection = RecordingConnection()

    load_lance_extension(connection)

    assert connection.loaded == [str(artifact.resolve())]


def test_load_lance_extension_never_falls_back_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(LANCE_EXTENSION_PATH_ENV, raising=False)

    with pytest.raises(ValueError, match=LANCE_EXTENSION_PATH_ENV):
        load_lance_extension(RecordingConnection())
