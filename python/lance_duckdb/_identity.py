# SPDX-FileCopyrightText: 2026 lance-duckdb contributors
# SPDX-License-Identifier: Apache-2.0

"""Credential-free Lance dataset identity helpers.

These helpers are deliberately independent of execution and lease state.  The
native Lance operators own snapshot/mutation/vacuum lifetimes; Python only
needs a stable representation when constructing REST table identities or
diagnostic names.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import SplitResult, unquote, urlsplit, urlunsplit


def normalize_dataset_uri(uri: str | os.PathLike[str]) -> str:
    """Return a credential-free, stable identity for a Lance dataset."""
    value = os.fspath(uri)
    if not isinstance(value, str):
        raise TypeError("Lance dataset URI must be a string or text path")
    if not value:
        raise ValueError("Lance dataset URI cannot be empty")
    if "\x00" in value:
        raise ValueError("Lance dataset URI cannot contain NUL bytes")
    parsed = urlsplit(value)
    if not parsed.scheme or (len(parsed.scheme) == 1 and value[1:2] == ":"):
        return str(Path(value).expanduser().resolve(strict=False))
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if scheme == "file" and hostname in {"", "localhost"}:
        return str(Path(unquote(parsed.path)).expanduser().resolve(strict=False))
    parsed_port = parsed.port
    default_port = (scheme == "http" and parsed_port == 80) or (
        scheme == "https" and parsed_port == 443
    )
    port = f":{parsed_port}" if parsed_port is not None and not default_port else ""
    host = f"[{hostname}]" if ":" in hostname else hostname
    # Omit userinfo, query, and fragment so credentials cannot enter names or
    # logs.  s3a/s3n are aliases for the same object-store identity.
    if scheme in {"s3a", "s3n"}:
        scheme = "s3"
    clean = SplitResult(scheme, host + port, parsed.path.rstrip("/"), "", "")
    return urlunsplit(clean)


def canonicalize_dataset_uri(uri: str | os.PathLike[str]) -> str:
    """Resolve local paths once while preserving remote access details."""
    value = os.fspath(uri)
    if not isinstance(value, str):
        raise TypeError("Lance dataset URI must be a string or text path")
    if not value:
        raise ValueError("Lance dataset URI cannot be empty")
    if "\x00" in value:
        raise ValueError("Lance dataset URI cannot contain NUL bytes")
    parsed = urlsplit(value)
    if not parsed.scheme or (len(parsed.scheme) == 1 and value[1:2] == ":"):
        return str(Path(value).expanduser().resolve(strict=False))
    if parsed.scheme.lower() == "file" and (parsed.hostname or "").lower() in {
        "",
        "localhost",
    }:
        return str(Path(unquote(parsed.path)).expanduser().resolve(strict=False))
    return value
