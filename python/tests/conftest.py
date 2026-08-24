# SPDX-FileCopyrightText: 2026 lance-duckdb contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

import pytest


@pytest.fixture(autouse=True)
def isolated_vane_runner(monkeypatch: pytest.MonkeyPatch):
    """Keep plugin tests independent from process-global Vane runner state."""
    try:
        import vane
    except ModuleNotFoundError as exc:
        if exc.name != "vane":
            raise
        yield
        return

    vane.teardown_runner()
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    try:
        yield
    finally:
        vane.teardown_runner()


@pytest.fixture(scope="session")
def ray_local():
    """Own the optional real-Ray cluster used by signed-artifact tests."""
    import vane

    try:
        import ray
    except ModuleNotFoundError as exc:
        if exc.name != "ray":
            raise
        pytest.skip("ray is not installed")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Tip: In future versions of Ray")
        if ray.is_initialized():
            ray.shutdown()
        ray.init(
            address="local",
            ignore_reinit_error=True,
            include_dashboard=False,
            logging_level="info",
            log_to_driver=True,
        )
    try:
        yield
    finally:
        try:
            vane.teardown_runner()
        finally:
            ray.shutdown()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "real_ray: starts and uses a real Ray cluster")
    config.addinivalue_line(
        "markers", "ray_cluster_owner: owns the lifecycle of a Ray cluster"
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if {"ray_runner", "ray_write_runner"}.intersection(item.fixturenames):
            item.add_marker(pytest.mark.real_ray)
            item.add_marker(pytest.mark.ray_cluster_owner)
