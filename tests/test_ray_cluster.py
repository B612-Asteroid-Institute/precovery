from __future__ import annotations

import pytest

from precovery import ray_cluster


def test_explicit_serial_mode_does_not_initialize_ray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ray_cluster.ray,
        "is_initialized",
        lambda: (_ for _ in ()).throw(AssertionError("Ray should not be inspected")),
    )

    assert ray_cluster.initialize_use_ray(num_cpus=1) is False
    assert ray_cluster.initialize_use_ray(num_cpus=0) is False


def test_connects_to_existing_cluster_with_dashboard_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(ray_cluster.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray_cluster.ray, "init", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(ray_cluster.ray, "cluster_resources", lambda: {"CPU": 4.0})

    assert ray_cluster.initialize_use_ray(num_cpus=4) is True
    assert calls == [{"address": "auto", "include_dashboard": False}]


def test_falls_back_to_bounded_local_cluster_and_forwards_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def _init(**kwargs: object) -> None:
        calls.append(kwargs)
        if kwargs.get("address") == "auto":
            raise ConnectionError("no cluster")

    monkeypatch.setattr(ray_cluster.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray_cluster.ray, "init", _init)
    monkeypatch.setattr(ray_cluster.ray, "cluster_resources", lambda: {"CPU": 3.0})

    runtime_env = {"excludes": ["runtime_artifacts/**"]}
    assert (
        ray_cluster.initialize_use_ray(
            num_cpus=3,
            object_store_bytes=1234,
            runtime_env=runtime_env,
            include_dashboard=True,
        )
        is True
    )
    assert calls == [
        {
            "address": "auto",
            "runtime_env": runtime_env,
            "include_dashboard": True,
        },
        {
            "num_cpus": 3,
            "object_store_memory": 1234,
            "runtime_env": runtime_env,
            "include_dashboard": True,
        },
    ]


def test_reuses_initialized_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ray_cluster.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        ray_cluster.ray,
        "init",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("ray.init called")),
    )
    monkeypatch.setattr(ray_cluster.ray, "cluster_resources", lambda: {"CPU": 2.0})

    assert ray_cluster.initialize_use_ray(num_cpus=None) is True
