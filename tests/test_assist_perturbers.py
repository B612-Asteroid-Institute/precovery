"""Tests for precovery.search.assist_perturbers."""

from __future__ import annotations

import numpy as np

from precovery.search import assist_perturbers
from precovery.search.assist_perturbers import (
    is_assist_perturber,
    perturber_warnings_for_orbit_ids,
    perturber_warning_or_none,
)


def test_is_assist_perturber_matches() -> None:
    assert is_assist_perturber("pluto") is True
    assert is_assist_perturber("Pluto") is True
    assert is_assist_perturber("16") is True
    assert is_assist_perturber("Psyche") is True
    assert is_assist_perturber("16 Psyche") is True
    assert is_assist_perturber("4 Vesta") is True


def test_is_assist_perturber_non_match() -> None:
    assert is_assist_perturber("2019 QU127") is False
    assert is_assist_perturber("00000") is False
    assert is_assist_perturber("random") is False


def test_is_assist_perturber_normalize_false() -> None:
    assert is_assist_perturber("pluto", normalize=False) is True
    assert is_assist_perturber("Pluto", normalize=False) is False


def test_perturber_warning_or_none_returns_message_when_perturber() -> None:
    msg = perturber_warning_or_none("pluto")
    assert msg is not None
    assert "ASSIST perturber" in msg
    assert "propagation" in msg.lower()


def test_perturber_warning_or_none_returns_none_when_not_perturber() -> None:
    assert perturber_warning_or_none("2019 QU127") is None
    assert perturber_warning_or_none("00000") is None


def test_perturber_warnings_for_orbit_ids_vectorized_output() -> None:
    out = perturber_warnings_for_orbit_ids(["pluto", "2019 QU127", "16 Psyche"])
    assert out[0] is not None
    assert out[1] is None
    assert out[2] is not None


def test_perturber_warnings_uses_vectorized_adam_assist_when_available(monkeypatch) -> None:
    calls = {"n": 0}

    def _fake_is_perturber(values, *, normalize=True):
        calls["n"] += 1
        arr = np.asarray(values, dtype=object)
        out = np.empty(arr.shape, dtype=object)
        out[:] = None
        out[arr == "pluto"] = "pluto"
        return out

    monkeypatch.setattr(assist_perturbers, "_is_perturber", _fake_is_perturber)
    out = perturber_warnings_for_orbit_ids(["pluto", "x"])
    assert calls["n"] == 1
    assert out[0] is not None
    assert out[1] is None
