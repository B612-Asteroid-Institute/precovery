from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from experiments.covariance_precovery.data.lazy_blobs import (
    LazyBlobConfig,
    ensure_blob_local,
    gcs_blob_uri,
    local_blob_path,
)


@dataclass(frozen=True)
class _FakeCopyTool:
    calls: list[tuple[str, Path]]

    def cp(self, src: str, dst: Path, recursive: bool = False) -> None:  # noqa: ARG002
        self.calls.append((str(src), Path(dst)))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"blob-bytes")


def test_local_blob_path_resolves_under_data_root() -> None:
    with TemporaryDirectory() as td:
        db_dir = Path(td)
        p = local_blob_path(db_dir=db_dir, data_uri="nsc/2019-08/frames_00000001.data")
        assert p == db_dir / "data" / "nsc" / "2019-08" / "frames_00000001.data"


@pytest.mark.parametrize("bad", ["../x", "a/../b", "a/./b", ""])
def test_local_blob_path_rejects_traversal(bad: str) -> None:
    with TemporaryDirectory() as td:
        db_dir = Path(td)
        with pytest.raises(ValueError):
            _ = local_blob_path(db_dir=db_dir, data_uri=bad)


def test_gcs_blob_uri_appends_data_prefix() -> None:
    uri = gcs_blob_uri(data_uri="ztf/2020-01/frames_00000002.data", gcs_root="gs://root")
    assert uri == "gs://root/data/ztf/2020-01/frames_00000002.data"


def test_ensure_blob_local_downloads_missing_to_expected_path() -> None:
    with TemporaryDirectory() as td:
        db_dir = Path(td)
        tool = _FakeCopyTool(calls=[])
        cfg = LazyBlobConfig(gcs_root="gs://root")

        out = ensure_blob_local(
            db_dir=db_dir, data_uri="nsc/2019-08/frames_00000001.data", cfg=cfg, copy_tool=tool
        )
        assert out.exists()
        assert out.read_bytes() == b"blob-bytes"
        assert out == db_dir / "data" / "nsc" / "2019-08" / "frames_00000001.data"
        # Ensure we attempted to copy from the expected GCS object.
        assert tool.calls and tool.calls[0][0] == "gs://root/data/nsc/2019-08/frames_00000001.data"


def test_ensure_blob_local_noops_when_present() -> None:
    with TemporaryDirectory() as td:
        db_dir = Path(td)
        dst = db_dir / "data" / "nsc" / "2019-08" / "frames_00000001.data"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"already")

        tool = _FakeCopyTool(calls=[])
        out = ensure_blob_local(
            db_dir=db_dir,
            data_uri="nsc/2019-08/frames_00000001.data",
            cfg=LazyBlobConfig(gcs_root="gs://root"),
            copy_tool=tool,
        )
        assert out.read_bytes() == b"already"
        assert tool.calls == []

