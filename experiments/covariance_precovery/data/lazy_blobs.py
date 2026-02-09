from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from .gcs_copy import CopyTool, default_copy_tool
from .subset_db import GCS_ROOT


def _validate_data_uri(data_uri: str) -> str:
    """
    Validate that `data_uri` is a safe, relative path under the DB's `data/` root.
    """
    s = str(data_uri).strip().lstrip("/")
    if not s:
        raise ValueError("data_uri must be non-empty")
    # Defend against path traversal.
    parts = [p for p in s.split("/") if p]
    if any(p == "." or p == ".." for p in parts):
        raise ValueError(f"Invalid data_uri (path traversal): {data_uri!r}")
    # Ensure we keep the normalized (no leading slashes) form.
    return "/".join(parts)


def local_blob_path(*, db_dir: Path, data_uri: str) -> Path:
    """
    Map `frames.data_uri` into the expected local path for FrameDB:
      <db_dir>/data/<data_uri>
    """
    rel = _validate_data_uri(data_uri)
    return Path(db_dir) / "data" / rel


def gcs_blob_uri(*, data_uri: str, gcs_root: str = GCS_ROOT) -> str:
    rel = _validate_data_uri(data_uri)
    # Production layout: <gcs_root>/data/<dataset>/<YYYY-MM>/frames_*.data
    return f"{str(gcs_root).rstrip('/')}/data/{rel}"


@dataclass(frozen=True)
class LazyBlobConfig:
    gcs_root: str = GCS_ROOT
    lock_timeout_sec: float = 600.0
    poll_sec: float = 0.25


class _FileLock:
    """
    Very small, cross-process lock based on O_EXCL lockfiles.

    This is intentionally minimal; it is sufficient to prevent multiple concurrent
    downloads of the same blob on a single machine.
    """

    def __init__(self, lock_path: Path, *, timeout_sec: float, poll_sec: float):
        self.lock_path = Path(lock_path)
        self.timeout_sec = float(timeout_sec)
        self.poll_sec = float(poll_sec)
        self._fd: int | None = None

    def __enter__(self) -> "_FileLock":
        t0 = time.monotonic()
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                self._fd = int(fd)
                try:
                    os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
                except Exception:
                    # Best-effort only; do not fail locking if write fails.
                    pass
                return self
            except FileExistsError:
                if (time.monotonic() - t0) >= float(self.timeout_sec):
                    raise TimeoutError(f"Timed out waiting for lock: {self.lock_path}")
                time.sleep(float(self.poll_sec))

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self._fd is not None:
            try:
                os.close(int(self._fd))
            except Exception:
                pass
            self._fd = None
        try:
            self.lock_path.unlink(missing_ok=True)
        except Exception:
            pass


def ensure_blob_local(
    *,
    db_dir: Path,
    data_uri: str,
    cfg: LazyBlobConfig | None = None,
    copy_tool: CopyTool | None = None,
) -> Path:
    """
    Ensure the `.data` blob referenced by `data_uri` exists under `<db_dir>/data/`.

    When missing, downloads it from GCS (using `cfg.gcs_root`) to the exact path expected
    by `FrameDB.get_observations`:
      path = os.path.join(db.frames.data_root, data_uri)

    Returns the local path.
    """
    cfg = cfg or LazyBlobConfig()
    copy_tool = copy_tool or default_copy_tool()

    dst = local_blob_path(db_dir=Path(db_dir), data_uri=str(data_uri))
    if dst.exists():
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dst.with_suffix(dst.suffix + ".lock")

    with _FileLock(lock_path, timeout_sec=float(cfg.lock_timeout_sec), poll_sec=float(cfg.poll_sec)):
        # Another process may have downloaded it while we waited.
        if dst.exists():
            return dst

        src = gcs_blob_uri(data_uri=str(data_uri), gcs_root=str(cfg.gcs_root))
        tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
        try:
            if tmp.exists():
                tmp.unlink()
            copy_tool.cp(src, tmp)
            tmp.replace(dst)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    return dst

