from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CopyTool:
    name: str

    def cp(self, src: str, dst: Path, recursive: bool = False) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if self.name == "gcloud":
            cmd = ["gcloud", "storage", "cp"]
            if recursive:
                cmd.append("--recursive")
            cmd.extend([src, str(dst)])
        elif self.name == "gsutil":
            cmd = ["gsutil", "-m", "cp"]
            if recursive:
                cmd.append("-r")
            cmd.extend([src, str(dst)])
        else:
            raise ValueError(f"Unknown copy tool: {self.name}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            joined = " ".join(shlex.quote(c) for c in cmd)
            raise RuntimeError(f"Copy failed: {joined}") from exc


def default_copy_tool() -> CopyTool:
    """
    Prefer `gcloud storage` (GA). Fall back to `gsutil` if needed.
    """
    for tool in ("gcloud", "gsutil"):
        try:
            if tool == "gcloud":
                subprocess.run(
                    ["gcloud", "storage", "--help"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            else:
                subprocess.run(
                    ["gsutil", "help"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            return CopyTool(tool)
        except Exception:
            continue
    raise RuntimeError("Neither `gcloud storage` nor `gsutil` is available.")

