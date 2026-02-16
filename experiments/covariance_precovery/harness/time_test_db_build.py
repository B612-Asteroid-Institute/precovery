from __future__ import annotations

import tempfile
import time
from pathlib import Path

from precovery.ingest import index
from precovery.precovery_db import PrecoveryDatabase


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    data_dir = repo / "tests" / "data" / "index" / "dataset_500"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        t0 = time.perf_counter()
        index(
            out_dir=tmp,
            dataset_id="dataset_500",
            dataset_name="dataset_500",
            data_dir=str(data_dir),
            nside=32,
        )
        t1 = time.perf_counter()
        _db = PrecoveryDatabase.from_dir(str(tmp), mode="r", allow_version_mismatch=True)
        t2 = time.perf_counter()

        print(f"index() sec: {t1 - t0:.3f}")
        print(f"from_dir() sec: {t2 - t1:.3f}")
        print("index.db size bytes:", (tmp / "index.db").stat().st_size)


if __name__ == "__main__":
    main()

