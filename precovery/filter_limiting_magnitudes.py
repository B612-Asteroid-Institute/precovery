from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv


class FilterLimitingMagnitudes(qv.Table):
    """
    Per-observatory limiting magnitudes for a canonical bandpass filter_id.

    Notes
    -----
    - `filter_id` must be a canonical bandpass filter ID as used by `adam_core.photometry`,
      e.g. "DECam_g", "LSST_r", "V".
    - This table is intended to be stored as Parquet in the DB directory and loaded once
      at DB open to avoid any runtime DB queries in the precovery hot path.
    """

    obscode = qv.LargeStringColumn()
    filter_id = qv.LargeStringColumn()
    limiting_mag = qv.Float64Column()
    mag_system = qv.LargeStringColumn(nullable=True)

    def static_code_filter_map(self) -> dict[str, float]:
        """
        Build a simple in-memory map keyed by "obscode|filter_id".
        """
        sep = pa.scalar("|", type=pa.large_string())
        keys = pc.binary_join_element_wise(self.obscode, self.filter_id, sep)
        return {str(k): float(v) for k, v in zip(keys.to_pylist(), self.limiting_mag.to_pylist())}
