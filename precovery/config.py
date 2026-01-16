from __future__ import annotations

import inspect
import json

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


class Config:
    def __init__(
        self,
        nside: int = 32,
        data_file_max_size: int = int(1e9),
        build_version: str = __version__,
        limiting_magnitudes_parquet_file: str | None = "limiting_magnitudes.parquet",
        faint_frame_skip_margin_mag: float = 0.0,
    ):
        """
        Precovery Database Configuration

        Parameters
        ----------
        nside : int
            Observations are indexed in HealpixFrames for each unique mjd and observatory_code.
            The nside parameter sets the size of the healpixelization.
        data_file_max_size : int
            Maximum size in bytes of the binary files to which the indexed observations are
            saved.
        """
        self.build_version = build_version
        self.nside = nside
        self.data_file_max_size = data_file_max_size
        # Generated-once cache file stored in the DB directory (Parquet).
        # If present, it will be loaded once at DB open and used for fast faint-frame
        # skipping without any subsequent SQL queries.
        self.limiting_magnitudes_parquet_file = limiting_magnitudes_parquet_file
        # If predicted_mag > (limit + margin), treat it as too faint to be detectable.
        self.faint_frame_skip_margin_mag = faint_frame_skip_margin_mag

        return

    def to_json(self, out_file):
        """
        Save Config as .json.

        Parameters
        ----------
        out_file : str
            Desired path and name of file to which to save config.
        """
        with open(out_file, "w", encoding="utf-8") as file:
            json.dump(self.__dict__, file, ensure_ascii=False, indent=4)
        return

    @classmethod
    def from_json(cls, in_file: str) -> "Config":
        """
        Load Config from .json.

        Parameters
        ----------
        in_file : str
            Desired path and name of file from which to load config.
        """
        with open(in_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Allow forward/backward compatible config.json by ignoring unknown keys.
        # (Useful during migrations when older DBs may have extra config entries.)
        params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        filtered = {k: v for k, v in data.items() if k in params}
        return cls(**filtered)


DefaultConfig = Config()
