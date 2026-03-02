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
        # Backend selection for precovery/search.
        backend: str = "duckdb_parquet",
        # DuckDB parquet dataset path (required when backend == "duckdb_parquet").
        detections_parquet: str | None = None,
        faint_frame_skip_margin_mag: float = 0.30,
        # Asymmetric magnitude residual thresholds (in magnitudes).
        # mag_residual = observed_mag - predicted_mag
        # - If mag_residual > max_mag_residual_fainter_mag => too faint => reject
        # - If mag_residual < -max_mag_residual_brighter_mag => too bright => reject
        max_mag_residual_fainter_mag: float | None = None,
        max_mag_residual_brighter_mag: float | None = None,
        # Optional Stage-3 uncertainty budget (arcsec, major axis 1-sigma in tangent plane).
        # When set, orbit-target predictions above this threshold are skipped before Stage 4.
        max_on_sky_sigma_major_arcsec: float | None = None,
        # Optional Stage-2 pre-propagation viability policy.
        preprop_viability_policy: str = "off",
        preprop_short_arc_days_threshold: float = 14.0,
        preprop_time_limit_days_short_arc: float = 30.0,
        preprop_time_limit_days_default: float = 90.0,
        preprop_max_sigma_r_over_r: float | None = None,
        preprop_max_covariance_condition: float | None = None,
        # Legacy knob retained for backwards compatibility; no longer used for policy decisions.
        preprop_viability_min_score: float = 0.25,
        preprop_fail_open_on_scoring_error: bool = False,
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
        self.backend = backend
        self.detections_parquet = detections_parquet
        # If predicted_mag > (limit + margin), treat it as too faint to be detectable.
        self.faint_frame_skip_margin_mag = faint_frame_skip_margin_mag
        self.max_mag_residual_fainter_mag = max_mag_residual_fainter_mag
        self.max_mag_residual_brighter_mag = max_mag_residual_brighter_mag
        self.max_on_sky_sigma_major_arcsec = max_on_sky_sigma_major_arcsec
        self.preprop_viability_policy = preprop_viability_policy
        self.preprop_short_arc_days_threshold = preprop_short_arc_days_threshold
        self.preprop_time_limit_days_short_arc = preprop_time_limit_days_short_arc
        self.preprop_time_limit_days_default = preprop_time_limit_days_default
        self.preprop_max_sigma_r_over_r = preprop_max_sigma_r_over_r
        self.preprop_max_covariance_condition = preprop_max_covariance_condition
        self.preprop_viability_min_score = preprop_viability_min_score
        self.preprop_fail_open_on_scoring_error = preprop_fail_open_on_scoring_error

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
        # Be permissive about extra keys so newer DBs can be opened by older code.
        # Unknown keys are preserved as attributes on the returned Config instance.
        sig = inspect.signature(cls.__init__)
        allowed = {k for k in sig.parameters.keys() if k != "self"}
        filtered = {k: v for k, v in data.items() if k in allowed}
        cfg = cls(**filtered)
        for k, v in data.items():
            if k not in allowed:
                setattr(cfg, k, v)
        return cfg


DefaultConfig = Config()
