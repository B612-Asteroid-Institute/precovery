import dataclasses
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pyarrow as pa
import quivr as qv
from adam_core.coordinates import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.coordinates.residuals import Residuals
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.observations import Exposures, PointSourceDetections
from adam_core.orbits.ephemeris import Ephemeris as EphemerisQv
from adam_core.time import Timestamp

from .precovery_db import FrameCandidate, PrecoveryCandidate


def drop_duplicates(
    table: qv.AnyTable,
    subset: Optional[List[str]] = None,
    keep: Literal["first", "last"] = "first",
) -> qv.AnyTable:
    """
    Drop duplicate rows from a `~quivr.Table`. This function is similar to
    `~pandas.DataFrame.drop_duplicates` but it supports nested columns (representing
    nested tables).

    Parameters
    ----------
    table : `~quivr.Table`
        Table to drop duplicate rows from.
    subset : list of str, optional
        Subset of columns to consider when dropping duplicates. If not specified then
        all columns are used.
    keep : {'first', 'last'}, default 'first'
        If there are duplicate rows then keep the first or last row.

    Returns
    -------
    table : `~quivr.Table`
        Table with duplicate rows removed.
    """
    # Flatten the table so nested columns are dot-delimited at the top level
    flattened_table = table.flattened_table()

    # If subset is not specified then use all the columns
    if subset is None:
        subset = [c for c in flattened_table.column_names]

    # Add an index column to the flattened table
    flattened_table = flattened_table.add_column(
        0, "index", pa.array(np.arange(len(flattened_table)))
    )

    if keep not in ["first", "last"]:
        raise ValueError(f"keep must be 'first' or 'last', got {keep}.")

    agg_func = keep
    indices = (
        flattened_table.group_by(subset, use_threads=False)
        .aggregate([("index", agg_func)])
        .column(f"index_{agg_func}")
    )

    # Take the indices from the flattened table and use them to index into the original table
    return table.take(indices)


class PrecoveryCandidatesQv(qv.Table):
    # copy all the fields from PrecoveryCandidate
    mjd = Timestamp.as_column()
    ra_deg = qv.Float64Column()
    dec_deg = qv.Float64Column()
    ra_sigma_arcsec = qv.Float64Column()
    dec_sigma_arcsec = qv.Float64Column()
    mag = qv.Float64Column()
    mag_sigma = qv.Float64Column()
    exposure_mjd_start = Timestamp.as_column()
    exposure_mjd_mid = Timestamp.as_column()
    filter = qv.StringColumn()
    obscode = qv.StringColumn()
    exposure_id = qv.StringColumn()
    exposure_duration = qv.Float64Column()
    observation_id = qv.StringColumn()
    healpix_id = qv.Int64Column()
    pred_ra_deg = qv.Float64Column()
    pred_dec_deg = qv.Float64Column()
    pred_vra_degpday = qv.Float64Column()
    pred_vdec_degpday = qv.Float64Column()
    delta_ra_arcsec = qv.Float64Column()
    delta_dec_arcsec = qv.Float64Column()
    distance_arcsec = qv.Float64Column()
    dataset_id = qv.StringColumn()

    @classmethod
    def from_precovery_candidates(
        cls, precovery_candidates: List[PrecoveryCandidate]
    ) -> "PrecoveryCandidatesQv":
        field_dict: Dict[str, Any] = {
            field.name: [] for field in dataclasses.fields(PrecoveryCandidate)
        }

        # Iterate over each candidate and convert to dictionary
        for candidate in precovery_candidates:
            candidate_dict = dataclasses.asdict(candidate)
            for key, value in candidate_dict.items():
                field_dict[key].append(value)

        return cls.from_kwargs(
            mjd=Timestamp.from_mjd(field_dict["mjd"], scale="utc"),
            ra_deg=field_dict["ra_deg"],
            dec_deg=field_dict["dec_deg"],
            ra_sigma_arcsec=field_dict["ra_sigma_arcsec"],
            dec_sigma_arcsec=field_dict["dec_sigma_arcsec"],
            mag=field_dict["mag"],
            mag_sigma=field_dict["mag_sigma"],
            exposure_mjd_start=Timestamp.from_mjd(
                field_dict["exposure_mjd_start"], scale="utc"
            ),
            exposure_mjd_mid=Timestamp.from_mjd(
                field_dict["exposure_mjd_mid"], scale="utc"
            ),
            filter=field_dict["filter"],
            obscode=field_dict["obscode"],
            exposure_id=field_dict["exposure_id"],
            exposure_duration=field_dict["exposure_duration"],
            observation_id=field_dict["observation_id"],
            healpix_id=field_dict["healpix_id"],
            pred_ra_deg=field_dict["pred_ra_deg"],
            pred_dec_deg=field_dict["pred_dec_deg"],
            pred_vra_degpday=field_dict["pred_vra_degpday"],
            pred_vdec_degpday=field_dict["pred_vdec_degpday"],
            delta_ra_arcsec=field_dict["delta_ra_arcsec"],
            delta_dec_arcsec=field_dict["delta_dec_arcsec"],
            distance_arcsec=field_dict["distance_arcsec"],
            dataset_id=field_dict["dataset_id"],
        )

    def point_source_detections(self) -> PointSourceDetections:
        return PointSourceDetections.from_kwargs(
            id=self.observation_id,
            exposure_id=self.exposure_id,
            ra=self.ra_deg,
            dec=self.dec_deg,
            ra_sigma=self.ra_sigma_arcsec,
            dec_sigma=self.dec_sigma_arcsec,
            mag=self.mag,
            mag_sigma=self.mag_sigma,
            time=self.mjd,
        )

    def exposures(self) -> Exposures:
        unique = drop_duplicates(self, subset=["exposure_id"])

        return Exposures.from_kwargs(
            id=unique.exposure_id,
            start_time=unique.exposure_mjd_start,
            observatory_code=unique.obscode,
            filter=unique.filter.to_pylist(),
            duration=unique.exposure_duration,
        )

    def predicted_ephemeris(self, orbit_ids=None) -> EphemerisQv:
        origin = Origin.from_kwargs(code=["SUN" for i in range(len(self.mjd))])
        frame = "ecliptic"
        if orbit_ids is None:
            orbit_ids = [str(i) for i in range(len(self.mjd))]
        return EphemerisQv.from_kwargs(
            orbit_id=orbit_ids,
            coordinates=SphericalCoordinates.from_kwargs(
                lon=self.pred_ra_deg,
                lat=self.pred_dec_deg,
                vlon=self.pred_vra_degpday,
                vlat=self.pred_vdec_degpday,
                time=self.mjd,
                origin=origin,
                frame=frame,
            ),
        )

    def residuals(self) -> Residuals:
        origin = Origin.from_kwargs(code=["SUN" for i in range(len(self.mjd))])
        frame = "ecliptic"

        # Create a Coordinates object for the observations - we need
        # these to calculate residuals
        obs_coords_spherical = SphericalCoordinates.from_kwargs(
            lon=self.ra_deg,
            lat=self.dec_deg,
            covariance=CoordinateCovariances.from_sigmas(
                np.stack(
                    [
                        np.array(
                            [
                                np.nan,
                                sig_lon.as_py(),
                                sig_lat.as_py(),
                                np.nan,
                                np.nan,
                                np.nan,
                            ],
                            np.float64,
                        )
                        for sig_lon, sig_lat in zip(
                            self.ra_sigma_arcsec,
                            self.dec_sigma_arcsec,
                        )
                    ]
                )
            ),
            time=self.mjd,
            origin=origin,
            frame=frame,
        )

        return Residuals.calculate(
            obs_coords_spherical, self.predicted_ephemeris().coordinates
        )


class FrameCandidatesQv(qv.Table):
    # copy all the fields from FrameCandidate
    exposure_mjd_start = Timestamp.as_column()
    exposure_mjd_mid = Timestamp.as_column()
    filter = qv.StringColumn()
    obscode = qv.StringColumn()
    exposure_id = qv.StringColumn()
    exposure_duration = qv.Float64Column()
    healpix_id = qv.Int64Column()
    pred_ra_deg = qv.Float64Column()
    pred_dec_deg = qv.Float64Column()
    pred_vra_degpday = qv.Float64Column()
    pred_vdec_degpday = qv.Float64Column()
    dataset_id = qv.StringColumn()

    @classmethod
    def from_frame_candidates(
        cls, precovery_candidates: List[FrameCandidate]
    ) -> "PrecoveryCandidatesQv":
        field_dict: Dict[str, Any] = {
            field.name: [] for field in dataclasses.fields(PrecoveryCandidate)
        }

        # Iterate over each candidate and convert to dictionary
        for candidate in precovery_candidates:
            candidate_dict = dataclasses.asdict(candidate)
            for key, value in candidate_dict.items():
                field_dict[key].append(value)

        return cls.from_kwargs(
            exposure_mjd_start=Timestamp.from_mjd(
                field_dict["exposure_mjd_start"], scale="utc"
            ),
            exposure_mjd_mid=Timestamp.from_mjd(
                field_dict["exposure_mjd_mid"], scale="utc"
            ),
            filter=field_dict["filter"],
            obscode=field_dict["obscode"],
            exposure_id=field_dict["exposure_id"],
            exposure_duration=field_dict["exposure_duration"],
            healpix_id=field_dict["healpix_id"],
            pred_ra_deg=field_dict["pred_ra_deg"],
            pred_dec_deg=field_dict["pred_dec_deg"],
            pred_vra_degpday=field_dict["pred_vra_degpday"],
            pred_vdec_degpday=field_dict["pred_vdec_degpday"],
            dataset_id=field_dict["dataset_id"],
        )

    def exposures(self) -> Exposures:
        unique = drop_duplicates(self, subset=["exposure_id"])

        return Exposures.from_kwargs(
            id=unique.exposure_id,
            start_time=unique.exposure_mjd_start,
            observatory_code=unique.obscode,
            filter=unique.filter.to_pylist(),
            duration=unique.exposure_duration,
        )

    def predicted_ephemeris(self, orbit_ids=None) -> EphemerisQv:
        origin = Origin.from_kwargs(
            code=["SUN" for i in range(len(self.exposure_mjd_mid))]
        )
        frame = "ecliptic"
        if orbit_ids is None:
            orbit_ids = [str(i) for i in range(len(self.exposure_mjd_mid))]
        return EphemerisQv.from_kwargs(
            orbit_id=orbit_ids,
            coordinates=SphericalCoordinates.from_kwargs(
                lon=self.pred_ra_deg,
                lat=self.pred_dec_deg,
                vlon=self.pred_vra_degpday,
                vlat=self.pred_vdec_degpday,
                time=self.exposure_mjd_mid,
                origin=origin,
                frame=frame,
            ),
        )
