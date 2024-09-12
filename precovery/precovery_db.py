import dataclasses
import itertools
import logging
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import quivr as qv
from adam_core.coordinates import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.coordinates.residuals import Residuals
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.observations import Exposures, PointSourceDetections
from adam_core.observers import Observers
from adam_core.orbits import Orbits as AdamOrbits
from adam_core.orbits.ephemeris import Ephemeris as EphemerisQv
from adam_core.time import Timestamp

from .config import Config, DefaultConfig
from .frame_db import FrameDB, FrameIndex, HealpixFrame
from .healpix_geom import radec_to_healpixel
from .observation import Observation, ObservationArray
from .orbit import Ephemeris, EpochTimescale, Orbit, PropagationMethod
from .spherical_geom import haversine_distance_deg
from .utils_qv import drop_duplicates
from .version import __version__

DEGREE = 1.0
ARCMIN = DEGREE / 60
ARCSEC = ARCMIN / 60

CANDIDATE_K = 15
CANDIDATE_NSIDE = 2**CANDIDATE_K

logging.basicConfig()
logger = logging.getLogger("precovery")


@dataclasses.dataclass
class PrecoveryCandidate:
    mjd: float
    ra_deg: float
    dec_deg: float
    ra_sigma_arcsec: float
    dec_sigma_arcsec: float
    mag: float
    mag_sigma: float
    exposure_mjd_start: float
    exposure_mjd_mid: float
    filter: str
    obscode: str
    exposure_id: str
    exposure_duration: float
    observation_id: str
    healpix_id: int
    pred_ra_deg: float
    pred_dec_deg: float
    pred_vra_degpday: float
    pred_vdec_degpday: float
    delta_ra_arcsec: float
    delta_dec_arcsec: float
    distance_arcsec: float
    dataset_id: str

    @classmethod
    def from_observed_ephem(
        cls, obs: Observation, frame: HealpixFrame, ephem: Ephemeris
    ):
        # Calculate the HEALpixel ID for the predicted ephemeris of
        # the orbit with a high nside value (k=15, nside=2**15) The
        # indexed observations are indexed to a much lower nside but
        # we may decide in the future to re-index the database using
        # different values for that parameter. As long as we return a
        # Healpix ID generated with nside greater than the indexed
        # database then we can always down-sample the ID to a lower
        # nside value
        healpix_id = int(radec_to_healpixel(ephem.ra, ephem.dec, nside=CANDIDATE_NSIDE))

        return cls(
            # Observation data:
            observation_id=obs.id.decode(),
            mjd=obs.mjd,
            ra_deg=obs.ra,
            dec_deg=obs.dec,
            ra_sigma_arcsec=obs.ra_sigma / ARCSEC,
            dec_sigma_arcsec=obs.dec_sigma / ARCSEC,
            mag=obs.mag,
            mag_sigma=obs.mag_sigma,
            # Exposure data:
            exposure_mjd_start=frame.exposure_mjd_start,
            exposure_mjd_mid=frame.exposure_mjd_mid,
            filter=frame.filter,
            obscode=frame.obscode,
            exposure_id=frame.exposure_id,
            exposure_duration=frame.exposure_duration,
            dataset_id=frame.dataset_id,
            # Ephemeris data:
            healpix_id=healpix_id,
            pred_ra_deg=ephem.ra,
            pred_dec_deg=ephem.dec,
            pred_vra_degpday=ephem.ra_velocity,
            pred_vdec_degpday=ephem.dec_velocity,
            # Data on the distance between the observation and ephemeris:
            delta_ra_arcsec=(obs.ra - ephem.ra) / ARCSEC,
            delta_dec_arcsec=(obs.dec - ephem.dec) / ARCSEC,
            distance_arcsec=haversine_distance_deg(obs.ra, ephem.ra, obs.dec, ephem.dec)
            / ARCSEC,
        )

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class FrameCandidate:
    exposure_mjd_start: float
    exposure_mjd_mid: float
    filter: str
    obscode: str
    exposure_id: str
    exposure_duration: float
    healpix_id: int
    pred_ra_deg: float
    pred_dec_deg: float
    pred_vra_degpday: float
    pred_vdec_degpday: float
    dataset_id: str

    @classmethod
    def from_frame(cls, frame: HealpixFrame, ephem: Ephemeris):
        # Calculate the HEALpixel ID for the predicted ephemeris of
        # the orbit with a high nside value (k=15, nside=2**15) The
        # indexed observations are indexed to a much lower nside but
        # we may decide in the future to re-index the database using
        # different values for that parameter. As long as we return a
        # Healpix ID generated with nside greater than the indexed
        # database then we can always down-sample the ID to a lower
        # nside value
        healpix_id = int(radec_to_healpixel(ephem.ra, ephem.dec, nside=CANDIDATE_NSIDE))
        return cls(
            # Exposure data:
            exposure_mjd_start=frame.exposure_mjd_start,
            exposure_mjd_mid=frame.exposure_mjd_mid,
            filter=frame.filter,
            obscode=frame.obscode,
            exposure_id=frame.exposure_id,
            exposure_duration=frame.exposure_duration,
            dataset_id=frame.dataset_id,
            healpix_id=healpix_id,
            # Ephemeris data:
            pred_ra_deg=ephem.ra,
            pred_dec_deg=ephem.dec,
            pred_vra_degpday=ephem.ra_velocity,
            pred_vdec_degpday=ephem.dec_velocity,
        )

    def to_dict(self):
        return dataclasses.asdict(self)


def sift_candidates(
    candidates: List[Union[PrecoveryCandidate, FrameCandidate]]
) -> Tuple[List[PrecoveryCandidate], List[FrameCandidate]]:
    """
    Separates candidates into precovery and frame candidates and sorts them.

    Sort candidates by ascending MJD. For precovery candidates, use the MJD of the observation.
    For frame candidates, use the MJD at the midpoint of the exposure.

    Parameters
    ----------
    candidates : List[Union[PrecoveryCandidate, FrameCandidate]]
        List of candidates to sort.

    Returns
    -------
    Tuple[List[PrecoveryCandidate], List[FrameCandidate]]
    """
    precovery_candidates = []
    frame_candidates = []
    for candidate in candidates:
        if isinstance(candidate, PrecoveryCandidate):
            precovery_candidates.append(candidate)
        elif isinstance(candidate, FrameCandidate):
            frame_candidates.append(candidate)
        else:
            raise TypeError(f"Unexpected candidate type: {type(candidate)}")

    precovery_candidates = sorted(
        precovery_candidates, key=lambda c: (c.mjd, c.observation_id)
    )
    frame_candidates = sorted(frame_candidates, key=lambda c: c.exposure_mjd_mid)

    return precovery_candidates, frame_candidates


class PrecoveryCandidatesQv(qv.Table):
    # copy all the fields from PrecoveryCandidate
    time = Timestamp.as_column()
    ra_deg = qv.Float64Column()
    dec_deg = qv.Float64Column()
    ra_sigma_arcsec = qv.Float64Column()
    dec_sigma_arcsec = qv.Float64Column()
    mag = qv.Float64Column()
    mag_sigma = qv.Float64Column()
    exposure_time_start = Timestamp.as_column()
    exposure_time_mid = Timestamp.as_column()
    filter = qv.LargeStringColumn()
    obscode = qv.LargeStringColumn()
    exposure_id = qv.LargeStringColumn()
    exposure_duration = qv.Float64Column()
    observation_id = qv.LargeStringColumn()
    healpix_id = qv.Int64Column()
    pred_ra_deg = qv.Float64Column()
    pred_dec_deg = qv.Float64Column()
    pred_vra_degpday = qv.Float64Column()
    pred_vdec_degpday = qv.Float64Column()
    delta_ra_arcsec = qv.Float64Column()
    delta_dec_arcsec = qv.Float64Column()
    distance_arcsec = qv.Float64Column()
    dataset_id = qv.LargeStringColumn()
    orbit_id = qv.LargeStringColumn()

    @classmethod
    def from_dataclass(
        cls,
        precovery_candidates: List[PrecoveryCandidate],
        source_orbit_id: str,
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
            time=Timestamp.from_mjd(field_dict["mjd"], scale="utc"),
            ra_deg=field_dict["ra_deg"],
            dec_deg=field_dict["dec_deg"],
            ra_sigma_arcsec=field_dict["ra_sigma_arcsec"],
            dec_sigma_arcsec=field_dict["dec_sigma_arcsec"],
            mag=field_dict["mag"],
            mag_sigma=field_dict["mag_sigma"],
            exposure_time_start=Timestamp.from_mjd(
                field_dict["exposure_mjd_start"], scale="utc"
            ),
            exposure_time_mid=Timestamp.from_mjd(
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
            orbit_id=[source_orbit_id for i in range(len(field_dict["mjd"]))],
        )

    def to_dataclass(self) -> List[PrecoveryCandidate]:
        return [
            PrecoveryCandidate(
                mjd=cand.time.mjd()[0].as_py(),
                ra_deg=cand.ra_deg[0].as_py(),
                dec_deg=cand.dec_deg[0].as_py(),
                ra_sigma_arcsec=cand.ra_sigma_arcsec[0].as_py(),
                dec_sigma_arcsec=cand.dec_sigma_arcsec[0].as_py(),
                mag=cand.mag[0].as_py(),
                mag_sigma=cand.mag_sigma[0].as_py(),
                filter=cand.filter[0].as_py(),
                obscode=cand.obscode[0].as_py(),
                exposure_id=cand.exposure_id[0].as_py(),
                exposure_mjd_start=cand.exposure_time_start.mjd()[0].as_py(),
                exposure_mjd_mid=cand.exposure_time_mid.mjd()[0].as_py(),
                exposure_duration=cand.exposure_duration[0].as_py(),
                observation_id=cand.observation_id[0].as_py(),
                healpix_id=cand.healpix_id[0].as_py(),
                pred_ra_deg=cand.pred_ra_deg[0].as_py(),
                pred_dec_deg=cand.pred_dec_deg[0].as_py(),
                pred_vra_degpday=cand.pred_vra_degpday[0].as_py(),
                pred_vdec_degpday=cand.pred_vdec_degpday[0].as_py(),
                delta_ra_arcsec=cand.delta_ra_arcsec[0].as_py(),
                delta_dec_arcsec=cand.delta_dec_arcsec[0].as_py(),
                distance_arcsec=cand.distance_arcsec[0].as_py(),
                dataset_id=cand.dataset_id[0].as_py(),
            )
            for cand in self
        ]

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
            time=self.time,
        )

    def exposures(self) -> Exposures:
        unique = drop_duplicates(self, subset=["exposure_id"])

        return Exposures.from_kwargs(
            id=unique.exposure_id,
            start_time=unique.exposure_time_start,
            observatory_code=unique.obscode,
            filter=unique.filter.to_pylist(),
            duration=unique.exposure_duration,
        )

    def predicted_ephemeris(self, orbit_ids=None) -> EphemerisQv:
        """
        Return the predicted ephemeris for these candidates.

        Parameters
        ----------
        orbit_ids : Optional[List[str]], optional
            Orbit IDs to use for the predicted ephemeris. If None, a unique
            orbit ID will be generated for each candidate.

        Returns
        -------
        EphemerisQv
            Predicted ephemeris for these candidates.
        """
        if orbit_ids is None:
            orbit_ids = [str(i) for i in range(len(self.time))]
        return EphemerisQv.from_kwargs(
            orbit_id=orbit_ids,
            coordinates=SphericalCoordinates.from_kwargs(
                lon=self.pred_ra_deg,
                lat=self.pred_dec_deg,
                vlon=self.pred_vra_degpday,
                vlat=self.pred_vdec_degpday,
                time=self.time,
                origin=Origin.from_kwargs(
                    code=self.obscode,
                ),
                frame="equatorial",
            ),
        )

    def to_residuals(self) -> Residuals:
        """
        Compute the residuals between the observations and the predicted ephemeris.

        Returns
        -------
        Residuals
            Residuals between the observations and the predicted ephemeris.
        """
        return Residuals.calculate(
            self.to_spherical_coordinates(), self.predicted_ephemeris().coordinates
        )

    def to_spherical_coordinates(self) -> SphericalCoordinates:
        """
        Convert the observations to a SphericalCoordinates object.

        Returns
        -------
        SphericalCoordinates
            Observations represented as a SphericalCoordinates object.
        """
        # Create a 2D array of sigmas for the observations
        # Convert arcseconds to degrees
        sigmas = np.full((len(self.time), 6), np.nan)
        sigmas[:, 1] = self.ra_sigma_arcsec.to_numpy(zero_copy_only=False) / 3600
        sigmas[:, 2] = self.dec_sigma_arcsec.to_numpy(zero_copy_only=False) / 3600

        # Create a Coordinates object for the observations - we need
        # these to calculate residuals
        return SphericalCoordinates.from_kwargs(
            lon=self.ra_deg,
            lat=self.dec_deg,
            time=self.time,
            covariance=CoordinateCovariances.from_sigmas(sigmas),
            origin=Origin.from_kwargs(
                code=self.obscode,
            ),
            frame="equatorial",
        )

    def get_observers(self) -> Observers:
        """
        Get the sorted observers for these candidates. Observers
        can be used with an `~adam_core.propagator.Propagator` to
        generate predicted ephemerides at the same time as the
        observations.

        Returns
        -------
        Observers (N)
            Observers for these candidates sorted by time and
            observatory code.
        """
        observers = Observers.empty()
        for obscode in self.obscode.unique():
            self_obs = self.select("obscode", obscode)
            times = self_obs.time
            observers = qv.concatenate(
                [observers, Observers.from_code(obscode.as_py(), times)]
            )

        return observers.sort_by(
            [
                "coordinates.time.days",
                "coordinates.time.nanos",
                "code",
            ]
        )


class FrameCandidatesQv(qv.Table):
    # copy all the fields from FrameCandidate
    exposure_time_start = Timestamp.as_column()
    exposure_time_mid = Timestamp.as_column()
    filter = qv.LargeStringColumn()
    obscode = qv.LargeStringColumn()
    exposure_id = qv.LargeStringColumn()
    exposure_duration = qv.Float64Column()
    healpix_id = qv.Int64Column()
    pred_ra_deg = qv.Float64Column()
    pred_dec_deg = qv.Float64Column()
    pred_vra_degpday = qv.Float64Column()
    pred_vdec_degpday = qv.Float64Column()
    dataset_id = qv.LargeStringColumn()
    orbit_id = qv.LargeStringColumn()

    @classmethod
    def from_dataclass(
        cls,
        precovery_candidates: List[FrameCandidate],
        source_orbit_id: str,
    ) -> "FrameCandidatesQv":
        field_dict: Dict[str, Any] = {
            field.name: [] for field in dataclasses.fields(PrecoveryCandidate)
        }

        # Iterate over each candidate and convert to dictionary
        for candidate in precovery_candidates:
            candidate_dict = dataclasses.asdict(candidate)
            for key, value in candidate_dict.items():
                field_dict[key].append(value)

        return cls.from_kwargs(
            exposure_time_start=Timestamp.from_mjd(
                field_dict["exposure_mjd_start"], scale="utc"
            ),
            exposure_time_mid=Timestamp.from_mjd(
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
            orbit_id=[
                source_orbit_id for i in range(len(field_dict["exposure_mjd_start"]))
            ],
        )

    def to_dataclass(self) -> List[FrameCandidate]:
        return [
            FrameCandidate(
                filter=cand.filter[0].as_py(),
                obscode=cand.obscode[0].as_py(),
                exposure_id=cand.exposure_id[0].as_py(),
                exposure_mjd_start=cand.exposure_time_start.mjd()[0].as_py(),
                exposure_mjd_mid=cand.exposure_time_mid.mjd()[0].as_py(),
                exposure_duration=cand.exposure_duration[0].as_py(),
                healpix_id=cand.healpix_id[0].as_py(),
                pred_ra_deg=cand.pred_ra_deg[0].as_py(),
                pred_dec_deg=cand.pred_dec_deg[0].as_py(),
                pred_vra_degpday=cand.pred_vra_degpday[0].as_py(),
                pred_vdec_degpday=cand.pred_vdec_degpday[0].as_py(),
                dataset_id=cand.dataset_id[0].as_py(),
            )
            for cand in self
        ]

    def exposures(self) -> Exposures:
        unique = drop_duplicates(self, subset=["exposure_id"])

        return Exposures.from_kwargs(
            id=unique.exposure_id,
            start_time=unique.exposure_time_start,
            observatory_code=unique.obscode,
            filter=unique.filter.to_pylist(),
            duration=unique.exposure_duration,
        )

    def predicted_ephemeris(self, orbit_ids=None) -> EphemerisQv:
        origin = Origin.from_kwargs(
            code=["SUN" for i in range(len(self.exposure_time_mid))]
        )
        frame = "ecliptic"
        if orbit_ids is None:
            orbit_ids = [str(i) for i in range(len(self.exposure_time_mid))]
        return EphemerisQv.from_kwargs(
            orbit_id=orbit_ids,
            coordinates=SphericalCoordinates.from_kwargs(
                lon=self.pred_ra_deg,
                lat=self.pred_dec_deg,
                vlon=self.pred_vra_degpday,
                vlat=self.pred_vdec_degpday,
                time=self.exposure_time_mid,
                origin=origin,
                frame=frame,
            ),
        )


class PrecoveryDatabase:
    def __init__(self, frames: FrameDB, config: Config = DefaultConfig):
        self.frames = frames
        self._exposures_by_obscode: dict = {}
        self.config = config

    @classmethod
    def from_dir(
        cls,
        directory: str,
        create: bool = False,
        mode: str = "r",
        allow_version_mismatch: bool = False,
    ):
        if not os.path.exists(directory):
            if create:
                return cls.create(directory)

        try:
            config = Config.from_json(os.path.join(directory, "config.json"))
        except FileNotFoundError:
            if not create:
                raise Exception("No config file found and create=False")
            config = DefaultConfig
            config.to_json(os.path.join(directory, "config.json"))

        if config.build_version != __version__:
            if not allow_version_mismatch:
                raise Exception(
                    f"Version mismatch: \nRunning version: {__version__}\nDatabase"
                    f" version: {config.build_version}\nUse allow_version_mismatch=True"
                    " to ignore this error."
                )
            else:
                logger.warning(
                    f"Version mismatch: \nRunning version: {__version__}\nDatabase"
                    f" version: {config.build_version}\nallow_version_mismatch=True, so"
                    " continuing."
                )

        frame_idx_db = "sqlite:///" + os.path.join(directory, "index.db")
        frame_idx = FrameIndex(frame_idx_db, mode=mode)

        data_path = os.path.join(directory, "data")
        frame_db = FrameDB(
            frame_idx, data_path, config.data_file_max_size, config.nside, mode=mode
        )
        return cls(frame_db, config)

    @classmethod
    def create(
        cls,
        directory: str,
        nside: int = DefaultConfig.nside,
        data_file_max_size: int = DefaultConfig.data_file_max_size,
    ):
        """
        Create a new database on disk in the given directory.
        """
        os.makedirs(directory, exist_ok=True)

        frame_idx_db = "sqlite:///" + os.path.join(directory, "index.db")
        frame_idx = FrameIndex(frame_idx_db, mode="w")

        config = Config(nside=nside, data_file_max_size=data_file_max_size)
        config.to_json(os.path.join(directory, "config.json"))

        data_path = os.path.join(directory, "data")
        os.makedirs(data_path, exist_ok=True)

        frame_db = FrameDB(frame_idx, data_path, data_file_max_size, nside)

        return cls(frame_db, config)

    def precover(
        self,
        orbit: AdamOrbits,
        tolerance: float = 30 * ARCSEC,
        start_mjd: Optional[float] = None,
        end_mjd: Optional[float] = None,
        window_size: int = 7,
        datasets: Optional[set[str]] = None,
    ) -> Tuple[PrecoveryCandidatesQv, FrameCandidatesQv]:
        """
        Find observations which match orbit in the database. Observations are
        searched in descending order by mjd.

        orbit: The orbit to match.

        start_mjd: Only consider observations from after this epoch
        (inclusive). If None, find all.

        end_mjd: Only consider observations from before this epoch (inclusive).
        If None, find all.

        datasets: Only consider observations from the indicated datasets.

        Returns
        -------
        Tuple[List[PrecoveryCandidate], List[FrameCandidate]]
            Precovery candidate observations and frame candidates.
        """
        # basically:
        """
        find all windows between start and end of given size
        for each window:
            propagate to window center
            for each unique epoch,obscode in window:
                propagate to epoch
                find frames which match healpix of propagation
                for each matching frame
                    find matching observations
                    for each matching observation
                        yield match
        """
        orbit_id = orbit.orbit_id[0].as_py()
        orbit = Orbit.from_adam_core(orbit_id=1, ac_orbits=orbit)

        if datasets is not None:
            self._warn_for_missing_datasets(datasets)

        if start_mjd is None or end_mjd is None:
            first, last = self.frames.idx.mjd_bounds(datasets=datasets)
            if start_mjd is None:
                start_mjd = first
            if end_mjd is None:
                end_mjd = last

        logger.info(
            "precovering orbit %s from %.5f to %.5f, window=%d, datasets=%s",
            orbit_id,
            start_mjd,
            end_mjd,
            window_size,
            datasets or "all",
        )

        windows = self.frames.idx.window_centers(
            start_mjd, end_mjd, window_size, datasets=datasets
        )

        # group windows by obscodes so that many windows can be searched at once
        matches = []
        for obscode, obs_windows in itertools.groupby(
            windows, key=lambda pair: pair[1]
        ):
            mjds = [window[0] for window in obs_windows]
            matches_window = self._check_windows(
                mjds,
                obscode,
                orbit,
                tolerance,
                start_mjd=start_mjd,
                end_mjd=end_mjd,
                window_size=window_size,
                datasets=datasets,
            )
            matches += list(matches_window)

        precovery_candidates, frame_candidates = sift_candidates(matches)

        # convert these to our new output formats
        return PrecoveryCandidatesQv.from_dataclass(
            precovery_candidates, source_orbit_id=orbit_id
        ), FrameCandidatesQv.from_dataclass(frame_candidates, source_orbit_id=orbit_id)

    def _check_windows(
        self,
        window_midpoints: Iterable[float],
        obscode: str,
        orbit: Orbit,
        tolerance: float,
        start_mjd: Optional[float] = None,
        end_mjd: Optional[float] = None,
        window_size: int = 7,
        datasets: Optional[set[str]] = None,
    ) -> Iterable[Union[PrecoveryCandidate, FrameCandidate]]:
        """
        Find all observations that match orbit within a list of windows
        """
        # Propagate the orbit with n-body to every window center
        # Since the window midpoints are calculated from the observations
        # in the database then they are in the UTC timescale so let's use that
        orbit_propagated = orbit.propagate(
            window_midpoints,
            PropagationMethod.N_BODY,
            time_scale=EpochTimescale.UTC,
        )

        # Calculate the location of the orbit on the sky with n-body propagation
        # Again, we do this in the UTC timescale to match the observations in the database
        window_ephems = orbit.compute_ephemeris(
            obscode,
            window_midpoints,
            PropagationMethod.N_BODY,
            time_scale=EpochTimescale.UTC,
        )
        window_healpixels = radec_to_healpixel(
            np.array([w.ra for w in window_ephems]),
            np.array([w.dec for w in window_ephems]),
            self.frames.healpix_nside,
        ).astype(int)

        # Using the propagated orbits, check each window. Propagate the orbit from the center of
        # window using 2-body to find any HealpixFrames where a detection could have occured
        for window_midpoint, window_ephem, window_healpixel, orbit_window in zip(
            window_midpoints, window_ephems, window_healpixels, orbit_propagated
        ):
            start_mjd_window = window_midpoint - (window_size / 2)
            end_mjd_window = window_midpoint + (window_size / 2)

            # Check if start_mjd_window is not earlier than start_mjd (if defined)
            # If start_mjd_window is earlier, then set start_mjd_window to start_mjd
            if (start_mjd is not None) and (start_mjd_window < start_mjd):
                logger.debug(
                    f"Window start MJD [UTC] ({start_mjd_window}) is earlier than"
                    f" desired start MJD [UTC] ({start_mjd})."
                )
                start_mjd_window = start_mjd

            # Check if end_mjd_window is not later than end_mjd (if defined)
            # If end_mjd_window is later, then set end_mjd_window to end_mjd
            if (end_mjd is not None) and (end_mjd_window > end_mjd):
                logger.debug(
                    f"Window end MJD [UTC] ({end_mjd_window}) is later than desired end"
                    f" MJD [UTC] ({end_mjd})."
                )
                end_mjd_window = end_mjd

            yield from self._check_window(
                window_start=start_mjd_window,
                window_end=end_mjd_window,
                orbit=orbit_window,
                obscode=obscode,
                tolerance=tolerance,
                datasets=datasets,
            )

    def _check_window(
        self,
        window_start: float,
        window_end: float,
        orbit: Orbit,
        obscode: str,
        tolerance: float,
        datasets: Optional[set[str]],
    ):
        # Gather all MJDs and their associated healpixels in the window.
        timestamps: list[float] = []
        observation_healpixels: list[set[int]] = []
        for timestamp, healpixels in self.frames.idx.propagation_targets(
            window_start,
            window_end,
            obscode,
            datasets,
        ):
            logger.debug("mjd=%.6f:\thealpixels with data: %r", timestamp, healpixels)
            timestamps.append(timestamp)
            observation_healpixels.append(healpixels)

        # Propagate the orbit with 2-body dynamics to every timestamp in the
        # window.
        ephems = orbit.compute_ephemeris(
            obscode,
            timestamps,
            PropagationMethod.TWO_BODY,
            time_scale=EpochTimescale.UTC,
        )

        # Convert those ephemerides to healpixels.
        ephem_healpixels = radec_to_healpixel(
            ra=np.array([w.ra for w in ephems]),
            dec=np.array([w.dec for w in ephems]),
            nside=self.frames.healpix_nside,
        ).astype(int)

        # Check which ephemerides land within data
        for timestamp, observation_healpixel_set, ephem_healpixel in zip(
            timestamps, observation_healpixels, ephem_healpixels
        ):
            if ephem_healpixel in observation_healpixel_set:
                # We have a match! Check the frame.
                yield from self._check_frame(
                    orbit=orbit,
                    healpixel=ephem_healpixel,
                    obscode=obscode,
                    mjd=timestamp,
                    tolerance=tolerance,
                    datasets=datasets,
                )

    def _check_frame(
        self,
        orbit: Orbit,
        healpixel: int,
        obscode: str,
        mjd: float,
        tolerance: float,
        datasets: Optional[set[str]],
    ) -> Iterator[Union[PrecoveryCandidate, FrameCandidate]]:
        """
        Deeply inspect all frames that match the given obscode, mjd, and healpix to
        see if they contain observations which match the ephemeris.
        """
        # Compute the position of the ephem carefully.
        ephem = orbit.compute_ephemeris(
            obscode,
            [mjd],
            PropagationMethod.N_BODY,
            time_scale=EpochTimescale.UTC,
        )[0]

        frames = self.frames.idx.get_frames(obscode, mjd, healpixel, datasets)
        logger.debug(
            "checking frames for healpix=%d obscode=%s mjd=%f",
            healpixel,
            obscode,
            mjd,
        )
        for f in frames:
            any_matches = False
            for match in self._find_matches_in_frame(f, orbit, ephem, tolerance):
                any_matches = True
                yield match
            # If no observations were found in this frame then we
            # yield a frame candidate if desired
            # Note that for the frame candidate we report the predicted
            # ephemeris at the exposure midpoint not at the observation
            # times which may differ from the exposure midpoint time
            if not any_matches:
                logger.debug("no observations found in this frame")
                frame_candidate = FrameCandidate.from_frame(f, ephem)
                yield frame_candidate

    def _find_matches_in_frame(
        self, frame: HealpixFrame, orbit: Orbit, ephem: Ephemeris, tolerance: float
    ) -> Iterator[PrecoveryCandidate]:
        """
        Find all sources in a single frame which match ephem.
        """
        logger.debug("checking frame: %s", frame)

        # Gather all observations.
        observations = ObservationArray(list(self.frames.iterate_observations(frame)))
        n_obs = len(observations)

        if not np.all(observations.values["mjd"] == ephem.mjd):
            # Data has per-observation MJDs. We'll need propagate to every observation's time.
            for match in self._find_matches_using_per_obs_timestamps(
                frame, orbit, observations, tolerance
            ):
                yield (match)
            return

        # Compute distance between our single ephemeris and all observations
        distances = ephem.distance(observations)

        # Gather ones within the tolerance
        observations.values = observations.values[distances < tolerance]
        matching_observations = observations.to_list()

        logger.debug(
            f"checked {n_obs} observations in frame {frame} and found {len(matching_observations)}"
        )

        for obs in matching_observations:
            yield PrecoveryCandidate.from_observed_ephem(obs, frame, ephem)

    def _find_matches_using_per_obs_timestamps(
        self,
        frame: HealpixFrame,
        orbit: Orbit,
        observations: ObservationArray,
        tolerance: float,
    ) -> Iterator[PrecoveryCandidate]:
        """If a frame has per-source observation times, we need to
        propagate to every time in the frame, so

        """
        # First, propagate to the mean MJD using N-body propagation.
        mean_mjd = observations.values["mjd"].mean()
        mean_orbit_state = orbit.propagate(
            epochs=[mean_mjd],
            method=PropagationMethod.N_BODY,
            time_scale=EpochTimescale.UTC,
        )[0]

        # Next, compute an ephemeris for every observation in the
        # frame. Use 2-body propagation from the mean position to the
        # individual timestamps.
        ephemerides = mean_orbit_state.compute_ephemeris(
            obscode=frame.obscode,
            epochs=observations.values["mjd"],
            method=PropagationMethod.TWO_BODY,
            time_scale=EpochTimescale.UTC,
        )

        # Now check distances.
        ephem_position_ra = np.array([e.ra for e in ephemerides])
        ephem_position_dec = np.array([e.dec for e in ephemerides])

        distances = haversine_distance_deg(
            ephem_position_ra,
            observations.values["ra"],
            ephem_position_dec,
            observations.values["dec"],
        )

        # Gather any matches.
        for index in np.where(distances < tolerance)[0]:
            obs = Observation(*observations.values[index])
            ephem = ephemerides[index]
            yield PrecoveryCandidate.from_observed_ephem(obs, frame, ephem)

    def find_observations_in_region(
        self, ra: float, dec: float, obscode: str
    ) -> Iterator[Observation]:
        """Gets all the Observations within the same Healpixel as a
        given RA, Dec for a particular observatory (specified as an obscode).

        """
        frames = self.frames.get_frames_for_ra_dec(ra, dec, obscode)
        for f in frames:
            yield from self.frames.iterate_observations(f)

    def find_observations_in_radius(
        self, ra: float, dec: float, tolerance: float, obscode: str
    ) -> Iterator[Observation]:
        """Gets all the Observations within a radius (in degrees) of
        a particular RA and Dec at a specific observatory.

        This method is approximate, and does not correctly find
        Observations that are within the radius, but on a different
        healpixel.

        """
        for o in self.find_observations_in_region(ra, dec, obscode):
            if haversine_distance_deg(o.ra, ra, o.dec, dec) <= tolerance:
                yield o

    def all_datasets(self) -> set[str]:
        """
        Returns the set of all dataset ID strings loaded in the database.
        """
        return set(self.frames.idx.get_dataset_ids())

    def _warn_for_missing_datasets(self, datasets: set[str]):
        """Log some warning messages if the given set includes
        dataset IDs which are not present in the database.

        """
        any_missing = False
        for ds in datasets:
            if not self.frames.has_dataset(ds):
                any_missing = True
                logger.warn(f'dataset "{ds}" is not in the database')
        if any_missing:
            logger.warn(f"datasets in the databse: {self.all_datasets()}")
