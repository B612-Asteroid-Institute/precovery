import logging
import os
from typing import Optional, Tuple, Type

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.coordinates import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.coordinates.residuals import Residuals
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.dynamics.ephemeris import generate_ephemeris_2body
from adam_core.dynamics.propagation import propagate_2body
from adam_core.observations import Exposures, PointSourceDetections
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.propagator import Propagator
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
from .config import Config, DefaultConfig
from .frame_db import FrameDB, FrameIndex, GenericFrame, HealpixFrame, WindowCenters
from .healpix_geom import radec_to_healpixel
from .observation import ObservationsTable
from .spherical_geom import haversine_distance_deg

DEGREE = 1.0
ARCMIN = DEGREE / 60
ARCSEC = ARCMIN / 60

CANDIDATE_K = 15
CANDIDATE_NSIDE = 2**CANDIDATE_K

logging.basicConfig()
logger = logging.getLogger("precovery")


class PrecoveryCandidates(qv.Table):

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
        unique = self.drop_duplicates(subset=["exposure_id"])

        return Exposures.from_kwargs(
            id=unique.exposure_id,
            start_time=unique.exposure_time_start,
            observatory_code=unique.obscode,
            filter=unique.filter.to_pylist(),
            duration=unique.exposure_duration,
        )

    def predicted_ephemeris(self, orbit_ids=None) -> Ephemeris:
        """
        Return the predicted ephemeris for these candidates.

        Parameters
        ----------
        orbit_ids : Optional[List[str]], optional
            Orbit IDs to use for the predicted ephemeris. If None, a unique
            orbit ID will be generated for each candidate.

        Returns
        -------
        Ephemeris
            Predicted ephemeris for these candidates.
        """
        if orbit_ids is None:
            orbit_ids = [str(i) for i in range(len(self.time))]
        return Ephemeris.from_kwargs(
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


class FrameCandidates(qv.Table):

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

    def exposures(self) -> Exposures:
        unique = self.drop_duplicates(subset=["exposure_id"])

        return Exposures.from_kwargs(
            id=unique.exposure_id,
            start_time=unique.exposure_time_start,
            observatory_code=unique.obscode,
            filter=unique.filter.to_pylist(),
            duration=unique.exposure_duration,
        )

    def predicted_ephemeris(self, orbit_ids=None) -> Ephemeris:
        origin = Origin.from_kwargs(
            code=["SUN" for i in range(len(self.exposure_time_mid))]
        )
        frame = "ecliptic"
        if orbit_ids is None:
            orbit_ids = [str(i) for i in range(len(self.exposure_time_mid))]
        return Ephemeris.from_kwargs(
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


def candidates_from_ephem(
    obs: ObservationsTable, ephem: Ephemeris, frame: HealpixFrame
) -> PrecoveryCandidates:
    """
    Generates PrecoveryCandidates from constituent observations, ephem, and frame data

    Parameters
    ----------
    obs : ObservationsTable
        Observations that matched and are now candidates
    ephem : Ephemeris
        Matching ephemeris for each observation from which the residuals are calculated
    frame : HealpixFrame
        Frames include the exposure metadata which is part of the candidate

    Returns
    -------
    PrecoveryCandidates
        PrecoveryCandidates table with the matching observations
    """
    assert len(frame) == 1, "frame should have only one entry"
    assert len(obs) == len(
        ephem
    ), "Observations and ephemeris must be the same length. If ephem is identical, use ephem.take() to repeat the same ephemeris for each observation."  # noqa: E501

    healpix = radec_to_healpixel(
        ephem.coordinates.lon.to_numpy(),
        ephem.coordinates.lat.to_numpy(),
        nside=CANDIDATE_NSIDE,
    )

    exposure_time_start = Timestamp.from_mjd(
        pa.repeat(frame.exposure_mjd_start[0].as_py(), len(obs)), scale="utc"
    )

    exposure_time_mid = Timestamp.from_mjd(
        pa.repeat(frame.exposure_mjd_mid[0].as_py(), len(obs)), scale="utc"
    )

    frame_filter = pa.repeat(frame.filter[0].as_py(), len(obs))
    obscode = pa.repeat(frame.obscode[0].as_py(), len(obs))
    exposure_id = pa.repeat(frame.exposure_id[0].as_py(), len(obs))
    exposure_duration = pa.repeat(frame.exposure_duration[0].as_py(), len(obs))
    delta_ra_arcsec = pc.divide(pc.subtract(obs.ra, ephem.coordinates.lon), ARCSEC)
    delta_dec_arcsec = pc.divide(pc.subtract(obs.dec, ephem.coordinates.lat), ARCSEC)

    distance_arcsec = (
        haversine_distance_deg(
            obs.ra.to_numpy(),
            ephem.coordinates.lon.to_numpy(),
            obs.dec.to_numpy(),
            ephem.coordinates.lat.to_numpy(),
        )
        / ARCSEC
    )

    dataset_id = pa.repeat(frame.dataset_id[0].as_py(), len(obs))
    orbit_id = pa.repeat(ephem.orbit_id[0].as_py(), len(obs))

    return PrecoveryCandidates.from_kwargs(
        time=obs.time,
        ra_deg=obs.ra,
        dec_deg=obs.dec,
        ra_sigma_arcsec=pc.divide(obs.ra_sigma, ARCSEC),
        dec_sigma_arcsec=pc.divide(obs.dec_sigma, ARCSEC),
        mag=obs.mag,
        mag_sigma=obs.mag_sigma,
        exposure_time_start=exposure_time_start,
        exposure_time_mid=exposure_time_mid,
        filter=frame_filter,
        obscode=obscode,
        exposure_id=exposure_id,
        exposure_duration=exposure_duration,
        observation_id=obs.id,
        healpix_id=healpix,
        pred_ra_deg=ephem.coordinates.lon,
        pred_dec_deg=ephem.coordinates.lat,
        pred_vra_degpday=ephem.coordinates.vlon,
        pred_vdec_degpday=ephem.coordinates.vlat,
        delta_ra_arcsec=delta_ra_arcsec,
        delta_dec_arcsec=delta_dec_arcsec,
        distance_arcsec=distance_arcsec,
        dataset_id=dataset_id,
        orbit_id=orbit_id,
    )


def frame_candidates_from_frame(frame: HealpixFrame, ephem: Ephemeris):
    # Calculate the HEALpixel ID for the predicted ephemeris of
    # the orbit with a high nside value (k=15, nside=2**15) The
    # indexed observations are indexed to a much lower nside but
    # we may decide in the future to re-index the database using
    # different values for that parameter. As long as we return a
    # Healpix ID generated with nside greater than the indexed
    # database then we can always down-sample the ID to a lower
    # nside value
    healpix_id = int(
        radec_to_healpixel(
            ephem.coordinates.lon[0].as_py(),
            ephem.coordinates.lat[0].as_py(),
            nside=CANDIDATE_NSIDE,
        )
    )
    return FrameCandidates.from_kwargs(
        exposure_time_start=Timestamp.from_mjd(frame.exposure_mjd_start, scale="utc"),
        exposure_time_mid=Timestamp.from_mjd(frame.exposure_mjd_mid, scale="utc"),
        filter=frame.filter,
        obscode=frame.obscode,
        exposure_id=frame.exposure_id,
        exposure_duration=frame.exposure_duration,
        dataset_id=frame.dataset_id,
        healpix_id=[healpix_id],
        pred_ra_deg=ephem.coordinates.lon,
        pred_dec_deg=ephem.coordinates.lat,
        pred_vra_degpday=ephem.coordinates.vlon,
        pred_vdec_degpday=ephem.coordinates.vlat,
        orbit_id=ephem.orbit_id,
    )


def find_healpixel_matches(
    propagation_targets: GenericFrame, ephems: Ephemeris, nside: int
) -> GenericFrame:
    """
    Find the healpixels that match between the propagation targets and the 2 body ephemeris

    Parameters
    ----------
    propagation_targets : PropagationTargets
        Propagation targets to match
    ephems : Ephemeris
        Ephemeris to match

    Returns
    -------
    PropagationTargets
        Propagation targets that match the ephemeris
    """
    # Sort them both by time
    propagation_targets = propagation_targets.sort_by(["time.days", "time.nanos"])
    ephems = ephems.sort_by(["coordinates.time.days", "coordinates.time.nanos"])

    propagation_target_times = propagation_targets.time.rescale("utc")
    ephem_times = ephems.coordinates.time.rescale("utc")

    # quickly check to make sure times are equal
    assert pc.all(
        propagation_target_times.equals(ephem_times, precision="ms")
    ).as_py(), "Propagation targets and ephemeris must have matching times"

    # Calculate the healpixels for the ephemeris
    ephem_healpixels = radec_to_healpixel(
        ephems.coordinates.lon.to_numpy(),
        ephems.coordinates.lat.to_numpy(),
        nside=nside,
    )

    # Find the matching healpixels
    mask = pc.equal(propagation_targets.healpixel, ephem_healpixels)
    filtered_targets = propagation_targets.apply_mask(mask)

    return filtered_targets


def generate_ephem_for_per_obs_timestamps(
    orbit: Orbits, observations: ObservationsTable, obscode: str, propagator: Propagator
) -> Ephemeris:
    """
    Use 2 body propagation to generate ephemeris for unique time observations
    """
    # The observations may not be centered on the exposure time
    # So we want to make our 2 body propagation start from the mean
    mean_mjd = pc.mean(observations.time.mjd()).as_py()
    mean_time = Timestamp.from_mjd([mean_mjd], scale="utc")
    mean_orbit_state = propagator.propagate_orbits(orbit, mean_time)
    propagated_orbits = propagate_2body(
        mean_orbit_state, observations.time, propagator, obscode
    )
    observers = Observers.from_code(obscode, observations.time)
    ephemeris = generate_ephemeris_2body(propagated_orbits, observers)
    return ephemeris


def find_observation_matches(
    observations: ObservationsTable, ephems: Ephemeris, tolerance_deg: float
) -> Tuple[ObservationsTable, Ephemeris]:
    """
    Find the observations that match the ephemeris within the given tolerance

    Parameters
    ----------
    observations : ObservationsTable
        Observations to match with ephemeris (in pairs)
    ephems : Ephemeris
        Ephemeris to match with observations (in pairs)
    tolerance_deg : float
        Tolerance in degrees for matching
    Returns
    -------
    Tuple[ObservationsTable, Ephemeris]
        Observations and ephemeris that match within the given tolerance
    """
    assert len(ephems) == len(
        observations
    ), "Ephemeris must be the same length as observations"
    assert pc.all(
        ephems.coordinates.time.equals(observations.time, precision="ms")
    ).as_py(), "Ephemeris and observations must have matching times"
    # Check for bizarrely large tolerance which might have been
    # sent in as arcseconds instead of degrees
    if tolerance_deg > 2:
        logger.warning(
            "Tolerance is very large, did you pass in arcseconds instead of degrees?"
        )

    distances = haversine_distance_deg(
        observations.ra.to_numpy(),
        ephems.coordinates.lon.to_numpy(),
        observations.dec.to_numpy(),
        ephems.coordinates.lat.to_numpy(),
    )

    mask = pc.less(distances, tolerance_deg)
    matching_observations = observations.apply_mask(mask)
    matching_ephems = ephems.apply_mask(mask)
    return matching_observations, matching_ephems


def check_window(
    db_dir: str,
    window: WindowCenters,
    orbit: Orbits,
    tolerance: float,
    propagator_class: Type[Propagator],
    datasets: Optional[set[str]] = None,
) -> Tuple[PrecoveryCandidates, FrameCandidates]:
    """
    Check a single window for precovery candidates

    Parameters
    ----------
    db_dir : str
        Directory of the database
    window : WindowCenters
        Window to check
    orbit : Orbits
        Orbit to propagate
    tolerance : float
        Tolerance in degrees for matching
    propagator_class : Type[Propagator]
        Propagator class to use for propagation
    datasets : Optional[set[str]], optional
        Datasets to consider, by default None

    Returns
    -------
    Tuple[PrecoveryCandidates, FrameCandidates]
        Precovery candidates and frame candidates
    """
    assert len(window) == 1, "Use _check_windows for multiple windows"
    assert len(orbit) == 1, "_check_window only support one orbit for now"
    logger.info(
        f"check_window orbit: {orbit.orbit_id[0].as_py()} obscode: {window.obscode[0].as_py()} window: {window.window_start().mjd()[0].as_py()} to {window.window_end().mjd()[0].as_py()}"
    )
    db = PrecoveryDatabase.from_dir(db_dir, mode="r", allow_version_mismatch=True)
    obscode = window.obscode[0].as_py()
    propagation_targets = db.frames.idx.propagation_targets(
        window,
        datasets,
    )
    if len(propagation_targets) == 0:
        logger.debug(
            f"No propagation targets found for window {window.window_start().mjd()[0].as_py()} to {window.window_end().mjd()[0].as_py()}"
        )
        return PrecoveryCandidates.empty(), FrameCandidates.empty()

    logger.debug(f"Found {len(propagation_targets)} propagation targets")
    times = propagation_targets.time

    # First make sure our orbit it n-body propagated to the window center
    time_utc = orbit.coordinates.time.rescale("utc")
    if not (pc.all(time_utc.equals(window.time, precision="ms")).as_py()):
        propagator = propagator_class()
        orbit = propagator.propagate_orbits(orbit, window.time)

    # create our observers from the individual frame times
    observers = Observers.from_code(obscode, times)
    ## first propagate with 2_body
    propagated_orbits = propagate_2body(orbit, times)

    # generate ephemeris
    ephems = generate_ephemeris_2body(propagated_orbits, observers)
    frames_to_check = find_healpixel_matches(
        propagation_targets, ephems, db.frames.healpix_nside
    )
    logger.debug(f"Found {len(frames_to_check)} healpixel matches")
    candidates = PrecoveryCandidates.empty()
    frame_candidates = FrameCandidates.empty()
    for frame in frames_to_check:
        candidates_healpixel, frame_candidates_healpixel = db._check_frames(
            orbit=orbit,
            generic_frames=frame,
            obscode=obscode,
            tolerance=tolerance,
            datasets=datasets,
            propagator_class=propagator_class,
        )
        candidates = qv.concatenate([candidates, candidates_healpixel])
        frame_candidates = qv.concatenate(
            [frame_candidates, frame_candidates_healpixel]
        )
    return candidates, frame_candidates


check_window_remote = ray.remote(check_window)


class PrecoveryDatabase:
    def __init__(self, frames: FrameDB, directory: str, config: Config = DefaultConfig):
        self.frames = frames
        self._exposures_by_obscode: dict = {}
        self.config = config
        self.directory: str = directory

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

        frame_idx_db = "sqlite:///" + os.path.join(directory, "index.db")
        frame_idx = FrameIndex(frame_idx_db, mode=mode)

        data_path = os.path.join(directory, "data")
        frame_db = FrameDB(
            frame_idx, data_path, config.data_file_max_size, config.nside, mode=mode
        )
        return cls(frame_db, directory, config)

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

        return cls(frame_db, directory, config)

    def precover(
        self,
        orbit: Orbits,
        tolerance: float = 30 * ARCSEC,
        start_mjd: Optional[float] = None,
        end_mjd: Optional[float] = None,
        window_size: int = 7,
        datasets: Optional[set[str]] = None,
        propagator_class: Optional[Type[Propagator]] = None,
        max_processes: Optional[int] = None,
    ) -> Tuple[PrecoveryCandidates, FrameCandidates]:
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
        Tuple[PrecoveryCandidates, FrameCandidates]
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

        if propagator_class is None:
            raise ValueError("A propagator must be provided to run precovery")

        assert len(orbit) == 1, "Use precovery_many for multiple orbits"

        orbit_id = orbit.orbit_id[0].as_py()

        # Normalize the orbit timescale to utc for comparisons
        orbit = orbit.set_column(
            "coordinates.time", orbit.coordinates.time.rescale("utc")
        )

        if datasets is not None:
            self._warn_for_missing_datasets(datasets)

        if start_mjd is None or end_mjd is None:
            first, last = self.frames.idx.mjd_bounds(datasets=datasets)
            if start_mjd is None:
                start_mjd = first
            if end_mjd is None:
                end_mjd = last

        logger.info(
            f"precovering orbit {orbit_id} from {start_mjd} to {end_mjd}, window={window_size}, datasets={datasets or 'all'}"
        )

        windows = self.frames.idx.window_centers(
            start_mjd, end_mjd, window_size, datasets=datasets
        )
        logger.info(f"Searching {len(windows)} windows")
        candidates = PrecoveryCandidates.empty()
        frame_candidates = FrameCandidates.empty()

        # group windows by obscodes so that many windows can be searched at once
        for obscode in windows.obscode.unique():
            obscode_windows = windows.select("obscode", obscode)
            logger.info(
                f"searching {len(obscode_windows)} windows for obscode {obscode}"
            )

            candidates_obscode, frame_candidates_obscode = self._check_windows(
                obscode_windows,
                orbit,
                tolerance,
                propagator_class,
                datasets=datasets,
                max_processes=max_processes,
            )
            candidates = qv.concatenate([candidates, candidates_obscode])
            frame_candidates = qv.concatenate(
                [frame_candidates, frame_candidates_obscode]
            )

        # convert these to our new output formats
        return candidates, frame_candidates

    def _check_windows(
        self,
        windows: WindowCenters,
        orbit: Orbits,
        tolerance: float,
        propagator_class: Type[Propagator],
        datasets: Optional[set[str]] = None,
        max_processes: Optional[int] = None,
    ) -> Tuple[PrecoveryCandidates, FrameCandidates]:
        """
        Find all observations that match orbit within a list of windows
        """
        assert len(orbit) == 1, "_check_windows only support one orbit for now"
        windows = windows.sort_by(
            [("time.days", "descending"), ("time.nanos", "descending")]
        )
        logger.info(
            f"_check_windows orbit: {orbit.orbit_id[0].as_py()} windows: {len(windows)} obscode: {windows.obscode.unique().to_pylist()}"
        )

        precovery_candidates = PrecoveryCandidates.empty()
        frame_candidates = FrameCandidates.empty()

        if max_processes is not None and max_processes > 1:
            initialize_use_ray(num_cpus=max_processes)
            futures = []
            for window in windows:
                futures.append(
                    check_window_remote.remote(
                        self.directory,
                        window,
                        # Note: There is no speed benefit to pre-propagating
                        # the orbit to the window center here, since we do
                        # the n-body propagation inside the worker and the
                        # delay to start the job offsets any advantage
                        # from prepropagation.
                        orbit,
                        tolerance,
                        propagator_class,
                        datasets,
                    )
                )

                if len(futures) >= max_processes * 2:
                    finished, futures = ray.wait(futures, num_returns=1)
                    precovery_candidates_window, frame_candidates_window = ray.get(
                        finished[0]
                    )
                    precovery_candidates = qv.concatenate(
                        [precovery_candidates, precovery_candidates_window]
                    )
                    frame_candidates = qv.concatenate(
                        [frame_candidates, frame_candidates_window]
                    )

            while len(futures) > 0:
                finished, futures = ray.wait(futures, num_returns=1)
                precovery_candidates_window, frame_candidates_window = ray.get(
                    finished[0]
                )
                precovery_candidates = qv.concatenate(
                    [precovery_candidates, precovery_candidates_window]
                )
                frame_candidates = qv.concatenate(
                    [frame_candidates, frame_candidates_window]
                )
        else:
            propagator = propagator_class()
            for window in windows:
                # For single process, we propagate the orbit
                # to the window center in a loop to avoid
                # duplicating the n-body propagation inside
                # check_window
                orbit = propagator.propagate_orbits(orbit, window.time)
                candidates_window, frame_candidates_window = check_window(
                    self.directory,
                    window,
                    orbit=orbit,
                    tolerance=tolerance,
                    propagator_class=propagator_class,
                    datasets=datasets,
                )
                precovery_candidates = qv.concatenate(
                    [precovery_candidates, candidates_window]
                )
                frame_candidates = qv.concatenate(
                    [frame_candidates, frame_candidates_window]
                )
        return precovery_candidates, frame_candidates

    def _check_frames(
        self,
        orbit: Orbits,
        generic_frames: GenericFrame,
        obscode: str,
        tolerance: float,
        datasets: Optional[set[str]],
        propagator_class: Type[Propagator],
    ) -> Tuple[PrecoveryCandidates, FrameCandidates]:
        """
        Deeply inspect all frames that match the given obscode, mjd, and healpix to
        see if they contain observations which match the ephemeris.
        """
        frames = HealpixFrame.empty()
        for generic_frame in generic_frames:
            frames = qv.concatenate(
                [
                    frames,
                    self.frames.idx.get_frames(
                        obscode,
                        generic_frame.time.mjd()[0].as_py(),
                        generic_frame.healpixel[0].as_py(),
                        datasets,
                    ),
                ]
            )
        unique_frame_times = frames.exposure_mid_timestamp().unique()
        observers = Observers.from_code(obscode, unique_frame_times)
        # Compute the position of the ephem carefully.
        propagator = propagator_class()
        ephemeris = propagator.generate_ephemeris(orbit, observers)
        precovery_candidates = PrecoveryCandidates.empty()
        frame_candidates = FrameCandidates.empty()
        for f in frames:
            matching_ephem = ephemeris.apply_mask(
                ephemeris.coordinates.time.equals(
                    f.exposure_mid_timestamp(), precision="us"
                )
            )
            # If we don't have at least one matching ephem, it implies
            # our propagated times are not matching the frame times well enough
            assert len(matching_ephem) == 1, "No matching ephemeris found, should be 1"
            matches = self.find_matches_in_frame(
                f, orbit, matching_ephem, tolerance, propagator
            )
            # If no observations were found in this frame then we
            # return frame candidates
            # Note that for the frame candidate we report the predicted
            # ephemeris at the exposure midpoint not at the observation
            # times which may differ from the exposure midpoint time
            if len(matches) == 0:
                frame_candidates = qv.concatenate(
                    [frame_candidates, frame_candidates_from_frame(f, matching_ephem)]
                )
            else:
                precovery_candidates = qv.concatenate([precovery_candidates, matches])
        return precovery_candidates, frame_candidates

    def find_matches_in_frame(
        self,
        frame: HealpixFrame,
        orbit: Orbits,
        frame_ephem: Ephemeris,
        tolerance: float,
        propagator: Propagator,
    ) -> PrecoveryCandidates:
        """
        Find all sources in a single frame which match ephem.
        """
        assert len(frame_ephem) == 1, "ephem should have only one entry"

        # Gather all observations.
        observations: ObservationsTable = self.frames.get_observations(frame)
        # Check if the observations have per-observation MJDs.
        # If so we use 2 body to generate unique ephemeris for each
        if len(observations.time.unique()) > 1:
            per_obs_ephem = generate_ephem_for_per_obs_timestamps(
                orbit, observations, frame.obscode, propagator
            )
            matching_observations, matching_ephem = find_observation_matches(
                observations, per_obs_ephem, tolerance
            )
        # Otherwise the default state is to use the same ephemeris
        # for each observation in the frame
        else:
            repeated_ephem = Ephemeris.from_pyarrow(
                frame_ephem.table.take(np.zeros(len(observations), dtype=int))
            )
            matching_observations, matching_ephem = find_observation_matches(
                observations, repeated_ephem, tolerance
            )

        if len(matching_observations) == 0:
            return PrecoveryCandidates.empty()

        candidates = candidates_from_ephem(matching_observations, matching_ephem, frame)
        return candidates

    def find_observations_in_region(
        self, ra: float, dec: float, obscode: str
    ) -> ObservationsTable:
        """Gets all the Observations within the same Healpixel as a
        given RA, Dec for a particular observatory (specified as an obscode).

        """
        frames = self.frames.get_frames_for_ra_dec(ra, dec, obscode)
        observations = ObservationsTable.empty()
        for f in frames:
            observations = qv.concatenate(
                [observations, self.frames.get_observations(f)]
            )
        return observations

    def find_observations_in_radius(
        self, ra: float, dec: float, tolerance: float, obscode: str
    ) -> ObservationsTable:
        """Gets all the Observations within a radius (in degrees) of
        a particular RA and Dec at a specific observatory.

        This method is approximate, and does not correctly find
        Observations that are within the radius, but on a different
        healpixel.

        """
        obs_in_region = self.find_observations_in_region(ra, dec, obscode)
        mask = pc.less_equal(
            haversine_distance_deg(
                obs_in_region.ra.to_numpy(), ra, obs_in_region.dec.to_numpy(), dec
            ),
            tolerance,
        )
        return obs_in_region.apply_mask(mask)

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
