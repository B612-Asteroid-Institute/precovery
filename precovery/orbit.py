import enum
from os import getenv
from typing import Iterable, List, Optional

import numpy as np
import numpy.typing as npt
import requests as req
from astropy.time import Time

from .observation import ObservationArray
from .spherical_geom import haversine_distance_deg

try:
    import pyoorb
except ImportError:
    raise ImportError(
        "pyoorb must be installed separately.\n"
        "Install via conda with 'conda install -c conda-forge pyoorb'\n"
        "or via source (see example at \n"
        "https://github.com/B612-Asteroid-Institute/precovery/blob/main/Dockerfile)"
    )

pyoorb_initialized = False

DEGREE = 1.0
ARCMIN = DEGREE / 60
ARCSEC = ARCMIN / 60


def _ensure_pyoorb_initialized(*args, **kwargs):
    """Make sure that pyoorb is initialized."""
    global pyoorb_initialized
    if not pyoorb_initialized:
        pyoorb.pyoorb.oorb_init(*args, **kwargs)
        pyoorb_initialized = True


class OrbitElementType(enum.Enum):
    CARTESIAN = 1
    COMETARY = 2
    KEPLERIAN = 3


class EpochTimescale(enum.Enum):
    UTC = 1
    UT1 = 2
    TT = 3
    TAI = 4


class PropagationIntegrator(enum.Enum):
    N_BODY = 1
    TWO_BODY = 2


class Orbit:
    def __init__(self, orbit_id: int, state_vector: npt.NDArray[np.float64]):
        """
        Create a new Orbit.

        state_vector is a pretty opaque blob. It should be the structure that
        pyoorb expects - a 12-element vector of doubles.
        """
        self.orbit_id = orbit_id
        self._state_vector = state_vector
        self._orbit_type = OrbitElementType(int(state_vector[0][7]))
        self._epoch_timescale = EpochTimescale(int(state_vector[0][9]))
        self._epoch = state_vector[0][8]

    @classmethod
    def cometary(
        cls,
        orbit_id: int,
        perihelion_au: float,
        eccentricity: float,
        inclination_deg: float,
        ascending_node_longitude_deg: float,
        periapsis_argument_deg: float,
        perihelion_epoch_mjd: float,
        osculating_element_epoch_mjd: float,
        epoch_timescale: EpochTimescale,
        abs_magnitude: float,
        photometric_slope_parameter: float,
    ):
        # Orbits class takes in degrees, but state vectors are given in radians
        state_vector = np.array(
            [
                [
                    orbit_id,
                    perihelion_au,
                    eccentricity,
                    np.deg2rad(inclination_deg),
                    np.deg2rad(ascending_node_longitude_deg),
                    np.deg2rad(periapsis_argument_deg),
                    perihelion_epoch_mjd,
                    OrbitElementType.COMETARY.value,
                    osculating_element_epoch_mjd,
                    epoch_timescale.value,
                    abs_magnitude,
                    photometric_slope_parameter,
                ]
            ],
            dtype=np.double,
            order="F",
        )

        return cls(orbit_id, state_vector)

    @classmethod
    def keplerian(
        cls,
        orbit_id: int,
        semimajor_axis_au: float,
        eccentricity: float,
        inclination_deg: float,
        ascending_node_longitude_deg: float,
        periapsis_argument_deg: float,
        mean_anomaly_deg: float,
        osculating_element_epoch_mjd: float,
        epoch_timescale: EpochTimescale,
        abs_magnitude: float,
        photometric_slope_parameter: float,
    ):
        # Orbits class takes in degrees, but state vectors are given in radians
        state_vector = np.array(
            [
                [
                    orbit_id,
                    semimajor_axis_au,
                    eccentricity,
                    np.deg2rad(inclination_deg),
                    np.deg2rad(ascending_node_longitude_deg),
                    np.deg2rad(periapsis_argument_deg),
                    np.deg2rad(mean_anomaly_deg),
                    OrbitElementType.KEPLERIAN.value,
                    osculating_element_epoch_mjd,
                    epoch_timescale.value,
                    abs_magnitude,
                    photometric_slope_parameter,
                ]
            ],
            dtype=np.double,
            order="F",
        )

        return cls(orbit_id, state_vector)

    @classmethod
    def cartesian(
        cls,
        orbit_id: int,
        x: float,
        y: float,
        z: float,
        vx: float,
        vy: float,
        vz: float,
        osculating_element_epoch_mjd: float,
        epoch_timescale: EpochTimescale,
        abs_magnitude: float,
        photometric_slope_parameter: float,
    ):
        state_vector = np.array(
            [
                [
                    orbit_id,
                    x,
                    y,
                    z,
                    vx,
                    vy,
                    vz,
                    OrbitElementType.CARTESIAN.value,
                    osculating_element_epoch_mjd,
                    epoch_timescale.value,
                    abs_magnitude,
                    photometric_slope_parameter,
                ]
            ],
            dtype=np.double,
            order="F",
        )

        return cls(orbit_id, state_vector)

    def propagate(
        self,
        epochs: Iterable[float],
        method: PropagationIntegrator = PropagationIntegrator.N_BODY,
        time_scale: EpochTimescale = EpochTimescale.UTC,
    ) -> List["Orbit"]:
        _ensure_pyoorb_initialized(error_verbosity=1)

        if method == PropagationIntegrator.N_BODY:
            dynmodel = "N"
        elif method == PropagationIntegrator.TWO_BODY:
            dynmodel = "2"
        else:
            raise ValueError("unexpected propagation method %r" % method)

        orbits = []
        for epoch in epochs:
            epoch_array = np.array(
                [epoch, time_scale.value], dtype=np.double, order="F"
            )

            result, err = pyoorb.pyoorb.oorb_propagation(
                in_orbits=self._state_vector,
                in_epoch=epoch_array,
                in_dynmodel=dynmodel,
            )
            assert err == 0, "There was an issue with the pyoorb orbit propagation"

            # Pyoorb wants radians as inputs for orbits but outputs propagated orbits as degrees
            # See here: https://github.com/oorb/oorb/blob/master/python/pyoorb.f90#L347
            # Note that time of perihelion passage also is converted to a degree.
            if (self._orbit_type == OrbitElementType.KEPLERIAN) or (
                self._orbit_type == OrbitElementType.COMETARY
            ):
                result[:, [3, 4, 5, 6]] = np.radians(result[:, [3, 4, 5, 6]])

            orbits.append(Orbit(int(result[0][0]), result))

        return orbits

    def compute_ephemeris(
        self,
        obscode: str,
        epochs: Iterable[float],
        method: PropagationIntegrator = PropagationIntegrator.N_BODY,
        time_scale: EpochTimescale = EpochTimescale.UTC,
    ) -> List["Ephemeris"]:
        """
        Compute ephemeris for the orbit, propagated to an epoch, and observed from
        a location represented by obscode.

        obscode should be a Minor Planet Center observatory code.
        """
        _ensure_pyoorb_initialized(error_verbosity=1)
        epochs_array = np.array(
            [[epoch, time_scale.value] for epoch in epochs],
            dtype=np.double,
            order="F",
        )
        # print(len(epochs))
        # print(epochs_array.shape)

        if method == PropagationIntegrator.N_BODY:
            dynmodel = "N"
        elif method == PropagationIntegrator.TWO_BODY:
            dynmodel = "2"
        else:
            raise ValueError("unexpected propagation method %r" % method)

        eph, err = pyoorb.pyoorb.oorb_ephemeris_basic(
            in_orbits=self._state_vector,
            in_obscode=obscode,
            in_date_ephems=epochs_array,
            in_dynmodel=dynmodel,
        )
        # print(epochs_array.shape)
        # print(eph.shape)
        assert err == 0, "There was an issue with the pyoorb ephemeris generation"
        return [
            Ephemeris.from_pyoorb_vector(eph[0, i, :])
            for i in range(epochs_array.shape[0])
        ]

    def precover_remote(
        self,
        tolerance: float = 30 * ARCSEC,
        max_matches: Optional[int] = None,
        start_mjd: Optional[float] = None,
        end_mjd: Optional[float] = None,
        window_size: Optional[float] = None,
    ):
        """
        Find observations which match orbit in the database. Observations are
        searched in descending order by mjd.

        Expects three environment variables:
        PRECOVERY_API_SINGLEORBIT_URL
        PRECOVERY_API_SINGLEORBIT_USERNAME
        PRECOVERY_API_SINGLEORBIT_PASSWORD

        max_matches: End once this many matches have been found. If None, find
        all matches.

        start_mjd: Only consider observations from after this epoch
        (inclusive). If None, find all.

        end_mjd: Only consider observations from before this epoch (inclusive).
        If None, find all.

        window_size: UNIMPLEMENTED
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

        # Check for environment set vars
        precovery_singleorbit_url = getenv("PRECOVERY_API_SINGLEORBIT_URL")
        api_username = getenv("PRECOVERY_API_SINGLEORBIT_USERNAME")
        # Note Password suffix results in encryption and storage in systems keys for conda env
        # (https://anaconda-project.readthedocs.io/en/latest/user-guide/tasks/work-with-variables.html#adding-an-encrypted-variable)
        api_password = getenv("PRECOVERY_API_SINGLEORBIT_PASSWORD")

        if not (precovery_singleorbit_url and api_username and api_password):
            raise ValueError(
                """one of required environment variables unset, expecting PRECOVERY_API_SINGLEORBIT_URL,
            PRECOVERY_API_SINGLEORBIT_USERNAME, PRECOVERY_API_SINGLEORBIT_PASSWORD"""
            )

        if self._orbit_type == OrbitElementType.KEPLERIAN:
            orbit_type = "kep"
            state_vector = {
                "a": self._state_vector[0][1],
                "e": self._state_vector[0][2],
                "i": self._state_vector[0][3],
                "an": self._state_vector[0][4],
                "ap": self._state_vector[0][5],
                "ma": self._state_vector[0][6],
            }
        elif self._orbit_type == OrbitElementType.COMETARY:
            orbit_type = "com"
            state_vector = {
                "q": self._state_vector[0][1],
                "e": self._state_vector[0][2],
                "i": self._state_vector[0][3],
                "an": self._state_vector[0][4],
                "ap": self._state_vector[0][5],
                "tp": self._state_vector[0][6],
            }
        elif self._orbit_type == OrbitElementType.CARTESIAN:
            orbit_type = "car"
            state_vector = {
                "x": self._state_vector[0][1],
                "y": self._state_vector[0][2],
                "z": self._state_vector[0][3],
                "vx": self._state_vector[0][4],
                "vy": self._state_vector[0][5],
                "vz": self._state_vector[0][6],
            }
        else:
            raise ValueError("orbit type improperly defined %r" % self._orbit_type)

        # Compile request dictionary

        if self._epoch_timescale == EpochTimescale.UTC:
            scale = "utc"
        elif self._epoch_timescale == EpochTimescale.UT1:
            scale = "ut1"
        elif self._epoch_timescale == EpochTimescale.TT:
            scale = "tt"
        elif self._epoch_timescale == EpochTimescale.TAI:
            scale = "tai"
        mjd_tdb = Time(self._epoch, scale=scale, format="mjd").tdb.mjd
        post_json = {
            "tolerance": tolerance,
            "orbit_type": orbit_type,
            "mjd_tdb": mjd_tdb,
        }
        post_json = post_json | state_vector
        if max_matches:
            post_json["max_matches"] = max_matches
        if start_mjd:
            post_json["start_mjd"] = start_mjd
        if end_mjd:
            post_json["end_mjd"] = end_mjd
        if window_size:
            post_json["window_size"] = window_size

        precovery_req = req.post(
            precovery_singleorbit_url, json=post_json, auth=(api_username, api_password)
        )
        return precovery_req.json()


class Ephemeris:
    def __init__(self, mjd, ra, dec, ra_velocity, dec_velocity):
        self.mjd = mjd
        self.ra = ra
        self.dec = dec
        self.ra_velocity = ra_velocity
        self.dec_velocity = dec_velocity

    @classmethod
    def from_pyoorb_vector(cls, raw_data: npt.NDArray[np.float64]):
        mjd = raw_data[0]
        ra = raw_data[1]
        dec = raw_data[2]
        # oorb returns vracos(dec), so lets remove the cos(dec) term
        ra_velocity = raw_data[3] / np.cos(np.radians(dec))  # deg per day
        dec_velocity = raw_data[4]  # deg per day
        return cls(mjd, ra, dec, ra_velocity, dec_velocity)

    def __str__(self):
        return f"<Ephemeris ra={self.ra:.4f} dec={self.dec:.4f} mjd={self.mjd:.6f}>"

    def distance(self, observations: ObservationArray) -> npt.NDArray[np.float64]:
        return haversine_distance_deg(
            self.ra, observations.values["ra"], self.dec, observations.values["dec"]
        )
