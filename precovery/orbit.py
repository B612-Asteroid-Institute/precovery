import enum
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pyoorb

from .spherical_geom import propagate_linearly

pyoorb_initialized = False


def _ensure_pyoorb_initialized(*args, **kwargs):
    """Make sure that pyoorb is initialized."""
    global pyoorb_initialized
    if not pyoorb_initialized:
        pyoorb.pyoorb.oorb_init(*args, **kwargs)


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
        state_vector = np.array(
            [
                [
                    orbit_id,
                    perihelion_au,
                    eccentricity,
                    inclination_deg,
                    ascending_node_longitude_deg,
                    periapsis_argument_deg,
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
        state_vector = np.array(
            [
                [
                    orbit_id,
                    semimajor_axis_au,
                    eccentricity,
                    inclination_deg,
                    ascending_node_longitude_deg,
                    periapsis_argument_deg,
                    mean_anomaly_deg,
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
        self, epoch: float, method: PropagationIntegrator = PropagationIntegrator.N_BODY
    ) -> "Orbit":
        _ensure_pyoorb_initialized()

        epoch_array = np.array(
            [epoch, self._epoch_timescale.value], dtype=np.double, order="F"
        )

        if method == PropagationIntegrator.N_BODY:
            dynmodel = "N"
        elif method == PropagationIntegrator.TWO_BODY:
            dynmodel = "2"
        else:
            raise ValueError("unexpected propagation method %r" % method)

        result, err = pyoorb.pyoorb.oorb_propagation(
            in_orbits=self._state_vector,
            in_epoch=epoch_array,
            in_dynmodel=dynmodel,
        )
        assert err == 0
        return Orbit(int(result[0][0]), result)

    def compute_ephemeris(
        self,
        obscode: str,
        epoch: float,
        method: PropagationIntegrator = PropagationIntegrator.N_BODY,
    ) -> "Ephemeris":
        """
        Compute ephemeris for the orbit, propagated to an epoch, and observed from
        a location represented by obscode.

        obscode should be a Minor Planet Center observatory code.
        """
        _ensure_pyoorb_initialized()
        epoch_array = np.array(
            [[epoch, self._epoch_timescale.value]], dtype=np.double, order="F"
        )

        if method == PropagationIntegrator.N_BODY:
            dynmodel = "N"
        elif method == PropagationIntegrator.TWO_BODY:
            dynmodel = "2"
        else:
            raise ValueError("unexpected propagation method %r" % method)

        eph, err = pyoorb.pyoorb.oorb_ephemeris_basic(
            in_orbits=self._state_vector,
            in_obscode=obscode,
            in_date_ephems=epoch_array,
            in_dynmodel=dynmodel,
        )
        assert err == 0
        return Ephemeris(eph[0][0])


class Ephemeris:
    def __init__(self, raw_data: npt.NDArray[np.float64]):
        self._raw_data = raw_data

        self.mjd = raw_data[0]

        self.ra = raw_data[1]
        self.dec = raw_data[2]
        self.ra_velocity = raw_data[3]  # deg per day
        self.dec_velocity = raw_data[4]  # deg per day

    def __str__(self):
        return f"<Ephemeris ra={self.ra:.4f} dec={self.dec:.4f} mjd={self.mjd:.6f}>"

    def approximately_propagate(
        self, obscode: str, orbit: Orbit, timedelta: float
    ) -> Tuple[float, float]:
        """
        Roughly propagate the ephemeris to a new epoch, 'timedelta' days away along.

        If timedelta is small and self.ra_velocity and self.dec_velocity are
        small (indicating relatively slow motion across the sky), this uses a
        linear motion approximation.

        Otherwise, it uses a 2-body integration of the orbit.

        Accuracy will decrease as timedelta increases.
        """
        if timedelta <= 3.0 and self.ra_velocity < 1 and self.dec_velocity < 1:
            # Naively calculate the approximate location of the ephemeris
            # when the orbit is "propagated" as naively as possible to the
            # epoch of the observation: just use linear motion across the
            # sky. This is a bad model for long or curving arcs, but it's
            # roughly okay over very short time intervals, like <= 3 days.
            approx_ra_rad, approx_dec_rad = propagate_linearly(
                np.deg2rad(self.ra),
                np.deg2rad(self.ra_velocity),
                np.deg2rad(self.dec),
                np.deg2rad(self.dec_velocity),
                timedelta,
            )
            approx_ra = np.rad2deg(approx_ra_rad)
            approx_dec = np.rad2deg(approx_dec_rad)
        else:
            approx_ephem = orbit.compute_ephemeris(
                obscode,
                self.mjd + timedelta,
                method=PropagationIntegrator.TWO_BODY,
            )
            approx_ra = approx_ephem.ra
            approx_dec = approx_ephem.dec

        return approx_ra, approx_dec
