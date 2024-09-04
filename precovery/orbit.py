import enum
from typing import Iterable, List, Union

import numpy as np
import numpy.typing as npt
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.cometary import CometaryCoordinates
from adam_core.coordinates.keplerian import KeplerianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.dynamics.ephemeris import generate_ephemeris_2body
from adam_core.dynamics.propagation import propagate_2body
from adam_core.observers import Observers
from adam_core.orbits import Ephemeris as AdamEphemeris
from adam_core.orbits import Orbits as AdamOrbits
from adam_core.propagator import Propagator
from adam_core.propagator.adam_assist import ASSISTPropagator
from adam_core.time import Timestamp

from .observation import ObservationArray
from .spherical_geom import haversine_distance_deg

DEGREE = 1.0
ARCMIN = DEGREE / 60
ARCSEC = ARCMIN / 60


class OrbitElementType(enum.Enum):
    CARTESIAN = 1
    COMETARY = 2
    KEPLERIAN = 3


class EpochTimescale(enum.Enum):
    UTC = 1
    UT1 = 2
    TT = 3
    TAI = 4


class PropagationMethod(enum.Enum):
    N_BODY = 1
    TWO_BODY = 2


class PropagatorClass(enum.Enum):
    ASSIST = 1
    PYOORB = 2


class Orbit:
    def __init__(
        self,
        orbit_id: int,
        state_vector: npt.NDArray[np.float64],
        propagator: Union[PropagatorClass, Propagator] = PropagatorClass.ASSIST,
    ):
        """
        Create a new Orbit.

        state_vector is a pretty opaque blob. This is a relic
        of using pyoorb for propagation.
        """
        self.orbit_id = orbit_id
        self._state_vector = state_vector
        self._orbit_type = OrbitElementType(int(state_vector[0][7]))
        self._epoch_timescale = EpochTimescale(int(state_vector[0][9]))
        self._epoch = state_vector[0][8]
        if propagator == PropagatorClass.ASSIST:
            self._propagator = ASSISTPropagator()
        elif propagator is Propagator:
            self._propagator = propagator
        else:
            raise ValueError("unexpected propagator %r" % propagator)

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

    @classmethod
    def from_adam_core(
        self,
        orbit_id: int,
        ac_orbits: AdamOrbits,
        timescale: EpochTimescale = EpochTimescale.UTC,
        orbit_type: OrbitElementType = OrbitElementType.CARTESIAN,
        absolute_magnitude: float = 20.0,
        photometric_slope_parameter: float = 0.15,
    ):
        """
        Create an Orbit from a 1-length adam_core Orbits table.
        """

        if len(ac_orbits) != 1:
            raise ValueError("expected 1-length orbits table")
        if orbit_type == OrbitElementType.CARTESIAN:
            return self.cartesian(
                orbit_id,
                ac_orbits.coordinates.x[0].as_py(),
                ac_orbits.coordinates.y[0].as_py(),
                ac_orbits.coordinates.z[0].as_py(),
                ac_orbits.coordinates.vx[0].as_py(),
                ac_orbits.coordinates.vy[0].as_py(),
                ac_orbits.coordinates.vz[0].as_py(),
                ac_orbits.coordinates.time[0]
                .rescale(timescale.name.lower())
                .mjd()
                .to_numpy(False)[0],
                timescale,
                absolute_magnitude,
                photometric_slope_parameter,
            )
        elif orbit_type == OrbitElementType.COMETARY:
            coords = ac_orbits.coordinates.to_cometary()
            return self.cometary(
                orbit_id,
                coords.q[0].as_py(),
                coords.e[0].as_py(),
                coords.i[0].as_py(),
                coords.raan[0].as_py(),
                coords.ap[0].as_py(),
                coords.tp[0].as_py(),
                coords.time[0].rescale(timescale.name.lower()).mjd().to_numpy(False)[0],
                timescale,
                absolute_magnitude,
                photometric_slope_parameter,
            )
        elif orbit_type == OrbitElementType.KEPLERIAN:
            coords = ac_orbits.coordinates.to_keplerian()
            return self.keplerian(
                orbit_id,
                coords.a[0].as_py(),
                coords.e[0].as_py(),
                coords.i[0].as_py(),
                coords.raan[0].as_py(),
                coords.ap[0].as_py(),
                coords.M[0].as_py(),
                coords.time[0].rescale(timescale.name.lower()).mjd().to_numpy(False)[0],
                timescale,
                absolute_magnitude,
                photometric_slope_parameter,
            )

    def to_adam_core(
        self,
        object_id: str = "ObjIdUnset",
    ):
        """
        Create an Orbit from a 1-length adam_core Orbits table.
        """
        times = Timestamp.from_mjd(
            [self._epoch], scale=self._epoch_timescale.name.lower()
        )
        origin = Origin.from_kwargs(code=["SUN" for i in range(len(times))])
        frame = "ecliptic"
        coordinates = None
        if self._orbit_type == OrbitElementType.CARTESIAN:
            coordinates = CartesianCoordinates.from_kwargs(
                time=times,
                x=[self._state_vector[0][1]],
                y=[self._state_vector[0][2]],
                z=[self._state_vector[0][3]],
                vx=[self._state_vector[0][4]],
                vy=[self._state_vector[0][5]],
                vz=[self._state_vector[0][6]],
                covariance=None,
                origin=origin,
                frame=frame,
            )
        elif self._orbit_type == OrbitElementType.COMETARY:
            coordinates = CometaryCoordinates.from_kwargs(
                time=times,
                q=[self._state_vector[0][1]],
                e=[self._state_vector[0][2]],
                i=[np.rad2deg(self._state_vector[0][3])],
                raan=[np.rad2deg(self._state_vector[0][4])],
                ap=[np.rad2deg(self._state_vector[0][5])],
                tp=[self._state_vector[0][6]],
                origin=origin,
                frame=frame,
            ).to_cartesian()

        elif self._orbit_type == OrbitElementType.KEPLERIAN:
            coordinates = KeplerianCoordinates.from_kwargs(
                time=times,
                a=[self._state_vector[0][1]],
                e=[self._state_vector[0][2]],
                i=[np.rad2deg(self._state_vector[0][3])],
                raan=[np.rad2deg(self._state_vector[0][4])],
                ap=[np.rad2deg(self._state_vector[0][5])],
                M=[np.rad2deg(self._state_vector[0][6])],
                origin=origin,
                frame=frame,
            ).to_cartesian()
        return AdamOrbits.from_kwargs(
            orbit_id=[str(self.orbit_id)],
            object_id=[object_id],
            coordinates=coordinates,
        )

    def propagate(
        self,
        epochs: Iterable[float],
        method: PropagationMethod = PropagationMethod.N_BODY,
        time_scale: EpochTimescale = EpochTimescale.UTC,
    ) -> List["Orbit"]:

        orbits = []
        propagated_orbits = AdamOrbits.empty()
        times = Timestamp.from_mjd(epochs, scale=time_scale.name.lower())
        ac_orbits = self.to_adam_core(str(self.orbit_id))
        if method == PropagationMethod.N_BODY:
            propagated_orbits = self._propagator.propagate_orbits(ac_orbits, times)
        elif method == PropagationMethod.TWO_BODY:
            propagated_orbits = propagate_2body(ac_orbits, times)
        else:
            raise ValueError("unexpected propagation method %r" % method)
        for i in range(len(propagated_orbits)):
            orbits.append(
                Orbit.from_adam_core(
                    int(propagated_orbits[i].orbit_id[0].as_py()), propagated_orbits[i]
                )
            )
        return orbits

    def compute_ephemeris(
        self,
        obscode: str,
        epochs: Iterable[float],
        method: PropagationMethod = PropagationMethod.N_BODY,
        time_scale: EpochTimescale = EpochTimescale.UTC,
    ) -> List["Ephemeris"]:
        """
        Compute ephemeris for the orbit, propagated to an epoch, and observed from
        a location represented by obscode.

        obscode should be a Minor Planet Center observatory code.
        ephemeris = AdamEphemeris.empty()
        """
        propagated_orbits = None
        times = Timestamp.from_mjd(epochs, scale=time_scale.name.lower())
        ac_orbits = self.to_adam_core(str(self.orbit_id))
        # create observers
        observers = Observers.from_code(obscode, times)
        if method == PropagationMethod.N_BODY:
            ephemeris = self._propagator.generate_ephemeris(ac_orbits, observers)
        elif method == PropagationMethod.TWO_BODY:
            # first propagate with 2_body
            propagated_orbits = propagate_2body(ac_orbits, times)
            # generate ephemeris
            ephemeris = generate_ephemeris_2body(propagated_orbits, observers)
        else:
            raise ValueError("unexpected propagation method %r" % method)

        return [Ephemeris.from_adam_core(ephemeris[i]) for i in range(len(ephemeris))]


class Ephemeris:
    def __init__(
        self,
        mjd: float,
        ra: float,
        dec: float,
        ra_velocity: float,
        dec_velocity: float,
        mag: float,
    ):
        self.mjd = mjd
        self.ra = ra
        self.dec = dec
        self.ra_velocity = ra_velocity
        self.dec_velocity = dec_velocity
        self.mag = mag

    @classmethod
    def from_pyoorb_vector(cls, raw_data: npt.NDArray[np.float64]):
        mjd = raw_data[0]
        ra = raw_data[1]
        dec = raw_data[2]
        # oorb returns vracos(dec), so lets remove the cos(dec) term
        ra_velocity = raw_data[3] / np.cos(np.radians(dec))  # deg per day
        dec_velocity = raw_data[4]  # deg per day
        mag = raw_data[9]
        return cls(mjd, ra, dec, ra_velocity, dec_velocity, mag)

    @classmethod
    def from_adam_core(cls, ac_ephemeris: AdamEphemeris, mag: float = 20.0):
        mjd = ac_ephemeris.coordinates.time[0].mjd().to_numpy(False)[0]
        ra = ac_ephemeris.coordinates.lon[0].as_py()
        dec = ac_ephemeris.coordinates.lat[0].as_py()
        ra_velocity = ac_ephemeris.coordinates.vlon[0].as_py()
        dec_velocity = ac_ephemeris.coordinates.vlat[0].as_py()
        return cls(mjd, ra, dec, ra_velocity, dec_velocity, mag)

    def __str__(self):
        return f"<Ephemeris ra={self.ra:.4f} dec={self.dec:.4f} mjd={self.mjd:.6f}>"

    def distance(self, observations: ObservationArray) -> npt.NDArray[np.float64]:
        return haversine_distance_deg(
            self.ra, observations.values["ra"], self.dec, observations.values["dec"]
        )
