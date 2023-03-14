import dataclasses
import struct

import numpy as np

from .sourcecatalog import SourceObservation

# mjd, ra, dec, ra_sigma, dec_sigma, mag, mag_sigma, id
DATA_LAYOUT = "<dddddddl"


@dataclasses.dataclass
class Observation:
    mjd: float
    ra: float
    dec: float
    ra_sigma: float
    dec_sigma: float
    mag: float
    mag_sigma: float
    id: bytes

    data_layout = struct.Struct(DATA_LAYOUT)
    datagram_size = struct.calcsize(DATA_LAYOUT)

    def to_bytes(self) -> bytes:
        prefix = self.data_layout.pack(
            self.mjd,
            self.ra,
            self.dec,
            self.ra_sigma,
            self.dec_sigma,
            self.mag,
            self.mag_sigma,
            len(self.id),
        )
        return prefix + self.id

    @classmethod
    def from_srcobs(cls, so: SourceObservation):
        """
        Cast a SourceObservation to an Observation.
        """
        return cls(
            mjd=so.mjd,
            ra=so.ra,
            dec=so.dec,
            ra_sigma=so.ra_sigma,
            dec_sigma=so.dec_sigma,
            mag=so.mag,
            mag_sigma=so.mag_sigma,
            id=so.id,
        )


class ObservationArray:
    """A collection of Observations stored together in a numpy array."""

    dtype = np.dtype(
        [
            ("mjd", np.float64),
            ("ra", np.float64),
            ("dec", np.float64),
            ("ra_sigma", np.float64),
            ("dec_sigma", np.float64),
            ("mag", np.float64),
            ("mag_sigma", np.float64),
            ("id", np.object_),
        ]
    )

    def __init__(self, observations: list[Observation]):
        self.values = np.array(
            [dataclasses.astuple(o) for o in observations], dtype=self.dtype
        )

    def __len__(self) -> int:
        return len(self.values)

    def to_list(self):
        return [Observation(*row) for row in self.values]
