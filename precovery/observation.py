import struct
from typing import List

import quivr as qv
from adam_core.time import Timestamp

from .healpix_geom import radec_to_healpixel
from .sourcecatalog import SourceObservation

# mjd, ra, dec, ra_sigma, dec_sigma, mag, mag_sigma, id
DATA_LAYOUT = "<dddddddl"


class ObservationsTable(qv.Table):
    """
    Quivr representation of observation
    """

    id = qv.BinaryColumn()
    time = Timestamp.as_column()
    ra = qv.Float64Column()
    dec = qv.Float64Column()
    ra_sigma = qv.Float64Column()
    dec_sigma = qv.Float64Column()
    mag = qv.Float64Column()
    mag_sigma = qv.Float64Column()

    data_layout = struct.Struct(DATA_LAYOUT)
    datagram_size = struct.calcsize(DATA_LAYOUT)

    def healpixel(self, nside):
        return radec_to_healpixel(self.ra, self.dec, nside)

    def to_bytes(self) -> List[bytes]:
        mjds = self.time.mjd().to_pylist()
        as_dicts = self.table.to_pylist()
        return [
            self.data_layout.pack(
                mjd,
                row["ra"],
                row["dec"],
                row["ra_sigma"],
                row["dec_sigma"],
                row["mag"],
                row["mag_sigma"],
                len(row["id"]),
            )
            + row["id"]
            for mjd, row in zip(mjds, as_dicts)
        ]

    @classmethod
    def from_srcobs(cls, srcobs: List[SourceObservation]):
        if len(srcobs) == 0:
            return ObservationsTable.empty()

        (mjds, ras, decs, ra_sigmas, dec_sigmas, mags, mag_sigmas, ids) = zip(
            *[
                (
                    so.mjd,
                    so.ra,
                    so.dec,
                    so.ra_sigma,
                    so.dec_sigma,
                    so.mag,
                    so.mag_sigma,
                    so.id,
                )
                for so in srcobs
            ]
        )
        return ObservationsTable.from_kwargs(
            id=ids,
            time=Timestamp.from_mjd(mjds, scale="utc"),
            ra=ras,
            dec=decs,
            ra_sigma=ra_sigmas,
            dec_sigma=dec_sigmas,
            mag=mags,
            mag_sigma=mag_sigmas,
        )
