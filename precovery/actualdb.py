import dataclasses
import glob
import os
import struct
from typing import Optional

import sqlalchemy
from sqlalchemy.sql import func as sqlfunc

from . import sourcecatalog


@dataclasses.dataclass
class ExposureMeta:
    id: Optional[int]
    obscode: str
    catalog_id: str
    mjd: float
    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float

    data_uri: str
    data_offset: int
    data_length: int


class ExposureIndex:
    def __init__(self, db_engine):
        self.db = db_engine
        self.dbconn = self.db.connect()
        self.initialize_tables()

    @classmethod
    def open(cls, db_uri):
        engine = sqlalchemy.create_engine(db_uri)
        return cls(engine)

    def close(self):
        self.dbconn.close()

    def exposure_bundles(self, window_size_days):
        """
        Returns a generator which yields bundles of exposures along with a time
        (mjd float) which represents the central epoch for the bundle.

        Each bundle of exposures will be nearby in time. They will be within
        window_size_days of each other.
        """
        select_stmt = sqlalchemy.select(
            sqlfunc.min(self.exps.c.mjd, type=sqlalchemy.Float),
            sqlfunc.max(self.exps.c.mjd, type=sqlalchemy.Float),
        )
        first, last = self.dbconn.execute(select_stmt).fetchone()

        window_start = first
        window_center = first + window_size_days / 2
        window_end = first + window_size_days
        while window_start < last:
            select_stmt = sqlalchemy.select(self.exps).where(
                self.exps.c.mjd >= window_start, self.exps.c.mjd < window_end
            )
            exposures = self.dbconn.execute(select_stmt)
            bundle = [ExposureMeta(*exp) for exp in exposures]
            yield (window_center, bundle)

            window_start += window_size_days
            window_center += window_size_days
            window_end += window_size_days

    def add_exposure(self, exp: ExposureMeta):
        insert = self.exps.insert().values(
            obscode=exp.obscode,
            catalog_id=exp.catalog_id,
            mjd=exp.mjd,
            ra_min=exp.ra_min,
            ra_max=exp.ra_max,
            dec_min=exp.dec_min,
            dec_max=exp.dec_max,
            data_uri=exp.data_uri,
            data_offset=exp.data_offset,
            data_length=exp.data_length,
        )
        self.dbconn.execute(insert)

    def initialize_tables(self):
        self._metadata = sqlalchemy.MetaData()
        self.exps = sqlalchemy.Table(
            "exposures",
            self._metadata,
            sqlalchemy.Column(
                "id",
                sqlalchemy.Integer,
                sqlalchemy.Sequence("exposure_id_seq"),
                primary_key=True,
            ),
            sqlalchemy.Column("obscode", sqlalchemy.String),
            sqlalchemy.Column("catalog_id", sqlalchemy.String),
            sqlalchemy.Column("mjd", sqlalchemy.Float),
            sqlalchemy.Column("ra_min", sqlalchemy.Float),
            sqlalchemy.Column("ra_max", sqlalchemy.Float),
            sqlalchemy.Column("dec_min", sqlalchemy.Float),
            sqlalchemy.Column("dec_max", sqlalchemy.Float),
            sqlalchemy.Column("data_uri", sqlalchemy.String),
            sqlalchemy.Column("data_offset", sqlalchemy.Integer),
            sqlalchemy.Column("data_length", sqlalchemy.Integer),
        )
        self._metadata.create_all(self.db)


class ExposureDatabase:
    def __init__(self, idx: ExposureIndex, data_root: str, data_file_max_size=1e9):
        self.idx = idx
        self.data_root = data_root
        self.data_file_max_size = data_file_max_size
        self.data_files: dict = {}  # basename -> open file
        self._open_data_files()
        self.n_data_files = 0

    def close(self):
        self.idx.close()
        for f in self.data_files.values():
            f.close()

    def load_hdf5(self, hdf5_file, limit):
        for exp_metadict, exp_data in sourcecatalog.iterate_exposures(hdf5_file, limit):
            data_uri, offset, length = self.store_data(exp_data)
            exp_meta = ExposureMeta(
                id=None,
                obscode=exp_metadict["obscode"],
                catalog_id=exp_metadict["exposure_id"],
                mjd=exp_metadict["mjd"],
                ra_min=exp_metadict["ra_min"],
                ra_max=exp_metadict["ra_max"],
                dec_min=exp_metadict["dec_min"],
                dec_max=exp_metadict["dec_max"],
                data_uri=data_uri,
                data_offset=offset,
                data_length=length,
            )
            self.idx.add_exposure(exp_meta)

    def _open_data_files(self):
        matcher = os.path.join(self.data_root, "exposures*.data")
        files = glob.glob(matcher)
        for f in files:
            abspath = os.path.abspath(f)
            name = os.path.basename(f)
            self.data_files[name] = open(abspath, "a+b")
        self.n_data_files = len(self.data_files) - 1
        if self.n_data_files <= 0:
            self.new_data_file()

    def _current_data_file_name(self):
        return "exposures_{:08d}.data".format(self.n_data_files)

    def _current_data_file_full(self):
        return os.path.abspath(
            os.path.join(self.data_root, self._current_data_file_name())
        )

    def _current_data_file_size(self):
        return os.stat(self._current_data_file_name()).st_size

    def _current_data_file(self):
        return self.data_files[self._current_data_file_name()]

    def iterate_observations(self, exp: ExposureMeta):
        """
        Iterate over the observations stored in an exposure.
        """
        f = self.data_files[exp.data_uri]
        f.seek(exp.data_offset)
        data_layout = struct.Struct("<ddl")
        datagram_size = struct.calcsize("<ddl")
        bytes_read = 0
        while bytes_read < exp.data_length:
            raw = f.read(datagram_size)
            ra, dec, id_size = data_layout.unpack(raw)
            id = f.read(id_size).decode("utf8")
            bytes_read += datagram_size + id_size
            yield (ra, dec, id)

    def store_data(self, exp_data):
        """
        Write observations in exp_data to disk. The write goes to the latest (or
        "current") file in the database. Data is written as a sequence of
        packed RA, Dec, and ID values.

        RA and Dec are encoded as double-length floating point values. ID is
        encoded as a length-prefixed string: a long integer (little-endian)
        which indicates the ID's length in bytes, followed by the UTF8-encoded
        bytes of the ID.
        """
        data_layout = struct.Struct("<ddl")
        f = self._current_data_file()
        f.seek(0, 2)  # seek to end
        start_pos = f.tell()

        for (ra, dec, id_bytes) in exp_data:
            data = data_layout.pack(ra, dec, len(id_bytes))
            f.write(data + id_bytes)

        end_pos = f.tell()
        length = end_pos - start_pos
        data_uri = self._current_data_file_name()

        if end_pos > self.data_file_max_size:
            self.new_data_file()

        return data_uri, start_pos, length

    def new_data_file(self):
        """
        Create a new empty database data file on disk, and update the database's
        state to write to it.
        """
        self.n_data_files += 1
        f = open(self._current_data_file_full(), "a+b")
        self.data_files[self._current_data_file_name()] = f
