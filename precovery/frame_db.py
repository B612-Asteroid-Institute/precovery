import dataclasses
import glob
import os
import struct
from typing import Iterable, Iterator, List, Optional, Tuple

import sqlalchemy as sq
from sqlalchemy.sql import func as sqlfunc

from . import sourcecatalog


@dataclasses.dataclass
class HealpixFrame:
    id: Optional[int]
    obscode: str
    catalog_id: str
    mjd: float
    healpixel: int
    data_uri: str
    data_offset: int
    data_length: int


@dataclasses.dataclass
class FrameBundleDescription:
    obscode: str
    common_epoch: float
    healpixel: int
    n_frames: int


@dataclasses.dataclass
class Observation:
    ra: float
    dec: float
    id: bytes

    data_layout = struct.Struct("<ddl")
    datagram_size = struct.calcsize("<ddl")

    def to_bytes(self) -> bytes:
        prefix = self.data_layout.pack(self.ra, self.dec, len(self.id))
        return prefix + self.id

    @classmethod
    def from_srcobs(cls, so: sourcecatalog.SourceObservation):
        """
        Cast a SourceObservation to an Observation.
        """
        return cls(ra=so.ra, dec=so.dec, id=so.id)


class FrameIndex:
    def __init__(self, db_engine):
        self.db = db_engine
        self.dbconn = self.db.connect()
        self.initialize_tables()

    @classmethod
    def open(cls, db_uri):
        engine = sq.create_engine(db_uri)
        return cls(engine)

    def close(self):
        self.dbconn.close()

    def frame_bundles(
        self, window_size_days: int
    ) -> Iterator[Tuple[float, List[HealpixFrame]]]:
        """
        Returns a generator which yields bundles of frames along with a time
        (mjd float) which represents the central epoch for the bundle.

        Each bundle of frames will be nearby in time. They will be within
        window_size_days of each other.
        """
        first, last = self.mjd_bounds()

        window_start = first
        window_center = first + window_size_days / 2
        window_end = first + window_size_days

        while window_start < last:
            select_stmt = sq.select(self.frames).where(
                self.frames.c.mjd >= window_start, self.frames.c.mjd < window_end
            )
            frames = self.dbconn.execute(select_stmt)
            bundle = [HealpixFrame(*frame) for frame in frames]
            yield (window_center, bundle)

            window_start += window_size_days
            window_center += window_size_days
            window_end += window_size_days

    def mjd_bounds(self) -> Tuple[float, float]:
        """
        Returns the minimum and maximum mjd of all frames in the index.
        """
        select_stmt = sq.select(
            sqlfunc.min(self.frames.c.mjd, type=sq.Float),
            sqlfunc.max(self.frames.c.mjd, type=sq.Float),
        )
        first, last = self.dbconn.execute(select_stmt).fetchone()
        return first, last

    def frame_bundle_epochs(
        self, window_size_days: int
    ) -> Iterator[FrameBundleDescription]:
        """
        Returns an iterator which yields descriptions of bundles of frames with
        a common epoch.
        """
        first, _ = self.mjd_bounds()
        offset = -first + window_size_days / 2
        # select
        #    obscode,
        #    (cast(mjd - first + (window_size_days / 2) as int) / windows_size_days)
        #     * window_size_days + first as common_epoch
        # from frames;
        subq = sq.select(
            self.frames.c.obscode,
            self.frames.c.healpixel,
            (
                sq.cast(
                    self.frames.c.mjd + offset,
                    sq.Integer,
                )
                / window_size_days
            )
            * window_size_days
            + first,
        ).subquery()
        select_stmt = (
            sq.select(
                subq.c.obscode,
                subq.c.common_epoch,
                subq.c.healpixel,
                sqlfunc.count(1).label("n_frames"),
            )
            .group_by(
                subq.c.obscode,
                subq.c.common_epoch,
                subq.c.healpixel,
            )
            .order_by(subq.c.common_epoch)
        )

        results = self.dbconn.execute(select_stmt)
        for row in results:
            yield FrameBundleDescription(*row)

    def all_frames(self) -> Iterator[HealpixFrame]:
        """
        Returns all frames in the index, sorted by obscode, mjd, and healpixel.
        """
        stmt = sq.select(
            self.frames.c.id,
            self.frames.c.obscode,
            self.frames.c.catalog_id,
            self.frames.c.mjd,
            self.frames.c.healpixel,
            self.frames.c.data_uri,
            self.frames.c.data_offset,
            self.frames.c.data_length,
        ).order_by(self.frames.c.obscode, self.frames.c.mjd, self.frames.c.healpixel)
        result = self.dbconn.execute(stmt)
        for row in result:
            yield HealpixFrame(*row)

    def add_frame(self, frame: HealpixFrame):
        insert = self.frames.insert().values(
            obscode=frame.obscode,
            catalog_id=frame.catalog_id,
            mjd=frame.mjd,
            healpixel=int(frame.healpixel),
            data_uri=frame.data_uri,
            data_offset=frame.data_offset,
            data_length=frame.data_length,
        )
        self.dbconn.execute(insert)

    def initialize_tables(self):
        self._metadata = sq.MetaData()
        self.frames = sq.Table(
            "frames",
            self._metadata,
            sq.Column(
                "id",
                sq.Integer,
                sq.Sequence("frame_id_seq"),
                primary_key=True,
            ),
            sq.Column("obscode", sq.String),
            sq.Column("catalog_id", sq.String),
            sq.Column("mjd", sq.Float),
            sq.Column("healpixel", sq.Integer),
            sq.Column("data_uri", sq.String),
            sq.Column("data_offset", sq.Integer),
            sq.Column("data_length", sq.Integer),
        )
        self._metadata.create_all(self.db)


class FrameDB:
    def __init__(self, idx: FrameIndex, data_root: str, data_file_max_size=1e9):
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
        for src_frame in sourcecatalog.iterate_frames(hdf5_file, limit):
            observations = [Observation.from_srcobs(o) for o in src_frame.observations]
            data_uri, offset, length = self.store_observations(observations)

            frame = HealpixFrame(
                id=None,
                obscode=src_frame.obscode,
                catalog_id=src_frame.exposure_id,
                mjd=src_frame.mjd,
                healpixel=src_frame.healpixel,
                data_uri=data_uri,
                data_offset=offset,
                data_length=length,
            )
            self.idx.add_frame(frame)

    def _open_data_files(self):
        matcher = os.path.join(self.data_root, "frames*.data")
        files = glob.glob(matcher)
        for f in files:
            abspath = os.path.abspath(f)
            name = os.path.basename(f)
            self.data_files[name] = open(abspath, "a+b")
        self.n_data_files = len(self.data_files) - 1
        if self.n_data_files <= 0:
            self.new_data_file()

    def _current_data_file_name(self):
        return "frames_{:08d}.data".format(self.n_data_files)

    def _current_data_file_full(self):
        return os.path.abspath(
            os.path.join(self.data_root, self._current_data_file_name())
        )

    def _current_data_file_size(self):
        return os.stat(self._current_data_file_name()).st_size

    def _current_data_file(self):
        return self.data_files[self._current_data_file_name()]

    def iterate_observations(self, exp: HealpixFrame) -> Iterator[Observation]:
        """
        Iterate over the observations stored in a frame.
        """
        f = self.data_files[exp.data_uri]
        f.seek(exp.data_offset)
        data_layout = struct.Struct("<ddl")
        datagram_size = struct.calcsize("<ddl")
        bytes_read = 0
        while bytes_read < exp.data_length:
            raw = f.read(datagram_size)
            ra, dec, id_size = data_layout.unpack(raw)
            id = f.read(id_size)
            bytes_read += datagram_size + id_size
            yield Observation(ra, dec, id)

    def store_observations(self, observations: Iterable[Observation]):
        """
        Write observations in exp_data to disk. The write goes to the latest (or
        "current") file in the database. Data is written as a sequence of
        packed RA, Dec, and ID values.

        RA and Dec are encoded as double-length floating point values. ID is
        encoded as a length-prefixed string: a long integer (little-endian)
        which indicates the ID's length in bytes, followed by the UTF8-encoded
        bytes of the ID.
        """
        f = self._current_data_file()
        f.seek(0, 2)  # seek to end
        start_pos = f.tell()

        for obs in observations:
            f.write(obs.to_bytes())

        end_pos = f.tell()
        length = end_pos - start_pos
        data_uri = self._current_data_file_name()

        f.flush()
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

    def defragment(self, new_index: FrameIndex, new_db: "FrameDB"):
        cur_key = ("", 0.0, 0)
        observations = []
        last_catalog_id = ""

        for frame in self.idx.all_frames():
            if cur_key[0] == "":
                # First iteration
                observations = list(self.iterate_observations(frame))
                cur_key = (frame.obscode, frame.mjd, frame.healpixel)
                last_catalog_id = frame.catalog_id

            elif (frame.obscode, frame.mjd, frame.healpixel) == cur_key:
                # Extending existing frame
                observations.extend(self.iterate_observations(frame))

            else:
                # On to the next key. Flush what we have and reset.
                data_uri, offset, length = new_db.store_observations(observations)
                new_frame = HealpixFrame(
                    id=None,
                    obscode=cur_key[0],
                    mjd=cur_key[1],
                    healpixel=cur_key[2],
                    catalog_id=last_catalog_id,
                    data_uri=data_uri,
                    data_offset=offset,
                    data_length=length,
                )
                new_db.idx.add_frame(new_frame)

                observations = list(self.iterate_observations(frame))
                cur_key = (frame.obscode, frame.mjd, frame.healpixel)
            last_catalog_id = frame.catalog_id

        # Last frame was unflushed, so do that now.
        data_uri, offset, length = new_db.store_observations(observations)
        frame = HealpixFrame(
            id=None,
            obscode=cur_key[0],
            mjd=cur_key[1],
            healpixel=cur_key[2],
            catalog_id=last_catalog_id,
            data_uri=data_uri,
            data_offset=offset,
            data_length=length,
        )
        new_db.idx.add_frame(frame)
