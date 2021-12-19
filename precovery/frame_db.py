import dataclasses
import glob
import itertools
import logging
import os
import struct
from typing import Iterable, Iterator, Optional, Set, Tuple

import sqlalchemy as sq
from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn
from sqlalchemy.sql import func as sqlfunc

from . import sourcecatalog
from .orbit import Ephemeris
from .spherical_geom import haversine_distance_deg

# ra, dec, ra_sigma, dec_sigma, id
DATA_LAYOUT = "<ddddl"

logger = logging.getLogger("frame_db")

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
    start_epoch: float
    end_epoch: float
    healpixel: int
    n_frames: int

    def epoch_midpoint(self) -> float:
        return (self.start_epoch + self.end_epoch) / 2.0


@dataclasses.dataclass
class FrameWindow:
    start_epoch: float
    end_epoch: float
    n_frames: int

    def epoch_midpoint(self) -> float:
        return (self.start_epoch + self.end_epoch) / 2.0


@dataclasses.dataclass
class Observation:
    ra: float
    dec: float
    ra_sigma: float
    dec_sigma: float
    id: bytes

    data_layout = struct.Struct(DATA_LAYOUT)
    datagram_size = struct.calcsize(DATA_LAYOUT)

    def to_bytes(self) -> bytes:
        prefix = self.data_layout.pack(self.ra, self.dec, self.ra_sigma, self.dec_sigma, len(self.id))
        return prefix + self.id

    @classmethod
    def from_srcobs(cls, so: sourcecatalog.SourceObservation):
        """
        Cast a SourceObservation to an Observation.
        """
        return cls(ra=so.ra, dec=so.dec, ra_sigma=so.ra_sigma, dec_sigma=so.dec_sigma, id=so.id)

    def matches(self, ephem: Ephemeris, tolerance: float) -> bool:
        distance = haversine_distance_deg(self.ra, ephem.ra, self.dec, ephem.dec)
        logger.debug(
            "%.4f, %.4f -> %.4f, %.4f = %.6f\t(tol=%.6f)",
            self.ra,
            self.dec,
            ephem.ra,
            ephem.dec,
            distance,
            tolerance,
        )
        return distance < tolerance


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

    def window_centers(
        self, start_mjd: float, end_mjd: float, window_size_days: int
    ) -> Iterator[Tuple[float, str]]:
        """
        Return the midpoint and obscode of all time windows with data in them.
        """
        offset = -start_mjd + window_size_days / 2

        # select distinct
        #     (cast(mjd - first + (window_size_days / 2) as int) / windows_size_days)
        #     * window_size_days + first as common_epoch
        # from frames;
        stmt = (
            sq.select(
                (
                    (
                        sq.cast(
                            self.frames.c.mjd + offset,
                            sq.Integer,
                        )
                        / window_size_days
                    )
                    * window_size_days
                    + start_mjd
                ).label("common_epoch"),
                self.frames.c.obscode,
            )
            .distinct()
            .where(
                self.frames.c.mjd >= start_mjd,
                self.frames.c.mjd <= end_mjd,
            )
            .order_by("common_epoch")
        )
        rows = self.dbconn.execute(stmt)
        for mjd, obscode in rows:
            yield (mjd, obscode)

    def propagation_targets(
        self, start_mjd: float, end_mjd: float, obscode: str
    ) -> Iterator[Tuple[float, Set[int]]]:
        """
        Yields (mjd, {healpixels}) pairs for the given obscode in the given range
        of MJDs.

        The yielded mjd is a float MJD epoch timestamp to propagate to, and
        {healpixels} is a set of integer healpixel IDs.
        """
        select_stmt = (
            sq.select(
                self.frames.c.mjd,
                self.frames.c.healpixel,
            )
            .where(
                self.frames.c.obscode == obscode,
                self.frames.c.mjd >= start_mjd,
                self.frames.c.mjd <= end_mjd,
            )
            .distinct()
            .order_by(
                self.frames.c.mjd.desc(),
                self.frames.c.healpixel,
            )
        )

        rows = self.dbconn.execute(select_stmt)
        for mjd, group in itertools.groupby(rows, key=lambda pair: pair[0]):
            healpixels = set(pixel for mjd, pixel in group)
            yield (mjd, healpixels)

    def get_frames(
        self, obscode: str, mjd: float, healpixel: int
    ) -> Iterator[HealpixFrame]:
        """
        Yield all the frames which are for given obscode, MJD, healpix.
        """
        select_stmt = sq.select(
            self.frames.c.id,
            self.frames.c.obscode,
            self.frames.c.catalog_id,
            self.frames.c.mjd,
            self.frames.c.healpixel,
            self.frames.c.data_uri,
            self.frames.c.data_offset,
            self.frames.c.data_length,
        ).where(
            self.frames.c.obscode == obscode,
            self.frames.c.healpixel == int(healpixel),
            self.frames.c.mjd >= mjd - 0.001,
            self.frames.c.mjd <= mjd + 0.001,
        )
        rows = self.dbconn.execute(select_stmt)
        for r in rows:
            yield HealpixFrame(*r)

    def n_frames(self) -> int:
        select_stmt = sq.select(sqlfunc.count(self.frames.c.id))
        row = self.dbconn.execute(select_stmt).fetchone()
        return row[0]

    def n_bytes(self) -> int:
        select_stmt = sq.select(sqlfunc.sum(self.frames.c.data_length))
        row = self.dbconn.execute(select_stmt).fetchone()
        return row[0]

    def n_unique_frames(self) -> int:
        """
        Count the number of unique (obscode, mjd, healpixel) triples in the index.

        This is not the same as the number of total frames, because those
        triples might have multiple data URIs and offsets.
        """
        subq = (
            sq.select(self.frames.c.obscode, self.frames.c.mjd, self.frames.c.healpixel)
            .distinct()
            .subquery()
        )
        stmt = sq.select(sqlfunc.count(subq.c.obscode))
        row = self.dbconn.execute(stmt).fetchone()
        return row[0]

    def frames_for_bundle(
        self, bundle: FrameBundleDescription
    ) -> Iterator[HealpixFrame]:
        """
        Yields frames which match a particular bundle.
        """
        select_stmt = (
            sq.select(self.frames)
            .where(
                self.frames.c.mjd >= bundle.start_epoch,
                self.frames.c.mjd <= bundle.end_epoch,
            )
            .order_by(self.frames.c.mjd.desc())
        )
        rows = self.dbconn.execute(select_stmt)
        for row in rows:
            yield HealpixFrame(*row)

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

    def frame_bundles(
        self, window_size_days: int, mjd_start: float, mjd_end: float
    ) -> Iterator[FrameBundleDescription]:
        """
        Returns an iterator which yields descriptions of bundles of frames with
        a common epoch between start and end (inclusive).
        """
        first, _ = self.mjd_bounds()
        offset = -first + window_size_days / 2
        # select
        #    obscode,
        #    (cast(mjd - first + (window_size_days / 2) as int) / windows_size_days)
        #     * window_size_days + first as common_epoch
        # from frames;
        subq = (
            sq.select(
                self.frames.c.obscode,
                self.frames.c.healpixel,
                self.frames.c.mjd,
                (
                    (
                        sq.cast(
                            self.frames.c.mjd + offset,
                            sq.Integer,
                        )
                        / window_size_days
                    )
                    * window_size_days
                    + first
                ).label("common_epoch"),
            )
            .where(
                self.frames.c.mjd >= mjd_start,
                self.frames.c.mjd <= mjd_end,
            )
            .subquery()
        )
        select_stmt = (
            sq.select(
                subq.c.common_epoch,
                subq.c.obscode,
                sqlfunc.min(subq.c.mjd).label("start_epoch"),
                sqlfunc.max(subq.c.mjd).label("end_epoch"),
                subq.c.healpixel,
                sqlfunc.count(1).label("n_frames"),
            )
            .group_by(
                subq.c.obscode,
                subq.c.common_epoch,
                subq.c.healpixel,
            )
            .order_by(
                subq.c.common_epoch.desc(),
                subq.c.obscode,
                subq.c.healpixel,
            )
        )
        logger.debug("executing query: %s", select_stmt)
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
    def __init__(
        self,
        idx: FrameIndex,
        data_root: str,
        data_file_max_size=1e9,
        healpix_nside: int = 32,
    ):
        self.idx = idx
        self.data_root = data_root
        self.data_file_max_size = data_file_max_size
        self.data_files: dict = {}  # basename -> open file
        self._open_data_files()
        self.n_data_files = 0
        self.healpix_nside = healpix_nside

    def close(self):
        self.idx.close()
        for f in self.data_files.values():
            f.close()

    def load_hdf5(
            self,
            hdf5_file: str,
            skip: int = 0,
            limit: Optional[int] = None,
            key: str = "data",
            chunksize: int = 100000
        ):
        """
        Load data from an NSC HDF5 catalog file.

        hdf5_file: Path to a file on disk.
        skip: Number of frames to skip in the file.
        limit: Maximum number of frames to load from the file. None means no limit.
        key: Name of observations table in the hdf5 file.
        chunksize: Load observations in chunks of this size and then iterate over the chunks
            to load observations.
        """
        for src_frame in sourcecatalog.iterate_frames(
            hdf5_file,
            limit,
            nside=self.healpix_nside,
            skip=skip,
            key=key,
            chunksize=chunksize
        ):
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
        data_layout = struct.Struct(DATA_LAYOUT)
        datagram_size = struct.calcsize(DATA_LAYOUT)
        bytes_read = 0
        while bytes_read < exp.data_length:
            raw = f.read(datagram_size)
            ra, dec, ra_sigma, dec_sigma, id_size = data_layout.unpack(raw)
            id = f.read(id_size)
            bytes_read += datagram_size + id_size
            yield Observation(ra, dec, ra_sigma, dec_sigma, id)

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

        n_bytes = self.idx.n_bytes()
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.completed} / {task.total}  {task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            read_frames = progress.add_task(
                "loading frames...",
                total=self.idx.n_frames(),
            )
            written_frames = progress.add_task(
                "writing frames...",
                total=self.idx.n_unique_frames(),
            )
            bytes_scanned = progress.add_task(
                "reading bytes...",
                total=n_bytes,
            )
            bytes_migrated = progress.add_task(
                "migrating bytes...",
                total=n_bytes,
            )
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
                    progress.update(written_frames, advance=1)
                    progress.update(bytes_migrated, advance=length)

                last_catalog_id = frame.catalog_id
                progress.update(bytes_scanned, advance=frame.data_length)
                progress.update(read_frames, advance=1)

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
            progress.update(written_frames, advance=1)
            progress.update(bytes_migrated, advance=length)
