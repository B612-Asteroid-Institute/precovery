import dataclasses
import glob
import itertools
import logging
import os
import struct
import warnings
from typing import Iterable, Iterator, Optional, Set, Tuple

import sqlalchemy as sq
from astropy.time import Time
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.sql import func as sqlfunc

from . import sourcecatalog

# mjd, ra, dec, ra_sigma, dec_sigma, mag, mag_sigma, id
DATA_LAYOUT = "<dddddddl"

logger = logging.getLogger("frame_db")


@dataclasses.dataclass
class HealpixFrame:
    id: Optional[int]
    dataset_id: str
    obscode: str
    exposure_id: str
    filter: str
    exposure_mjd_start: float
    exposure_mjd_mid: float
    exposure_duration: float
    healpixel: int
    data_uri: str
    data_offset: int
    data_length: int


@dataclasses.dataclass
class Dataset:
    id: str
    name: Optional[str]
    reference_doi: Optional[str]
    documentation_url: Optional[str]
    sia_url: Optional[str]


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
    def from_srcobs(cls, so: sourcecatalog.SourceObservation):
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


class FrameIndex:
    def __init__(self, db_engine):
        self.db = db_engine
        self.dbconn = self.db.connect()
        self.initialize_tables()

    @classmethod
    def open(cls, db_uri, mode: str = "r"):
        if (mode != "r") and (mode != "w"):
            err = "mode should be one of {'r', 'w'}"
            raise ValueError(err)

        if db_uri.startswith("sqlite:///") and (mode == "r"):
            db_uri += "?mode=ro"

        engine = sq.create_engine(db_uri, connect_args={"timeout": 60})

        # Check if fast_query index exists (older databases may not have it)
        # if it doesn't throw a warning with the command to create it
        con = engine.connect()
        curs = con.execute(
            sq.text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='frames';"
            )
        )
        table_names = [row[0] for row in curs.fetchall()]
        if "frames" in table_names:
            curs = con.execute(
                sq.text("SELECT name FROM sqlite_master WHERE type = 'index';")
            )
            index_names = [row[0] for row in curs.fetchall()]
            if "fast_query" not in index_names:
                warning = (
                    "No fast_query index exists on the frames table. This may cause"
                    " significant performance issues.To create the index run the"
                    " following SQL command:\n   CREATE INDEX fast_query ON frames"
                    " (exposure_mjd_mid, healpixel, obscode);\n"
                )
                warnings.warn(warning, UserWarning)
        con.close()

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
                            self.frames.c.exposure_mjd_mid + offset,
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
                self.frames.c.exposure_mjd_mid >= start_mjd,
                self.frames.c.exposure_mjd_mid <= end_mjd,
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
                self.frames.c.exposure_mjd_mid,
                self.frames.c.healpixel,
            )
            .where(
                self.frames.c.obscode == obscode,
                self.frames.c.exposure_mjd_mid >= start_mjd,
                self.frames.c.exposure_mjd_mid <= end_mjd,
            )
            .distinct()
            .order_by(
                self.frames.c.exposure_mjd_mid.desc(),
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

        MJDs are checked to within +- 1e-7 days or 8.64 ms. Any frames that
        are within 8.64 ms of the given mjd will be returned. This does not garauntee
        that they will represent the desired exposure time and may lead to multiple
        matches computed at the wrong observation time.
        """
        select_stmt = sq.select(
            self.frames.c.id,
            self.frames.c.dataset_id,
            self.frames.c.obscode,
            self.frames.c.exposure_id,
            self.frames.c.filter,
            self.frames.c.exposure_mjd_start,
            self.frames.c.exposure_mjd_mid,
            self.frames.c.exposure_duration,
            self.frames.c.healpixel,
            self.frames.c.data_uri,
            self.frames.c.data_offset,
            self.frames.c.data_length,
        ).where(
            self.frames.c.obscode == obscode,
            self.frames.c.healpixel == int(healpixel),
            self.frames.c.exposure_mjd_mid >= mjd - 1e-7,
            self.frames.c.exposure_mjd_mid <= mjd + 1e-7,
        )
        result = self.dbconn.execute(select_stmt)
        # Turn result into a list so we can iterate over it twice: once
        # to check the MJDs for uniqueness and a second time to actually
        # yield the individual rows
        rows = list(result)

        # Loop through rows and track midpoint MJDs
        mjds_mid = set()
        for r in rows:
            # id, dataset_id, obscode, exposure_id, filter, exposure_mjd_start, exposure_mjd_mid,
            # exposure_duration, healpixel, data uri, data offset, data length
            mjds_mid.add(r[6])

        if len(mjds_mid) > 1:
            logger.warn(
                f"Query returned non-unique MJDs for mjd: {mjd}, healpix:"
                f" {int(healpixel)}, obscode: {obscode}."
            )

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
        Count the number of unique (obscode, exposure_mjd_mid, healpixel) triples in the index.

        This is not the same as the number of total frames, because those
        triples might have multiple data URIs and offsets.
        """
        subq = (
            sq.select(
                self.frames.c.obscode,
                self.frames.c.exposure_mjd_mid,
                self.frames.c.healpixel,
            )
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
                self.frames.c.exposure_mjd_mid >= bundle.start_epoch,
                self.frames.c.exposure_mjd_mid <= bundle.end_epoch,
            )
            .order_by(self.frames.c.exposure_mjd_mid.desc())
        )
        rows = self.dbconn.execute(select_stmt)
        for row in rows:
            yield HealpixFrame(*row)

    def mjd_bounds(self) -> Tuple[float, float]:
        """
        Returns the minimum and maximum exposure_mjd_mid of all frames in the index.
        """
        select_stmt = sq.select(
            sqlfunc.min(self.frames.c.exposure_mjd_mid, type=sq.Float),
            sqlfunc.max(self.frames.c.exposure_mjd_mid, type=sq.Float),
        )
        first, last = self.dbconn.execute(select_stmt).fetchone()
        if first is None or last is None:

            raise ValueError(
                "the database has no data entered, and so no minimum and maximum can be computed"
            )
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
                self.frames.c.exposure_mjd_mid,
                (
                    (
                        sq.cast(
                            self.frames.c.exposure_mjd_mid + offset,
                            sq.Integer,
                        )
                        / window_size_days
                    )
                    * window_size_days
                    + first
                ).label("common_epoch"),
            )
            .where(
                self.frames.c.exposure_mjd_mid >= mjd_start,
                self.frames.c.exposure_mjd_mid <= mjd_end,
            )
            .subquery()
        )
        select_stmt = (
            sq.select(
                subq.c.common_epoch,
                subq.c.obscode,
                sqlfunc.min(subq.c.exposure_mjd_mid).label("start_epoch"),
                sqlfunc.max(subq.c.exposure_mjd_mid).label("end_epoch"),
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
            self.frames.c.dataset_id,
            self.frames.c.obscode,
            self.frames.c.exposure_id,
            self.frames.c.filter,
            self.frames.c.exposure_mjd_start,
            self.frames.c.exposure_mjd_mid,
            self.frames.c.exposure_duration,
            self.frames.c.healpixel,
            self.frames.c.data_uri,
            self.frames.c.data_offset,
            self.frames.c.data_length,
        ).order_by(
            self.frames.c.obscode,
            self.frames.c.exposure_mjd_mid,
            self.frames.c.healpixel,
        )
        result = self.dbconn.execute(stmt)
        for row in result:
            yield HealpixFrame(*row)

    def add_frames(self, frames: list[HealpixFrame]):
        """
        Add one or more frames to the index.
        """
        insert = sqlite_insert(self.frames)
        values = [
            {
                "dataset_id": frame.dataset_id,
                "obscode": frame.obscode,
                "exposure_id": frame.exposure_id,
                "filter": frame.filter,
                "exposure_mjd_start": frame.exposure_mjd_start,
                "exposure_mjd_mid": frame.exposure_mjd_mid,
                "exposure_duration": frame.exposure_duration,
                "healpixel": int(frame.healpixel),
                "data_uri": frame.data_uri,
                "data_offset": frame.data_offset,
                "data_length": frame.data_length,
            }
            for frame in frames
        ]
        self.dbconn.execute(insert, values)

    def add_dataset(self, dataset: Dataset):
        """
        Safely UPSERTs a dataset
        """
        insert = (
            sqlite_insert(self.datasets)
            .values(
                id=dataset.id,
                name=dataset.name,
                reference_doi=dataset.reference_doi,
                documentation_url=dataset.documentation_url,
                sia_url=dataset.sia_url,
            )
            .on_conflict_do_nothing(index_elements=["id"])
        )
        self.dbconn.execute(insert)

    def get_dataset_ids(self):
        unique_datasets_stmt = sq.select(
            self.datasets.c.id,
        ).distinct()
        dataset_ids = list(self.dbconn.execute(unique_datasets_stmt))
        dataset_ids = {dataset_id[0] for dataset_id in dataset_ids}
        return dataset_ids

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
            sq.Column("dataset_id", sq.String),
            sq.Column("obscode", sq.String),
            sq.Column("exposure_id", sq.String),
            sq.Column("filter", sq.String),
            sq.Column("exposure_mjd_start", sq.Float),
            sq.Column("exposure_mjd_mid", sq.Float),
            sq.Column("exposure_duration", sq.Float),
            sq.Column("healpixel", sq.Integer),
            sq.Column("data_uri", sq.String),
            sq.Column("data_offset", sq.Integer),
            sq.Column("data_length", sq.Integer),
            # Create index on midpoint mjd, healpixel, obscode
            sq.Index(
                "fast_query", "exposure_mjd_mid", "healpixel", "obscode", unique=True
            ),
        )

        self.datasets = sq.Table(
            "datasets",
            self._metadata,
            sq.Column(
                "id",
                sq.String,
                primary_key=True,
                index=True,
            ),
            sq.Column("name", sq.String, nullable=True),
            sq.Column("reference_doi", sq.String, nullable=True),
            sq.Column("documentation_url", sq.String, nullable=True),
            sq.Column("sia_url", sq.String, nullable=True),
        )
        self._metadata.create_all(self.db)


class FrameDB:
    def __init__(
        self,
        idx: FrameIndex,
        data_root: str,
        data_file_max_size: float = 1e9,
        healpix_nside: int = 32,
        mode: str = "r",
    ):
        self.idx = idx
        self.data_root = data_root
        self.data_file_max_size = data_file_max_size
        self.data_files: dict = {}  # basename -> open file
        self.n_data_files: dict = (
            {}
        )  # Dictionary map of how many files for each dataset, and month
        self.mode = mode
        self._open_data_files()
        self.healpix_nside = healpix_nside

    def close(self):
        self.idx.close()
        for f in self.data_files.values():
            f.close()

    def add_dataset(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        reference_doi: Optional[str] = None,
        documentation_url: Optional[str] = None,
        sia_url: Optional[str] = None,
    ):
        """Add a new (empty) dataset to the database.

        If a dataset with the provided dataset_id already exists,
        nothing happens.

        Parameters
        ----------
        csv_file : str
            Path to a file on disk.
        dataset_id : str
            Name of dataset (should be the same for each observation file that
            comes from the same dataset).
        name : str, optional
            User-friendly name of the dataset.
        reference_doi : str, optional
            DOI of the reference paper for the dataset.
        documentation_url : str, optional
            URL of any documentation describing the dataset.
        sia_url : str, optional
            Simple Image Access URL for accessing images for this particular dataset.

        """
        if self.has_dataset(dataset_id):
            logger.info(
                f"{dataset_id} dataset already has an entry in the datasets table."
            )
            return

        logger.info(f"Adding new entry into datasets table for dataset {dataset_id}.")
        dataset = Dataset(
            id=dataset_id,
            name=name,
            reference_doi=reference_doi,
            documentation_url=documentation_url,
            sia_url=sia_url,
        )
        self.idx.add_dataset(dataset)

    def has_dataset(self, dataset_id: str) -> bool:
        return dataset_id in self.idx.get_dataset_ids()

    def add_frames(self, dataset_id: str, frames: Iterator[sourcecatalog.SourceFrame]):
        """Adds many SourceFrames to the database. This includes both
        writing binary observation data and storing frames in the
        FrameIndex.

        """
        healpix_frames = []
        for src_frame in frames:
            observations = [Observation.from_srcobs(o) for o in src_frame.observations]
            year_month_str = self._compute_year_month_str(src_frame)

            # Write observations to disk
            data_uri, offset, length = self.store_observations(
                observations, dataset_id, year_month_str
            )

            frame = HealpixFrame(
                id=None,
                dataset_id=dataset_id,
                obscode=src_frame.obscode,
                exposure_id=src_frame.exposure_id,
                filter=src_frame.filter,
                exposure_mjd_start=src_frame.exposure_mjd_start,
                exposure_mjd_mid=src_frame.exposure_mjd_mid,
                exposure_duration=src_frame.exposure_duration,
                healpixel=src_frame.healpixel,
                data_uri=data_uri,
                data_offset=offset,
                data_length=length,
            )

            healpix_frames.append(frame)

        self.idx.add_frames(healpix_frames)

    @staticmethod
    def _compute_year_month_str(frame: sourcecatalog.SourceFrame) -> str:
        """Gets the Year and Month parts of a SourceFrame's exposure
        midpoint timestamp, and returns them in "YYYY-MM" format.

        This is used as a partitioning key for observation data files.

        """
        time = Time(frame.exposure_mjd_mid, format="mjd", scale="utc")
        return time.strftime("%Y-%m")

    def load_csv(
        self,
        csv_file: str,
        dataset_id: str,
        skip: int = 0,
        limit: Optional[int] = None,
    ):
        """
        Load data from a CSV observation file.

        Parameters
        ----------
        csv_file : str
            Path to a file on disk.
        dataset_id : str
            Name of dataset (should be the same for each observation file that
            comes from the same dataset).
        skip : int, optional
            Number of frames to skip in the file.
        limit : int, optional
            Maximum number of frames to load from the file. None means no limit.
        """
        frames = sourcecatalog.iterate_frames(
            csv_file,
            limit,
            nside=self.healpix_nside,
            skip=skip,
        )
        self.add_frames(dataset_id, frames)

    def _open_data_files(self):
        matcher = os.path.join(self.data_root, "**/frames*.data")
        files = sorted(glob.glob(matcher, recursive=True))
        for f in files:
            abspath = os.path.abspath(f)
            name = os.path.basename(f)
            year_month_str = os.path.basename(os.path.dirname(abspath))
            dataset_id = os.path.basename(os.path.dirname(os.path.dirname(abspath)))
            data_uri = f"{dataset_id}/{year_month_str}/{name}"
            if dataset_id not in self.n_data_files.keys():
                self.n_data_files[dataset_id] = {}
            if year_month_str not in self.n_data_files[dataset_id].keys():
                self.n_data_files[dataset_id][year_month_str] = 0
            self.data_files[data_uri] = open(
                abspath, "rb" if self.mode == "r" else "a+b"
            )
            self.n_data_files[dataset_id][year_month_str] += 1

    def _current_data_file_name(self, dataset_id: str, year_month_str: str):
        return "{}/{}/frames_{:08d}.data".format(
            dataset_id, year_month_str, self.n_data_files[dataset_id][year_month_str]
        )

    def _current_data_file_full(self, dataset_id: str, year_month_str: str):
        return os.path.abspath(
            os.path.join(
                self.data_root, self._current_data_file_name(dataset_id, year_month_str)
            )
        )

    def _current_data_file_size(self, dataset_id: str, year_month_str: str):
        return os.stat(self._current_data_file_name(dataset_id, year_month_str)).st_size

    def _current_data_file(self, dataset_id: str, year_month_str: str):
        return self.data_files[self._current_data_file_name(dataset_id, year_month_str)]

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
            (
                mjd,
                ra,
                dec,
                ra_sigma,
                dec_sigma,
                mag,
                mag_sigma,
                id_size,
            ) = data_layout.unpack(raw)
            id = f.read(id_size)
            bytes_read += datagram_size + id_size
            yield Observation(mjd, ra, dec, ra_sigma, dec_sigma, mag, mag_sigma, id)

    def store_observations(
        self, observations: Iterable[Observation], dataset_id: str, year_month_str: str
    ):
        """
        Write observations in exp_data to disk. The write goes to the latest (or
        "current") file in the database. Data is written as a sequence of
        packed RA, Dec, and ID values.

        RA and Dec are encoded as double-length floating point values. ID is
        encoded as a length-prefixed string: a long integer (little-endian)
        which indicates the ID's length in bytes, followed by the UTF8-encoded
        bytes of the ID.
        """
        f = None
        try:
            f = self._current_data_file(dataset_id, year_month_str)
        except KeyError as ke:  # NOQA: F841
            self.new_data_file(dataset_id, year_month_str)
            f = self._current_data_file(dataset_id, year_month_str)

        if hasattr(observations, "__len__"):
            logger.debug(f"Writing {len(observations)} observations to {f.name}")  # type: ignore
        else:
            logger.debug(f"Writing stream of observations to {f.name}")

        f.seek(0, 2)  # seek to end
        start_pos = f.tell()

        for obs in observations:
            f.write(obs.to_bytes())

        end_pos = f.tell()
        length = end_pos - start_pos
        data_uri = self._current_data_file_name(dataset_id, year_month_str)

        f.flush()
        if end_pos > self.data_file_max_size:
            self.new_data_file(dataset_id, year_month_str)

        return data_uri, start_pos, length

    def new_data_file(self, dataset_id: str, year_month_str: str):
        """
        Create a new empty database data file on disk, and update the database's
        state to write to it.
        """
        if dataset_id not in self.n_data_files.keys():
            self.n_data_files[dataset_id] = {}
        if year_month_str not in self.n_data_files[dataset_id].keys():
            self.n_data_files[dataset_id][year_month_str] = 0
        self.n_data_files[dataset_id][year_month_str] += 1
        current_data_file = self._current_data_file_full(dataset_id, year_month_str)
        os.makedirs(os.path.dirname(current_data_file), exist_ok=True)
        f = open(current_data_file, "a+b")
        print("Opened new data file: {}".format(current_data_file))
        self.data_files[self._current_data_file_name(dataset_id, year_month_str)] = f

    def defragment(self, new_index: FrameIndex, new_db: "FrameDB"):
        # dataset, obscode, filter, exposure_mjd_start, exposure_mjd_mid, exposure_duration, healpixel
        cur_key = ("", "", "", 0.0, 0.0, 0.0, 0)
        observations = []
        last_exposure_id = ""

        for frame in self.idx.all_frames():
            year_month_str = "-".join(
                Time(frame.exposure_mjd_mid, format="mjd", scale="utc").isot.split("-")[
                    :2
                ]
            )
            if cur_key[0] == "":
                # First iteration
                observations = list(self.iterate_observations(frame))
                cur_key = (
                    frame.dataset_id,
                    frame.obscode,
                    frame.filter,
                    frame.exposure_mjd_start,
                    frame.exposure_mjd_mid,
                    frame.exposure_duration,
                    frame.healpixel,
                )
                last_exposure_id = frame.exposure_id
            elif (
                frame.dataset_id,
                frame.obscode,
                frame.filter,
                frame.exposure_mjd_start,
                frame.exposure_mjd_mid,
                frame.exposure_duration,
                frame.healpixel,
            ) == cur_key:
                # Extending existing frame
                observations.extend(self.iterate_observations(frame))

            else:
                # On to the next key. Flush what we have and reset.
                data_uri, offset, length = new_db.store_observations(
                    observations, frame.dataset_id, year_month_str
                )
                new_frame = HealpixFrame(
                    id=None,
                    dataset_id=cur_key[0],
                    obscode=cur_key[1],
                    filter=cur_key[2],
                    exposure_mjd_start=cur_key[3],
                    exposure_mjd_mid=cur_key[4],
                    exposure_duration=cur_key[5],
                    healpixel=cur_key[6],
                    exposure_id=last_exposure_id,
                    data_uri=data_uri,
                    data_offset=offset,
                    data_length=length,
                )
                new_db.idx.add_frames([new_frame])

                observations = list(self.iterate_observations(frame))
                cur_key = (
                    frame.dataset_id,
                    frame.obscode,
                    frame.filter,
                    frame.exposure_mjd_start,
                    frame.exposure_mjd_mid,
                    frame.exposure_duration,
                    frame.healpixel,
                )

            last_exposure_id = frame.exposure_id

        # Last frame was unflushed, so do that now.
        data_uri, offset, length = new_db.store_observations(
            observations, frame.dataset_id, year_month_str
        )
        frame = HealpixFrame(
            id=None,
            dataset_id=cur_key[0],
            obscode=cur_key[1],
            filter=cur_key[2],
            exposure_mjd_start=cur_key[3],
            exposure_mjd_mid=cur_key[4],
            exposure_duration=cur_key[5],
            healpixel=cur_key[6],
            exposure_id=last_exposure_id,
            data_uri=data_uri,
            data_offset=offset,
            data_length=length,
        )
        new_db.idx.add_frames([frame])
