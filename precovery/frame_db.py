import dataclasses
import glob
import itertools
import logging
import os
import struct
import warnings
from typing import Iterable, Iterator, List, Optional, Set, Tuple

import sqlalchemy as sq
from astropy.time import Time
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.sql import func as sqlfunc
from sqlalchemy.sql.expression import ColumnClause

from . import healpix_geom, sourcecatalog
from .observation import DATA_LAYOUT, Observation

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
class HealpixFrameKey:
    """Represents the keys used to identify a unique
    HealpixFrame. Note that the FrameDB might actually contain
    multiple HealpixFrames with the same HealpixFrameKey, since when
    HealpixFrames are written into the database they are not
    automatically merged and compacted.
    """

    dataset_id: str
    obscode: str
    filter: str
    exposure_mjd_start: float
    exposure_mjd_mid: float
    exposure_duration: float
    healpixel: int

    @staticmethod
    def _sql_order_by_clauses(db_table: sq.Table) -> List[ColumnClause]:
        return [
            db_table.c.dataset_id,
            db_table.c.obscode,
            db_table.c.filter,
            db_table.c.exposure_mjd_start,
            db_table.c.exposure_mjd_mid,
            db_table.c.exposure_duration,
            db_table.c.healpixel,
        ]


@dataclasses.dataclass
class Dataset:
    id: str
    name: Optional[str]
    reference_doi: Optional[str]
    documentation_url: Optional[str]
    sia_url: Optional[str]


class FrameIndex:
    def __init__(self, db_uri: str, mode: str = "r"):
        self.db_uri = db_uri
        self.mode = mode

        if self.mode not in {"r", "w"}:
            raise ValueError(f"mode {self.mode} must be one of {{'r', 'w'}}")

        if self.db_uri.startswith("sqlite:///") and (self.mode == "r"):
            if not self.db_uri.endswith("?mode=ro"):
                self.db_uri += "?mode=ro"

        # future=True is required here for SQLAlchemy 2.0 API usage
        # while we migrate from 1.x up. Version 2 is incompatible with
        # dagster, so we need to actually pin to 1.x, but future=True
        # lets us use the 2.0 API in this code.
        self.db = sq.create_engine(db_uri, connect_args={"timeout": 60}, future=True)
        self.dbconn = self.db.connect()

        if self.mode == "w":
            self._create_tables()
        if self.mode == "r":
            self._load_tables()
        self._check_fast_query()

    def _load_tables(self):
        """
        Reflect the metadata and assign references to table objects
        """
        self._metadata = sq.MetaData()
        self._metadata.reflect(bind=self.db)
        self.frames = self._metadata.tables["frames"]
        self.datasets = self._metadata.tables["datasets"]

    def _check_fast_query(self):
        curs = self.dbconn.execute(
            sq.text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='frames';"
            )
        )
        table_names = [row[0] for row in curs.fetchall()]
        if "frames" in table_names:
            curs = self.dbconn.execute(
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

    def close(self):
        self.dbconn.close()

    def window_centers(
        self,
        start_mjd: float,
        end_mjd: float,
        window_size_days: int,
        datasets: Optional[set[str]] = None,
    ) -> list[Tuple[str, list[float]]]:
        """Return the midpoint and obscode of all time windows with data in them.

        If datasets is provided, it will be applied as a filter on the
        datasets used to find data.

        """
        offset = -start_mjd + window_size_days / 2

        # Subquery to calculate common_epoch and order them
        subquery = (
            sq.select(
                self.frames.c.obscode,
                sq.cast(
                    (
                        sq.cast(
                            self.frames.c.exposure_mjd_mid + offset,
                            sq.Integer,
                        )
                        / window_size_days
                    )
                    * window_size_days
                    + start_mjd,
                    sq.Float,
                ).label("common_epoch"),
            )
            .where(
                self.frames.c.exposure_mjd_mid >= start_mjd,
                self.frames.c.exposure_mjd_mid <= end_mjd,
            )
            .distinct()
            .order_by("common_epoch")
        )

        if datasets is not None:
            subquery = subquery.where(self.frames.c.dataset_id.in_(list(datasets)))

        subquery = subquery.alias("subquery")

        # Main query to concatenate common_epoch values into a comma-separated string
        stmt = sq.select(
            subquery.c.obscode,
            sqlfunc.group_concat(subquery.c.common_epoch, ",").label(
                "common_epoch_times"
            ),
        ).group_by(subquery.c.obscode)

        windows_by_obscode = []
        rows = self.dbconn.execute(stmt).fetchall()
        for row in rows:
            common_epochs = list(map(float, row.common_epoch_times.split(",")))
            windows_by_obscode.append((row.obscode, common_epochs))

        return windows_by_obscode

    def propagation_targets(
        self,
        start_mjd: float,
        end_mjd: float,
        obscode: str,
        datasets: Optional[set[str]] = None,
    ) -> Iterator[Tuple[float, Set[int]]]:
        """Yields (mjd, {healpixels}) pairs for the given obscode in the given range
        of MJDs.

        The yielded mjd is a float MJD epoch timestamp to propagate to, and
        {healpixels} is a set of integer healpixel IDs.

        An optional set of dataset IDs can be provided to filter the
        data that gets scanned.

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
        if datasets is not None:
            select_stmt = select_stmt.where(
                self.frames.c.dataset_id.in_(list(datasets))
            )

        rows = self.dbconn.execute(select_stmt)
        for mjd, group in itertools.groupby(rows, key=lambda pair: pair[0]):
            healpixels = set(pixel for mjd, pixel in group)
            yield (mjd, healpixels)

    def get_frames(
        self,
        obscode: str,
        mjd: float,
        healpixel: int,
        datasets: Optional[set[str]] = None,
    ) -> list[HealpixFrame]:
        """
        Yield all the frames which are for given obscode, MJD, healpix.

        MJDs are checked to within +- 1e-7 days or 8.64 ms. Any frames that
        are within 8.64 ms of the given mjd will be returned. This does not garauntee
        that they will represent the desired exposure time and may lead to multiple
        matches computed at the wrong observation time.

        An optional set of dataset IDs can be provided to filter the
        data that gets scanned.
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
        if datasets is not None:
            select_stmt = select_stmt.where(
                self.frames.c.dataset_id.in_(list(datasets))
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

        return [HealpixFrame(*r) for r in rows]

    def get_frames_by_id(
        self,
        ids: List[int],
    ) -> Iterator[HealpixFrame]:
        """
        Yield all the frames with the given IDs. Frames are returned in the order
        they are given in the list.
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
            self.frames.c.id.in_(ids),
        )

        result = self.dbconn.execute(select_stmt)
        rows = list(result)
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

    def mjd_bounds(self, datasets: Optional[set[str]] = None) -> Tuple[float, float]:
        """
        Returns the minimum and maximum exposure_mjd_mid of all frames in the index.

        Parameters
        ----------
        datasets : set[str], optional
            If provided, only consider frames from the given datasets.
        """
        select_stmt = sq.select(
            sqlfunc.min(self.frames.c.exposure_mjd_mid, type=sq.Float),
            sqlfunc.max(self.frames.c.exposure_mjd_mid, type=sq.Float),
        )
        if datasets is not None:
            select_stmt = select_stmt.where(
                self.frames.c.dataset_id.in_(list(datasets))
            )
        first, last = self.dbconn.execute(select_stmt).fetchone()
        return first, last

    def all_frames_by_key(self) -> Iterator[Tuple[HealpixFrameKey, List[HealpixFrame]]]:
        """Returns all frames in the index, sorted according to the
        HealpixFrameKey class's instructions.

        Returned frames are grouped by HealpixFrameKey.

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
            *HealpixFrameKey._sql_order_by_clauses(self.frames),
        )
        result = self.dbconn.execute(stmt)
        cur_key: Optional[HealpixFrameKey] = None
        cur_frames: List[HealpixFrame] = []

        def frame_for_row(row: sq.engine.Row) -> HealpixFrame:
            # Little helper to cast the results of the above query
            return HealpixFrame(
                id=None,
                dataset_id=row.dataset_id,
                obscode=row.obscode,
                filter=row.filter,
                exposure_mjd_start=row.exposure_mjd_start,
                exposure_mjd_mid=row.exposure_mjd_mid,
                exposure_duration=row.exposure_duration,
                healpixel=row.healpixel,
                exposure_id=row.exposure_id,
                data_uri=row.data_uri,
                data_offset=row.data_offset,
                data_length=row.data_length,
            )

        for row in result:
            key = HealpixFrameKey(
                dataset_id=row.dataset_id,
                obscode=row.obscode,
                filter=row.filter,
                exposure_mjd_start=row.exposure_mjd_start,
                exposure_mjd_mid=row.exposure_mjd_mid,
                exposure_duration=row.exposure_duration,
                healpixel=row.healpixel,
            )
            if cur_key is None:
                # First iteration
                cur_key = key
                cur_frames = [frame_for_row(row)]
            elif key == cur_key:
                # Continuing a previous key
                cur_frames.append(frame_for_row(row))
            else:
                # New key - flush what we have and reset
                assert cur_key is not None
                yield cur_key, cur_frames
                cur_key = key
                cur_frames = [frame_for_row(row)]
        # End of iteration - yield the last pair
        if cur_key is not None:
            yield cur_key, cur_frames

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
        if len(values) > 0:
            self.dbconn.execute(insert, values)
            self.dbconn.commit()
        else:
            logger.warning("No frames to add")

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
        self.dbconn.commit()

    def get_dataset_ids(self):
        unique_datasets_stmt = sq.select(
            self.datasets.c.id,
        ).distinct()
        dataset_ids = list(self.dbconn.execute(unique_datasets_stmt))
        dataset_ids = {dataset_id[0] for dataset_id in dataset_ids}
        return dataset_ids

    def _create_tables(self):
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
            sq.Column("dataset_id", sq.String, nullable=False),
            sq.Column("obscode", sq.String, nullable=False),
            sq.Column("exposure_id", sq.String, nullable=False),
            sq.Column("filter", sq.String),
            sq.Column("exposure_mjd_start", sq.Float, nullable=False),
            sq.Column("exposure_mjd_mid", sq.Float, nullable=False),
            sq.Column("exposure_duration", sq.Float, nullable=False),
            sq.Column("healpixel", sq.Integer, nullable=False),
            sq.Column("data_uri", sq.String, nullable=False),
            sq.Column("data_offset", sq.Integer, nullable=False),
            sq.Column("data_length", sq.Integer, nullable=False),
            # Create index on midpoint mjd, healpixel, obscode
            sq.Index("fast_query", "exposure_mjd_mid", "healpixel", "obscode"),
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

    def frames_for_healpixel(
        self, healpixel: int, obscode: str
    ) -> Iterator[HealpixFrame]:
        """
        Yields all the frames which are for a single healpixel-obscode pair.
        """
        stmt = sq.select("*").where(
            self.frames.c.healpixel == int(healpixel),
            self.frames.c.obscode == obscode,
        )
        rows = self.dbconn.execute(stmt).fetchall()
        for row in rows:
            yield HealpixFrame(*row)


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
        frames = sourcecatalog.frames_from_csv_file(
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

    def get_frames_for_ra_dec(self, ra: float, dec: float, obscode: str):
        """Yields all frames that overlap given ra, dec for a given
        obscode, using nside for the healpix resolution.

        """
        logger.debug(
            f"checking frames for ra={ra} dec={dec} obscode={obscode} nside={self.healpix_nside}"
        )
        healpixel = healpix_geom.radec_to_healpixel(ra, dec, self.healpix_nside)
        for frame in self.idx.frames_for_healpixel(healpixel, obscode):
            yield frame

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
            logger.info(f"Writing {len(observations)} observations to {f.name}")  # type: ignore
        else:
            logger.info(f"Writing stream of observations to {f.name}")

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
        self.data_files[self._current_data_file_name(dataset_id, year_month_str)] = f

    def defragment(self, new_index: FrameIndex, new_db: "FrameDB"):
        """Deduplicates HealpixFrames by writing a new database.

        Iterates over all frames in the index, grouped by a common key
        (see the HealpixFrameKey for the grouping).

        For each frame key, all associated observations are gathered
        and held in memory, and then reserialized to disk. A new
        HealpixFrame is then created and inserted into the index,
        pointing to the new data serialization.

        This function does not modify the current FrameDB. It works by
        writing to a separate (presumably empty) FrameDB and
        FrameIndex.
        """
        for key, frames in self.idx.all_frames_by_key():
            year_month_str = "-".join(
                Time(key.exposure_mjd_mid, format="mjd", scale="utc").isot.split("-")[
                    :2
                ]
            )
            observations: List[Observation] = []
            for frame in frames:
                observations.extend(self.iterate_observations(frame))
                last_exposure_id = frame.exposure_id

            # Re-store the observations now that we have all of them for a common HealpixFrameKey
            data_uri, offset, length = new_db.store_observations(
                observations, frame.dataset_id, year_month_str
            )
            new_frame = HealpixFrame(
                id=None,
                # Fields common to all frames:
                dataset_id=key.dataset_id,
                obscode=key.obscode,
                filter=key.filter,
                exposure_mjd_start=key.exposure_mjd_start,
                exposure_mjd_mid=key.exposure_mjd_mid,
                exposure_duration=key.exposure_duration,
                healpixel=key.healpixel,
                # Pick one exposure_id - the last will suffice
                exposure_id=last_exposure_id,
                # Use the new data locators
                data_uri=data_uri,
                data_offset=offset,
                data_length=length,
            )
            new_db.idx.add_frames([new_frame])
