import dataclasses
import glob
import logging
import os
import struct
import warnings
from typing import Iterator, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import sqlalchemy as sq
from adam_core.time import Timestamp
from astropy.time import Time
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Row
from sqlalchemy.sql import func as sqlfunc
from sqlalchemy.sql.expression import ColumnClause

from . import healpix_geom, sourcecatalog
from .observation import DATA_LAYOUT, ObservationsTable

logger = logging.getLogger("frame_db")


class HealpixFrame(qv.Table):
    """
    Represents a HealpixFrame in a Quivr table.
    """

    id = qv.Int64Column(nullable=True)
    dataset_id = qv.LargeStringColumn()
    obscode = qv.LargeStringColumn()
    exposure_id = qv.LargeStringColumn()
    filter = qv.LargeStringColumn()
    exposure_mjd_start = qv.Float64Column()
    exposure_mjd_mid = qv.Float64Column()
    exposure_duration = qv.Float64Column()
    healpixel = qv.Int64Column()
    data_uri = qv.LargeStringColumn()
    data_offset = qv.Int64Column()
    data_length = qv.Int64Column()

    def exposure_mid_timestamp(self) -> Timestamp:
        return Timestamp.from_mjd(self.exposure_mjd_mid, scale="utc")


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


class GenericFrame(qv.Table):
    """
    Represents combinations of healpixels and times.

    Real frames refer to intersections of exposure data and healpixels.
    Generic frames represents the idea of a healpixel at a particular time
    Used for representing propagation targets and querying the frames table.

    The real frames table may have multiple frames for a single time/healpixel
    combination, especially since these aren't limited by obscode.
    """

    time = Timestamp.as_column()
    healpixel = qv.Int64Column()


class WindowCenters(qv.Table):
    """
    Represents the centers of time windows with data in them.

    This is used for propagating to the centers of time windows with data in them.
    """

    obscode = qv.LargeStringColumn()
    time = Timestamp.as_column()
    window_size_days = qv.Int16Column()

    def window_start(self):
        return self.time.add_fractional_days(
            pc.negate(pc.divide(pc.cast(self.window_size_days, pa.float64()), 2.0))
        )

    def window_end(self):
        return self.time.add_fractional_days(
            pc.divide(pc.cast(self.window_size_days, pa.float64()), 2)
        )


class FrameIndex:
    def __init__(self, db_uri: str, mode: str = "r"):
        self.db_uri = db_uri
        self.mode = mode

        if self.mode not in {"r", "w"}:
            raise ValueError(f"mode {self.mode} must be one of {{'r', 'w'}}")

        if self.db_uri.startswith("sqlite:///") and (self.mode == "r"):
            if not self.db_uri.endswith("?mode=ro"):
                self.db_uri += "?mode=ro"

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
    ) -> WindowCenters:
        """Return the midpoint and obscode of all time windows with data in them."""

        # Build base query with minimal columns
        query = sq.select(
            self.frames.c.obscode,
            self.frames.c.exposure_mjd_mid,
        ).where(
            (self.frames.c.exposure_mjd_mid < end_mjd)
            & (self.frames.c.exposure_mjd_mid >= start_mjd)
        )

        if datasets is not None:
            query = query.where(self.frames.c.dataset_id.in_(list(datasets)))

        # Execute query and fetch results in chunks
        chunk_size = 100000
        obscodes: List[str] = []
        mjds: List[float] = []
        result = self.dbconn.execution_options(stream_results=True).execute(query)

        while True:
            chunk: Sequence[Row[Tuple[str, float]]] = result.fetchmany(chunk_size)
            if not chunk:
                break
            # Unzip the chunk directly into the lists
            chunk_obscodes, chunk_mjds = zip(*chunk)
            obscodes.extend(chunk_obscodes)
            mjds.extend(chunk_mjds)

        if not mjds:  # or `if not mjds:` - they'll have the same length
            return WindowCenters.empty()

        # Process results using PyArrow for better performance
        mjds_arr = pa.array(mjds)
        window_ids = pc.floor(
            pc.divide(
                pc.subtract(mjds_arr, pa.scalar(start_mjd)), pa.scalar(window_size_days)
            )
        )

        # Group by obscode and window_id using PyArrow
        unique_pairs = set((obs, wid.as_py()) for obs, wid in zip(obscodes, window_ids))

        if not unique_pairs:
            return WindowCenters.empty()

        final_obscodes, final_window_ids = zip(*unique_pairs)

        # Calculate window centers
        window_starts = [start_mjd + wid * window_size_days for wid in final_window_ids]
        window_center_mjds = [ws + window_size_days / 2 for ws in window_starts]

        # Return as WindowCenters
        window_centers: WindowCenters = WindowCenters.from_kwargs(
            obscode=final_obscodes,
            time=Timestamp.from_mjd(window_center_mjds, scale="utc"),
            window_size_days=pa.repeat(window_size_days, len(window_center_mjds)),
        ).sort_by(["time.days", "time.nanos"])
        return window_centers

    def propagation_targets(
        self,
        window: WindowCenters,
        datasets: Optional[set[str]] = None,
    ) -> GenericFrame:
        """
        Return the healpixels that were observed at least once by the given
        observatory in the given time window.

        Parameters
        ----------
        window : WindowCenters
            The time window and observatory to consider.
        datasets: set[str], optional
            If provided, only consider frames from the given datasets.

        Returns
        -------
        GenericFrame
            A table of healpixels and times.

        """
        logger.debug(
            f"Selecting propagation targets for {window.obscode[0].as_py()} at {window.time.mjd()[0].as_py()}"
        )
        assert len(window) == 1
        obscode = window.obscode[0].as_py()
        start_mjd = window.window_start().mjd()[0].as_py()
        end_mjd = window.window_end().mjd()[0].as_py()
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
        if datasets is not None and len(datasets) > 0:
            select_stmt = select_stmt.where(
                self.frames.c.dataset_id.in_(list(datasets))
            )

        rows = self.dbconn.execute(select_stmt).fetchall()
        if len(rows) == 0:
            return GenericFrame.empty()
        # turn rows into columns
        times, healpixels = zip(*rows)
        return GenericFrame.from_kwargs(
            time=Timestamp.from_mjd(times, scale="utc"),
            healpixel=healpixels,
        )

    def get_frames(
        self,
        obscode: str,
        mjd: float,
        healpixel: int,
        datasets: Optional[set[str]] = None,
    ) -> HealpixFrame:
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

        (
            ids,
            dataset_ids,
            obscodes,
            exposure_ids,
            filters,
            exposure_mjd_starts,
            exposure_mjd_mids,
            exposure_durations,
            healpixels,
            data_uris,
            data_offsets,
            data_lengths,
        ) = zip(*result)

        healpixel_frames = HealpixFrame.from_kwargs(
            id=ids,
            dataset_id=dataset_ids,
            obscode=obscodes,
            exposure_id=exposure_ids,
            filter=filters,
            exposure_mjd_start=exposure_mjd_starts,
            exposure_mjd_mid=exposure_mjd_mids,
            exposure_duration=exposure_durations,
            healpixel=healpixels,
            data_uri=data_uris,
            data_offset=data_offsets,
            data_length=data_lengths,
        )

        if len(pc.unique(healpixel_frames.exposure_mjd_mid)) > 1:

            logger.warning(
                f"Query returned non-unique MJDs for mjd: {mjd}, healpix:"
                f" {int(healpixel)}, obscode: {obscode}."
            )

        return healpixel_frames

    def get_frames_by_id(
        self,
        ids: List[int],
    ) -> HealpixFrame:
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

        (
            frame_ids,
            dataset_ids,
            obscodes,
            exposure_ids,
            filters,
            exposure_mjd_starts,
            exposure_mjd_mids,
            exposure_durations,
            healpixels,
            data_uris,
            data_offsets,
            data_lengths,
        ) = zip(*result)

        return HealpixFrame.from_kwargs(
            id=frame_ids,
            dataset_id=dataset_ids,
            obscode=obscodes,
            exposure_id=exposure_ids,
            filter=filters,
            exposure_mjd_start=exposure_mjd_starts,
            exposure_mjd_mid=exposure_mjd_mids,
            exposure_duration=exposure_durations,
            healpixel=healpixels,
            data_uri=data_uris,
            data_offset=data_offsets,
            data_length=data_lengths,
        )

    def n_frames(self) -> int:
        select_stmt = sq.select(sqlfunc.count(self.frames.c.id))
        row = self.dbconn.execute(select_stmt).fetchone()
        if row is None or row[0] is None:
            return 0
        return row[0]

    def n_bytes(self) -> int:
        select_stmt = sq.select(sqlfunc.sum(self.frames.c.data_length))
        row = self.dbconn.execute(select_stmt).fetchone()
        if row is None or row[0] is None:
            return 0
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
        if row is None or row[0] is None:
            return 0
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
        bounds = self.dbconn.execute(select_stmt).fetchone()
        if bounds is None:
            raise Exception("No frames in the index")
        first, last = bounds
        return first, last

    def all_frames(self) -> HealpixFrame:
        """
        Yields all the frames in the index.
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
        )
        result = self.dbconn.execute(stmt)

        (
            ids,
            dataset_ids,
            obscodes,
            exposure_ids,
            filters,
            exposure_mjd_starts,
            exposure_mjd_mids,
            exposure_durations,
            healpixels,
            data_uris,
            data_offsets,
            data_lengths,
        ) = zip(*result)

        return HealpixFrame.from_kwargs(
            id=ids,
            dataset_id=dataset_ids,
            obscode=obscodes,
            exposure_id=exposure_ids,
            filter=filters,
            exposure_mjd_start=exposure_mjd_starts,
            exposure_mjd_mid=exposure_mjd_mids,
            exposure_duration=exposure_durations,
            healpixel=healpixels,
            data_uri=data_uris,
            data_offset=data_offsets,
            data_length=data_lengths,
        )

    def add_frames(self, frames: HealpixFrame):
        """
        Add one or more frames to the index.
        """
        insert = sqlite_insert(self.frames)
        values = frames.table.drop_columns("id").to_pylist()
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
            # Add the window centers index
            sq.Index("window_centers_idx", "exposure_mjd_mid", "dataset_id", "obscode"),
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

    def frames_for_healpixel(self, healpixel: int, obscode: str) -> HealpixFrame:
        """
        Yields all the frames which are for a single healpixel-obscode pair.
        """
        stmt = sq.select("*").where(
            self.frames.c.healpixel == int(healpixel),
            self.frames.c.obscode == obscode,
        )
        rows = self.dbconn.execute(stmt).fetchall()
        (
            ids,
            dataset_ids,
            obscodes,
            exposure_ids,
            filters,
            exposure_mjd_starts,
            exposure_mjd_mids,
            exposure_durations,
            healpixels,
            data_uris,
            data_offsets,
            data_lengths,
        ) = zip(*rows)

        return HealpixFrame.from_kwargs(
            id=ids,
            dataset_id=dataset_ids,
            obscode=obscodes,
            exposure_id=exposure_ids,
            filter=filters,
            exposure_mjd_start=exposure_mjd_starts,
            exposure_mjd_mid=exposure_mjd_mids,
            exposure_duration=exposure_durations,
            healpixel=healpixels,
            data_uri=data_uris,
            data_offset=data_offsets,
            data_length=data_lengths,
        )


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
        # self.data_files: dict = {}  # basename -> open file
        self.n_data_files: dict = (
            {}
        )  # Dictionary map of how many files for each dataset, and month
        self.mode = mode
        self._open_data_files()
        self.healpix_nside = healpix_nside

    def close(self):
        self.idx.close()

    #     for f in self.data_files.values():
    #         f.close()

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

    def add_frames(
        self, dataset_id: str, src_frames: Iterator[sourcecatalog.SourceFrame]
    ):
        """Adds many SourceFrames to the database. This includes both
        writing binary observation data and storing frames in the
        FrameIndex.

        """
        healpix_frames = HealpixFrame.empty()
        for src_frame in src_frames:
            observations: ObservationsTable = ObservationsTable.from_srcobs(
                src_frame.observations
            )
            year_month_str = self._compute_year_month_str(src_frame)

            # Write observations to disk
            data_uri, offset, length = self.store_observations(
                observations, dataset_id, year_month_str
            )
            frame = HealpixFrame.from_kwargs(
                id=[None],
                dataset_id=[dataset_id],
                obscode=[src_frame.obscode],
                exposure_id=[src_frame.exposure_id],
                filter=[src_frame.filter],
                exposure_mjd_start=[src_frame.exposure_mjd_start],
                exposure_mjd_mid=[src_frame.exposure_mjd_mid],
                exposure_duration=[src_frame.exposure_duration],
                healpixel=[src_frame.healpixel],
                data_uri=[data_uri],
                data_offset=[offset],
                data_length=[length],
            )

            healpix_frames = qv.concatenate([healpix_frames, frame])
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
            year_month_str = os.path.basename(os.path.dirname(abspath))
            dataset_id = os.path.basename(os.path.dirname(os.path.dirname(abspath)))
            if dataset_id not in self.n_data_files.keys():
                self.n_data_files[dataset_id] = {}
            if year_month_str not in self.n_data_files[dataset_id].keys():
                self.n_data_files[dataset_id][year_month_str] = 0
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
        # return self.data_files[self._current_data_file_name(dataset_id, year_month_str)]
        return self._current_data_file_name(dataset_id, year_month_str)

    def get_observations(self, exp: HealpixFrame) -> ObservationsTable:
        """
        Get the observations for a given HealpixFrame as a Quivr Table
        """
        assert len(exp) == 1
        data_uri = exp.data_uri[0].as_py()
        data_offset = exp.data_offset[0].as_py()
        data_length = exp.data_length[0].as_py()

        path = os.path.abspath(os.path.join(self.data_root, data_uri))

        with open(path, "rb") as f:
            # f = self.data_files[data_uri]
            f.seek(data_offset)
            data_layout = struct.Struct(DATA_LAYOUT)
            datagram_size = struct.calcsize(DATA_LAYOUT)
            bytes_read = 0
            observations = []
            while bytes_read < data_length:
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
                observations.append(
                    (mjd, ra, dec, ra_sigma, dec_sigma, mag, mag_sigma, id)
                )
            (mjds, ras, decs, ra_sigmas, dec_sigmas, mags, mag_sigmas, ids) = zip(
                *observations
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
        self, observations: ObservationsTable, dataset_id: str, year_month_str: str
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
        # f = None
        # try:
        #     f = self._current_data_file(dataset_id, year_month_str)
        # except KeyError as ke:  # NOQA: F841
        #     self.new_data_file(dataset_id, year_month_str)
        #     f = self._current_data_file(dataset_id, year_month_str)

        try:
            path = self._current_data_file_full(dataset_id, year_month_str)
        except KeyError as ke:  # NOQA: F841
            self.new_data_file(dataset_id, year_month_str)
            path = self._current_data_file_full(dataset_id, year_month_str)

        with open(path, "a+b") as f:
            if hasattr(observations, "__len__"):
                logger.info(f"Writing {len(observations)} observations to {f.name}")  # type: ignore
            else:
                logger.info(f"Writing stream of observations to {f.name}")

            f.seek(0, 2)  # seek to end
            start_pos = f.tell()

            for obs_bytes in observations.to_bytes():
                f.write(obs_bytes)

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
        # touch the file
        with open(current_data_file, "a+b") as f:  # NOQA: F841
            pass

        # f = open(current_data_file, "a+b")
        # self.data_files[self._current_data_file_name(dataset_id, year_month_str)] = f
        # self.data_files[self._current_data_file_name(dataset_id, year_month_str)] = f
