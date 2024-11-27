import json

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


class Config:
    def __init__(
        self,
        nside: int = 32,
        data_file_max_size: int = int(1e9),
        build_version: str = __version__,
    ):
        """
        Precovery Database Configuration

        Parameters
        ----------
        nside : int
            Observations are indexed in HealpixFrames for each unique mjd and observatory_code.
            The nside parameter sets the size of the healpixelization.
        data_file_max_size : int
            Maximum size in bytes of the binary files to which the indexed observations are
            saved.
        """
        self.build_version = build_version
        self.nside = nside
        self.data_file_max_size = data_file_max_size

        return

    def to_json(self, out_file):
        """
        Save Config as .json.

        Parameters
        ----------
        out_file : str
            Desired path and name of file to which to save config.
        """
        with open(out_file, "w", encoding="utf-8") as file:
            json.dump(self.__dict__, file, ensure_ascii=False, indent=4)
        return

    @classmethod
    def from_json(cls, in_file: str) -> "Config":
        """
        Load Config from .json.

        Parameters
        ----------
        in_file : str
            Desired path and name of file from which to load config.
        """
        with open(in_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


DefaultConfig = Config()
