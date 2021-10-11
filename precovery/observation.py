import numpy as np


class Observation:
    def __init__(self, ra: float, dec: float, timestamp: int, id: str):
        self.ra = ra
        self.dec = dec
        self.timestamp = timestamp
        self.id = id

        self.ra_rad = np.deg2rad(ra)
        self.dec_rad = np.deg2rad(dec)

    def __repr__(self):
        return f"Observation({self.ra}, {self.dec}, {self.timestamp}, {self.id})"
