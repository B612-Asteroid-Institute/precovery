from typing import List

import pandas as pd

from .precovery_db import FrameCandidate, PrecoveryCandidate


def candidates_to_dataframe(candidates: List[PrecoveryCandidate]):
    """Convert a list of precovery candidates to a pandas dataframe."""
    if len(candidates) == 0:
        return pd.DataFrame()
    else:
        return pd.DataFrame([c.to_dict() for c in candidates])


def frames_to_dataframe(frames: List[FrameCandidate]):
    """Convert a list of frame candidates to a pandas dataframe."""
    if len(frames) == 0:
        return pd.DataFrame()
    else:
        return pd.DataFrame([c.to_dict() for c in frames])
