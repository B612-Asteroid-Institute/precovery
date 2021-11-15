import logging

import numpy as np
import pandas as pd

from precovery import orbit, precovery_db

# Load some orbits.
df = pd.read_csv("mpc_orbits.csv")
orbits = [
    orbit.Orbit(i, np.array([row[1:].values], dtype=np.float64, order="F"))
    for i, row in df.iterrows()
]

# Load the database.
db = precovery_db.PrecoveryDatabase.from_dir("db")

# Check for precovery matches for each orbit.
# for o in orbits:
#     matches = [m for m in db.precover(o, max_matches=3)]
#     print(o, len(matches), matches)


# do that again, but with debug logging

precovery_db.logger.setLevel(logging.DEBUG)
list(db.precover(orbits[0]))
