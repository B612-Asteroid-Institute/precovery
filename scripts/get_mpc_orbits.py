from thor.backend import PYOORB
from thor.orbits import Orbits


def load_orbits():
    po = PYOORB()
    orbits = Orbits.from_csv("~/code/b612/thor/thor/testing/data/orbits.csv")
    pyoorb_orbits = po._configureOrbits(
        orbits.cartesian, orbits.epochs.tt.mjd, "cartesian", "TT", orbits.H, orbits.G
    )
    return pyoorb_orbits
