import numpy as np

from precovery._exposure import _Exposure
from precovery.observation import Observation


def test_exposure_cone_search():
    # Make a grid of observations from in the box [10, 12] with resolution 0.1 in both RA and DEC
    exposure_observations = []
    for ra in np.arange(10, 12, 0.1):
        for dec in np.arange(10, 12, 0.1):
            id = f"{ra}-{dec}"
            exposure_observations.append(Observation(ra, dec, 0, id))

    exposure = _Exposure(0, "obs", exposure_observations)

    # Try a cone search for everything within 0.51 of 11, 11. We should get many results.
    results = list(exposure.cone_search(11, 11, 0.51))
    assert len(results) > 10
    # ... but not everything should match.
    assert len(results) < len(exposure_observations)
    # The center should match
    assert any(np.isclose(x.ra, 11) and np.isclose(x.dec, 11) for x in results)
    # So should points 0.5 degrees off
    assert any(np.isclose(x.ra, 11.5) and np.isclose(x.dec, 11) for x in results)
    assert any(np.isclose(x.ra, 10.5) and np.isclose(x.dec, 11) for x in results)
    assert any(np.isclose(x.ra, 11) and np.isclose(x.dec, 11.5) for x in results)
    assert any(np.isclose(x.ra, 11) and np.isclose(x.dec, 10.5) for x in results)
    # But not points further off
    assert not any(np.isclose(x.ra, 11.6) and np.isclose(x.dec, 11) for x in results)

    # Try another few cone searches which should match nothing
    results = list(exposure.cone_search(8, 8, 0.01))
    assert len(results) == 0
    results = list(exposure.cone_search(12, 12, 0.01))
    assert len(results) == 0
    results = list(exposure.cone_search(-11, -11, 0.5))
    assert len(results) == 0
    results = list(exposure.cone_search(11.05, 11.05, 0.0001))
    assert len(results) == 0

    # Try a cone search which should match everything
    results = list(exposure.cone_search(11.0, 11.0, 2.5))
    assert len(results) == len(exposure_observations)
