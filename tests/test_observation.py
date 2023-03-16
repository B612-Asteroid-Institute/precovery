import numpy.testing as numpy_testing

from precovery.observation import Observation, ObservationArray

from .testutils import make_sourceobs


def test_observation_array_initialization():
    src_observations = [
        make_sourceobs(mjd=1, ra=2, dec=3),
        make_sourceobs(mjd=4, ra=5, dec=6),
        make_sourceobs(mjd=7, ra=8, dec=9),
    ]
    obs = [Observation.from_srcobs(so) for so in src_observations]
    obs_array = ObservationArray(obs)

    assert len(obs_array.values) == 3
    numpy_testing.assert_array_equal(obs_array.values["mjd"], [1, 4, 7])


def test_observation_to_list():
    src_observations = [
        make_sourceobs(mjd=1, ra=2, dec=3),
        make_sourceobs(mjd=4, ra=5, dec=6),
        make_sourceobs(mjd=7, ra=8, dec=9),
    ]
    obs = [Observation.from_srcobs(so) for so in src_observations]
    obs_array = ObservationArray(obs)

    have = obs_array.to_list()
    assert have == obs
