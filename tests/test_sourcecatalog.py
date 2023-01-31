import random
from typing import Optional, Tuple

import healpy

from precovery.sourcecatalog import SourceFrame, SourceObservation, bundle_into_frames


def make_sourceobs(
    exposure_id: str = "exposure",
    id: Optional[bytes] = None,
    healpixel: Optional[int] = None,
    nside: int = 32,
    ra: float = 1.0,
    dec: float = 2.0,
) -> SourceObservation:

    if id is None:
        id = random.randbytes(16)

    if healpixel is not None:
        ra, dec = radec_for_healpixel(healpixel, nside)

    return SourceObservation(
        exposure_id=exposure_id,
        obscode="obs",
        id=id,
        mjd=50000,
        ra=ra,
        dec=dec,
        ra_sigma=3.0,
        dec_sigma=4.0,
        mag=5.0,
        mag_sigma=6.0,
        filter="filter",
        exposure_mjd_start=50000,
        exposure_mjd_mid=50000,
        exposure_duration=30,
    )


def radec_for_healpixel(healpixel: int, nside: int) -> Tuple[float, float]:
    """
    Compute the ra and dec associated with a healpixel
    """
    return healpy.pix2ang(nside=nside, ipix=healpixel, nest=True, lonlat=True)


def test_sourceframe_from_observation():
    obs = make_sourceobs()
    frame = SourceFrame.from_observation(obs, healpixel=200)

    assert frame.exposure_id == obs.exposure_id
    assert frame.obscode == obs.obscode
    assert frame.filter == obs.filter
    assert frame.exposure_mjd_start == obs.exposure_mjd_start
    assert frame.exposure_mjd_mid == obs.exposure_mjd_mid
    assert frame.exposure_duration == obs.exposure_duration
    assert frame.healpixel == 200

    assert len(frame.observations) == 0


class TestBundleObservationsIntoFrames:
    def test_empty_observation_iterator(self):
        observations = []
        frames = bundle_into_frames(observations)

        result = list(frames)
        assert len(result) == 0

    def test_single_exposure(self):
        """Bundle a single exposure"""
        observations = [make_sourceobs()]
        frames = bundle_into_frames(observations)

        result = list(frames)

        assert len(result) == 1
        frame = result[0]
        assert len(frame.observations) == 1
        assert frame.observations[0] == observations[0]

    def test_single_frame(self):
        """Bundle observations related to a single SourceFrame"""

        # Use a constant exposure ID and healpixel, so everything gets one frame.
        exp_id = "exposure"
        healpixel = 1

        observations = [
            make_sourceobs(exposure_id=exp_id, healpixel=healpixel),
            make_sourceobs(exposure_id=exp_id, healpixel=healpixel),
            make_sourceobs(exposure_id=exp_id, healpixel=healpixel),
            make_sourceobs(exposure_id=exp_id, healpixel=healpixel),
        ]
        frames = bundle_into_frames(observations)

        result = list(frames)

        assert len(result) == 1
        frame = result[0]

        assert len(frame.observations) == 4
        assert frame.observations[0] == observations[0]

    def test_multiple_frames_one_exposure(self):
        # Use a constant exposure ID, but variable healpixels
        exp_id = "exposure"

        observations = [
            make_sourceobs(exposure_id=exp_id, healpixel=1),
            make_sourceobs(exposure_id=exp_id, healpixel=1),
            make_sourceobs(exposure_id=exp_id, healpixel=1),
            make_sourceobs(exposure_id=exp_id, healpixel=2),
            make_sourceobs(exposure_id=exp_id, healpixel=2),
            make_sourceobs(exposure_id=exp_id, healpixel=3),
        ]
        frames = bundle_into_frames(observations)

        result = list(frames)

        assert len(result) == 3
        frame1 = result[0]

        assert len(frame1.observations) == 3
        assert frame1.healpixel == 1
        assert frame1.observations[0] == observations[0]

        frame2 = result[1]
        assert len(frame2.observations) == 2
        assert frame2.healpixel == 2
        assert frame2.observations[0] == observations[3]

        frame3 = result[2]
        assert len(frame3.observations) == 1
        assert frame3.healpixel == 3
        assert frame3.observations[0] == observations[5]

    def test_scrambled_order_within_exposure(self):
        exp_id = "exposure"

        observations = [
            make_sourceobs(exposure_id=exp_id, healpixel=2),
            make_sourceobs(exposure_id=exp_id, healpixel=1),
            make_sourceobs(exposure_id=exp_id, healpixel=3),
            make_sourceobs(exposure_id=exp_id, healpixel=1),
            make_sourceobs(exposure_id=exp_id, healpixel=2),
            make_sourceobs(exposure_id=exp_id, healpixel=1),
        ]
        frames = bundle_into_frames(observations)

        result = list(frames)

        assert len(result) == 3

        assert any(f.healpixel == 1 for f in result)
        assert any(f.healpixel == 2 for f in result)
        assert any(f.healpixel == 3 for f in result)

        for f in result:
            if f.healpixel == 1:
                assert len(f.observations) == 3
            elif f.healpixel == 2:
                assert len(f.observations) == 2
            elif f.healpixel == 3:
                assert len(f.observations) == 1
            else:
                assert False, f"unexpected healpixel: {f.healpixel}"

    def test_multiple_exposures(self):
        observations = [
            # Frame 1:
            make_sourceobs(exposure_id="e1", healpixel=1),
            make_sourceobs(exposure_id="e1", healpixel=1),
            # Frame 2:
            make_sourceobs(exposure_id="e1", healpixel=2),
            # Frame 3:
            make_sourceobs(exposure_id="e2", healpixel=2),
            # Frame 4:
            make_sourceobs(exposure_id="e2", healpixel=1),
            # Frame 5:
            make_sourceobs(exposure_id="e3", healpixel=3),
        ]
        frames = bundle_into_frames(observations)

        result = list(frames)

        assert len(result) == 5
