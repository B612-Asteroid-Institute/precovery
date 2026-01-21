from __future__ import annotations

"""
Preset limiting magnitude tables for precovery.

Why this exists
---------------
`precovery` can optionally skip deep inspection of frames when an object's predicted
magnitude is fainter than a (code, filter) limiting magnitude. This file provides a
*reasonable default* set of limiting magnitudes for the specific MPC observatory codes
and canonical bandpass `filter_id`s we support via `adam_core.photometry.bandpasses`.

Important notes / caveats
-------------------------
- "Limiting magnitude" is not uniquely defined across surveys (5σ vs 10σ, aperture vs
  PSF, single-visit vs coadd, sky brightness/seeing dependence, moving vs stationary
  sources). These presets are intended to be conservative *and* to have traceable
  references.
- For SkyMapper, a commonly cited table is reported at 10σ; we convert to an
  approximate 5σ depth by adding Δm = 2.5 log10(10/5) ≈ 0.753 mag.

Primary sources used
--------------------
- ATLAS (ATLAS_c / ATLAS_o): ATLAS technical specifications (30s exposures, 5σ, dark sky)
  `https://atlas.fallingstar.com/specifications.php`
- NSC DR2 (DECam_* and VR/Y): NSC landing page includes the DR2 "Median Depth (95th%)"
  table (works even when legacy/dlweb hosts are blocked on some networks):
  `https://datalab.noirlab.edu/data/nsc`
  - Paper (NSC DR2, stable fallback): `https://arxiv.org/abs/2011.08868`
- ZTF (ZTF_g/r/i): ZTF performance summaries reporting ~30s 5σ depths near ~20.5 mag
  `https://www.aanda.org/articles/aa/full_html/2025/02/aa50388-24/aa50388-24.html`
  and ZTF/IPAC communications (g~20.8, r~20.6 typical medians)
  `https://ztf.ipac.caltech.edu/news/8`
- SkyMapper (SkyMapper_u/v/g/r/i/z): SkyMapper depth tables (10σ) in literature,
  e.g. MNRAS 499(1)1005
  `https://academic.oup.com/mnras/article/499/1/1005/5905421`
"""

import argparse
import math
import os
from typing import Iterable

from .filter_limiting_magnitudes import FilterLimitingMagnitudes


def _snr_mag_offset(snr_from: float, snr_to: float) -> float:
    """
    Convert a magnitude depth between SNR thresholds under background-limited scaling:
        m_to = m_from + 2.5 log10(snr_from / snr_to)
    """
    if snr_from <= 0 or snr_to <= 0:
        raise ValueError("SNR values must be positive")
    return 2.5 * math.log10(snr_from / snr_to)


def build_default_limiting_magnitudes_table() -> FilterLimitingMagnitudes:
    """
    Build a default `FilterLimitingMagnitudes` table for the observatory codes we use.

    Returns
    -------
    FilterLimitingMagnitudes
        Rows are (obscode, filter_id, limiting_mag, mag_system="AB").
    """
    mag_system = "AB"
    rows: list[tuple[str, str, float, str]] = []

    def add(obscode: str, filter_id: str, limiting_mag: float) -> None:
        rows.append((str(obscode), str(filter_id), float(limiting_mag), mag_system))

    # ATLAS (codes used by our canonical band map)
    for code in ("T05", "T08", "M22", "W68"):
        add(code, "ATLAS_c", 19.7)
        add(code, "ATLAS_o", 19.7)

    # NSC DR2 "Median Depth (95th %)" (point sources), used as a pragmatic proxy.
    add("W84", "DECam_u", 22.6)
    add("W84", "DECam_g", 23.6)
    add("W84", "DECam_r", 23.2)
    add("W84", "DECam_i", 22.8)
    add("W84", "DECam_z", 22.3)
    add("W84", "DECam_Y", 21.0)
    add("W84", "DECam_VR", 23.3)

    add("V00", "BASS_g", 23.6)
    add("V00", "BASS_r", 23.2)
    add("695", "Mosaic3_z", 22.3)

    # ZTF (I41)
    add("I41", "ZTF_g", 20.8)
    add("I41", "ZTF_r", 20.6)
    add("I41", "ZTF_i", 20.0)

    # SkyMapper (Q55): published 10σ depths converted to ~5σ by +0.753 mag.
    dm_10_to_5 = _snr_mag_offset(10.0, 5.0)
    add("Q55", "SkyMapper_u", 19.12 + dm_10_to_5)
    add("Q55", "SkyMapper_v", 19.26 + dm_10_to_5)
    add("Q55", "SkyMapper_g", 20.83 + dm_10_to_5)
    add("Q55", "SkyMapper_r", 20.43 + dm_10_to_5)
    add("Q55", "SkyMapper_i", 19.45 + dm_10_to_5)
    add("Q55", "SkyMapper_z", 18.69 + dm_10_to_5)

    obscodes, filter_ids, limiting_mags, mag_systems = zip(*rows)
    return FilterLimitingMagnitudes.from_kwargs(
        obscode=list(obscodes),
        filter_id=list(filter_ids),
        limiting_mag=list(limiting_mags),
        mag_system=list(mag_systems),
    )


def write_default_parquet(out_file: str) -> str:
    table = build_default_limiting_magnitudes_table()
    out_path = os.path.abspath(os.path.expanduser(out_file))
    table.to_parquet(out_path)
    return out_path


def _main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a default limiting magnitudes parquet file for precovery."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output parquet file path (a FilterLimitingMagnitudes table).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(write_default_parquet(args.out))


if __name__ == "__main__":
    _main()

