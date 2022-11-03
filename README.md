# precovery: fast asteroid precovery at scale
#### A Python package by the Asteroid Institute, a program of the B612 Foundation 
[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-blue)](https://img.shields.io/badge/Python-3.7%2B-blue)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/414402801.svg)](https://zenodo.org/badge/latestdoi/414402801)  
[![Python Package with conda](https://github.com/B612-Asteroid-Institute/precovery/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/B612-Asteroid-Institute/precovery/actions/workflows/python-package-conda.yml)
[![Publish Python Package to conda](https://github.com/B612-Asteroid-Institute/precovery/actions/workflows/python-publish-conda.yml/badge.svg)](https://github.com/B612-Asteroid-Institute/precovery/actions/workflows/python-publish-conda.yml)  
[![Anaconda-Server Badge](https://anaconda.org/asteroid-institute/precovery/badges/version.svg)](https://anaconda.org/asteroid-institute/precovery)
[![Anaconda-Server Badge](https://anaconda.org/asteroid-institute/precovery/badges/platforms.svg)](https://anaconda.org/asteroid-institute/precovery)
[![Anaconda-Server Badge](https://anaconda.org/asteroid-institute/precovery/badges/downloads.svg)](https://anaconda.org/asteroid-institute/precovery)  

## Installation 

### Conda

To get the latest released version and install it into a conda environment:  
`conda install -c asteroid-institute precovery`  

### Source

To install the bleeding edge source code, clone this repository and then:  

If you use conda to manage packages and environments:  
`conda install -c defaults -c conda-forge --file requirements.txt`  
`pip install . --no-deps`  

If you would rather download dependencies with pip:  
`pip install -r requirements-pip.txt`  
`pip install . --no-deps`  

Note that, `openorb` is not available on the Python Package Index and so it would still need
to be installed either via source install or via conda.

## Usage

After the precovery datasets are indexed and ready, you can run precovery jobs.

```python

from precovery import precover
from precovery.orbit import Orbit, EpochTimeScale


db_dir = "/path/to/db/folder/"

orbit = Orbit.keplerian(
    0,
    2.269057465131142,
    0.1704869454928905,
    21.27981352885659,
    281.533811391701,
    7.854179343480579,
    98.55494515731131,
    57863.,
    EpochTimescale.UTC,
    20,
    0.15
)

results = precover(orbit, db_dir, tolerance=10/3600)

for result in results:
    print(result)

```

## Dataset Format

Currently the precovery services can index observation datasets that have been formatted with the following schema:

| Column | Description |
| ------ | ----------- |
| mjd_utc | Modified Julian Date from UTC |
| ra | right ascension |
| ra_sigma | right ascension uncertainty. If null, we assume a standard sigma for orbit determination. |
| dec | declination |
| dec_sigma | declination uncertainty.  If null, we assume a standard sigma for orbit determination. |
| mag | visible magnitude  |
| mag_sigma | magnitude uncertainty |
| filter | wavelength filter |
| observatory_code | The observatory that the observation came from |
| obs_id | Unique ID for the observation |
| exposure_id | Unique exposure id from the observatory. |

