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

### Docker
You can build and use precovery using the Dockerfile and docker-compose.yml
`docker compose build`
`docker compose run -it precovery bash`

### Conda

To get the latest released version and install it into a conda environment:  
`conda install -c asteroid-institute precovery`  

### Source

To install the bleeding edge source code, clone this repository and then:  

`pip install .`  

**openorb**

Note that, `openorb` is not available on the Python Package Index and so you wil need
to install it via source or conda. See the `Dockerfile` for example of how to build on Ubuntu linux.

## Observation Schema
### Input CSV

`precovery` expects a specific set of columns to be able to index observations into a search 
efficient format. Input files should be sorted by ascending time.

|Name|Unit|Type|Description|
|---|---|---|---|
| obs_id      | None | str |Unique observation ID for the observation |
| exposure_id | None | str |Exposure or Image ID from which observation was measured |
| ra  | degree | float | Right Ascension (J2000) |
| dec  | degree | float | Declination (J2000) |
| ra_sigma  | degree | float | 1-sigma uncertainty in Right Ascension |
| dec_sigma  | degree | float | 1-sigma uncertainty in Declination |
| mag  | None | float | Photometric magnitude measured for observation |
| mag_sigma  | None | float | 1-sigma uncertainty in photmetric magnitude |
| filter | None | str | Filter/bandpass in which the observation was made |
| mjd_start_utc | days | float | Start MJD of the exposure in UTC |
| mjd_mid_utc | days | float | Midpoint MJD of the exposure in UTC (the observation is assumed to reported at this time)|
| exposure_duration | seconds | float | The length of the exposure |
| observatory_code | None | str | MPC observatory code for the observatory/observing program |
