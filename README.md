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

## Developer Setup

This project uses [pre-commit](https://pre-commit.com/) to run linters and code formatters.

pre-commit sets the versions of these code analysis tools, and serves as an entrypoint for running them.;

### pre-commit installation

pre-commit is installed automatically inside the Docker container.

If you're developing on yur local machine without Docker, install it
using either `pip install pre-commit` or `conda install -c conda-forge
pre-commit`. Then, install the hooks with `pre-commit install-hooks`,
run from the root of this repository. This will install all the
linters and tools in an isolated environment.

### Running pre-commit
There are two ways you may choose to run pre-commit. You can run it
manually, or you can run it automatically before every commit.

pre-commit generally only checks files that you have changed. It does
this by comparing against git. This means that `pre-commit` will
**only** check files you have staged (ones you have `git add`-ed). It
will check the staged versions of those files.

#### Running pre-commit manually

Run `pre-commit run` to run linters against any files that you have
changed.

Run `pre-commit run --all-files` to run linters against all files in
the entire repository.

If you use a docker container for all development, you can use
`docker-compose run precovery pre-commit run [--all-files]` to run
within the container.

#### Running pre-commit automatically before every commit

Run `pre-commit install` to set up git hooks. These will block any
commits if your changes don't pass the lint tests.

Sometimes, you might not pass lint but need to commit anyway. If you
have automatic pre-commit enabled, this can get in the way.

You can disable all checks by using `git commit --no-verify`. You can
disable a single check by using a `SKIP` environment variable. For
example, to disable the `mypy` checks, use `SKIP=mypy git commit`.

You can skip multiple linters by passing a comma-separated list. For
example, `SKIP=mypy,black,flake8 git commit`.

The values you pass to SKIP are the pre-commit hook IDs. These can be
found in `.pre-commit-config.yaml`.


## Observation Schema
### Input CSV

`precovery` expects a specific set of columns to be able to index observations into a search
efficient format. Input files should be sorted by ascending time.

|Name|Unit|Type|Description|
|---|---|---|---|
| obs_id | None | str |Unique observation ID for the observation |
| exposure_id | None | str | Exposure or Image ID from which observation was measured |
| mjd | days | float | MJD of the observation in UTC** |
| ra  | degree | float | Right Ascension (J2000) |
| dec  | degree | float | Declination (J2000) |
| ra_sigma  | degree | float | 1-sigma uncertainty in Right Ascension |
| dec_sigma  | degree | float | 1-sigma uncertainty in Declination |
| mag  | None | float | Photometric magnitude measured for observation |
| mag_sigma  | None | float | 1-sigma uncertainty in photometric magnitude |
| filter | None | str | Filter/bandpass in which the observation was made |
| exposure_mjd_start | days | float | Start MJD of the exposure in UTC |
| exposure_mjd_mid | days | float | Midpoint MJD of the exposure in UTC |
| exposure_duration | seconds | float | The length of the exposure |
| observatory_code | None | str | MPC observatory code for the observatory/observing program |

Note:  
** The time at which the observation is reported may be different than the exposure midpoint time to account for effects such as shutter motion

## Precovery Results

|Name|Unit|Type|Description|
|---|---|---|---|
| mjd | days | float | MJD of the observation in UTC** |
| ra_deg  | degree | float | Right Ascension (J2000) |
| dec_deg  | degree | float | Declination (J2000) |
| ra_sigma_arcsec  | arcsecond | float | 1-sigma uncertainty in Right Ascension |
| dec_sigma_arcsec  | arcsecond | float | 1-sigma uncertainty in Declination |
| mag  | None | float | Photometric magnitude measured for observation |
| mag_sigma  | None | float | 1-sigma uncertainty in photometric magnitude |
| filter | None | str | Filter/bandpass in which the observation was made |
| obscode | None | str | MPC observatory code for the observatory/observing program |
| exposure_id | None | str | Exposure or Image ID from which observation was measured |
| exposure_mjd_start | days | float | Start MJD of the exposure in UTC |
| exposure_mjd_mid | days | float | Midpoint MJD of the exposure in UTC |
| exposure_duration | seconds | float | The length of the exposure |
| observation_id | None | str |Unique observation ID for the observation |
| healpix_id | None | int | ID of the HEALPixel onto which the exposure was mapped |
| pred_ra_deg  | degree | float | Predicted Right Ascension (J2000) of the object at the time of the observation |
| pred_dec_deg  | degree | float | Predicted Declination (J2000) of the object at the time of the observation |
| pred_vra_degpday  | degree / day| float | Predicted velocity in Right Ascension (J2000) of the object at the time of the observation |
| pred_vdec_degpday  | degree /day | float | Predicted velocity in Declination (J2000) of the object at the time of the observation |
| delta_ra_arcsec  | arcsecond | float | Difference between predicted and observed Right Ascension (predicted - observed) |
| delta_dec_arcsec  | arcsecond | float | Difference between predicted and observed Declination (predicted - observed) |  
| distance_arcsec  | arcsecond | float | Angular offset between the predicted location of the object and the obervation |
| dataset_id  | None | str | Dataset ID from where the observation was precovered |

Note:  
** The time at which the observation is reported may be different than the exposure midpoint time to account for effects such as shutter motion
