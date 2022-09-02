# precovery: fast asteroid precovery at scale
#### A Python package by the Asteroid Institute, a program of the B612 Foundation 
[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-blue)](https://img.shields.io/badge/Python-3.7%2B-blue)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)  
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
