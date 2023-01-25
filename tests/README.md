## Precovery Unit Tests

### Testing Data
The testing data includes two files:
- `data/sample_orbits.csv`: Keplerian and Cometary orbital elements of a representative set of small bodies. Produced by `get_orbits.py`.
- `data/index/observations.csv`: Synthetic and indexing-ready observations of each of the orbits in `data/sample_orbits.csv`.

### Scripts
#### `get_orbits.py`:
Queries JPL's Small Body Database Browser for the Keplerian and Cometary elements for a set of representative small bodies. These orbits are stored in a CSV in the data directory contained within this folder. Osculating elements change over time and the best-fit orbits for these objects will change as more observations are made. As a result, the orbits queried and downloaded represent only a snapshot of our knowledge of these objects at the time. This script should only be run when new objects need to be added or when signficant time has passed that more observations of these objects could have impacted their best-fit orbits. The actual values of the orbital elements *should* not matter for any testing conducted here, they are simply used as a snapshotted input to create synthetic data.

Usage: ```python get_orbits.py```  
This saves orbits to `data/sample_orbits.csv`.

#### `make_observations.py`:
Produces 3 synthetic observations CSV files containing observations for each object in the `sample_orbits.csv` file. The observations have the following properties:
- 4 observations each day (a "quad" tracklet) for 15 days for observatories '500', 'I11', 'I41', 'F51'.
- Observations from '500' and 'I11' are split into their own datasets. While observations from 'I41' and 'F51' are combined into a single dataset.
- Each observatory has an observation window that starts at 0, 10, 20, and, 30 days after the initial epoch at which the orbits are defined for '500', 'I11', 'I41', 'F51', respectively. This, by design, creates periods of overlap and non-overlap for each dataset and spreads observations over a 2 calendar month period.
- The first observation on any night is always an integer number of days from the reference epoch at which the orbit is defined plus a ~few hours time shift to account for the location of the observatory relative to GMT. Observations 2, 3, and 4 in the same night as the first, are set to be 30, 60, and 90 minutes from the time of the first observation, respectively.
- The exposure times increase for each observation in the quad tracklet: 30, 60, 90, and 120 seconds.
- **No** astrometric errors are added.
- The observation times are reported at a random time between the start of the exposure and the end of the exposure.


This script should only be run when the underlying orbits are updated or when changes need to be made to the observation properties such as the cadence. As each orbit is stored with Keplerian and Cometary elements in `sample_orbits.csv`, the element type used to initialize the internal orbits class used to create the observations can be specified with the optional argument, `--orbit_type`. The default is currently set to "keplerian". Please note that the behavior of PYOORB given different orbital element representations may not be consistent.

Usage: ```python make_observations.py```  
This saves observations to `data/index/dataset_500/dataset_500_observations.csv`, `data/index/dataset_I11/dataset_I11_observations.csv`, `data/index/dataset_I41+F51/dataset_I41+F51_observations.csv`.
