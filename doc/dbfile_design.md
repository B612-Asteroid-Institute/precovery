The database is composed of _exposures_, which are groups of observations with
a common epoch and observation location. The exposures are further grouped into
_epoch bundles_, which are exposures which have close epochs.

Each epoch bundle is a separate file in Apache Arrow format.
