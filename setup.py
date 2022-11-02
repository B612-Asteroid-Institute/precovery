from setuptools import setup

setup(
    install_requires=[
        "numpy",
        "numba",
        "sqlalchemy",
        "rich",
        "tables",
        "pandas",
        "healpy",
        "requests",
        # "openorb", # Must be installed separately.
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "setuptools>=45",
            "wheel",
            "setuptools_scm>=6.0",
        ]
    },
    use_scm_version={
        "write_to": "precovery/version.py",
        "write_to_template": "__version__ = '{version}'",
    },
)
