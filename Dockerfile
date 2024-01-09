FROM ubuntu:jammy

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get upgrade -y \
    && apt-get install libcurl4-openssl-dev libssl-dev git curl unzip -y

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-transport-https \
    build-essential \
    ca-certificates \
    gnupg \
    gfortran \
    liblapack-dev \
    pkg-config \
    libcfitsio-dev \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    tzdata \
    unzip \
    wget \
    software-properties-common

# # set 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 3.11 doesn't come with pip automatically
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

RUN pip install -v cython==0.29.36 setuptools wheel
RUN chmod 777 /opt

# Install numpy
RUN git clone https://github.com/numpy/numpy.git /opt/numpy
RUN cd /opt/numpy && git checkout v1.24.4 && git submodule update --init
RUN cd /opt/numpy && python3 setup.py build --cpu-baseline=native install

# Install openorb
# TODO: We need a more robust way to be appropriately natively compiled pyoorb installed
# including data file generation
RUN git clone https://github.com/B612-Asteroid-Institute/oorb.git /opt/oorb
RUN cd /opt/oorb && git checkout fork
RUN cd /opt/oorb && ./configure gfortran opt --with-pyoorb --with-f2py=/usr/local/bin/f2py --with-python=python3
# Add '-march=native' to compiler options by running a sed
# script directly on the Makefile.includse file. This is a
# hack to get around the fact that the configure script
# doesn't support this option.
RUN sed -i 's/FCOPTIONS = .*/FCOPTIONS = $(FCOPTIONS_OPT_GFORTRAN) -march=native/g' /opt/oorb/Makefile.include
# --no-build-isolation is needed because we need to ensure we use
# the same version of numpy as the one we compiled previously so
# that it matches the version of f2py we passed in to ./configure.
RUN pip install --no-build-isolation -v /opt/oorb

# Generate the data files
RUN cd /opt/oorb && make ephem
RUN cd /opt/oorb/data && ./getBC430
RUN cd /opt/oorb/data && ./updateOBSCODE
ENV OORBROOT=/opt/oorb
ENV OORB_DATA=/opt/oorb/data

# Install precovery
RUN mkdir -p /code/precovery
# Install precovery dependencies
COPY pyproject.toml /code/pyproject.toml
WORKDIR /code
RUN --mount=type=cache,target=/root/.cache/pip \
    SETUPTOOLS_SCM_PRETEND_VERSION=1 pip install -e .[tests]

# Install pre-commit hooks
COPY .pre-commit-config.yaml /code/.pre-commit-config.yaml
# pre-commit only wants to run inside a git repo
RUN git init && \
    git add .pre-commit-config.yaml && \
    pre-commit install-hooks && \
    rm -rf .git

# Install precovery
ADD . /code/
RUN SETUPTOOLS_SCM_PRETEND_VERSION=1 pip install -e .[tests]
