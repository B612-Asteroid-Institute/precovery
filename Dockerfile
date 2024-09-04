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
