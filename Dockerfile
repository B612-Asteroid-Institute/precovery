FROM ubuntu:jammy

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update -y && apt-get install -y \
    git \
    curl \
    python3=3.10.* \
    python3-dev=3.10.* \
    python3-pip \
    unzip \
    git

# Install pyoorb from B612's fork, which includes a patch to handle fortran errors better
ENV OORB_TAG=v1.2.1a1.dev2
ENV OORB_VERSION="pyoorb-1.2.1a1.dev2+66b7753.dirty"

# Install oorb data
RUN curl -fL -o /tmp/oorb_data.zip \
    "https://github.com/B612-Asteroid-Institute/oorb/releases/download/${OORB_TAG}/oorb_data.zip"
RUN unzip -d /opt/oorb_data /tmp/oorb_data.zip
ENV OORB_DATA=/opt/oorb_data

# Install pyoorb
RUN --mount=type=cache,target=/root/.cache/pip \
    export WHEEL_NAME="${OORB_VERSION}-cp310-cp310-manylinux_2_17_$(uname -m).manylinux2014_$(uname -m).whl" && \
    pip install "https://github.com/B612-Asteroid-Institute/oorb/releases/download/${OORB_TAG}/${WHEEL_NAME}"

# Install precovery
RUN mkdir -p /code/precovery
# Install precovery dependencies
COPY setup.cfg /code/setup.cfg
COPY setup.py /code/setup.py
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
