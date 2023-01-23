FROM ubuntu:jammy

RUN apt-get update -y && apt-get install -y \
    git \
    curl \
    python3=3.10.* \
    python3-dev=3.10.* \
    python3-pip \
    unzip

# Install pyoorb from B612's fork, which includes a patch to handle fortran errors better
ENV OORB_TAG=v1.2.1a1.dev2
ENV OORB_VERSION="pyoorb-1.2.1a1.dev2+66b7753.dirty"

# Install oorb data
RUN curl -fL -o /tmp/oorb_data.zip \
    "https://github.com/B612-Asteroid-Institute/oorb/releases/download/${OORB_TAG}/oorb_data.zip"
RUN unzip -d /opt/oorb_data /tmp/oorb_data.zip
ENV OORB_DATA=/opt/oorb_data

# Install pyoorb
RUN export WHEEL_NAME="${OORB_VERSION}-cp310-cp310-manylinux_2_17_$(uname -m).manylinux2014_$(uname -m).whl" && \
    pip install "https://github.com/B612-Asteroid-Institute/oorb/releases/download/${OORB_TAG}/${WHEEL_NAME}"

# Install precovery
RUN mkdir /code/
ADD . /code/
WORKDIR /code/

RUN SETUPTOOLS_SCM_PRETEND_VERSION=1 pip install -e .[tests]
