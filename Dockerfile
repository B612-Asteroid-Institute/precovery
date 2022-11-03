FROM ubuntu:jammy

RUN apt-get update
RUN apt-get install git python3 pip -y


# Build OpenOrb
# The following are build requirements for oorb
RUN apt-get install gfortran curl liblapack-dev python3-numpy wget -y
RUN git clone https://github.com/oorb/oorb.git /oorb/
WORKDIR /oorb/
RUN ./configure gfortran opt  --with-pyoorb --with-python=python3 --with-f2py=/usr/bin/f2py3 --prefix=/usr/local
RUN make
# Fetch the ephemeris data
RUN make ephem
RUN cd /oorb/data && ./getBC430
RUN cd /oorb/data && ./updateOBSCODE
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/oorb/python/"
ENV PYTHONPATH="$PYTHONPATH:/oorb/python/"
ENV OORB_DATA="/oorb/data"
ENV OORB_CONF="/oorb/main/oorb.conf"

RUN wget https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT
RUN /oorb/bin/oorb --task=mpcorb --mpcorb=MPCORB.DAT > /oorb/data/MPCORB.orb

# Put everything where it belongs
RUN make install

RUN mkdir /code/
ADD . /code/
WORKDIR /code/
RUN pip install .