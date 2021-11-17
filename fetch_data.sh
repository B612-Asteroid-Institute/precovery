#!/bin/bash
set -e

mkdir -p data

# Temporarily disable hash checks, because the data file is so big that the CLI
# barfs and aborts without special packages installed.
CHECK_HASHES_VALUE=$(gcloud config get-value storage/check_hashes)
if [[ $CHECK_HASHES_VALUE != False ]]; then
    gcloud config set storage/check_hashes False
fi

# Actually do the download
gcloud alpha storage cp gs://precovery-tutorial/nsc_dr2_observations_2019-08-09_2019-09-09.h5 ./data/nsc_dr2_observations_2019-08-09_2019-09-09.h5

# reset storage/check_hashes to prior value
gcloud config set storage/check_hashes $CHECK_HASHES_VALUE
