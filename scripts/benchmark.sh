#!/bin/bash

# Set default healpix order to 10
: "${ORDER:=10}"
: "${N_ITER:=8}"

while getopts ":d:o:n:" opt; do
    case $opt in
	d)
	    DB_DIR=$OPTARG
	    ;;
	n)
	    N_ITER=$OPTARG
	    ;;
	o)
	    ORDER=$OPTARG
	    ;;
	\?)
	    echo "Invalid option -$OPTARG" >&2
	    exit 1
	    ;;
    esac
done

shift $((OPTIND-1))

if [[ -z "${DB_DIR}" ]]; then
    DB_DIR=$(mktemp -d)
    CLEANUP_DB_DIR=true
fi

DATA_FILE=$1

if [[ -z "${DATA_FILE}" ]]; then
    echo "Usage: $0 <data file> [-d <database directory>] [-n <order>]"
    exit 1
fi

NOW=$(date +"%Y-%m-%d_%H-%M-%S")

echo "Creating precovery database in ${DB_DIR} with ORDER=${ORDER}..."
python scripts/measure_precovery_db_creation_time.py "${DB_DIR}" "${DATA_FILE}" "testdata" -p "${ORDER}" -n "${N_ITER}" -o "precovery_create_benchmark_${NOW}.json"

echo "Running precovery benchmark on ${DB_DIR}..."
python scripts/measure_precover_search_time.py "${DB_DIR}" -o "precovery_search_benchmark_${NOW}.json"

if [[ "${CLEANUP_DB_DIR}" == "true" ]]; then
    echo "Cleaning up temporary database directory..."
    rm -rf "${DB_DIR}"
fi
