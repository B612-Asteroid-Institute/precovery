#!/bin/bash
res=$(python -m cProfile -s tottime sq-execute-test.py $@)
if [ "$?" == "0" ]; then
    echo "${res}" | head -n 26
fi
