#!/usr/bin/env bash
res=$(python3 -m cProfile -s tottime precovery-test.py $@)
if [ "$?" == "0" ]; then
    echo "${res}" | head -n 53
fi
