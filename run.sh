#!/bin/bash
if [ -z "${CSLAIER_CONFIG+x}" ] ; then
    export CSLAIER_CONFIG=$(pwd)/cslaier.cfg
fi

OPT=${1:-0}

if [ $OPT = '-profiler' ]; then
    python $(pwd)/src/profiler.py
else
    python $(pwd)/src/main.py
fi
