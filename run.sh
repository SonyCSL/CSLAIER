#!/bin/bash
if [ -z "${DEEPSTATION_CONFIG+x}" ] ; then
    export DEEPSTATION_CONFIG=$(pwd)/deepstation.cfg
fi

OPT=${1:-0}

if [ $OPT = '-profiler' ]; then
    python $(pwd)/src/profiler.py
else
    python $(pwd)/src/main.py
fi
