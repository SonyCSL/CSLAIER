#!/bin/bash
if [ -z "${DEEPSTATION_CONFIG+x}" ] ; then
    export DEEPSTATION_CONFIG=$(pwd)/deepstation.cfg
fi
python $(pwd)/src/main.py
