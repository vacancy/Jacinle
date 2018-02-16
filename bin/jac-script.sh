#!/bin/bash -E

JACROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../ && pwd )"
SCRIPT=$( basename ${BASH_SOURCE[0]} )
SCRIPT=${SCRIPT:4}

export PYTHONPATH=$JACROOT:./:$PYTHONPATH
exec python3 "$JACROOT/scripts/$SCRIPT.py" $@

