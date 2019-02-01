#!/bin/bash -E

JACROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../ && pwd )"
SCRIPT=$( basename ${BASH_SOURCE[0]} )
SCRIPT=${SCRIPT:4}

source $JACROOT/bin/_jac-init.sh $JACROOT
exec python3 "$JACROOT/scripts/$SCRIPT.py" "$@"

