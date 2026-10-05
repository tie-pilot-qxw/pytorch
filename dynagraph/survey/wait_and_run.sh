#!/bin/bash
# Wait for the parameter-count pass to finish, then automatically start the full real-tensor run.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
while pgrep -f validate_lists.py > /dev/null; do sleep 20; done
if [ ! -s model_sizes.json ]; then
    echo "!! model_sizes.json was not produced, exiting"; exit 1
fi
echo "parameter counts done, $(python -c "import json;print(len(json.load(open('model_sizes.json'))))") models have a parameter count"
exec bash "$HERE/run_real.sh"
