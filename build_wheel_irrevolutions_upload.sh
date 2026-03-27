#!/bin/bash

# Abort on error
set -e
python3 -m pip install --upgrade build
# Build wheel
echo "Building wheel for irrevolutions..."
python3 -m build --wheel

# Find latest built wheel
WHEEL_FILE=$(ls -t dist/irrevolutions-*.whl | head -n1)

# Define remote destination
REMOTE_USER=leonbala
REMOTE_HOST=irene-fr.ccc.cea.fr
REMOTE_PATH=/ccc/work/cont003/gen14282/leonbala/wheels/

echo "Sending $WHEEL_FILE to $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
scp "$WHEEL_FILE" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

echo "Done."