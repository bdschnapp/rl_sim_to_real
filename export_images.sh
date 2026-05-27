#!/bin/bash

mkdir -p build-images || true
cd build-images

docker save --output ${HA_RUNTIME_IMAGE_NAME}.tar ${HA_RUNTIME_IMAGE} ${HA_RUNTIME_IMAGE_LATEST}