#!/bin/bash

docker run --rm -it --gpus all \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $PWD:/workspace \
  -v ~/data:/workspace/DATA \
  --device /dev/dri \
  --shm-size=8g \
  --name hanelso-alexnet \
  u2204_c118_p23
