#!/bin/bash

# Run OpenSource Art!

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# DEEPDREAM_INPUT is the first argument
# If not defined, set as '/deepdream/deepdream/sky1024px.jpg'
# DEEPDREAM_MODELS is defined with container /deepdream/caffe/models
# Data is mapped at /data, including subfolders for input and output
# DEEPDREAM_FRAMES is optional, int defaults to 5
# DEEPDREAM_SCALE_COEFF also optional, float defaults to 0.25

# export DEEPDREAM_IINPUT=/data/input_frames
# export DEEPDREAM_OUTPUT=/data/output_frames
# mkdir -p /data/inputs /data/outputs

# echo "Images Directory: ${DEEPDREAM_INPUT}"
# echo "Output frame Directory: ${DEEPDREAM_OUTPUT}"

sleep 5
echo "params ${@}"
python ${HERE}/deepdream.py "${@}"
