FROM python:3.7.7

# Additional updates to pip
RUN pip install --upgrade pip && \
    # pip install typing && \
    pip install ptpython 


FROM saturnism/deepdream


LABEL MAINTAINER vsochat@stanford.edu

# Unzip and wget dependencies
RUN apt-get update && apt-get install -y wget unzip


ADD . /deepdream/caffe/scripts
ADD ./deepdream.py /deepdream.py
ADD ./frames2movie.sh /frames2movie.sh

RUN apt-get install -y libav-tools
# RUN apt-get install avconv -y
# RUN apt install ffmpeg -y


ADD ./run.sh /run.sh
WORKDIR /deepdream/caffe/models

# Environment variables for deepdream
ENV DEEPDREAM_OUTPUT /data/output_frames
ENV DEEPDREAM_INPUT /data/input_frames

ENV DEEPDREAM_MODELS /deepdream/caffe
ENV CAFFE_SCRIPTS /deepdream/caffe/scripts


ADD ./download_model_from_gist.sh /deepdream/caffe/download_model_from_gist.sh

# Download extra models
# wget https://raw.githubusercontent.com/wiki/BVLC/caffe/Model-Zoo.md -O zoo.md
# cat zoo.md | grep -o -P '(?<=gist[.]github[.]com).*(?=[)])' # (gets most)
RUN cd /deepdream/caffe/models && \
    echo "Downloading extra models..." && \
    chmod u+x ${CAFFE_SCRIPTS}/download_zoo.sh && \
    /bin/bash ${CAFFE_SCRIPTS}/download_zoo.sh ${PWD}

# disable opencv camera driver
RUN ln /dev/null /dev/raw1394

WORKDIR / 
# WORKDIR /deepdream
ENTRYPOINT ["/bin/bash", "/run.sh"]





