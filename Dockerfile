# FROM python:3.7.7

# # Additional updates to pip
# RUN pip install --upgrade pip && \
#     # pip install typing && \
#     pip install ptpython
    


FROM saturnism/deepdream


LABEL MAINTAINER vsochat@stanford.edu

# Unzip and wget dependencies
RUN apt-get update && apt-get install -y wget unzip

# install python 3
# RUN apt-get install software-properties-common -y
# RUN add-apt-repository ppa:deadsnakes/ppa -y
# RUN apt-get update && apt install python3.7 -y
# RUN apt install python3-pip -y
# RUN update-alternatives  --set python /usr/bin/python3.4
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.4 1

# RUN rm -f /usr/bin/python && ln -s /usr/bin/python /usr/bin/python3

# # install ffmpeg
# RUN apt-get install software-properties-common -y
# RUN add-apt-repository ppa:mc3man/trusty-media -y
# RUN apt-get update && apt-get dist-upgrade -y
# RUN apt-get install ffmpeg -y

# RUN pip3 install --upgrade pip && \
#     pip3 install typing && \
#     pip3 install ptpython

ADD . /deepdream/caffe/scripts
ADD ./deepdream.py /deepdream.py
ADD ./frames2movie.sh /frames2movie.sh
ADD ./download_model_binary.py /download_model_binary.py


RUN apt-get update && apt-get install -y libav-tools
# RUN apt-get update && apt-get install -y mplayer
# RUN apt-get install avconv -y
# RUN apt install ffmpeg -y

# RUN easy_install pip
# RUN python -m pip install pyyaml

# RUN pip install urllib3[secure]
# RUN pip install --upgrade setuptools
# RUN apt-get install python-yaml -y
# RUN pip install PyYAML


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





