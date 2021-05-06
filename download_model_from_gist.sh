#!/usr/bin/env sh

GIST=$1
DIRNAME=${2:-./models}


if [ -z $GIST ]; then
  echo "usage: download_model_from_gist.sh <gist_id> <dirname>"
  exit
fi



LIST=($(echo $GIST  | tr '.' '\n'))

GIST_URL=${LIST[0]}


MODEL_DIR="$DIRNAME/${LIST[1]}"


if [ -d $MODEL_DIR ]; then
    echo "$MODEL_DIR already exists! Please make sure you're not overwriting anything important!"
    exit
fi

echo "Downloading Caffe model info to $MODEL_DIR ..."
mkdir -p $MODEL_DIR
wget https://gist.github.com/$GIST_URL/download -O $MODEL_DIR/gist.zip
unzip -j $MODEL_DIR/gist.zip -d $MODEL_DIR
rm $MODEL_DIR/gist.zip
echo "Done"
