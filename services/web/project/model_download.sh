#!/bin/bash

MODEL_DIR=./project/models
LOCAL_MODEL_ZIP=ml_hate_speech_classifier.zip
LOCAL_MODEL_DIR=ml_hate_speech_classifier
REMOTE_MODEL_URL=https://www.dropbox.com/s/fivwekb7kr71x61/ml_hate_speech_classifier.zip

if [ ! -d $MODEL_DIR ]; then
    mkdir $MODEL_DIR
fi

if [ -f $MODEL_DIR/$LOCAL_MODEL_ZIP ]; then
    echo "File $MODEL_DIR/$LOCAL_MODEL_ZIP exists, doing nothing."
else
    echo "No file $MODEL_DIR/$LOCAL_MODEL_ZIP, retrieving..."
    wget -O $MODEL_DIR/$LOCAL_MODEL_ZIP $REMOTE_MODEL_URL
    # do some error checking here ... did it download OK?
fi

if [ -d $MODEL_DIR/$LOCAL_MODEL_DIR ]; then
    echo "File $MODEL_DIR/$LOCAL_MODEL_DIR exists, doing nothing."
else
    echo "No directory $MODEL_DIR/$LOCAL_MODEL_DIR, unzipping..."
    unzip $MODEL_DIR/$LOCAL_MODEL_ZIP -d $MODEL_DIR
    # do some error checking here ... did it unzip OK?
fi

