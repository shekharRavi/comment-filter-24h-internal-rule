# pull official base image
FROM python:3.8.1-slim-buster

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update && apt-get install -y netcat && \
    apt-get install -y wget
#git



# Download model
# RUN wget https://www.dropbox.com/s/2r5s5hbovt9ig7y/pytorch_model_epoch_20_seqlen_256.bin
# RUN wget -O ./project/models/ml_hate_speech_classifier/pytorch_model_epoch_20_seqlen_256.bin  https://www.dropbox.com/s/2r5s5hbovt9ig7y/pytorch_model_epoch_20_seqlen_256.bin

RUN  echo $PWD

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt
#RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# copy project
COPY . /usr/src/app/

# run entrypoint.sh
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
