# syntax=docker/dockerfile:experimental
# pull official base image
FROM python:3.8.1-slim-buster

# set work directory
ENV workDir /usr/src/app
WORKDIR ${workDir}

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN \
#RUN --mount=type=cache,target=/root/.cache/apt \
    apt-get update && \
    apt-get install -y netcat && \
    apt-get install -y wget && \
    apt-get install -y unzip && \
    apt-get install -y gcc
#git

# install dependencies
RUN \
#RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip
COPY ./requirements.txt ${workDir}/
RUN \
#RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# copy project
COPY . ${workDir}/

# run entrypoint.sh
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
