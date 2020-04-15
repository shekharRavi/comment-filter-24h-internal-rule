# pull official base image
FROM python:3.8.1-slim-buster

# set work directory
ENV workDir /usr/src/app
WORKDIR ${workDir}

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update && \
    apt-get install -y netcat && \
    apt-get install -y wget && \
    apt-get install -y unzip
#git

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt ${workDir}/
RUN pip install -r requirements.txt
#RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# install models
ENV modelDir project/models
RUN mkdir project
RUN mkdir ${modelDir}
COPY ./${modelDir}/model_download.sh ${workDir}/${modelDir}/
RUN ./${modelDir}/model_download.sh

# copy project
COPY . ${workDir}/

# run entrypoint.sh
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
