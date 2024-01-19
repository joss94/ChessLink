FROM ubuntu:22.04

# Install linux packages
RUN apt-get update && \
    apt-get install -y sudo make git git-lfs python3-pip iputils-ping iproute2 nano ffmpeg libsm6 libxext6

COPY ./env/requirements.txt /env/requirements.txt
RUN python3 -m pip install -r /env/requirements.txt

RUN python3 -m pip install shapely

# Run the Nginx server
CMD cd /src;python3 -m web_app.upload_server
