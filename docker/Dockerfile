# FROM ubuntu:22.04
# FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04

ARG BASE_IMAGE=ubuntu:20.04

FROM ${BASE_IMAGE}

ARG PROJECT_NAME
ARG USER
ARG UID
ARG GID

# Install linux packages
RUN apt-get update \
    && apt-get install -yq tzdata \
    && apt-get install -y sudo make git git-lfs python3-pip iputils-ping iproute2 nano ffmpeg libsm6 libxext6 libcairo2-dev \
    && dpkg-reconfigure -f noninteractive tzdata \
    && ([ -z "$VGLUSERS_ID" ] || addgroup --gecos "" --gid ${VGLUSERS_ID} vglusers) \
    && (addgroup --gecos "" --force-badname --gid ${GID} ${USER} || true) \
    && adduser --gecos "" --force-badname --uid ${UID} --gid ${GID} ${USER} --disabled-password \
    && passwd -d ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USER}_no_password \
    && chmod 440 /etc/sudoers.d/${USER}_no_password

# Necessary to use bash as shell here for the variable substitution ${USER//./_}
SHELL ["/bin/bash", "-c"]
# Note: "|| true" necessary here to handle possible mv errors when user name does not include any "."
RUN mv -f /etc/sudoers.d/${USER}_no_password /etc/sudoers.d/${USER//./_}_no_password || true

COPY ./requirements.txt /env/requirements.txt
RUN python3 -m pip install -r /env/requirements.txt

WORKDIR "/workspace/${PROJECT_NAME}"

USER ${UID}:${GID}

# Use this for web app (run the Nginx server)
#CMD cd /src;python3 -m web_app.upload_server

# Use this for training (just leave container running for user to open cli)
CMD ["/bin/bash","-c","tail -F anything"]