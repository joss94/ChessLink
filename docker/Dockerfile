ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG USER
ARG UID
ARG GID
# Note: needed to add vglusers group to get access to GPU(s) on Paris servers
ARG VLGUSERS_ID

# Install linux packages
RUN apt-get update && \
    apt-get install -y sudo make git wget python3-pip curl libxrender1 libsm6 libxext6 xauth mesa-utils

RUN if [ -n "${VLGUSERS_ID}" ]; then \
        addgroup --gecos "" --gid ${VLGUSERS_ID} vglusers; \
    fi
# Note: "|| true" necessary here to handle possible errors when the GID is already in use.
RUN addgroup --gecos "" --force-badname --gid ${GID} ${USER} || true && \
    adduser --gecos "" --force-badname --uid ${UID} --gid ${GID} ${USER} --disabled-password && \
    passwd -d ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USER}_no_password && \
    chmod 440 /etc/sudoers.d/${USER}_no_password

# Necessary to use bash as shell here for the variable substitution ${USER//./_}
SHELL ["/bin/bash", "-c"]
# Note: "|| true" necessary here to handle possible mv errors when user name does not include any "."
RUN mv -f /etc/sudoers.d/${USER}_no_password /etc/sudoers.d/${USER//./_}_no_password || true

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Install miniconda
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -p /miniconda -b && \
    rm Miniconda3-py38_23.1.0-1-Linux-x86_64.sh && \
    export PATH=/miniconda/bin:$PATH && \
    conda update -y conda
