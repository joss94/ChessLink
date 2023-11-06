#!/bin/bash

USER=$(id -u -n)
USER_ID=$(id -u)
GROUP_ID=$(id -g)

IMAGE_NAME="test_${USER}"
CONTAINER_NAME="${IMAGE_NAME}_container"

REPO_PATH="$(realpath $(dirname "$0")/..)"
DATA_PATH="/home/josselin.manceau/CL/"dataset2
CONDA_ENV_NAME="jma_test"

echo ""
echo "-- Creating Docker container $CONTAINER_NAME --"
echo ""

VGLUSERS_ID=$(getent group vglusers | cut -d: -f3)
VGLUSERS_CLI_ARGS_CREATE=""
if [ ! -z "$VGLUSERS_ID" ]
then
    VGLUSERS_CLI_ARGS_CREATE="--group-add vglusers"
fi

docker run \
    -i -t \
    --runtime=nvidia \
    --gpus all \
    --shm-size=64gb \
    --net host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e PATH=/miniconda/bin:$PATH \
    -e QT_X11_NO_MITSHM=1 \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ${HOME}/.Xauthority:/root/.Xauthority \
    -v ${HOME}/.ssh:/home/$USER/.ssh \
    -v "$(dirname $REPO_PATH)":/workspace \
    -v $DATA_PATH:/data \
    -w /workspace \
    --name=$CONTAINER_NAME \
    --user $USER_ID:$GROUP_ID \
    --group-add video \
    --group-add sudo \
    $VGLUSERS_CLI_ARGS_CREATE \
    -d  $IMAGE_NAME

echo ""
echo "-- Docker container $CONTAINER_NAME is created successfully! --"
echo ""

echo ""
echo "-- Setting up Conda environment and Pre-commit inside Docker container $CONTAINER_NAME --"
echo ""

docker exec -it --user $USER_ID:$GROUP_ID $CONTAINER_NAME /workspace/$(basename $REPO_PATH)/docker/setup_container.sh

echo ""
echo "-- Conda environment and Pre-commit are set successfully inside Docker container $CONTAINER_NAME! --"
echo ""
