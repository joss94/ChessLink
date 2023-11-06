#!/bin/bash

USER=$(id -u -n)
USER_ID=$(id -u)
GROUP_ID=$(id -g)

BASE_IMAGE="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04"
IMAGE_NAME="test_${USER}"

REPO_PATH="$(realpath $(dirname "$0")/..)"

VLGUSERS_ID=$(getent group vglusers | cut -d: -f3)
VLGUSERS_CLI_ARGS=""
if [ ! -z "$VLGUSERS_ID" ]
then
    VLGUSERS_CLI_ARGS="--build-arg VLGUSERS_ID=$VLGUSERS_ID"
fi


echo ""
echo "-- Creating Docker image $IMAGE_NAME --"
echo ""

# In German AIME server, $VLGUSERS_CLI_ARGS is not needed
docker build \
    -t $IMAGE_NAME "$REPO_PATH/docker" \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    --build-arg USER=$USER \
    --build-arg UID=$USER_ID \
    --build-arg GID=$GROUP_ID \
    $VLGUSERS_CLI_ARGS

echo ""
echo "-- Docker image $IMAGE_NAME is created successfully! --"
echo ""
