#!/bin/bash
# Script to enter Docker container

USER=$(id -u -n)
USER_ID=$(id -u)
GROUP_ID=$(id -g)

IMAGE_NAME="test_${USER}"
CONTAINER_NAME="${IMAGE_NAME}_container"

docker start $CONTAINER_NAME
docker exec -it --user $USER_ID:$GROUP_ID $CONTAINER_NAME bash
