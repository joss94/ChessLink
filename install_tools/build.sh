#!/bin/bash
# Script to create Docker image and Docker container

USER=$(id -u -n)

IMAGE_NAME="test_${USER}"
CONTAINER_NAME="${IMAGE_NAME}_container"

if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    # Create Docker image on server
    ./docker/create_image.sh
else
    echo ""
    echo "-- Docker image $IMAGE_NAME exists --"
    echo ""
fi

if docker ps -a | grep -q $CONTAINER_NAME; then
    echo ""
    echo "-- Docker container $CONTAINER_NAME exists --"
    echo "-- Enter Docker container by running ./install_tools/exec.sh --"
    echo ""
else
    # Create Docker container for current user based on Docker image $IMAGE_NAME
    ./docker/create_container.sh
fi

echo ""
echo "***************************************************"
echo "***      Installation finished succesfully!     ***"
echo "***  You can enter Docker container by running  ***"
echo "***          ./install_tools/exec.sh            ***"
echo "***************************************************"
echo ""