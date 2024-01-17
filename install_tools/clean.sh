#!/bin/bash
# Script to remove Docker container and clean conda environment

USER=$(id -u -n)

IMAGE_NAME="test_${USER}"
CONTAINER_NAME="${IMAGE_NAME}_container"
CONDA_ENV_NAME=jma_test

CLEAN_IMAGE=true
CLEAN_ENV=true

usage()
{
    echo "usage: ./install_tools/clean.sh [options]"
    echo "options:"
    echo "  --clean_image: Clean Docker image $IMAGE_NAME as well as Docker container"
    echo "  --clean_env: Clean Conda environment $CONDA_ENV_NAME"
}

while [ "$1" != "" ]; do
    case $1 in
        --clean_image )
            shift
            CLEAN_IMAGE=true
            ;;
        --clean_env )
            shift
            CLEAN_ENV=true
            ;;
        -h | --help )
            usage
            exit
            ;;
        * )
            usage
            exit 1
            ;;
    esac
    shift
done

# Remove Docker container
if docker ps -a | grep -q $CONTAINER_NAME; then
    docker rm -f $CONTAINER_NAME
    echo "-- Docker container $CONTAINER_NAME has been removed successfully --"
fi

# Remove Docker image if argument --clean_image is used
if [ $CLEAN_IMAGE = true ] && [ "$(docker images -q $IMAGE_NAME 2> /dev/null)" != "" ]; then
    docker rmi $IMAGE_NAME
    echo "-- Docker image $IMAGE_NAME has been removed successfully --"
fi

# Remove Conda environment if argument --clean_env is used
if [ $CLEAN_ENV = true ] && [ -d "env/$CONDA_ENV_NAME/" ]; then
    echo "-- Removing Conda environment $CONDA_ENV_NAME --"
    rm -r "env/$CONDA_ENV_NAME/"
    echo "-- Conda environment has been removed successfully --"
fi
