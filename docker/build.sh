#!/bin/bash

export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
export VGLUSERS_ID=$(getent group vglusers | cut -d: -f3)

echo "Bringing container down"
docker compose -f ./docker/compose.yaml down

echo "Building container..."
docker compose -f ./docker/compose.yaml build #--no-cache
echo "Container build finished !"

echo "Firing up container"
docker compose -f ./docker/compose.yaml up -d
