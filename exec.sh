#!/bin/bash
# Script to enter Docker container

docker compose up -d
docker exec -it chesslink bash
