#!/bin/bash
docker pull docker.io/kumiori3/numerix:latest
docker stop numerix
docker rm numerix
docker run --init -p 8888:8888 --name numerix -v "$(pwd)":/root/shared docker.io/kumiori3/numerix:latest
