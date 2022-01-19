#!/bin/bash
docker build  -t fenicsx-dev -f Dockerfile.dev --no-cache .
# docker images
# docker tag <id> kumiori3/numerix:latest
# docker push kumiori3/numerix:latest
# latest: digest: sha256:14c60e8d454758c97b3bbbbb8a205c75e84e3934ed7edcbe2e0041c8bf3c1697 size: 4962