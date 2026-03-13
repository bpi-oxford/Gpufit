# syntax=docker/dockerfile:1
# Dockerfile — CUDA development image for building the pyGpufit wheel.
#
# The Gpufit source is bind-mounted read-only at /src/Gpufit.
# The built wheel is written to a bind-mounted /dist/gpufit.
#
# Build the image:
#   docker build --build-arg CUDA_VERSION=12.4.0 -t gpufit-builder:12.4.0 .
#
# Or via the helper script (recommended):
#   bash scripts/build_wheel.sh
#
# CUDA version / driver requirements:
#   CUDA 12.4 → driver >= 550.54 (Linux) / 551.61 (Windows)
#   CUDA 11.8 → driver >= 520.61 (Linux) / 522.06 (Windows)

ARG CUDA_VERSION=12.4.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        python3 \
        python3-pip \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

RUN python -m pip install --upgrade --no-cache-dir pip setuptools wheel

WORKDIR /build
