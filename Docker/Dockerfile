FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libxml2 \
    make \
    vim  \
    gcc g++ \
    gcc-8 g++-8 \
    git \
    python3 python3-pip python3-dev \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

CMD ["tail", "-f", "/dev/null"]
