FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    vim  \
    gcc-8 g++-8 \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


SHELL ["/bin/bash", "-c"]

CMD ["tail", "-f", "/dev/null"]
