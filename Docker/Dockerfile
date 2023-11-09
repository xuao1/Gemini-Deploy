FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    vim  \
    curl  \
    netbase \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


SHELL ["/bin/bash", "-c"]

CMD ["tail", "-f", "/dev/null"]
