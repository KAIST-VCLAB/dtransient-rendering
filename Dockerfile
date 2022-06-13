FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y alien \
                       dpkg-dev \
                       debhelper \
                       build-essential \
                       libtbb-dev \
                       libglfw3-dev \
                       python-dev \
                       libzip-dev \
                       python3-pip \
                       libopenexr-dev \
                       pkg-config \
                       libeigen3-dev \
                       git \
                       unzip \
                       wget
RUN pip3 install torch==1.11.0+cpu torchvision==0.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install cmake OpenEXR scikit-image pytest matplotlib