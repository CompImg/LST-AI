# Start with the base Ubuntu 20.04 LTS image
FROM ubuntu:22.04

# Prevents prompts from asking for user input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    cmake \
    g++ \
    build-essential \
    libpng-dev \
    libtiff-dev \
    uuid-dev \
    make \
    python3 \
    python3-pip \
    libgl1-mesa-dev

RUN wget https://github.com/InsightSoftwareConsortium/ITK/archive/refs/tags/v5.2.1.tar.gz && \
    tar -zxvf v5.2.1.tar.gz && \
    cd ITK-5.2.1 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install

# Set up the directory tree and clone the greedy repository
RUN mkdir /custom_apps
WORKDIR /custom_apps
RUN git clone https://github.com/pyushkevich/greedy
RUN echo $PWD
RUN mkdir -p greedy/build

# Set the build directory as the working directory
WORKDIR /custom_apps/greedy/build
RUN cmake ..

# Compile using make with the available number of CPU cores
RUN make -j$(nproc)

# Install HD-BET into the container
WORKDIR /custom_apps
RUN git clone https://github.com/MIC-DKFZ/HD-BET
WORKDIR /custom_apps/HD-BET
RUN pip3 install .

# Install LST-AI into the container
WORKDIR /custom_apps
RUN git clone https://github.com/jqmcginnis/lst_production
WORKDIR /custom_apps/lst_production
RUN pip install .

# Entrypoint to run the python script when the container starts
ENTRYPOINT [ "lst"]
