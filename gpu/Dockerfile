# Start with NVIDIA's CUDA-enabled Ubuntu 20.04 base image
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Prevents prompts from asking for user input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    cmake-curses-gui \
    g++ \
    make \
    libinsighttoolkit4-dev \
    python3 \
    python3-pip

# Set up the directory tree and clone the greedy repository
RUN mkdir custom_apps
WORKDIR /custom_apps
RUN mkdir -p greedy/build
RUN git clone https://github.com/pyushkevich/greedy greedy

# Set the build directory as the working directory
WORKDIR /custom_apps/greedy/build
RUN ccmake ../greedy

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
