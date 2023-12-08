# Start with the base Ubuntu 22.04 LTS image
# FROM ubuntu:22.04
FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04

# Prevents prompts from asking for user input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip

# copied from https://stackoverflow.com/a/76170605/3485363
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# RUN curl -sSL https://install.python-poetry.org | python3.10 - --preview
RUN pip3 install --upgrade requests
RUN ln -fs /usr/bin/python3.10 /usr/bin/python
RUN python --version

# Setup LST-AI
RUN mkdir -p /custom_apps/lst_directory

# Setup for greedy (Choose either pre-compiled or compilation from source)

# There are two possible ways of obtaining 'greedy'
# Download pre-compiled version via wget
# Compile greedy from source (which requires ITK and VTK)

# Option 1: Download pre-compiled version of greedy
# Download pre-compiled version of greedy and place in $PATH

# https://sourceforge.net/projects/greedy-reg/files/Nightly/greedy-nightly-MacOS-x86_64.dmg/download

# for MAC use this instead?
#WORKDIR /custom_apps/lst_directory
#RUN wget "https://sourceforge.net/projects/greedy-reg/files/Nightly/greedy-nightly-MacOS-x86_64.dmg/download" -o "greedy.dmg" && \
#    chmod +x greedy && \
#    mv greedy /usr/local/bin


WORKDIR /custom_apps/lst_directory
RUN wget "https://github.com/CompImg/LST-AI/releases/download/v1.0.0/greedy" && \
    chmod +x greedy && \
    mv greedy /usr/local/bin

# Option 2: Compile greedy from source
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libpng-dev \
#     libtiff-dev \
#     uuid-dev \
#     make \
#     cmake \
#     g++ \
#     libgl1-mesa-dev

# RUN wget https://github.com/InsightSoftwareConsortium/ITK/archive/refs/tags/v5.2.1.tar.gz && \
#     tar -zxvf v5.2.1.tar.gz && \
#     cd ITK-5.2.1 && \
#     mkdir build && \
#     cd build && \
#     cmake .. && \
#     make -j$(nproc) && \
#     make install

# # Download and extract VTK 9.1.0
# WORKDIR /opt
# RUN wget https://www.vtk.org/files/release/9.1/VTK-9.1.0.tar.gz && \
#     tar -xf VTK-9.1.0.tar.gz

# # Build VTK
# WORKDIR /opt/VTK-9.1.0/build
# RUN cmake .. && \
#     make -j$(nproc) && \
#     make install

# # Set up the VTK_DIR environment variable
# ENV VTK_DIR=/usr/local/lib/cmake/vtk-9.1

# Set up the directory tree and clone the greedy repository
# RUN mkdir /custom_apps
# WORKDIR /custom_apps
# RUN git clone https://github.com/pyushkevich/greedy
# RUN echo $PWD
# RUN mkdir -p greedy/build

# # Set the build directory as the working directory
# WORKDIR /custom_apps/greedy/build
# RUN cmake ..

# # Compile using make with the available number of CPU cores
# RUN make -j$(nproc)
# RUN make install

# Install HD-BET
WORKDIR /custom_apps/lst_directory
RUN git clone https://github.com/MIC-DKFZ/HD-BET
WORKDIR /custom_apps/lst_directory/HD-BET
RUN pip install -e .

# Retrieve model weights for HD-BET
WORKDIR /custom_apps/lst_directory/
RUN mkdir -p /root/hd-bet_params
RUN wget -O /root/hd-bet_params/data.zip \
    https://zenodo.org/api/records/2540695/files-archive
WORKDIR /root/hd-bet_params/
RUN unzip data.zip && rm data.zip

WORKDIR /custom_apps/lst_directory
ARG CACHEBUST=1
RUN git clone https://github.com/jqmcginnis/LST-AI/
WORKDIR /custom_apps/lst_directory/LST-AI
RUN git checkout 1c05bb5ac58144e20fd82fa9bca95c6d16728f8c
# pip or pip3 depending on your system
RUN pip install -e .

# Retrieve model weights and files for LST-AI
WORKDIR /custom_apps/lst_directory/
RUN wget -O /custom_apps/lst_directory/LST-AI/LST_AI/data.zip \
    https://github.com/CompImg/LST-AI/releases/download/v1.0.0/lst_data.zip
WORKDIR /custom_apps/lst_directory/LST-AI/LST_AI/
RUN unzip data.zip && rm data.zip

# Make directories for easily mounting data
# You may change these to your liking
RUN mkdir -p /custom_apps/lst_input
RUN mkdir -p /custom_apps/lst_output
RUN mkdir -p /custom_apps/lst_temp

# Entrypoint to run the python script when the container starts
ENTRYPOINT [ "lst" ]