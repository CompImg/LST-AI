# Use a CUDA 12.1 compatible base image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevents prompts from asking for user input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libeigen3-dev \
    python3-dev \
    unzip

# Setup Python 3.10
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN pip3 install --upgrade requests
RUN ln -fs /usr/bin/python3.10 /usr/bin/python
RUN python --version

# Install prerequisites for VTK and ITK
RUN apt-get update && \
    apt-get install -y cmake g++ git libgl1-mesa-dev libxt-dev libpng-dev

# Install additional dependencies for ITK
RUN apt-get install -y libexpat1-dev libgtest-dev libhdf5-dev libtiff-dev libvtkgdcm-dev

# Build VTK
WORKDIR /VTK
RUN git clone https://gitlab.kitware.com/vtk/vtk.git
WORKDIR /VTK/vtk
RUN git checkout v9.1.0

RUN mkdir VTK-build
WORKDIR /VTK/vtk/VTK-build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF ..
RUN make -j$(nproc)
RUN make install

# Build ITK
WORKDIR /ITK
RUN git clone https://github.com/InsightSoftwareConsortium/ITK.git .
RUN git checkout v5.2.1

RUN mkdir build
WORKDIR /ITK/build
RUN cmake -DModule_ITKPNG=ON \
          -DBUILD_TESTING=OFF \
          -DCMAKE_BUILD_TYPE=Release ..
RUN make -j$(nproc)
RUN make install

# Build Greedy
WORKDIR /greedy
RUN git clone https://github.com/pyushkevich/greedy .
RUN git checkout 1eafa4c6659b7a669fb299ce98d9531fc23e332a

RUN mkdir build
WORKDIR /greedy/build
RUN cmake -DITK_DIR=/ITK/build \
          -DVTK_DIR=/VTK/vtk/VTK-build \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_SHARED_LIBS=OFF ..
RUN make -j$(nproc)
RUN make install

# Setup LST-AI
WORKDIR /custom_apps/lst_directory
RUN git clone https://github.com/CompImg/LST-AI/
WORKDIR /custom_apps/lst_directory/LST-AI
RUN git pull origin main
RUN git checkout jqm/docker

RUN pip install -e .

# Retrieve model weights for LST-AI
WORKDIR /custom_apps/lst_directory/
RUN wget -O /custom_apps/lst_directory/LST-AI/LST_AI/data.zip \
    https://github.com/CompImg/LST-AI/releases/download/v1.1.0/lst_data.zip
WORKDIR /custom_apps/lst_directory/LST-AI/LST_AI/
RUN unzip data.zip && rm data.zip

# Install HD-BET
WORKDIR /custom_apps/lst_directory
RUN git clone https://github.com/MIC-DKFZ/HD-BET
WORKDIR /custom_apps/lst_directory/HD-BET
RUN git checkout 1fb6cb784a2c270b96f424e792de78fd74323365
RUN pip install -e .

# Retrieve model weights for HD-BET
WORKDIR /custom_apps/lst_directory/
RUN mkdir -p /root/hd-bet_params
RUN wget -O /root/hd-bet_params/data.zip \
    https://zenodo.org/api/records/2540695/files-archive
WORKDIR /root/hd-bet_params/
RUN unzip data.zip && rm data.zip

# Make directories for easily mounting data
RUN mkdir -p /custom_apps/lst_input
RUN mkdir -p /custom_apps/lst_output
RUN mkdir -p /custom_apps/lst_temp

# Entrypoint to run the python script when the container starts
ENTRYPOINT [ "lst" ]
