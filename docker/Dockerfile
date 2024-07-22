FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Prevents prompts from asking for user input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libeigen3-dev \
    unzip

# copied from https://stackoverflow.com/a/76170605/3485363
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN pip3 install --upgrade requests
RUN ln -fs /usr/bin/python3.10 /usr/bin/python
RUN python --version

# # Setup LST-AI
RUN mkdir -p /custom_apps/lst_directory

# Install prerequisites
# c.f. https://greedy.readthedocs.io/en/latest/install.html#compiling-from-source-code
RUN apt-get update && \
    apt-get install -y cmake g++ git

# Install additional dependencies for VTK
RUN apt-get install -y libgl1-mesa-dev libxt-dev

# Install libpng
RUN apt-get install -y libpng-dev

# Build VTK
# Download and unpack VTK
WORKDIR /VTK
RUN git clone https://gitlab.kitware.com/vtk/vtk.git
WORKDIR /VTK/vtk
RUN git checkout v9.1.0

# Create and navigate to the build directory for VTK
RUN mkdir VTK-build
WORKDIR /VTK/vtk/VTK-build
# ENV LD_LIBRARY_PATH=/VTK/vtk/VTK-build:$LD_LIBRARY_PATH

# Run CMake to configure and build VTK
RUN cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF ..
RUN make -j ${BUILD_JOBS}
RUN make install

# Build ITK
# c.f. https://itk.org/Wiki/ITK/Getting_Started/Build/Linux
# Clone the ITK repository
RUN git clone https://github.com/InsightSoftwareConsortium/ITK.git /ITK
WORKDIR /ITK

# Checkout the specific version
RUN git checkout v5.2.1

# Create and navigate to the build directory
RUN mkdir -p /ITK/build
WORKDIR /ITK/build

RUN apt-get install libexpat1-dev libgtest-dev libhdf5-dev libtiff-dev libvtkgdcm-dev -y

# Run CMake to configure and build ITK
RUN cmake -DModule_ITKPNG=ON \
          -DBUILD_TESTING=OFF \
          -DCMAKE_BUILD_TYPE=Release ..

# run build process
RUN make -j ${BUILD_JOBS}
RUN make install

# Clone the greedy repository
RUN git clone https://github.com/pyushkevich/greedy /greedy
WORKDIR /greedy
RUN git checkout 1eafa4c6659b7a669fb299ce98d9531fc23e332a

# Set the working directory to the build directory
WORKDIR /greedy/build

# Run ccmake from the build directory
RUN cmake -DITK_DIR=/ITK/build \
          -DVTK_DIR=/VTK/vtk/VTK-build \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_SHARED_LIBS=OFF \
           ..
RUN make -j ${BUILD_JOBS}
RUN make install

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
RUN apt-get install python3-dev -y
RUN git clone https://github.com/CompImg/LST-AI/

WORKDIR /custom_apps/lst_directory/LST-AI
RUN git pull origin main
RUN git checkout v1.1.0

# pip or pip3 depending on your system
RUN pip install -e .

# Retrieve model weights and files for LST-AI
WORKDIR /custom_apps/lst_directory/
RUN wget -O /custom_apps/lst_directory/LST-AI/LST_AI/data.zip \
    https://github.com/CompImg/LST-AI/releases/download/v1.1.0/lst_data.zip
WORKDIR /custom_apps/lst_directory/LST-AI/LST_AI/
RUN unzip data.zip && rm data.zip

# Make directories for easily mounting data
# You may change these to your liking
RUN mkdir -p /custom_apps/lst_input
RUN mkdir -p /custom_apps/lst_output
RUN mkdir -p /custom_apps/lst_temp

# Entrypoint to run the python script when the container starts
ENTRYPOINT [ "lst" ]