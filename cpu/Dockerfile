# Start with the base Ubuntu 20.04 LTS image
FROM ubuntu:22.04

# Prevents prompts from asking for user input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    python3 \
    python3-pip

RUN mkdir -p /custom_apps/lst_directory
WORKDIR /custom_apps/lst_directory
RUN git clone https://github.com/jqmcginnis/LST-AI
WORKDIR /custom_apps/lst_directory/LST-AI
RUN pip install -e .

# There are two possible ways of obtaining 'greedy'
# Download pre-compiled version via wget
# Compile greedy from source (which requires ITK and VTK)

# Option 1: Download pre-compiled version of greedy
# Download pre-compiled version of greedy and place in $PATH
WORKDIR /custom_apps/lst_directory
RUN wget "https://syncandshare.lrz.de/dl/fi65b93EmVE42LPbRtoVgR/greedy" && \
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

# Ensure model weights and files are available
# before first (offline) usage
WORKDIR /custom_apps/lst_directory/
RUN wget -O /custom_apps/lst_directory/LST-AI/LST_AI/data.zip \
    https://syncandshare.lrz.de/dl/fiPiTmWKv5Ga4S7YA7TNks/.dir
WORKDIR /custom_apps/lst_directory/LST-AI/LST_AI/
RUN unzip data.zip && rm data.zip

# Ensure weights are downloaded
RUN hd-bet -i /custom_apps/lst_directory/LST-AI/testing/segmentation/T1W.nii.gz -mode accurate 

# Make directories for mounting data
RUN mkdir -p /custom_apps/lst_input
RUN mkdir -p /custom_apps/lst_output
RUN mkdir -p /custom_apps/lst_temp

# Entrypoint to run the python script when the container starts
ENTRYPOINT [ "lst"]
