# Use an NVIDIA CUDA base image
FROM nvidia/cuda:11.0-base-ubuntu20.04

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone your repository
RUN git clone https://github.com/jqmcginnis/code.git /app

# Set the working directory
WORKDIR /app

# Install the package
RUN pip3 install .

# Default command to keep container running (for debug purposes, can be replaced)
CMD ["tail", "-f", "/dev/null"]
