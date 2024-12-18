#### Building the docker

Info: We are happy to support both ARM64 and AMD64 platforms with the newest docker container.

#### Guide on how to build the docker natively, and (tp push it to dockerhub)

To build and push a Docker image for both linux/amd64 and linux/arm64/v8 platforms and then push it to Docker Hub under the name jqmcginnis/lst-ai, you can follow these steps:

#### 1. Log in to dockerhub

Open your terminal and log in to your Docker Hub account using the Docker CLI:

```bash
docker login
```
Enter your Docker Hub username and password when prompted.

#### 2. Enable Buildx (if not already enabled)

Docker Buildx is an extended build feature that supports building multi-platform images. To ensure it is enabled, run:

```bash
docker buildx create --use --name mybuilder
```

#### 3. Start a New Buildx Builder Instance

This step ensures that the builder instance is started and uses the newly created builder:

```bash
docker buildx use mybuilder
docker buildx inspect --bootstrap
```

#### 4. Build and Push the Image

Navigate to the directory where your Dockerfile is located, then build and push the image for both platforms. Replace path/to/dockerfile with the actual path to your Dockerfile if it's not in the current directory:

```bash
docker buildx build --platform linux/amd64,linux/arm64/v8 -t jqmcginnis/lst-ai:v1.1.0 --push --build-arg BUILD_JOBS=8 .
```
This command will build the image for amd64 and arm64/v8 architectures and push it to Docker Hub under the repository jqmcginnis/lst-ai. It may take several hpurs (!).

#### 5. Verify the Push

Navigate to the directory where your Dockerfile is located, then build and push the image for both platforms. Replace path/to/dockerfile with the actual path to your Dockerfile if it's not in the current directory:

```bash
docker buildx build --platform linux/amd64,linux/arm64/v8 -t jqmcginnis/lst-ai:v1.1.0 --push .
```
This command will build the image for amd64 and arm64/v8 architectures and push it to Docker Hub under the repository jqmcginnis/lst-ai. It may take several hours (!).
