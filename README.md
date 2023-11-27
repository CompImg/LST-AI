# LST-AI - Deep Learning Ensemble for Accurate MS Lesion Segmentation

[![DOI](https://img.shields.io/badge/arXiv-https%3A%2F%2Fdoi.org%2F10.48550%2FarXiv.2303.15065-B31B1B)]([[https://doi.org/10.48550/arXiv.2303.15065](https://www.medrxiv.org/content/10.1101/2023.11.23.23298966v1)](https://www.medrxiv.org/content/10.1101/2023.11.23.23298966v1)) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Welcome to our codebase for LST-AI, the deep learning based successor of the original [Lesion Segmentation Toolbox (LST)](https://www.applied-statistics.de/lst.html) by [Schmidt et al.](https://www.sciencedirect.com/science/article/abs/pii/S1053811911013139) LST-AI was collaboratively developed by the Department of Neurology, Department of Neuroradiology, Klinikum rechts der Isar at the Technical University of Munich, and the Department of Computer Science at the Technical University of Munich. This repository offers user-friendly access to our newly released LST-AI MS lesion segmentation and annotation tool.

<img src="figures/header.png" alt="Overview" width="1000" height="600" title="Meet LST-AI.">


Disclaimer: LST-AI is a research-only tool for MS Lesion Segmentation and has not been validated, licensed or approved for any clinical usage.

## What is different, or: why should I switch?!
* Here, we present LST-AI, an advanced deep learning-based extension of the original LST with improved performance and additional features.
* LST-AI constitutes a completely new framework and has been developed from scratch.
* While LST depends on MATLAB, we offer LST-AI as a python-based tool which makes it available to the whole community.

## Usage
To allow the usage of LST-AI on different platforms and online-/offline usage, we provide LST-AI as a python package and Docker (CPU and GPU-Docker versions available).

### Installing the python package

LST-AI is a python based package, and requires python3, pip3, git and cmake. For Debian-based systems, you can install all required packages via `apt`:

```
apt-get update && apt-get install -y \
git \
cmake \
cmake-curses-gui \
g++ \
make \
libinsighttoolkit4-dev \
python3 \
python3-pip
```

Under the hood, LST also wraps [HD-BET](https://github.com/MIC-DKFZ/HD-BET) and [greedy](https://github.com/pyushkevich/greedy).
We guide you through the download / compilation for greedy and the installation for HD-BET in the following process. If you encounter issues specifically with these packages, let us know in an issue and/or consult the respective github repositories.

1. Make a new directory for LST-AI
```bash
mkdir lst_directory
cd lst_directory
```

2. We recommend setting up a virtual environment for LST-AI:
```
python3 -m venv /path/to/new/lst/virtual/environment
```

3. Activate your new environment, i.e. `(lst_env)`
```
source /path/to/new/lst/virtual/environment/bin/activate
```

4. Install LST-AI (and yeah, with `pip -e` option!):
```bash
git clone https://github.com/jqmcginnis/LST-AI
cd LST-AI
pip install -e .
cd ..
```

4. Install [HD-BET](https://github.com/MIC-DKFZ/HD-BET)
```bash
git clone https://github.com/MIC-DKFZ/HD-BET
```
```
cd HD-BET
pip install -e .
cd ..
```

6. Download or Compile and install greedy for your platform
  * 6.1 (Variant A): Download the built greedy tool and place into structure
    1) Download the tool
    ```bash
    wget "https://syncandshare.lrz.de/dl/fi65b93EmVE42LPbRtoVgR/greedy"
    ```
    2) and ensure it is a findable path:
    ```bash
    chmod +x greedy
    mkdir ~/bin
    mv greedy ~/bin
    export PATH="$HOME/bin:$PATH"
    ```
    Naturally, you can place the binary in ANY directory if you add it to your `.bashrc` and export the location to the `$PATH`.
  * 6.2 (Variant B): Compile, make and install the greedy tool (you will need to install both, VTK and ITK)
    ```
    wget https://github.com/InsightSoftwareConsortium/ITK/archive/refs/tags/v5.2.1.tar.gz
    tar -zxvf v5.2.1.tar.gz
    cd ITK-5.2.1
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    make install

    wget https://www.vtk.org/files/release/9.1/VTK-9.1.0.tar.gz
    tar -xf VTK-9.1.0.tar.gz
    cmake ..
    make -j$(nproc)
    make install

    git clone https://github.com/pyushkevich/greedy greedy
    cmake ../greedy
    make -j$(nproc)
    make install
    ```

### Usage of LST-AI

Once installed, lst can be used as a simple command line tool. LST-AI expects you to provide **zipped NIFTIs (*.nii.gz)** as input
and assumes the input images **NOT** to be **skull-stripped**. If you already have skull-strips, **do not forget** to provide the **--skull-stripped** option, otherwise the segmentation performance will be likely impacted.

LST requires you to provide a `--t1` T1w and `--flair` FLAIR image, and to specify an output path for the segmentation results `--output`.

#### Example usage:
```
(lst_env) jqm@workstation: lst --t1 t1.nii.gz --flair flair.nii.gz --output /mnt/data/lst/results --temp /mnt/data/lst/processing
```

#### Modes

We provide three different modes:

1. **Default Mode - Segmentation + Annotation**: In this mode, you only need to provide the T1w and FLAIR input images. LST-AI will automatically segment and annotate your lesions according to the MCDonald's criteria.

2. Segmentation Only: If you only care about the binary segmentation, and not about the annotation / class (perventricular, ...), this mode is for you. It will (only) save the binary segmentation mask. To execute it, provide the `--segmentation_only` flag to run it.

3. Annotation Only: If you already have a satisfactory binary segmentation mask for your T1w/FLAIR images, you can only use the annotation/region labeling function. Please provide your existing segmentation via `--existing_seg /path/to/binary/mask`, and provide the `--annotate_only` flag to run it.

#### Other (useful) settings

- If you would like to access intermediate pipeline results such as the skull-stripped T1w, FLAIR images in MNI152 space, please provide a temporary directory via `--temp `. Otherwise we create a temporary directory on the fly and remove it once the pipeline has finished.
- `--use_gpu`: Porvide this flag if you have access to a GPU.
- `-fast-mode`: Option to speed-up the skull-stripping (performed by HD-BET).
- `skull-stripping`: Bypass skull-stripping. Only use if your images are (actually) skull-stripped.


### Dockerfile and Dockerhub

While the installation and usage requires internet access to install python packages and to download the weights and atlas,
we understand that some researchers prefer to use lst-ai offline. Thus, we decided to provide lst-ai as a CPU-/GPU-enabled docker container, which can be (1) compiled using our scripts or (2) downloaded from [Dockerhub](https://hub.docker.com/u/jqmcginnis).

### Running the LST-AI Docker Container
Once you have built your Docker image, using the Dockerfile provided, you can run the container using the docker run command. Here are the steps to bind mount your files and retrieve the results:

#### Build the Docker Image
Clone the repository:
```bash
git clone https://github.com/jqmcginnis/LST-AI
cd LST-AI
```
If you haven't already, build your CPU or GPU Docker image:

```
cd cpu
docker build -t lst-ai_cpu:latest .
```
```
cd gpu
docker build -t lst-ai_gpu:latest .
```
#### Run the Docker Container with Bind Mounts
The primary mechanism for sharing files between your host system and the Docker container is the -v or --volume flag, which specifies a bind mount.

Here's a breakdown of how to use bind mounts:
```bash
docker run -v [path_on_host]:[path_in_container] [image_name]
```

Given an example command, the run command might look something like this:

```bash
docker run -v /mnt/data/lst_example/sub-123456/ses-20231101/anat:/data/anat \
           -v /mnt/data/lst_example/derivatives/sub-123456/ses-20231101/:/data/output \
           lstai:latest \
           --t1 /data/anat/sub-123456_ses-20231101_T1w.nii.gz \
           --flair /data/anat/sub-123456_ses-20231101_FLAIR.nii.gz \
           --output /data/output/
```

__Note__: Ensure your paths are absolute, as Docker requires absolute paths for bind mounts. Since you've bind-mounted your output directory to `/mnt/data/lst_example/derivatives/sub-123456/ses-20231101/` on your host, the results from the Docker container will be written directly to this directory. No additional steps are needed to retrieve the results, they will appear in this directory after the container has finished processing.

#### Extending and modifying LST-AI for your custom code and pipeline

We invite you to tailor LST-AI to your pipeline and application, please have a look at our [sources](LST-AI).

### Citation

If you use our tool, please cite us:
```
@article {Wiltgen2023.11.23.23298966,
	author = {Tun Wiltgen and Julian McGinnis and Sarah Schlaeger and CuiCi Voon and Achim Berthele and Daria Bischl and Lioba Grundl and Nikolaus Will and Marie Metz and David Schinz and Dominik Sepp and Philipp Prucker and Benita Schmitz-Koep and Claus Zimmer and Bjoern Menze and Daniel Rueckert and Bernhard Hemmer and Jan Kirschke and Mark Muehlau and Benedikt Wiestler},
	title = {LST-AI: a Deep Learning Ensemble for Accurate MS Lesion Segmentation},
	elocation-id = {2023.11.23.23298966},
	year = {2023},
	doi = {10.1101/2023.11.23.23298966},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2023/11/24/2023.11.23.23298966},
	eprint = {https://www.medrxiv.org/content/early/2023/11/24/2023.11.23.23298966.full.pdf},
	journal = {medRxiv}
}
```

We also kindly ask you to cite greedy and HD-BET, which we use in the LST-AI pipeline.

```
@article{yushkevich2016ic,
  title={IC-P-174: Fast Automatic Segmentation of Hippocampal Subfields and Medial Temporal Lobe Subregions In 3 Tesla and 7 Tesla T2-Weighted MRI},
  author={Yushkevich, Paul A and Pluta, John and Wang, Hongzhi and Wisse, Laura EM and Das, Sandhitsu and Wolk, David},
  journal={Alzheimer's \& Dementia},
  volume={12},
  pages={P126--P127},
  year={2016},
  publisher={Wiley Online Library}
}
```

```
@article{isensee2019automated,
  title={Automated brain extraction of multisequence MRI using artificial neural networks},
  author={Isensee, Fabian and Schell, Marianne and Pflueger, Irada and Brugnara, Gianluca and Bonekamp, David and Neuberger, Ulf and Wick, Antje and Schlemmer, Heinz-Peter and Heiland, Sabine and Wick, Wolfgang and others},
  journal={Human brain mapping},
  volume={40},
  number={17},
  pages={4952--4964},
  year={2019},
  publisher={Wiley Online Library}
}
```
