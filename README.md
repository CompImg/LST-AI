# LST-AI - Deep Learning Ensemble for Accurate MS Lesion Segmentation

> 2026-05-06: **LST-AI has received the [NeuroImage: Clinical Editors’ Choice Award 2025](https://www.sciencedirect.com/journal/neuroimage-clinical/about/editor-choice/neuroimage-clinical-editors-choice-award-2025-tun-wiltgen) 🏆**

Welcome to our codebase for LST-AI, the deep learning-based successor of the original [Lesion Segmentation Toolbox (LST)](https://www.applied-statistics.de/lst.html) by [Schmidt et al.](https://www.sciencedirect.com/science/article/abs/pii/S1053811911013139)
LST-AI was collaboratively developed by the Department of Neurology and Department of Neuroradiology, Klinikum rechts der Isar at the Technical University of Munich, and the Department of Computer Science at the Technical University of Munich.

<img src="figures/header.png" alt="Overview" width="1000" height="600" title="Meet LST-AI.">


**Disclaimer:** LST-AI is a research-only tool for MS Lesion Segmentation and has not been validated, licensed or approved for any clinical usage.

## What is different, why or when should I switch?!
* LST-AI is an advanced deep learning-based extension of the original LST with improved performance and additional features.
* LST-AI constitutes a completely new framework and has been developed from scratch.
* While LST depends on MATLAB, we offer LST-AI as a python-based tool which makes it available to the whole community.
* We suggest using LST or LST-AI according to your type of data:
    * A 3D T1-weighted and a 3D FLAIR sequence are available: (new) LST-AI
    * Only a 3D FLAIR sequence is available: (old) [LST](https://www.applied-statistics.de/lst.html) (LPA)
    * Only a 3D T1-weighted sequence is available: not covered by any LST(-AI) version
    * If a 3D T1-weighted and a non-3D FLAIR sequence are available, please try both (new) LST-AI or (old) [LST](https://www.applied-statistics.de/lst.html) (LGA or LPA)


## Usage
To allow the usage of LST-AI on different platforms and online/offline usage, we provide LST-AI as a Python package and Docker (CPU and GPU-Docker versions available).

### Installing the Python package

LST-AI installs from PyPI. Inference runs in **PyTorch** — there is no TensorFlow, no ONNX
Runtime and, since v2.0.0, no `onnx` dependency either: the released weights ship as `.pt`
checkpoints. `greedy` (registration) and HD-BET (skull stripping) install as wheels too,
so nothing is compiled.

```bash
python3 -m venv lst_env && source lst_env/bin/activate
pip install lst-ai
```

While 2.0.0 is in its release-candidate phase, pip needs to be told that a
pre-release is acceptable: `pip install --pre lst-ai` (or pin it explicitly,
`pip install lst-ai==2.0.0rc1`). Once the final 2.0.0 is published, the plain command
above is all there is.

To work on LST-AI itself, install the repository in editable mode instead:

```bash
python3 -m venv lst_env && source lst_env/bin/activate
git clone https://github.com/CompImg/LST-AI/ && cd LST-AI
pip install -e .
```

If your system Python is older than 3.10 (LST-AI's floor) or you simply want faster
installs, [`uv`](https://docs.astral.sh/uv/) is a good alternative — it fetches a
suitable interpreter itself, so nothing has to be installed system-wide:

```bash
uv venv --python 3.12 lst_env && source lst_env/bin/activate
uv pip install lst-ai
```

Note the `uv pip install` — a venv created by `uv venv` deliberately ships without
`pip`, so the plain `pip install` from above would fail inside it.

**GPU users, check your driver against PyTorch's default CUDA build.** The `torch`
wheels on PyPI now target CUDA 13, which needs an R580+ driver; on an older driver
(anything reporting CUDA ≤ 12.8 in `nvidia-smi`) the install succeeds but
`torch.cuda.is_available()` is `False`, so every GPU run fails at startup ("driver too
old") and only `--device cpu` works. Install torch from the index matching your driver
*before* installing `lst-ai`, e.g.:

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch
```

The GPU Docker image is pinned to cu126 and does not have this problem.

Either way, the install pulls in `torch`, `picsl-greedy`, `hd-bet` and FastSurfer, which every mode except
`--segment_only` needs to annotate lesions — see [Lesion annotation with
FastSurfer](#lesion-annotation-with-fastsurfer). The model bundle and atlas are
downloaded automatically on first run. Nothing else has to be set up by hand.

The bundle lands next to the installed `lst_ai` package when that directory is
writable (any venv install), and falls back to `~/.cache/lst_ai` when it is not
(e.g. a system-wide install). Set `LST_AI_DATA_DIR` to override the location
explicitly.

`picsl-greedy` ships official PyPI wheels for linux/arm64 since 1.4.0.1
([pyushkevich/greedy_python#6](https://github.com/pyushkevich/greedy_python/pull/6)), so
no platform needs a wheel workaround any more — the same install covers every
supported architecture.

### What happened to TensorFlow?

Inference was reimplemented natively in PyTorch. The **released weights are unchanged** —
every convolution kernel, gamma and beta matches the released `.h5` bit-for-bit
(max abs diff 0.0, all three ensemble members). Nothing was retrained.

Validated against the real released TensorFlow models on 10 subjects, full ensemble:

| | |
|---|---|
| Dice agreement with TensorFlow | **0.99565** |
| lesion volume difference | **−0.12 %** (worst −2.16 %) |
| lesion Dice vs consensus GT | TF 0.6829 · PyTorch 0.6835 |

No framework swap is bit-identical, but this is far closer than the intermediate ONNX
Runtime backend it replaces (Dice 0.9367, **+14.3 %** lesion volume). For scale, the
pipeline has never been bit-reproducible anyway: running the identical code in a container
rather than a virtualenv moves the result more (Dice 0.883) than this change does, because
greedy's registration is nondeterministic run to run.

See [docs/pytorch-reimplementation.md](docs/pytorch-reimplementation.md) for the
measurements, the cause of the ONNX drift, and guidance if you are comparing against
earlier results.

**Testing this release:** [docs/testing.md](docs/testing.md) is the validation checklist,
split into an aarch64 and an x86_64 track, with the expected run-to-run variation
quantified. Both tracks have end-to-end runs recorded (x86_64 including the CUDA image on
a real GPU); the remaining known gap is that CI builds and unit-tests but does not segment
a subject, so further "worked for me" reports — especially from aarch64 and CPU-only
machines — are genuinely useful.

### Lesion annotation with FastSurfer

Lesions are assigned to anatomical regions using a FastSurfer segmentation of the T1. This
is not an optional stage: every mode except `--segment_only` runs it, the default one
included, so FastSurfer is a hard requirement of a normal `lst` run — and both the pip
install and the Docker image bring it with them. There is nothing to install by hand.

FastSurfer is not published on PyPI, so `pip install` brings it in two parts, identically
on x86_64 and on arm64/aarch64:

- the Python dependencies of the segmentation path LST-AI drives are ordinary
  requirements in `setup.py`, resolved by pip like `greedy` and HD-BET;
- the FastSurfer source tree itself (~17 MB) is unpacked from the pinned release tarball
  into `<your environment>/share/lst-ai/FastSurfer-v2.5.4` — or into
  `~/.local/share/lst-ai/` when that prefix is not writable.

LST-AI finds it there on its own: no `FASTSURFER_HOME`, no `PATH` entry, nothing to add to
your `~/.bashrc`. The three VINN checkpoints (~65 MB) are fetched by FastSurfer on the
first run that annotates, the same way LST-AI fetches its own model bundle.

**This copy is the only one LST-AI uses.** A FastSurfer already on the machine — one
`FASTSURFER_HOME` points at, or a `run_fastsurfer.sh` on `PATH` — is ignored, and there is
no option to prefer it, so an annotation never depends on what a given machine happens to
have lying around. The version is pinned in
[lst_ai/fastsurfer.py](lst_ai/fastsurfer.py).

Two things you can still do by hand:

```bash
# fetch the checkpoints ahead of time, for a machine that will later be offline
python -m lst_ai.fastsurfer --checkpoints

# re-download the tree, if it was interrupted or something under it was edited
python -m lst_ai.fastsurfer --force
```

Setting `LST_AI_SKIP_FASTSURFER=1` before `pip install` skips the download at install
time — it defers it to the first run that annotates, it does not substitute another
FastSurfer. A failed download never fails the install, for the same reason: `--segment_only`
still works, and the download is retried on first use.

### Training your own models

`training/` trains the same network this package runs at inference — single-channel
(FLAIR) or dual-channel (FLAIR + T1). It imports `lst_ai.model` rather than copying it, so
what you train is what ships. See [training/README.md](training/README.md).

### Usage of LST-AI

Once installed, LST-AI can be used as a simple command line tool. LST-AI expects you to provide **zipped NIFTIs (*.nii.gz)** as input and assumes the input images **NOT** to be **skull-stripped**. If you already have skull-stripped images, **do not forget** to provide the **`--stripped`** option, otherwise, the segmentation performance will be severely affected.

LST-AI always requires you to provide a `--t1` T1w and `--flair` FLAIR image and to specify an output path for the segmentation results `--output`. If you would like to keep all processing files, for example, the segmentations and skull-stripped images in the MNI152 space, provide a directory via `--temp`.

#### Example usage:
```
(lst_env) jqm@workstation: lst --t1 t1.nii.gz --flair flair.nii.gz --output /mnt/data/lst/results --temp /mnt/data/lst/processing
```

#### Modes

We provide three different modes:

1. **Default Mode - Segmentation + Annotation**: In this mode, you only need to provide the T1w and FLAIR input images. LST-AI will automatically segment and annotate your lesions according to McDonald's criteria.

2. **Segmentation Only**: If you only care about the binary segmentation, and not about the annotation/class (perventricular, ...), this mode is for you. It will (only) save the binary segmentation mask. To execute it, provide the `--segment_only` flag to run it.

3. **Annotation Only**: If you already have a satisfactory binary segmentation mask for your T1w/FLAIR images, you can only use the annotation/region labeling function. Please provide your existing segmentation via `--existing_seg /path/to/binary/mask`, and provide the `--annotate_only` flag to run it. We assume that the lesion mask is provided in the FLAIR image space.

#### Other (useful) settings

- `--temp `: If you would like to access intermediate pipeline results such as the skull-stripped T1w, and FLAIR images in MNI152 space, please provide a temporary directory using this flag. Otherwise, we create a temporary directory on the fly and remove it once the pipeline has finished.
- `--device`: Provide an integer value (e.g. `0`) for a GPU ID or `cpu` if you do not have access to a GPU. **The default is `0`, i.e. GPU 0.** On a machine without a GPU — including the CPU Docker image, whose PyTorch is a CPU-only build — you must pass `--device cpu` explicitly.
- `--stripped`: Bypass skull-stripping. Only use if your images are (actually) skull-stripped. We cannot handle a mixture (e.g. skull-stripped T1w, but non-skull-stripped FLAIR) as of now.
- `--threshold`: Provide a value between `0` and `1` that defines the threshold which is applied to the lesion probability map generated by the ensemble network to create the binary lesion mask. The default setting is 0.5.
- `--lesion_threshold`: Minimum lesion size in **mm³**. Connected components smaller than this are dropped from the binary mask. The default is `0`, i.e. no size filtering.
- `--clipping`: This flag can be used to define lower and upper percentiles that are used as min & max for standardization of image intensities in pre-processing. Changing these parameters can have an effect on the sensitivity of the segmentation process (e.g., higher max can yield higher sensitivity). The default setting is `0.5 99.5`, which indicates that the min is defined as the 0.5 percentile and the max is defined as the 99.5 percentile.
- `--probability_map`: Save the lesion probability maps of the ensemble network and of each individual 3D UNet model of the ensemble network. The `--temp` flag must be set, otherwise the files will be removed as they are stored in the temporary directory along with the intermediate pipeline results.
- `--fast-mode`: Run HD-BET with a single model rather than its full ensemble, and without test-time augmentation. Faster skull-stripping, at some cost in mask quality.
- `--threads`: Number of threads used for registration. Defaults to all available cores.  


### Dockerfile and Dockerhub

While the installation and usage require internet access to install python packages and to download the weights and atlas, we understand that some researchers prefer to use lst-ai offline. Thus, we have decided to provide lst-ai as a CPU-/GPU-enabled docker container, which can be compiled using our scripts (for instructions please check the docker directory). If, instead of building the docker yourself, you would just rather use it, you can pull it from dockerhub instead.

Images are published to Docker Hub under `jqmcginnis/lst-ai`, one CUDA and one CPU
flavour per release, each as a multi-arch manifest (linux/amd64 + linux/arm64):

```bash
docker pull jqmcginnis/lst-ai:v2.0.0        # CUDA flavour
docker pull jqmcginnis/lst-ai:v2.0.0-cpu    # CPU flavour
```

`latest` and `latest-cpu` track the newest stable release. Pre-releases are published
only under their explicit tag (e.g. `v2.0.0rc1`, `v2.0.0rc1-cpu`) and never move
`latest`, so a plain `docker pull jqmcginnis/lst-ai` cannot land on a candidate. The
legacy `jqmcginnis/lst-ai_cpu` repository and the v1.x tags remain for the old
TensorFlow-based images.

#### Building the image yourself

One Dockerfile covers both flavours and both architectures — there is no separate CPU
file to drift out of sync. It defaults to a CUDA base:

```bash
# GPU (default): nvidia/cuda:12.6.3-runtime-ubuntu22.04 + the cu126 torch wheels
docker build -f docker/Dockerfile -t lst-ai:gpu .

# CPU-only: ~4 GB instead of ~17 GB, and no NVIDIA runtime needed to run it
docker build -f docker/Dockerfile -t lst-ai:cpu \
  --build-arg BASE_IMAGE=ubuntu:22.04 \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
```

Both flavours are built for `linux/amd64` and `linux/arm64` in CI. All weights — the LST-AI
ensemble, the atlas, HD-BET's five folds and FastSurfer's three VINN checkpoints — are
baked in at build time, so the container needs no network at run time. `greedy` installs
from PyPI on both architectures, pinned to one version via `--build-arg GREEDY_VERSION`
so the two images do not differ in which greedy produced the result.

#### Building behind a TLS-inspecting proxy

Many hospital and university networks inspect HTTPS traffic, re-signing it with their own
root certificate. Your machine trusts it; a fresh Docker container does not, so the build
fails at the first download with `CERTIFICATE_VERIFY_FAILED: unable to get local issuer
certificate`. The certificate has to be added manually:

**1. Copy your organisation's root certificate into `docker/certs/`**

That directory ships with the repository, empty. On Debian/Ubuntu hosts the certificate is
usually already installed locally:

```bash
cp /usr/local/share/ca-certificates/*.crt docker/certs/
```

Otherwise ask your IT department for it. Files must be PEM format
(they start with `-----BEGIN CERTIFICATE-----`) and end in `.crt`, or they are ignored.

**2. Build as usual**

```bash
docker build -f docker/Dockerfile -t lst-ai:gpu .
```

Everything you put in `docker/certs/` is gitignored, so your certificates cannot be
committed by accident. Do not push an image built this way to a public registry — it trusts
your organisation's CA.

### Running the LST-AI Docker Container
Once you have pulled (or built) your Docker image using the Dockerfile provided you can run the container using the `docker run` command. Here are the steps to bind mount your files and retrieve the results:

#### Run the Docker Container with Bind Mounts
The primary mechanism for sharing files between your host system and the Docker container is the `-v` or `--volume` flag, which specifies a bind mount.

Here's a breakdown of how to use bind mounts:
```bash
docker run -v [path_on_host]:[path_in_container] [image_name]
```

Given our provided GPU Dockerfile command, the run command might look something like this:

```bash
docker run --gpus all -v /home/ginnis/lst_in:/custom_apps/lst_input -v /home/ginnis/lst_out/:/custom_apps/lst_output -v /home/ginnis/lst_temp/:/custom_apps/lst_temp lst-ai:gpu --t1 /custom_apps/lst_input/t1.nii.gz --flair /custom_apps/lst_input/flair3d.nii.gz --output /custom_apps/lst_output --temp /custom_apps/lst_temp
```  

__Note__: Ensure your paths are absolute, as Docker requires absolute paths for bind mounts. Since you've bind-mounted your output directory to `/home/ginnis/lst_out/` on your host, the results from the Docker container will be written directly to this directory. No additional steps are needed to retrieve the results, they will appear in this directory after the container has finished processing.

#### Running on CPU
Run docker using the CPU image:

```bash
docker run -v /home/ginnis/lst_in:/custom_apps/lst_input -v /home/ginnis/lst_out/:/custom_apps/lst_output -v /home/ginnis/lst_temp/:/custom_apps/lst_temp lst-ai:cpu --t1 /custom_apps/lst_input/t1.nii.gz --flair /custom_apps/lst_input/flair3d.nii.gz --output /custom_apps/lst_output --temp /custom_apps/lst_temp --device cpu
```  

__Note__: Omit the `--gpus all` and add `--device cpu`, and replace use the cpu image `lst-ai:cpu` instead of the gpu image `lst-ai:gpu`.

#### Who owns the results: run as yourself

The container runs as root unless told otherwise. As root, everything it writes into your
bind-mounted output and temp directories comes out owned by `root`, and you need `sudo`
to clean it up.

FastSurfer makes this sharper: it *refuses to start* as root, precisely because
everything it writes would come out root-owned. LST-AI detects that case and passes
its `--allow_root`, so the annotation stage runs rather than aborting — but that only
unblocks the run, it does not change who ends up owning the files.

To own the results yourself, add `-u $(id -u):$(id -g)`:

```bash
docker run -u $(id -u):$(id -g) --gpus all -v /home/ginnis/lst_in:/in -v /home/ginnis/lst_out:/out lst-ai:gpu --t1 /in/t1.nii.gz --flair /in/flair3d.nii.gz --output /out
```

The bind-mounted directories have to be writable by that uid, which they are if they are
yours. Nothing else needs changing, and LST-AI drops
FastSurfer's `--allow_root` when it is not running as root.

#### Extending and modifying LST-AI for your custom code and pipeline

We invite you to tailor LST-AI to your pipeline and application, please have a look at our [sources](lst_ai).

### BIDS Compliance with LST-AI

To ensure maximum flexibility for our user base, LST-AI does not natively enforce BIDS-compliant file-naming conventions. This decision allows users to work seamlessly with both BIDS and non-BIDS datasets.

However, for those who wish to utilize LST-AI within a BIDS-compliant workflow, we have provided an [example repository](https://github.com/twiltgen/LST-AI_BIDS) that demonstrates the integration of LST-AI with BIDS-compliant data. This example reflects the BIDS-compliant usage of LST-AI that we are currently using in our internal database.

### Citation

Please consider citing [LST-AI](https://www.medrxiv.org/content/10.1101/2023.11.23.23298966) to support the development:
```
@article{wiltgen2024lst,
  title={LST-AI: A deep learning ensemble for accurate MS lesion segmentation},
  author={Wiltgen, Tun and McGinnis, Julian and Schlaeger, Sarah and Kofler, Florian and Voon, CuiCi and Berthele, Achim and Bischl, Daria and Grundl, Lioba and Will, Nikolaus and Metz, Marie and others},
  journal={NeuroImage: Clinical},
  pages={103611},
  year={2024},
  publisher={Elsevier}
}
```

Further, please also credit [greedy](https://greedy.readthedocs.io/en/latest/) and [HD-BET](https://github.com/MIC-DKFZ/HD-BET), used for preprocessing the image data, and — unless you run `--segment_only` — [FastSurfer](https://github.com/Deep-MI/FastSurfer), which provides the anatomical parcellation lesions are assigned to.

greedy
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

HD-BET:
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

FastSurfer (used by every mode except `--segment_only`). LST-AI calls FastSurfer with
`--seg_only`, so it uses the FastSurferVINN whole-brain segmentation network rather than
the surface pipeline; cite both the pipeline paper and the VINN paper, as FastSurfer
itself asks:
```
@article{henschel2020fastsurfer,
  title={FastSurfer -- A fast and accurate deep learning based neuroimaging pipeline},
  author={Henschel, Leonie and Conjeti, Sailesh and Estrada, Santiago and Diers, Kersten and Fischl, Bruce and Reuter, Martin},
  journal={NeuroImage},
  volume={219},
  pages={117012},
  year={2020},
  publisher={Elsevier},
  doi={10.1016/j.neuroimage.2020.117012}
}

@article{henschel2022fastsurfervinn,
  title={FastSurferVINN: Building resolution-independence into deep learning segmentation methods -- A solution for HighRes brain MRI},
  author={Henschel, Leonie and K{\"u}gler, David and Reuter, Martin},
  journal={NeuroImage},
  volume={251},
  pages={118933},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.neuroimage.2022.118933}
}
```
