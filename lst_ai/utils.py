from io import open
import os
import zipfile
from urllib import request

import nibabel as nib


def harmonize_affines(input_path, output_path):
    """Save ``input_path`` to ``output_path`` with its qform and sform set to the
    same (nibabel-resolved) affine.

    A NIfTI file stores two affines. ``nibabel`` (and therefore the rest of this
    pipeline) reads ``img.affine``, which prefers the sform when ``sform_code > 0``,
    while ``greedy`` (used for registration) reads the qform. When an earlier
    co-registration step (e.g. in SPM) updates only the sform, the two disagree
    and the resulting segmentation is mislocated in FLAIR space. Setting both
    forms to ``img.affine`` keeps every downstream tool consistent. When the two
    forms already agree this is a no-op. See
    https://github.com/CompImg/LST-AI/issues/44.
    """
    img = nib.load(input_path)
    affine = img.affine
    code = int(img.header["sform_code"]) or int(img.header["qform_code"]) or 1
    img.set_qform(affine, code=code)
    img.set_sform(affine, code=code)
    nib.save(img, output_path)

# Which release hosts the weights, kept separate from the package version in setup.py.
# They are not the same thing and should not be assumed to move together: shipping an
# additional model is a minor package release but needs a new bundle, while a pure code
# fix is a package release against an unchanged bundle. Bump this only when the bundle's
# contents actually change.
#
# STAGING: served from the fork while the first author reviews. Flip DATA_REPO to
# CompImg/LST-AI once upstream publishes the same assets.
DATA_REPO = "jqmcginnis/LST-AI"
DATA_RELEASE = "v2.0.0"

# Contents of lst_data.zip at DATA_RELEASE: the PyTorch ensemble
# (UNet3D_MS_final_mdl{A,B,C}.pt) plus the MNI atlas. No compiled 'binaries' -- greedy is
# the picsl-greedy pip package now. The .pt were exported from the ONNX graphs by
# `python -m lst_ai.weights` and are tensor-for-tensor identical to them; those graphs
# stay downloadable as lst_data_onnx.zip on the same release for provenance, though
# nothing in this package needs them. See docs/pytorch-reimplementation.md.
DATA_URL = f"https://github.com/{DATA_REPO}/releases/download/{DATA_RELEASE}/lst_data.zip"

# Where the bundle is unpacked to, and read back from: next to this package. Note this is
# *not* where the `lst` script lives -- setup.py ships it via scripts=, so it is installed
# into .../bin, a directory that has nothing to do with the package and that a non-root
# user cannot write to. Resolving from the package instead keeps the download target and
# the read location the same in every install, and lets a container bake the bundle in.
#
# The package directory is not always writable, though: a system-wide install (sudo pip,
# a managed site-packages) leaves a non-root user unable to download the bundle on first
# run. resolve_data_dir() therefore falls back to the user cache directory in that case,
# and LST_AI_DATA_DIR overrides the choice entirely.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_data_dir(package_dir=_PACKAGE_DIR, environ=os.environ):
    """Pick the directory the model bundle lives in.

    In order: the LST_AI_DATA_DIR environment variable if set; the package directory
    when the bundle is already there (baked-in Docker images, existing installs) or when
    it is writable (venv installs); otherwise the user cache directory
    ($XDG_CACHE_HOME/lst_ai or ~/.cache/lst_ai), for installs whose site-packages the
    user cannot write to.
    """
    override = environ.get("LST_AI_DATA_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if all(os.path.isdir(os.path.join(package_dir, d)) for d in ("atlas", "model")):
        return package_dir
    if os.access(package_dir, os.W_OK):
        return package_dir
    cache_root = environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(cache_root, "lst_ai")


DATA_DIR = resolve_data_dir()


def download_data(path=DATA_DIR):
    """
    Downloads required model weights, binaries and atlas files for usage.
    """
    url = DATA_URL

    extract_path = path  # This is the base directory.
    os.makedirs(extract_path, exist_ok=True)
    # The zip lands in the directory it will be extracted to, not the current working
    # directory, which may be read-only or shared.
    target_path = os.path.join(extract_path, "lst_data.zip")

    atlas_path = os.path.join(extract_path, 'atlas')
    model_path = os.path.join(extract_path, 'model')

    paths_to_check = [atlas_path, model_path]

    # Check if all paths exist.
    if not all(os.path.exists(path) for path in paths_to_check):
        print("Downloading data...")
        # Download the zip file if it doesn't exist.
        if not os.path.exists(target_path):
            with request.urlopen(url) as response, open(target_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)

        # Unzip the file to the base directory.
        with zipfile.ZipFile(target_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # Remove the ZIP file after extracting its contents.
        os.remove(target_path)
        print("Completed.")
