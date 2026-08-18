from io import open
import os
import zipfile
from urllib import request

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
# `python -m LST_AI.weights` and are tensor-for-tensor identical to them; those graphs
# stay downloadable as lst_data_onnx.zip on the same release for provenance, though
# nothing in this package needs them. See docs/pytorch-reimplementation.md.
DATA_URL = f"https://github.com/{DATA_REPO}/releases/download/{DATA_RELEASE}/lst_data.zip"


def download_data(path):
    """
    Downloads required model weights, binaries and atlas files for usage.
    """
    url = DATA_URL

    target_path = "lst_data.zip"
    extract_path = path  # This is the base directory.

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
