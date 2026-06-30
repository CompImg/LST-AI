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


def download_data(path):
    """
    Downloads required model weights, binaries and atlas files for usage.
    """
    url = "https://github.com/CompImg/LST-AI/releases/download/v1.0.0/lst_data.zip"

    target_path = "lst_data.zip"
    extract_path = path  # This is the base directory.

    atlas_path = os.path.join(extract_path, 'atlas')
    binary_path = os.path.join(extract_path, 'binaries')
    model_path = os.path.join(extract_path, 'model')

    paths_to_check = [atlas_path, binary_path, model_path]

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
