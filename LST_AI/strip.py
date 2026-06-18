import os
import subprocess
import shlex
import nibabel as nib
import numpy as np


def run_hdbet(input_image, output_image, device, mode="accurate"):
    """
    Runs HD-BET (v2, PyPI) to perform brain extraction on an input image.

    Parameters:
    input_image (str): Path to the input image file.
    output_image (str): Path for the brain-extracted output image. The binary brain
        mask is written alongside as ``<output>_bet.nii.gz`` (HD-BET v2 convention).
    device (str): GPU id (e.g. '0') or 'cpu'.
    mode (str, optional): 'accurate' (test-time augmentation on) or 'fast' (TTA off).
        Default 'accurate'. HD-BET v2 has a single model, so this maps to the
        --disable_tta flag rather than the v1 -mode option.

    Notes
    -----
    HD-BET v2 selects the GPU via CUDA_VISIBLE_DEVICES + ``-device cuda`` (it no
    longer takes a GPU index directly), so a numeric ``device`` is honoured by
    exposing only that GPU to the subprocess. CPU always disables TTA (recommended).
    """
    assert mode in ["accurate", "fast"], 'Unknown HD-BET mode. Choose "accurate" or "fast".'

    env = dict(os.environ)
    if "cpu" in str(device).lower():
        bet_call = f"hd-bet -i {input_image} -o {output_image} -device cpu --disable_tta --save_bet_mask"
    else:
        tta = "--disable_tta" if mode == "fast" else ""
        bet_call = f"hd-bet -i {input_image} -o {output_image} -device cuda {tta} --save_bet_mask"
        env["CUDA_VISIBLE_DEVICES"] = str(device)  # honour the requested GPU id

    subprocess.run(shlex.split(bet_call), check=True, env=env)


def apply_mask(input_image, mask, output_image):
    """
    Applies a brain mask to an input image and saves the result.

    Parameters:
    input_image (str): Path to the input image file.
    mask (str): Path to the brain mask (HD-BET ``*_bet.nii.gz``).
    output_image (str): Path for the masked (brain-extracted) output.

    The mask and the input image are expected to be in compatible format / alignment.
    """
    brain_mask_arr = nib.load(mask).get_fdata()
    image_nib = nib.load(input_image)
    image_arr = np.multiply(np.squeeze(image_nib.get_fdata()), np.squeeze(brain_mask_arr))
    nib.save(nib.Nifti1Image(image_arr.astype(np.float32), image_nib.affine, image_nib.header), output_image)
