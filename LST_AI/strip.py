import os
import nibabel as nib
import numpy as np


def run_hdbet(input_image, output_image, device, mode="accurate"):
    """
    Runs HD-BET to perform brain extraction on an input image.

    Uses ``brainles_hd_bet``, a pinned, pip-installable fork of HD-BET v1. Two reasons
    for that over ``hd-bet`` from PyPI:

      * it is the version LST-AI v1.0.0/v1.1.0 shipped against (the released README
        pinned HD-BET at commit ae16068, Aug 2023, which predates v2), so masks match
        the pipeline the released weights were validated with. HD-BET v2 is a different
        model and strips differently, which propagates into the segmentation;
      * it publishes a ``py3-none-any`` wheel with no compiled dependencies, so it
        installs on linux/arm64. ``hd-bet`` 2.0.1 ships an sdist only.

    Called through the Python API rather than the console script: ``brainles_hd_bet``
    0.0.11's ``hd-bet`` entry point is broken (it imports ``maybe_mkdir_p`` from
    ``utils``, which no longer exists), and going direct avoids a subprocess and a PATH
    dependency anyway.

    Parameters:
    input_image (str): Path to the input image file.
    output_image (str): Path for the brain-extracted output image. The binary brain
        mask is written alongside as ``<output>_bet.nii.gz``, keeping the filename the
        rest of the pipeline expects regardless of HD-BET's own convention.
    device (str): GPU id (e.g. '0') or 'cpu'.
    mode (str, optional): 'accurate' (test-time augmentation on) or 'fast' (TTA off).
    """
    from brainles_hd_bet import run_hd_bet

    assert mode in ["accurate", "fast"], 'Unknown HD-BET mode. Choose "accurate" or "fast".'

    on_cpu = "cpu" in str(device).lower()
    run_hd_bet(
        mri_fnames=[str(input_image)],
        output_fnames=[str(output_image)],
        mode=mode,
        device="cpu" if on_cpu else int(device),
        postprocess=False,
        do_tta=(mode == "accurate") and not on_cpu,   # TTA on CPU is prohibitively slow
        keep_mask=True,
        overwrite=True,
    )

    # HD-BET v1 writes the mask as <output>_mask.nii.gz; the pipeline expects
    # <output>_bet.nii.gz (the v2 name). Normalise so callers need not care.
    expected = str(output_image).replace(".nii.gz", "_bet.nii.gz")
    if not os.path.exists(expected):
        produced = str(output_image).replace(".nii.gz", "_mask.nii.gz")
        if not os.path.exists(produced):
            raise FileNotFoundError(
                f"HD-BET produced no brain mask for {input_image}; expected {produced}"
            )
        os.replace(produced, expected)


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
