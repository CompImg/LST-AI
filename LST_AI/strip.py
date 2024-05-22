import subprocess
import shlex
import nibabel as nib
import numpy as np

def run_hdbet(input_image, output_image, device, mode="accurate"):
    """
    Runs the HD-BET tool to perform brain extraction on an input image.

    Parameters:
    input_image (str): Path to the input image file.
    output_image (str): Path for the output image file.
    device (str): The device to use for computation, either a GPU device number or 'cpu'.
    mode (str, optional): Operation mode of HD-BET. Can be 'accurate' or 'fast'. Default is 'accurate'.

    Raises:
    AssertionError: If an unknown mode is provided.

    This function utilizes HD-BET, a tool for brain extraction from MRI images. Depending on the chosen mode
    and device, it executes the appropriate command.
    """
    assert mode in ["accurate","fast"], 'Unknown HD-BET mode. Please choose either "accurate" or "fast"'

    if "cpu" in str(device).lower():
        bet_call = f"hd-bet -i {input_image} -device cpu -mode {mode} -tta 0 -o {output_image}"
    else:
        bet_call = f"hd-bet -i {input_image} -device {device} -mode accurate -tta 1 -o {output_image}"

    subprocess.run(shlex.split(bet_call), check=True)

def apply_mask(input_image, mask, output_image):
    """
    Applies a mask to an input image and saves the result.

    Parameters:
    input_image (str): Path to the input image file.
    mask (str): Path to the mask image file.
    output_image (str): Path for the output image file where the masked image will be saved.

    This function loads a brain mask and an input image, applies the mask to the input image,
    and then saves the result. The mask and the input image are expected to be in a compatible format
    and spatial alignment.
    """
    brain_mask_arr = nib.load(mask).get_fdata()
    image_nib = nib.load(input_image)
    image_arr = np.multiply(np.squeeze(image_nib.get_fdata()), np.squeeze(brain_mask_arr))
    nib.save(nib.Nifti1Image(image_arr.astype(np.float32), image_nib.affine, image_nib.header), output_image)
