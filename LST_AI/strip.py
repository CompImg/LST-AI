import subprocess
import shlex
import nibabel as nib
import numpy as np

def run_hdbet(input_image, output_image, device, mode="accurate"):
    assert mode in ["accurate","fast"], 'Unkown HD-BET mode. Please choose either "accurate" or "fast"'

    if "cpu" in str(device).lower():
        bet_call = f"hd-bet -i {input_image} -device cpu -mode {mode} -tta 0 -o {output_image}"
    else:
        bet_call = f"hd-bet -i {input_image} -device {device} -mode accurate -tta 1 -o {output_image}"

    subprocess.run(shlex.split(bet_call), check=True)

def apply_mask(input_image, mask, output_image):
    brain_mask_arr = nib.load(mask).get_fdata()
    image_nib = nib.load(input_image)
    image_arr = np.multiply(image_nib.get_fdata(), brain_mask_arr)
    nib.save(nib.Nifti1Image(image_arr.astype(np.float32), image_nib.affine, image_nib.header), output_image)
