import subprocess
import shlex

def run_hdbet(input_image, output_image, use_gpu=False, fast_mode=False):
    if use_gpu:
        device = '1'
    else:
        device = 'cpu'

    if fast_mode:
        mode = 'fast'
    else:
        mode = 'accurate'

    bet_call = f"hd-bet -i {input_image} -device {device} -mode {mode} -tta 0 -o {output_image}"
    subprocess.run(shlex.split(bet_call), check=True)

    # check @Bene if this is required
    """
    brain_mask_arr = nib.load(atlas_mask).get_fdata()

    flair_nib = nib.load(mni_flair)
    flair_arr = np.multiply(flair_nib.get_fdata(), brain_mask_arr)
    nib.save(nib.Nifti1Image(flair_arr.astype(np.float32), flair_nib.affine, flair_nib.header), mni_flair)

    t1_nib = nib.load(mni_t1)
    t1_arr = np.multiply(t1_nib.get_fdata(), brain_mask_arr)
    nib.save(nib.Nifti1Image(t1_arr.astype(np.float32), t1_nib.affine, t1_nib.header), mni_t1)
    """
