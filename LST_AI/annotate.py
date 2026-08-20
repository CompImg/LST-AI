"""
This module contains label information for Output and MSMask.

Output labels:
    1  ==  Periventricular
    2  ==  Juxtacortical
    3  ==  Subcortical
    4  ==  Infratentorial

MSMask labels:
    1  ==  CSF
    2  ==  GM
    3  ==  WM
    4  ==  Ventricles
    5  ==  Infratentorial

"""
import os
import subprocess
import shutil
import sys

import nibabel as nib
import numpy as np
from skimage.measure import label
from skimage.morphology import dilation

from LST_AI.fastsurfer import ensure_fastsurfer

# FastSurfer/FreeSurfer aseg structure_label -> class_label mapping
# class_label key: 2=juxtacortical, 3=subcortical, 4=periventricular, 5=infratentorial
FASTSURFER_LABEL_MAPPING = {
    2: 3,   # Left-Cerebral-White-Matter
    3: 2,   # Left-Cortex
    4: 4,   # Left-Lateral-Ventricle
    7: 5,   # Left-Cerebellum-White-Matter
    8: 5,   # Left-Cerebellum-Cortex
    10: 3,  # Left-Thalamus
    11: 3,  # Left-Caudate
    12: 3,  # Left-Putamen
    13: 3,  # Left-Pallidum
    14: 4,  # 3rd-Ventricle
    15: 5,  # 4th-Ventricle
    16: 5,  # Brain-Stem
    17: 3,  # Left-Hippocampus
    18: 3,  # Left-Amygdala
    24: 4,  # CSF
    26: 3,  # Left-Accumbens-area
    28: 3,  # Left-VentralDC
    31: 4,  # Left-choroid-plexus
    41: 3,  # Right-Cerebral-White-Matter
    42: 2,  # Right-Cortex
    43: 4,  # Right-Lateral-Ventricle
    46: 5,  # Right-Cerebellum-White-Matter
    47: 5,  # Right-Cerebellum-Cortex
    49: 3,  # Right-Thalamus
    50: 3,  # Right-Caudate
    51: 3,  # Right-Putamen
    52: 3,  # Right-Pallidum
    53: 3,  # Right-Hippocampus
    54: 3,  # Right-Amygdala
    58: 3,  # Right-Accumbens-area
    60: 3,  # Right-VentralDC
    63: 4,  # Right-choroid-plexus
    77: 3,  # WM-hypointensities
}

def get_fastsurfer(im_path, seg_path, device):
    """
    This function runs the FastSurfer cross sectional pipeline for the provided image. 

    Parameters:
    -----------
    im_path : str
         Path of the T1w image for which FastSurfer segmentation should be calculated.
    seg_path : str
         Path where the FastSurfer segmentation should be saved.
    
    Returns:
    --------
    None 
        This function produces a FastSurfer segmentation. 
    """
          
    # define output folder "fastsurfer" for FastSurfer results, which simplifies data housekeeping
    sd_path = os.path.dirname(im_path)
    sid = 'fastsurfer'

    # The FastSurfer LST-AI ships and pins, installed with it (setup.py) -- never one the
    # machine happens to have. Resolved to an absolute path rather than left to PATH, so
    # neither the user's shell nor their environment decides which one annotates.
    fastsurfer_home = ensure_fastsurfer()

    # LST-AI's --device is a bare GPU id ('0') or 'cpu'; FastSurfer names devices the way
    # torch does, where a bare id is not a device at all ('0' raises "Invalid device
    # string"). Translate, as LST_AI/segment.py does, and pass a torch-style string
    # through untouched.
    dev = str(device).strip().lower()
    fs_device = dev if dev in ('cpu', 'mps', 'auto') or dev.startswith('cuda') else f'cuda:{dev}'

    # call FastSurfer cross sectional segmentation
    print(f'Running FastSurfer ...')
    cmd = [
        'timeout', '15000', str(fastsurfer_home / 'run_fastsurfer.sh'),
        '--t1', im_path,
        '--sd', sd_path,
        '--sid', sid,
        '--device', fs_device,
        # Without this run_fastsurfer.sh runs whatever `python3` resolves to on PATH,
        # which is this interpreter only when the environment happens to be activated --
        # and FastSurfer's dependencies were installed into *this* one.
        '--py', sys.executable,
        '--seg_only',
        '--threads', '4',
        '--no_cereb',
        '--no_hypothal',
        '--no_cc',
        '--no_biasfield',
        '--keepgeom',
    ]

    # FastSurfer refuses to start as root unless told otherwise, because everything it
    # writes would come out root-owned. Inside the Docker image LST-AI *is* root, so that
    # refusal is the whole annotation stage failing; opt in exactly where it applies.
    if os.geteuid() == 0:
        cmd.append('--allow_root')

    # Overwrite any inherited FASTSURFER_HOME with the tree we resolved. Unset, the script
    # derives it from its own location and gets the same answer; left as the user set it,
    # it would send this run off to whatever other FastSurfer that points at.
    env = {**os.environ, 'FASTSURFER_HOME': str(fastsurfer_home)}

    subprocess.run(cmd, check=True, env=env)

    # check if folder contains aseg.auto_noCCseg.mgz file, indicating that FastSurfer successfully finished
    # and convert to nifti
    seg_mgz_path = os.path.join(sd_path, sid, 'mri', 'aseg.auto_noCCseg.mgz')
    if os.path.exists(seg_mgz_path):
        seg_mgz = nib.load(seg_mgz_path)
        nib.save(seg_mgz, seg_path)
    else:
        raise ValueError(f'{seg_mgz_path}: FastSurfer segmentation failed!')
    
    # Housekeeping: remove the remaining FastSurfer output folder.
    fastsurfer_output_folder = os.path.join(sd_path, sid)
    if os.path.exists(fastsurfer_output_folder):
        shutil.rmtree(fastsurfer_output_folder)

def convert_labels(segmentation, label_mapping=FASTSURFER_LABEL_MAPPING):
    """
    This function converts the labels of a brain segmentation mask according to a label mapping. 
    The function takes as input the segmentation and a dictionary with the label mapping. 

    Parameters:
    -----------
    seg_mask : 3D array
        3D array containing the segmentation mask with original labels.
    label_mapping : dict
        Dictionary containing the mapping from original labels to new labels.
    
    Returns:
    --------
    out_mask :  
        This function produces a 3D array containing the segmentation mask with converted labels according to the provided mapping.
    """
    # Build a lookup table (LUT) for fast vectorized remapping
    max_label = max(label_mapping.keys())
    lut = np.zeros(max_label + 1, dtype=np.int32)
    for struct_label, class_label in label_mapping.items():
        lut[struct_label] = class_label

    # Apply LUT mapping to segmentation
    segmentation = segmentation.astype(np.int32)  # Ensure the segmentation mask is of integer type for indexing
    out_mask = lut[segmentation] 

    return out_mask

def annotate_lesions(t1w_im, lesion_mask, t1w_seg, lesion_mask_annotated, device, label_mapping=FASTSURFER_LABEL_MAPPING):
    """
    Annotate lesions in a given image using an atlas.

    Parameters:
    -----------
    t1w_im : str
        Path to the patient's T1-weighted image.
    lesion_mask : str
        Path to the lesion mask in the same space.
    t1w_seg : str
        Path to the FastSurfer segmentation.
    lesion_mask_annotated : str
        Path where the annotated lesion mask will be saved.
    device : str
        Device to run FastSurfer on (e.g., 'cpu' or '0').
    label_mapping : str
        Path to the label mapping file that defines the correspondence between atlas labels and lesion types.

    Description:
    ------------
    The function performs several tasks:
    1. Applies segmentation of T1w image using FastSurfer
    2. Converts the segmentation to a subject-specific MS region segmentation.
    3. Annotates lesions based on the overlap with the MS region segmentation.
    4. Saves the annotated lesion segmentation.

    Returns:
    --------
    None

    """
    if device == '0':
        device = f'cuda:{device}'

    # Load lesion segmentation
    les_seg_nib = nib.load(lesion_mask)
    les_seg = les_seg_nib.get_fdata()
    les_seg[les_seg > 0] = 1  # Make sure seg is binary

    # Call FastSurfer to segment the T1w image
    get_fastsurfer(t1w_im, t1w_seg, device)

    # Convert FastSurfer segmentation to subject-specific MS region segmentation
    seg_nib = nib.load(t1w_seg)
    seg_data = seg_nib.get_fdata()
    msmask = convert_labels(seg_data, label_mapping)
    # set all lesion voxels to 3 (WM) in the MSMask to avoid misclassification
    # but not for IT lesions, which are in the infratentorial region
    # First, define a temporary lesion mask in which the voxels that are in the infratentorial region are set to 5 (IT)
    les_seg_temp = np.copy(les_seg)

    # check which lesions overlap with the infratentorial region (label 5 in MSMask) and set them to 5 in the temporary lesion mask
    # use connected components to identify individual lesions
    les_temp_label = label(les_seg_temp, connectivity=3)
    for lesion_ctr in range(1, les_temp_label.max() + 1):
        # We create a temporary binary mask
        # for each lesion & dilate it by 1
        # (to "catch" adjacent structures)
        temp_mask = np.zeros(les_seg.shape)
        temp_mask[les_temp_label == lesion_ctr] = 1
        temp_mask_dil = dilation(
            temp_mask, footprint=np.ones((3, 3, 3))).astype(np.uint8)

        if 5 in msmask[temp_mask_dil == 1]:
            les_seg_temp[les_temp_label == lesion_ctr] = 5

    # Next, all lesion voxels that are not in the IT region still have label 1, 
    # and we can set them to 3 (WM) in the MSMask
    msmask[les_seg_temp == 1] = 3

    # save the converted segmentation as a temporary NIfTI file
    msmask_nib = nib.Nifti1Image(msmask.astype(np.uint8), seg_nib.affine, seg_nib.header)
    temp_seg_path = str(t1w_seg).replace('.nii.gz', '_MS-mask.nii.gz')
    nib.save(msmask_nib, temp_seg_path)

    les_seg_label = label(les_seg, connectivity=3)
    for lesion_ctr in range(1, les_seg_label.max() + 1):
        # We create a temporary binary mask
        # for each lesion & dilate it by 1
        # (to "catch" adjacent structures)
        temp_mask = np.zeros(les_seg.shape)
        temp_mask[les_seg_label == lesion_ctr] = 1
        temp_mask_dil = dilation(
            temp_mask, footprint=np.ones((3, 3, 3))).astype(np.uint8)

        if 4 in msmask[temp_mask_dil == 1]:
            les_seg[temp_mask == 1] = 1  # PV

        elif 5 in msmask[temp_mask_dil == 1]:
            les_seg[temp_mask == 1] = 4  # IT

        elif 2 in msmask[temp_mask_dil == 1]:
            les_seg[temp_mask == 1] = 2  # JC

        else:
            les_seg[temp_mask == 1] = 3  # SC

    # Saving & warping back to T1w space
    nib.save(nib.Nifti1Image(les_seg.astype(np.uint8),
                             les_seg_nib.affine, les_seg_nib.header),
                             lesion_mask_annotated)


if __name__ == "__main__":

    # Only for testing purposes
    lst_dir = os.getcwd()
    parent_directory = os.path.dirname(lst_dir)

    # annotate lesion test data
    t1w_native_path = os.path.join(parent_directory, "testing", "annotation",
                              "sub-msseg-test-center01-02_ses-01_space-mni_t1.nii.gz")
    seg_native_path = os.path.join(parent_directory, "testing", "annotation",
                              "sub-msseg-test-center01-02_ses-01_space-mni_seg-manual.nii.gz")
    annotated_seg_path = os.path.join(parent_directory, "annotated_segmentation.nii.gz")

    annotate_lesions(t1w_im=t1w_native_path,
                     lesion_mask=seg_native_path,
                     t1w_seg=t1w_native_path,
                     lesion_mask_annotated=annotated_seg_path,
                     label_mapping=FASTSURFER_LABEL_MAPPING)

    # check and remove testing results
    gt = os.path.join(parent_directory, "testing", "annotation",
                      "sub-msseg-test-center01-02_ses-01_space-mni_annotated_seg.nii.gz")
    array_gt = nib.load(gt).get_fdata()
    array_pred = nib.load(annotated_seg_path).get_fdata()
    os.remove(annotated_seg_path)
    np.testing.assert_array_equal(array_gt, array_pred)
