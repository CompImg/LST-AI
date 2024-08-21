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
import shlex
import subprocess
import os

import nibabel as nib
import numpy as np
from skimage.measure import label
from skimage.morphology import binary_dilation


def annotate_lesions(atlas_t1, atlas_mask, t1w_native, seg_native, out_atlas_warp,
                     out_atlas_mask_warped, out_annotated_native, n_threads=8):
    """
    Annotate lesions in a given image using an atlas.

    Parameters:
    -----------
    atlas_t1: str
        Path to the atlas T1-weighted image (in MNI space).
    atlas_mask: str
        Path to the atlas mask image.
    t1w_native: str
        Path to the T1-weighted image in native space.
    seg_native: str
        Path to the segmentation image in native space.
    out_atlas_warp: str
        Path where the warp from the atlas to the patient T1 should be saved.
    out_atlas_mask_warped: str
        Path where the warped atlas mask should be saved.
    out_annotated_native: str
        Path where the annotated lesion segmentation in native space should be saved.

    Description:
    ------------
    The function performs several tasks:
    1. Registers the atlas to the patient's T1 image using a two-step greedy algorithm.
    2. Warps the atlas mask to the patient's space.
    3. Annotates lesions based on the overlap with the atlas mask.
    4. Saves the annotated segmentation in native space.

    Returns:
    --------
    None

    """

    # Register Atlas -> Patient_T1 using greedy (two-step: rigid first, then deformable)
    deformable_call = (
        f"greedy -d 3 -m WNCC 2x2x2 -sv -n 100x50x10"
        f" -i {t1w_native} {atlas_t1}"
        f" -o {out_atlas_warp}"
        f" -threads {n_threads}"
    )
    subprocess.run(shlex.split(deformable_call), check=True)

    # Warp MSmask in patient space
    warp_call = (
        f"greedy -d 3 -rf {t1w_native} -ri LABEL 0.2vox"
        f" -rm {atlas_mask} {out_atlas_mask_warped}"
        f" -r {out_atlas_warp}"
        f" -threads {n_threads}"
    )

    subprocess.run(shlex.split(warp_call), check=True)

    # Load segmentation and msmask and location-label lesions
    seg_nib = nib.load(seg_native)
    seg = seg_nib.get_fdata()
    seg[seg > 0] = 1  # Make sure seg is binary
    msmask = nib.load(out_atlas_mask_warped).get_fdata()

    seg_label = label(seg, connectivity=3)
    for lesion_ctr in range(1, seg_label.max() + 1):
        # We create a temporary binary mask
        # for each lesion & dilate it by 1
        # (to "catch" adjacent structures)
        temp_mask = np.zeros(seg.shape)
        temp_mask[seg_label == lesion_ctr] = 1
        temp_mask_dil = binary_dilation(
            temp_mask, footprint=np.ones((3, 3, 3))).astype(np.uint8)

        if 4 in msmask[temp_mask_dil == 1]:
            seg[temp_mask == 1] = 1  # PV

        elif 2 in msmask[temp_mask_dil == 1]:
            seg[temp_mask == 1] = 2  # JC

        elif 5 in msmask[temp_mask_dil == 1]:
            seg[temp_mask == 1] = 4  # IT

        else:
            seg[temp_mask == 1] = 3  # SC

    # Saving & warping back to T1w space
    nib.save(nib.Nifti1Image(seg.astype(np.uint8),
                             seg_nib.affine, seg_nib.header),
                             out_annotated_native)


if __name__ == "__main__":

    # Only for testing purposes
    lst_dir = os.getcwd()
    parent_directory = os.path.dirname(lst_dir)
    atlas_t1w_path = os.path.join(parent_directory, "atlas", "sub-mni152_space-mni_t1.nii.gz")
    atlas_mask_path = os.path.join(parent_directory, "atlas", "sub-mni152_space-mni_msmask.nii.gz")
    out_atlas_warp_path = os.path.join(parent_directory, "warp_field.nii.gz")
    out_atlas_mask_warped_path = os.path.join(parent_directory,
                                              "tmp_mask.nii.gz")

    # annotate lesion test data
    t1w_native_path = os.path.join(parent_directory, "testing", "annotation",
                              "sub-msseg-test-center01-02_ses-01_space-mni_t1.nii.gz")
    seg_native_path = os.path.join(parent_directory, "testing", "annotation",
                              "sub-msseg-test-center01-02_ses-01_space-mni_seg-manual.nii.gz")
    annotated_seg_path = os.path.join(parent_directory, "annotated_segmentation.nii.gz")

    annotate_lesions(atlas_t1=atlas_t1w_path,
                     atlas_mask=atlas_mask_path,
                     t1w_native=t1w_native_path,
                     seg_native=seg_native_path,
                     out_atlas_warp=out_atlas_warp_path,
                     out_atlas_mask_warped=out_atlas_mask_warped_path,
                     out_annotated_native=annotated_seg_path,
                     n_threads=6)

    # check and remove testing results
    gt = os.path.join(parent_directory, "testing", "annotation",
                      "sub-msseg-test-center01-02_ses-01_space-mni_annotated_seg.nii.gz")
    array_gt = nib.load(gt).get_fdata()
    array_pred = nib.load(annotated_seg_path).get_fdata()
    os.remove(out_atlas_mask_warped_path)
    os.remove(out_atlas_warp_path)
    os.remove(annotated_seg_path)
    np.testing.assert_array_equal(array_gt, array_pred)
