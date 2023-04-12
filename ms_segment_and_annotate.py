"""
REQUIRES
> greedy
> HD-BET
> MNI atlas files
> Model files
"""

import argparse
import os
import subprocess
import shlex
import nibabel as nib
import numpy as np
from skimage.measure import label
from skimage.morphology import binary_dilation
import tensorflow as tf
import tensorflow_addons as tfa

"""
Output labels:
    1 == Periventricular
    2 == Juxtacortical
    3 == Subcortical
    4 == Infratentorial

MSMask labels:
    1 == CSF
    2 == GM
    3 == WM
    4 == Ventricles
    5 == Infratentorial
"""

"""
A COUPLE OF PARAMETERS
"""
input_shape = (192,192,192) #Expected input (single 3D volume) for the UNet
unet_mdls = ["mdl_best_val_dice_ep500.h5","UNet3D_MS_final_DBCE_both.h5","UNet3D_MS_final_nnUNet3D.h5"] #Here you can add / remove models
threshold = 0.5 #How to binarize the joint segmentation

def mni_registration(org_t1,org_flair,use_gpu):
    """
    INPUT:
    ------
    > Paths to T1 and FLAIR images (in native space)
    > Whether to use GPU (for HD-BET) or not
    
    RETURNS:
    --------
    > T1 and FLAIR, preprocessed in MNI space (rigid reg + skullstripping; file paths)
    > The affine to warp FLAIR->MNI (needed later to warp back the segmentation to FLAIR space; file path)
    """
 
    #Define some variables    
    script_dir = os.getcwd() + "/"
    atlas_t1 = script_dir + "sub-mni152_space-mni_t1.nii.gz"
    greedy = script_dir + "greedy"
    mni_t1 = script_dir + "mni_t1.nii.gz"
    mni_flair = script_dir + "mni_flair.nii.gz"
    flair_2_mni_affine = script_dir + "flair_2_mni_affine.mat"

    #Register T1 -> Atlas_T1 (6 DOF)
    rigid_call = greedy + " -d 3 -a -dof 6 -m NMI -ia-image-centers -n 100x50x10 -i " + atlas_t1 + " " + org_t1 + " -o " + script_dir + "tmp_affine.mat"
    subprocess.run(shlex.split(rigid_call))
    warp_call = greedy + " -d 3 -rf " + atlas_t1 + " -rm " + org_t1 + " " + mni_t1 + " -r " + script_dir + "tmp_affine.mat"
    subprocess.run(shlex.split(warp_call))

    #Register FLAIR -> T1 in atlas space and save the affine for later
    rigid_call = greedy + " -d 3 -a -dof 6 -m NMI -ia-image-centers -n 100x50x10 -i " + mni_t1 + " " + org_flair + " -o " + flair_2_mni_affine
    subprocess.run(shlex.split(rigid_call))
    warp_call = greedy + " -d 3 -rf " + atlas_t1 + " -rm " + org_flair + " " + mni_flair + " -r " + flair_2_mni_affine
    subprocess.run(shlex.split(warp_call))

    #Skullstrip (using HD-BET)
    if use_gpu:
        bet_call = "hd-bet -i " + mni_t1
    else:
        bet_call = "hd-bet -i " + mni_t1 + " -device cpu -mode fast -tta 0"
    subprocess.run(shlex.split(bet_call))

    #Mask images
    brain_mask_arr = nib.load(script_dir + "mni_t1_bet_mask.nii.gz").get_fdata()
    
    flair_nib = nib.load(mni_flair)
    flair_arr = np.multiply(flair_nib.get_fdata(),brain_mask_arr)
    nib.save(nib.Nifti1Image(flair_arr.astype(np.float32),flair_nib.affine,flair_nib.header),mni_flair)

    t1_nib = nib.load(mni_t1)
    t1_arr = np.multiply(t1_nib.get_fdata(),brain_mask_arr)
    nib.save(nib.Nifti1Image(t1_arr.astype(np.float32),t1_nib.affine,t1_nib.header),mni_t1)

    #Housekeeping
    os.remove(script_dir + "mni_t1_bet.nii.gz")
    os.remove(script_dir + "tmp_affine.mat")
    os.remove(script_dir + "mni_t1_bet_mask.nii.gz")

    return mni_t1, mni_flair, flair_2_mni_affine

def unet_segmentation(mni_t1,mni_flair):
    """
    INPUT:
    ------
    > Paths to T1 and FLAIR images (in MNI space)
    
    RETURNS:
    --------
    > Binary segmentation mask (in MNI space; file path)
    """

    def adapt_shape(img_arr):
        #Crops input image array to target shape; also returns information how to re-zero-pad
        difference_0 = img_arr.shape[0] - input_shape[0]
        difference_0_l = (difference_0 // 2)+(difference_0 % 2)
        difference_0_r = (difference_0 // 2)

        difference_1 = img_arr.shape[1] - input_shape[1]
        difference_1_l = (difference_1 // 2)+(difference_1 % 2)
        difference_1_r = (difference_1 // 2)

        difference_2 = img_arr.shape[2] - input_shape[2]
        difference_2_l = (difference_2 // 2)+(difference_2 % 2)
        difference_2_r = (difference_2 // 2)

        img_arr_cropped = img_arr[difference_0_l : img_arr.shape[0] - difference_0_r, difference_1_l : img_arr.shape[1] - difference_1_r,difference_2_l : img_arr.shape[2] - difference_2_r]

        return img_arr_cropped.astype(np.float32), [difference_0_l,difference_0_r,difference_1_l,difference_1_r,difference_2_l,difference_2_r]

    def preprocess_intensities(img_arr):
        #Standardize image intensities to [0;1]
        temp_bm = np.zeros(img_arr.shape)
        temp_bm[img_arr != 0] = 1
        img_arr = np.clip(img_arr, np.percentile(img_arr[temp_bm != 0],0.5),np.percentile(img_arr[temp_bm != 0],99.5) )
        img_arr -= img_arr[temp_bm == 1].min()
        img_arr = img_arr / img_arr[temp_bm == 1].max()
        img_arr *= temp_bm
        
        return img_arr.astype(np.float32)

    mni_seg = os.getcwd() + "/mni_seg.nii.gz"

    #Load and preprocess images
    t1_nib = nib.load(mni_t1)
    t1 = t1_nib.get_fdata()    
    flair = nib.load(mni_flair).get_fdata()

    t1, shape_lst = adapt_shape(t1)
    flair, _ = adapt_shape(flair)
    
    t1 = preprocess_intensities(t1)
    flair = preprocess_intensities(flair)
    
    joint_seg = np.zeros(t1.shape)
    for model in unet_mdls:
            mdl = tf.keras.models.load_model(os.getcwd() + "/" + model,compile=False)

            img_image = np.stack([flair,t1],axis=-1)
            img_image = np.expand_dims(img_image,axis=0)

            out_seg = mdl(img_image)[0] #Will return a len(2) list of [out_seg,out_ds]
            out_seg = np.squeeze(out_seg)

            out_binary = np.zeros(t1.shape)
            out_binary[out_seg > threshold] = 1

            joint_seg += out_seg
        
    joint_seg /= len(unet_mdls)

    out_binary = np.zeros(t1.shape)
    out_binary[joint_seg > threshold] = 1

    out_binary = np.pad(out_binary,((shape_lst[0],shape_lst[1]),(shape_lst[2],shape_lst[3]),(shape_lst[4],shape_lst[5])),'constant', constant_values=0.)
    nib.save(nib.Nifti1Image(out_binary.astype(np.uint8),t1_nib.affine,t1_nib.header),mni_seg)

    return mni_seg

def annotate_lesions(mni_t1,mni_seg,flair_2_mni_affine,org_flair,org_seg):
    """
    INPUT:
    ------
    > Paths to T1 and segmentation images (in MNI space)
    > Paths to FLAIR->MNI affine and native FLAIR
    > Path to where the labelled output segmentation (in FLAIR space) should be saved  
    """

    script_dir = os.getcwd() + "/"
    atlas_t1 = script_dir + "sub-mni152_space-mni_t1bet.nii.gz"
    atlas_warp = script_dir + "warp_field.nii.gz"
    atlas_mask = script_dir + "sub-mni152_space-mni_msmask.nii.gz"
    atlas_mask_warped = script_dir + "tmp_mask.nii.gz"
    greedy = script_dir + "greedy"

    #Register Atlas -> Patient_T1 using greedy (two-step: rigid first, then deformable)
    deformable_call = greedy + " -d 3 -m WNCC 2x2x2 -sv -n 100x50x10 -i " + mni_t1 + " " + atlas_t1 + " -o " + atlas_warp
    print(deformable_call)
    subprocess.run(shlex.split(deformable_call))

    #Warp MSmask in patient space
    warp_call = greedy + " -d 3 -rf " + mni_t1 + " -ri LABEL 0.2vox -rm " + atlas_mask + " " + atlas_mask_warped + " -r " + atlas_warp
    subprocess.run(shlex.split(warp_call))

    #Load segmentation and msmask and location-label lesions
    seg_nib = nib.load(mni_seg)
    seg = seg_nib.get_fdata()
    seg[seg > 0] = 1 #Make sure seg is binary
    msmask = nib.load(atlas_mask_warped).get_fdata()

    seg_label = label(seg,connectivity=3)
    for lesion_ctr in range(1,seg_label.max()+1):
        #We create a temporary binary mask for each leasion & dilate it by 1 (to "catch" adjacent structures)
        temp_mask = np.zeros(seg.shape)
        temp_mask[seg_label == lesion_ctr] = 1
        temp_mask_dil = binary_dilation(temp_mask,selem=np.ones((3,3,3))).astype(np.uint8)

        if 4 in msmask[temp_mask_dil == 1]:
            seg[temp_mask == 1] = 1 #PV

        elif 2 in msmask[temp_mask_dil == 1]:
            seg[temp_mask == 1] = 2 #JC

        elif 5 in msmask[temp_mask_dil == 1]:
            seg[temp_mask == 1] = 4 #IT

        else:
            seg[temp_mask == 1] = 3 #SC

    #Saving & warping back to FLAIR space
    nib.save(nib.Nifti1Image(seg.astype(np.uint8),seg_nib.affine,seg_nib.header), mni_seg)
    warp_back_call = greedy + " -d 3 -rf " + org_flair + " -ri LABEL 0.2vox -rm " + mni_seg + " " + org_seg + " -r " + flair_2_mni_affine + ",-1"
    subprocess.run(shlex.split(warp_back_call))

    #Housekeeping
    os.remove(atlas_warp)
    os.remove(atlas_mask_warped)
    os.remove(flair_2_mni_affine)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment and label MS lesions according to McDonald criteria')
    parser.add_argument('-t1',
                        dest='t1',
                        help='Path to T1 image',
                        type=str,
                        required=True)
    parser.add_argument('-flair',
                        dest='flair',
                        help='Path to FLAIR image',
                        type=str,
                        required=True)
    parser.add_argument('-seg',
                        dest='seg',
                        help='Path to output segmentation image (will be saved in FLAIR space)',
                        type=str,
                        required=True)
    parser.add_argument('--use-gpu',
                        action='store_true',
                        dest='use_gpu',
                        help='Use GPU (for HD-BET and segmentation)')
    args = parser.parse_args()

    #First, preprocess images to MNI space (rigid coregistration, skullstripping)
    mni_t1,mni_flair,flair_2_mni_affine = mni_registration(args.t1,args.flair,args.use_gpu)
    #Next, segment using model ensemble (still in MNI space)
    mni_seg = unet_segmentation(mni_t1,mni_flair)
    #Finally, label segmentation and warp back to FLAIR space
    annotate_lesions(mni_t1,mni_seg,flair_2_mni_affine,args.flair,args.seg)
    #Housekeeping
    os.remove(mni_t1)
    os.remove(mni_flair)
    os.remove(mni_seg)