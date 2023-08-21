import argparse
import os
import subprocess
import nibabel as nib
import numpy as np
from skimage.measure import label
from skimage.morphology import binary_dilation
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

# parse command line arguments
parser = argparse.ArgumentParser(description='Label MS lesions according to McDonald criteria')
parser.add_argument('--use-skullstripped',
                    action='store_true',
                    dest='use_bet',
                    help='Use if input T1 is skullstripped (Uses appropriate atlas template')
parser.add_argument('-t1',
                    dest='t1',
                    help='Path to T1 image',
                    type=str,
                    required=True)
parser.add_argument('-seg',
                    dest='seg',
                    help='Path to (binary) segmentation image (in T1 space)',
                    type=str,
                    required=True)

args = parser.parse_args()

if args.use_bet:
    atlas_t1 = os.getcwd() + "/sub-mni152_space-mni_t1bet.nii.gz"
else:
    atlas_t1 = os.getcwd() + "/sub-mni152_space-mni_t1.nii.gz"
atlas_mask = os.getcwd() + "/sub-mni152_space-mni_msmask.nii.gz"

#Register Atlas -> Patient_T1 using greedy (two-step: rigid first, then deformable)
rigid_call = [os.getcwd() + "/greedy","-d","3","-a","-m","NMI","-ia-image-centers","-n","100x50x10","-i",args.t1,atlas_t1,"-o",os.getcwd() + "/affine.mat"]
subprocess.run(rigid_call)
deformable_call = [os.getcwd() + "/greedy","-d","3","-m","WNCC","2x2x2","-sv","-n","100x50x10","-it",os.getcwd() + "/affine.mat","-i",args.t1,atlas_t1,"-o",os.getcwd() + "/warp.nii.gz"]
subprocess.run(deformable_call)

#Warp MSmask in patient space
warp_call = [os.getcwd() + "/greedy","-d","3","-rf",args.t1,"-ri","LABEL","0.2vox","-rm",atlas_mask,args.seg.replace(".nii.gz","_msmask.nii.gz"),"-r",os.getcwd() + "/warp.nii.gz",os.getcwd() + "/affine.mat"]
subprocess.run(warp_call)

#Load segmentation and msmask and location-label lesions
seg_nib = nib.load(args.seg)
seg = seg_nib.get_fdata()
seg[seg > 0] = 1 #Make sure seg is binary
msmask = nib.load(args.seg.replace(".nii.gz","_msmask.nii.gz")).get_fdata()


seg_label = label(seg,connectivity=3)
for lesion_ctr in range(1,seg_label.max()+1):
    #We create a temporary binary mask for each leasion & dilate it by 1 (to "catch" adjacent structures)
    temp_mask = np.zeros(seg.shape)
    temp_mask[seg_label == lesion_ctr] = 1
    temp_mask_dil = binary_dilation(temp_mask,footprint=np.ones((3,3,3))).astype(np.uint8)

    if 4 in msmask[temp_mask_dil == 1]:
        seg[temp_mask == 1] = 1 #PV

    elif 5 in msmask[temp_mask_dil == 1]:
        seg[temp_mask == 1] = 4 #IT

    elif 2 in msmask[temp_mask_dil == 1]:
        seg[temp_mask == 1] = 2 #JC

    else:
        seg[temp_mask == 1] = 3 #SC

#Saving & housekeeping
nib.save(nib.Nifti1Image(seg.astype(np.uint8),seg_nib.affine,seg_nib.header),args.seg.replace(".nii.gz","_labelled.nii.gz"))
os.remove(os.getcwd()+"/affine.mat")
os.remove(os.getcwd()+"/warp.nii.gz")