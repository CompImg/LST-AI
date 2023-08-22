import argparse
import os
import subprocess
import nibabel as nib
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

"""
HYPERPARAMETERS
"""
shape = (192,192,192) #of the input images to the UNet (we use full 3D volumes)
threshold = 0.5 # For binarizing segmentation map
unet_mdls = ["UNet3D_MS_final_mdlA.h5","UNet3D_MS_final_mdlB.h5","UNet3D_MS_final_mdlC.h5"]

"""
HELPER FUNCTIONS
"""
def adapt_shape(img_arr):
    """Crops input image array to target shape"""
    difference_0 = img_arr.shape[0] - shape[0]
    difference_0_l = (difference_0 // 2)+(difference_0 % 2)
    difference_0_r = (difference_0 // 2)

    difference_1 = img_arr.shape[1] - shape[1]
    difference_1_l = (difference_1 // 2)+(difference_1 % 2)
    difference_1_r = (difference_1 // 2)

    difference_2 = img_arr.shape[2] - shape[2]
    difference_2_l = (difference_2 // 2)+(difference_2 % 2)
    difference_2_r = (difference_2 // 2)

    img_arr_cropped = img_arr[difference_0_l : img_arr.shape[0] - difference_0_r, difference_1_l : img_arr.shape[1] - difference_1_r,difference_2_l : img_arr.shape[2] - difference_2_r]

    return img_arr_cropped.astype(np.float32), [difference_0_l,difference_0_r,difference_1_l,difference_1_r,difference_2_l,difference_2_r]

"""
MAIN
"""
# parse command line arguments
parser = argparse.ArgumentParser(description='3D-UNet-based MS lesion segmentation')
parser.add_argument('--use-gpu',
                    action='store_true',
                    dest='use_gpu',
                    help='Use GPU for HD-BET yes/no')
parser.add_argument('--skip-preproc',
                    action='store_true',
                    dest='skip_preproc',
                    help='Skip preprocessing and directly segment yes/no')
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
                    help='Path to save the resulting seg image to',
                    type=str,
                    required=True)

args = parser.parse_args()

if args.skip_preproc: #In this case, data _must_ be skullstripped and coregistered in MNI space
    f2_nib = nib.load(args.flair)
    t1_nib = nib.load(args.t1)
    f2,_ = adapt_shape(f2_nib.get_fdata())
    t1,shape_lst = adapt_shape(t1_nib.get_fdata())

    temp_bm = np.zeros(f2.shape)
    temp_bm[t1 != 0] = 1

else:
    #Register T1 -> Atlas_T1
    atlas_t1 = os.path.join(os.getcwd() , "atlas", "sub-mni152_space-mni_t1.nii.gz")
    rigid_call = [os.getcwd() + "/greedy","-d","3","-a","-dof","6","-m","NMI","-ia-image-centers","-n","100x50x10","-i",atlas_t1,args.t1,"-o",os.getcwd() + "/affine.mat"]
    subprocess.run(rigid_call)
    warp_call = [os.getcwd() + "/greedy","-d","3","-rf",atlas_t1,"-rm",args.t1,os.getcwd()+"/t1_tmp.nii.gz","-r", os.getcwd() + "/affine.mat"]
    subprocess.run(warp_call)

    #Register FLAIR -> T1 in atlas space
    rigid_call = [os.getcwd() + "/greedy","-d","3","-a","-dof","6","-m","NMI","-ia-image-centers","-n","100x50x10","-i",os.getcwd()+"/t1_tmp.nii.gz",args.flair,"-o",os.getcwd() + "/affine.mat"]
    subprocess.run(rigid_call)
    warp_call = [os.getcwd() + "/greedy","-d","3","-rf",atlas_t1,"-rm",args.flair,os.getcwd()+"/flair_tmp.nii.gz","-r",os.getcwd() + "/affine.mat"]
    subprocess.run(warp_call)

    #Skullstrip (using HD-BET)
    if args.use_gpu:
        bet_call = ["hd-bet","-i",os.getcwd()+"/t1_tmp.nii.gz"]
    else:
        bet_call = ["hd-bet","-i",os.getcwd()+"/t1_tmp.nii.gz","-device","cpu","-mode","fast","-tta","0"]
    subprocess.run(bet_call)
    os.remove(os.getcwd()+"/t1_tmp.nii.gz")
    os.rename(os.getcwd()+"/t1_tmp_bet.nii.gz",os.getcwd()+"/t1_tmp.nii.gz")

    #Load data
    f2_nib = nib.load(os.getcwd()+"/flair_tmp.nii.gz")
    t1_nib = nib.load(os.getcwd()+"/t1_tmp.nii.gz")
    f2,_ = adapt_shape(f2_nib.get_fdata())
    t1,shape_lst = adapt_shape(t1_nib.get_fdata())

    temp_bm = np.zeros(f2.shape)
    temp_bm[t1 != 0] = 1
    

#Clip to [0.05;0.995] and norm to [0;1] (inside brain!)
f2 = np.clip(f2, np.percentile(f2[temp_bm != 0],0.5),np.percentile(f2[temp_bm != 0],99.5) )
f2 -= f2[temp_bm == 1].min()
f2 = f2 / f2[temp_bm == 1].max()
f2 *= temp_bm
f2 = f2.astype(np.float32)

t1 = np.clip(t1, np.percentile(t1[temp_bm != 0],0.5),np.percentile(t1[temp_bm != 0],99.5) )
t1 -= t1[temp_bm == 1].min()
t1 = t1 / t1[temp_bm == 1].max()
t1 *= temp_bm
t1 = t1.astype(np.float32)

#Segment
joint_seg = np.zeros(t1.shape)
for model in unet_mdls:
    mdl = tf.keras.models.load_model(os.path.join(os.getcwd(), "model", model),compile=False)

    img_image = np.stack([f2,t1],axis=-1)
    img_image = np.expand_dims(img_image,axis=0)

    out_seg = mdl(img_image)[0] #Will return a len(2) list of [out_seg,out_ds]
    out_seg = np.squeeze(out_seg)

    joint_seg += out_seg
        
joint_seg /= len(unet_mdls)
seg = np.zeros(f2.shape)
seg[joint_seg > threshold] = 1
seg_mni = np.pad(seg,((shape_lst[0],shape_lst[1]),(shape_lst[2],shape_lst[3]),(shape_lst[4],shape_lst[5])),'constant', constant_values=0.)

if args.skip_preproc:
    #Save to ppFLAIR space
    nib.save(nib.Nifti1Image(seg_mni.astype(np.uint8),f2_nib.affine,f2_nib.header),args.seg)
else:
    #Save and warp segmentation back to patient FLAIR space
    nib.save(nib.Nifti1Image(seg_mni.astype(np.uint8),f2_nib.affine,f2_nib.header),os.getcwd()+"/seg.nii.gz")
    warp_back_call = [os.getcwd() + "/greedy","-d","3","-rf",args.flair,"-ri","LABEL","0.2vox","-rm",os.getcwd()+"/seg.nii.gz",args.seg,"-r",os.getcwd()+"/affine.mat,-1"]
    subprocess.run(warp_back_call)

    #Housekeeping
    os.remove(os.getcwd()+"/affine.mat")
    os.remove(os.getcwd()+"/flair_tmp.nii.gz")
    os.remove(os.getcwd()+"/t1_tmp.nii.gz")
    os.remove(os.getcwd()+"/seg.nii.gz")
    os.remove(os.getcwd()+"/t1_tmp_bet_mask.nii.gz")
