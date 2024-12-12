import os
import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from LST_AI.custom_tf import load_custom_model


# stolen and adapted from SCT 6.5 math.py line 421
def remove_small_objects(data, dim_lst, unit='mm3', thr=0):
    """Removes all unconnected objects smaller than the minimum specified size.

    Adapted from:
    https://github.com/ivadomed/ivadomed/blob/master/ivadomed/postprocessing.py#L327
    and
    https://github.com/ivadomed/ivadomed/blob/master/ivadomed/postprocessing.py#L224

    Args:
        data (ndarray): Input data.
        dim_lst (list): Dimensions of a voxel in mm.
        unit (str): Indicates the units of the objects: "mm3" or "vox"
        thr (float): Minimal object size to keep in input data.

    Returns:
        ndarray: Array with small objects.
    """
    print(f"Thresholding lesions at [mm3]:{np.prod(dim_lst)}")

    px, py, pz = dim_lst

    # structuring element that defines feature connections
    bin_structure = generate_binary_structure(3, 2)

    data_label, n = label(data, structure=bin_structure)

    if unit == 'mm3':
        size_min = np.round(thr / (px * py * pz))
    else:
        print('Please choose a different unit for removeSmall. Choices: vox or mm3')
        exit()

    for idx in range(1, n + 1):
        data_idx = (data_label == idx).astype(int)
        n_nonzero = np.count_nonzero(data_idx)

        if n_nonzero < size_min:
            data[data_label == idx] = 0

    return data



def unet_segmentation(model_path, mni_t1, mni_flair, output_segmentation_path, 
                      output_prob_path, output_prob1_path, output_prob2_path, output_prob3_path, 
                      device='cpu', probmap=False, input_shape=(192,192,192), threshold=0.5, clipping=(0.5,99.5), lesion_thr=0):
    """
    Segment medical images using ensemble of U-Net models.

    This function utilizes pre-trained U-Net models to perform segmentation
    on T1 and FLAIR images in the MNI space. The segmentation output is a
    binary lesion mask, which is then saved to a specified path.

    Parameters:
    -----------
    model_path : str
        Directory path containing the U-Net models.
    mni_t1 : str
        Path to the T1-weighted image in MNI space.
    mni_flair : str
        Path to the FLAIR image in MNI space.
    output_segmentation_path : str
        Path to save the resulting binary segmentation mask.
    input_shape : tuple of int, optional
        Expected shape of the input images for the U-Net model.
        Default is (192, 192, 192).
    threshold : float, optional
        Segmentation threshold to determine the binary mask from the U-Net's
        output. Pixels with values above this threshold in the U-Net output
        will be set to 1 in the binary mask, and others to 0. Default is 0.5.
    clipping : list of floats, optional
        Min and Max values for the np.clip option which applies clipping for 
        the standardization of image intensities. Default is min=0.5 and max=99.5.

    Returns:
    --------
    None
        The function saves the binary segmentation mask to the path specified
        in `output_segmentation_path`.

    """

    tf_device = '/CPU:0' if device == 'cpu' else f'/GPU:{device}'

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

    def preprocess_intensities(img_arr, clipping):
        #Standardize image intensities to [0;1]
        temp_bm = np.zeros(img_arr.shape)
        temp_bm[img_arr != 0] = 1
        img_arr = np.clip(img_arr, 
                          a_min=np.percentile(img_arr[temp_bm != 0],clipping[0]),
                          a_max=np.percentile(img_arr[temp_bm != 0],clipping[1]) )
        img_arr -= img_arr[temp_bm == 1].min()
        img_arr = img_arr / img_arr[temp_bm == 1].max()
        img_arr *= temp_bm

        return img_arr.astype(np.float32)

    # weight files
    unet_mdls = [
    "UNet3D_MS_final_mdlA.h5",
    "UNet3D_MS_final_mdlB.h5",
    "UNet3D_MS_final_mdlC.h5"
    ]
    unet_mdls = [os.path.join(model_path, x) for x in unet_mdls]

    # Load and preprocess images
    t1_nib = nib.load(mni_t1)
    t1 = t1_nib.get_fdata()
    flair_nib = nib.load(mni_flair)
    flair = flair_nib.get_fdata()

    t1, shape_lst = adapt_shape(t1)
    flair, _ = adapt_shape(flair)

    t1 = preprocess_intensities(t1, clipping)
    flair = preprocess_intensities(flair, clipping)

    joint_seg = np.zeros(t1.shape)
    print(f"Running segmentation on {tf_device}.")

    # define list of model probability maps
    output_prob_list = [output_prob1_path, output_prob2_path, output_prob3_path]

    for i, model in enumerate(unet_mdls):
        with tf.device(tf_device):
            print(f"Running model {i}. ")
            mdl = load_custom_model(model, compile=False)

        img_image = np.stack([flair, t1], axis=-1)
        img_image = np.expand_dims(img_image, axis=0)
        with tf.device(tf_device):
            out_seg = mdl(img_image)[0]  # Will return a len(2) list of [out_seg, out_ds]
        out_seg = np.squeeze(out_seg)

        if probmap:
            # add zero values to the probability map
            out_seg_pad = np.pad(out_seg,
                                 ((shape_lst[0], shape_lst[1]), (shape_lst[2], shape_lst[3]), (shape_lst[4], shape_lst[5])),
                                 'constant', constant_values=0.)
            # save the probability map
            nib.save(nib.Nifti1Image(out_seg_pad.astype(np.float32),
                                    flair_nib.affine,
                                    flair_nib.header),
                                    output_prob_list[i])

        joint_seg += out_seg

    joint_seg /= len(unet_mdls)

    if probmap:
        # add zero values to the probability map
        joint_seg_pad = np.pad(joint_seg,
                               ((shape_lst[0], shape_lst[1]), (shape_lst[2], shape_lst[3]), (shape_lst[4], shape_lst[5])),
                               'constant', constant_values=0.0)
        # save the probability map
        nib.save(nib.Nifti1Image(joint_seg_pad.astype(np.float32),
                                flair_nib.affine,
                                flair_nib.header),
                                output_prob_path)

    out_binary = np.zeros(t1.shape)
    out_binary[joint_seg > threshold] = 1

    out_binary = np.pad(
        out_binary,
        ((shape_lst[0], shape_lst[1]), (shape_lst[2], shape_lst[3]), (shape_lst[4], shape_lst[5])),
        'constant', constant_values=0.
    )

    out_binary = remove_small_objects(out_binary, flair_nib.header.get_zooms(), unit="mm3", thr=int(lesion_thr))

    nib.save(nib.Nifti1Image(out_binary.astype(np.uint8),
                             flair_nib.affine,
                             flair_nib.header),
                             output_segmentation_path)


if __name__ == "__main__":
    # Testing only
    # Working directory
    script_dir = os.getcwd()
    parent_dir = os.path.dirname(script_dir)

    model_dir = os.path.join(parent_dir, "model")
    test_dir = os.path.join(parent_dir, "testing", "annotation")
    t1_path = os.path.join(test_dir, "sub-msseg-test-center01-02_ses-01_space-mni_t1.nii.gz")
    flair_path = t1_path
    output_path = os.path.join(parent_dir, "testing", "seg_mni.nii.gz")
    gt_path = os.path.join(test_dir, "sub-msseg-test-center01-02_ses-01_space-mni_seg-unet.nii.gz")

    unet_segmentation(mni_t1=t1_path,
                      mni_flair=flair_path,
                      model_path=model_dir,
                      output_segmentation_path=output_path)

    # Check and remove testing results
    gt_data = nib.load(gt_path).get_fdata()
    pred_data = nib.load(output_path).get_fdata()
    os.remove(output_path)
    np.testing.assert_array_equal(gt_data, pred_data)
