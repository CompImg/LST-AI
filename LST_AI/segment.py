import os
import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Ensemble model basenames (extension chosen by backend: .onnx or .h5).
_MODEL_STEMS = ["UNet3D_MS_final_mdlA", "UNet3D_MS_final_mdlB", "UNet3D_MS_final_mdlC"]


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


def _make_inference(backend, model_path, device):
    """Return a callable ``run(stem, x) -> out_seg ndarray`` for the chosen backend.

    backend='onnx' (default): onnxruntime — CUDAExecutionProvider when ``device`` is a
        GPU id, else CPUExecutionProvider. Multi-arch, no TensorFlow.
    backend='tf' (legacy): TensorFlow/Keras via LST_AI.custom_tf.

    Both run the ensemble UNets that output ``[out_seg, out_ds...]`` and return
    out_seg (model output 0). Imports are lazy so an ONNX-only environment needs no
    TensorFlow (and vice versa).
    """
    if backend == 'onnx':
        try:
            import onnxruntime as ort
        except ImportError as exc:
            # onnxruntime is an install-time extra (see setup.py): 'onnxruntime' (CPU)
            # and 'onnxruntime-gpu' (CUDA) share the same namespace and can't coexist,
            # so a backend must be chosen explicitly.
            raise ImportError(
                "No ONNX runtime found for the 'onnx' segmentation backend. Install a "
                "backend:\n"
                '    pip install "lst-ai[cpu]"   # portable CPU\n'
                '    pip install "lst-ai[gpu]"   # NVIDIA CUDA (onnxruntime-gpu)\n'
                "(or `pip install onnxruntime` / `onnxruntime-gpu` directly)."
            ) from exc

        if str(device) == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            providers = [('CUDAExecutionProvider', {'device_id': int(device)}),
                         'CPUExecutionProvider']
        sessions = {}

        def run(stem, x):
            if stem not in sessions:
                sessions[stem] = ort.InferenceSession(
                    os.path.join(model_path, stem + '.onnx'), providers=providers)
            sess = sessions[stem]
            return sess.run(None, {sess.get_inputs()[0].name: x})[0]  # output 0 = out_seg

        return run

    if backend == 'tf':
        import tensorflow as tf
        from LST_AI.custom_tf import load_custom_model

        tf_device = '/CPU:0' if str(device) == 'cpu' else f'/GPU:{device}'

        def run(stem, x):
            with tf.device(tf_device):
                mdl = load_custom_model(os.path.join(model_path, stem + '.h5'), compile=False)
                out = mdl(x)
            return out[0] if isinstance(out, (list, tuple)) else out

        return run

    raise ValueError(f"Unknown backend '{backend}'. Use 'onnx' or 'tf'.")


def unet_segmentation(model_path, mni_t1, mni_flair, output_segmentation_path,
                      output_prob_path, output_prob1_path, output_prob2_path, output_prob3_path,
                      device='cpu', probmap=False, input_shape=(192, 192, 192), threshold=0.5,
                      clipping=(0.5, 99.5), lesion_thr=0, backend='onnx'):
    """
    Segment medical images using an ensemble of U-Net models.

    Uses pre-trained ensemble UNets to segment T1 and FLAIR images in MNI space; the
    output is a binary lesion mask saved to ``output_segmentation_path``. Inference
    runs through ``backend`` ('onnx', default — multi-arch, no TensorFlow; or 'tf').
    All pre/post-processing is backend-independent.

    Parameters
    ----------
    model_path : str
        Directory holding the ensemble weights (UNet3D_MS_final_mdl{A,B,C}.{onnx,h5}).
    mni_t1, mni_flair : str
        Skull-stripped T1 / FLAIR in MNI space.
    device : str
        GPU id (e.g. '0') or 'cpu'.
    backend : str
        'onnx' (default) or 'tf'.
    """

    def adapt_shape(img_arr):
        # Crops input image array to target shape; also returns how to re-zero-pad.
        difference_0 = img_arr.shape[0] - input_shape[0]
        difference_0_l = (difference_0 // 2) + (difference_0 % 2)
        difference_0_r = (difference_0 // 2)
        difference_1 = img_arr.shape[1] - input_shape[1]
        difference_1_l = (difference_1 // 2) + (difference_1 % 2)
        difference_1_r = (difference_1 // 2)
        difference_2 = img_arr.shape[2] - input_shape[2]
        difference_2_l = (difference_2 // 2) + (difference_2 % 2)
        difference_2_r = (difference_2 // 2)
        img_arr_cropped = img_arr[difference_0_l: img_arr.shape[0] - difference_0_r,
                                  difference_1_l: img_arr.shape[1] - difference_1_r,
                                  difference_2_l: img_arr.shape[2] - difference_2_r]
        return img_arr_cropped.astype(np.float32), [difference_0_l, difference_0_r,
                                                    difference_1_l, difference_1_r,
                                                    difference_2_l, difference_2_r]

    def preprocess_intensities(img_arr, clipping):
        # Standardize image intensities to [0;1] over the brain.
        temp_bm = np.zeros(img_arr.shape)
        temp_bm[img_arr != 0] = 1
        img_arr = np.clip(img_arr,
                          a_min=np.percentile(img_arr[temp_bm != 0], clipping[0]),
                          a_max=np.percentile(img_arr[temp_bm != 0], clipping[1]))
        img_arr -= img_arr[temp_bm == 1].min()
        img_arr = img_arr / img_arr[temp_bm == 1].max()
        img_arr *= temp_bm
        return img_arr.astype(np.float32)

    # Load and preprocess images
    t1_nib = nib.load(mni_t1)
    t1 = t1_nib.get_fdata()
    flair_nib = nib.load(mni_flair)
    flair = flair_nib.get_fdata()

    t1, shape_lst = adapt_shape(t1)
    flair, _ = adapt_shape(flair)
    t1 = preprocess_intensities(t1, clipping)
    flair = preprocess_intensities(flair, clipping)

    img_image = np.expand_dims(np.stack([flair, t1], axis=-1), axis=0).astype(np.float32)
    run = _make_inference(backend, model_path, device)
    print(f"Running segmentation (backend={backend}, device={device}).")

    joint_seg = np.zeros(t1.shape)
    output_prob_list = [output_prob1_path, output_prob2_path, output_prob3_path]

    for i, stem in enumerate(_MODEL_STEMS):
        print(f"Running model {i}. ")
        out_seg = np.squeeze(run(stem, img_image))

        if probmap:
            out_seg_pad = np.pad(out_seg,
                                 ((shape_lst[0], shape_lst[1]), (shape_lst[2], shape_lst[3]), (shape_lst[4], shape_lst[5])),
                                 'constant', constant_values=0.)
            nib.save(nib.Nifti1Image(out_seg_pad.astype(np.float32), flair_nib.affine, flair_nib.header),
                     output_prob_list[i])

        joint_seg += out_seg

    joint_seg /= len(_MODEL_STEMS)

    if probmap:
        joint_seg_pad = np.pad(joint_seg,
                               ((shape_lst[0], shape_lst[1]), (shape_lst[2], shape_lst[3]), (shape_lst[4], shape_lst[5])),
                               'constant', constant_values=0.0)
        nib.save(nib.Nifti1Image(joint_seg_pad.astype(np.float32), flair_nib.affine, flair_nib.header),
                 output_prob_path)

    out_binary = np.zeros(t1.shape)
    out_binary[joint_seg > threshold] = 1
    out_binary = np.pad(out_binary,
                        ((shape_lst[0], shape_lst[1]), (shape_lst[2], shape_lst[3]), (shape_lst[4], shape_lst[5])),
                        'constant', constant_values=0.)
    out_binary = remove_small_objects(out_binary, flair_nib.header.get_zooms(), unit="mm3", thr=int(lesion_thr))

    nib.save(nib.Nifti1Image(out_binary.astype(np.uint8), flair_nib.affine, flair_nib.header),
             output_segmentation_path)
