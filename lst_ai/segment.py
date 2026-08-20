import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure

# Ensemble model basenames. The v2.0.0 bundle ships .pt; a v1.3.0 .onnx bundle still
# loads, via the fallback in _make_inference.
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
    print(f"Thresholding lesions at [mm3]:{thr}")

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


def _make_inference(model_path, device):
    """Return a callable ``run(stem, x) -> out_seg ndarray``, running natively in PyTorch.

    Loads ``<stem>.pt`` when it exists -- what the v2.0.0 bundle ships -- and otherwise
    transfers the weights straight out of a legacy ``<stem>.onnx``, so a v1.3.0 bundle
    keeps working. That fallback needs the optional `onnx` package (a protobuf schema
    reader), not onnxruntime; no inference runtime other than PyTorch is involved. The
    two paths were checked tensor-for-tensor equal, so the source of the weights makes
    no difference to the output.

    The graphs are NDHWC and the module is NCDHW, so the input is permuted around the
    call and the output permuted back, leaving the rest of the pipeline unchanged.

    Running under PyTorch rather than ONNX Runtime also keeps the GPU footprint bounded
    via torch's caching allocator; the ONNX Runtime CUDA arena transiently grabbed ~40 GB
    at session init and OOM'd under GPU contention (tw/v200_updates). This module reaches
    the same place natively, so onnx2torch is no longer needed either -- the two agree
    exactly (Dice 1.000000, zero voxels differing).
    """
    import torch
    from lst_ai.model import NNUNet3D
    from lst_ai.weights import load_onnx_weights

    torch_device = torch.device('cpu' if str(device) == 'cpu' else f'cuda:{device}')
    models = {}

    def load(stem):
        checkpoint = os.path.join(model_path, stem + '.pt')
        legacy = os.path.join(model_path, stem + '.onnx')
        if os.path.exists(checkpoint):
            # weights_only=True: the checkpoint is a download, so loading it must not be
            # able to run code. The payload is tensors and ints only, so this suffices.
            ckpt = torch.load(checkpoint, map_location=torch_device, weights_only=True)
            cfg = dict(ckpt['config'])
            cfg['ds_layers'] = tuple(cfg.get('ds_layers', ()))
            mdl = NNUNet3D(**cfg)
            mdl.load_state_dict(ckpt['state_dict'])
        elif os.path.exists(legacy):
            variant = stem.rsplit('_', 1)[-1]        # UNet3D_MS_final_mdlA -> mdlA
            mdl = NNUNet3D.shipped(variant, in_channels=2)
            load_onnx_weights(mdl, legacy)
        else:
            raise FileNotFoundError(
                f"no weights for {stem} in {model_path}: expected {stem}.pt (the current "
                f"bundle) or {stem}.onnx (a legacy one). Run "
                "lst_ai.utils.download_data() to fetch them."
            )
        return mdl.to(torch_device).eval()

    def run(stem, x):
        if stem not in models:
            models[stem] = load(stem)
        xt = torch.from_numpy(x).permute(0, 4, 1, 2, 3).contiguous().to(torch_device)
        with torch.no_grad():
            out = models[stem](xt)
        out = out[0] if isinstance(out, (list, tuple)) else out
        return out.permute(0, 2, 3, 4, 1).cpu().numpy()

    return run


def unet_segmentation(model_path, mni_t1, mni_flair, output_segmentation_path,
                      output_prob_path, output_prob1_path, output_prob2_path, output_prob3_path,
                      device='cpu', probmap=False, input_shape=(192, 192, 192), threshold=0.5,
                      clipping=(0.5, 99.5), lesion_thr=0):
    """
    Segment medical images using an ensemble of U-Net models.

    Uses pre-trained ensemble UNets to segment T1 and FLAIR images in MNI space; the
    output is a binary lesion mask saved to ``output_segmentation_path``. Inference
    Inference runs natively in PyTorch; there is no TensorFlow or ONNX Runtime
    dependency. See docs/pytorch-reimplementation.md for how this compares to the
    TensorFlow results the released weights were trained under.

    Parameters
    ----------
    model_path : str
        Directory holding the ensemble weights (UNet3D_MS_final_mdl{A,B,C}.{onnx,h5}).
    mni_t1, mni_flair : str
        Skull-stripped T1 / FLAIR in MNI space.
    device : str
        GPU id (e.g. '0') or 'cpu'.
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
    run = _make_inference(model_path, device)
    print(f"Running segmentation (PyTorch, device={device}).")

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
