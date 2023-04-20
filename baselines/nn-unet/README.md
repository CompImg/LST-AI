## nn-unetV1 ms brain lesion segmentation
#### please note that nn-unet V2 is a reimplementation of the code, but not of the method - both yield the same results!

This repository features small utility script to allow to train a nn-unet for lesion segmentation,
based on the conventions of the [official nn-unet repository](https://github.com/MIC-DKFZ/nnUNet#run-inference) by Isensee et al.


### Setup the environment

Please setup a virtual environment as described [here](https://github.com/MIC-DKFZ/nnUNet#installation).

### Prepare training data

Input Directory Format:

```
    ├── sub-m991840
    │   └── ses-20110519
    │       ├── sub-m991840_ses-20110519_space-mni_flair.nii.gz
    │       ├── sub-m991840_ses-20110519_space-mni_seg-labelled.nii.gz
    │       ├── sub-m991840_ses-20110519_space-mni_seg.nii.gz
    │       ├── sub-m991840_ses-20110519_space-mni_synd2.nii.gz
    │       └── sub-m991840_ses-20110519_space-mni_t1.nii.gz
    ├── sub-m993488
    │   └── ses-20120216
    │       ├── sub-m993488_ses-20120216_space-mni_flair.nii.gz
    │       ├── sub-m993488_ses-20120216_space-mni_seg-labelled.nii.gz
    │       ├── sub-m993488_ses-20120216_space-mni_seg.nii.gz
    │       ├── sub-m993488_ses-20120216_space-mni_synd2.nii.gz
    │       └── sub-m993488_ses-20120216_space-mni_t1.nii.gz
```


To prepare training data, please run the following command. Please be aware that this only works for the tested structure,
where we utilize FLAIR and T1w as input channels. Other modalities will fail.

```
python3 create_trainset.py --input_directory /mnt/Drive4/julian/nnunet_ms_lesion/data/achieva/ --output_directory dataset -tn MSBrain_Lesion -tno 500
```
Afterwards, we can expect the following data directory structure:

```
conversion_dict_train.json  dataset_train.json  imagesTr  imagesTs  labelsTr
```

Please place the folder `Task500_MSBrainLesion` into the raw database:

e.g. /mnt/Drive4/julian/nnunet_paths/nnUNet_raw_data_base/nnUNet_raw_data

### Prepare test data

To create the test set, we have to follow a similar approach:

```
python3 create_testset.py --input_directory /mnt/Drive4/julian/nnunet_ms_lesion/data/achieva/ --output_directory dataset -tn MSBrain_Lesion -tno 500
```

Afterwards, we can expect the following data directory structure:

```
conversion_dict_test.json  dataset_test.json  imagesTr  imagesTs  labelsTr
```

Please place/merge the folder `Task500_MSBrainLesion` into the raw database (same location where the training samples live):

e.g. /mnt/Drive4/julian/nnunet_paths/nnUNet_raw_data_base/nnUNet_raw_data


### Starting the training

nn-unet provides different options for training: [2d, 3d_lowres, 3d_fullres, 3d_cascade]. Prepare the training by running the following command:

`nnUNet_plan_and_preprocess -t 501 -tl 32 -tf 32 --verify_dataset_integrity`, and specify the number of workers via `-tl` and `-tf`.

In this case, we use 2d and 3d_fullres options, which need to be independently trained for all (5) folds.

`
CUDA_VISIBLE_DEVICES=6 nnUNet_train 3d_fullres nnUnetTrainV2 501 3 --npz
`

Here 501 corresponds to the task id you assigned during the train/test set generation, and 3 to the fold id (run the same instructions for all folds in [0,1,2,3,4]).

### Resuming the training after N (50, 100, ...) epochs

As nn-unet takes a while to train, it's very practical that nnunet automatically saves checkpoints every 50 epochs. To continue training, simply add the `-c` option. E.g.

`CUDA_VISIBLE_DEVICES=6 nnUNet_train 3d_fullres nnUnetTrainV2 501 3 --npz -c`

### Choosing the best model

Run `nnUNet_find_best_configuration` to identify the best model based on five-fold cross validation. However, this also requires you having trained all five folds! You can also disable ensembling via `--disable_ensembling`.

```
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS 
```

### Inference

Now, we can finally run inference on the testset - we trained a 3d_fullres, so please use this model as input model:

```
nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_MSBrainLesion/imagesTs -o /mnt/Drive4/julian/out -t 500 -m 3d_fullres
```

### Evaluation of test set samples

Evaluation of DICE score: (c.f. [url](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/inference_example_Prostate.md))

nnUNet_evaluate_folder -ref FOLDER_WITH_GT -pred FOLDER_WITH_PREDICTIONS -l 1