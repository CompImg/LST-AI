import argparse
import pathlib
from pathlib import Path
import json
import os
import shutil
from collections import OrderedDict

# this script is employed to generate the nn-Unet based dataset format
# as described in this readme: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md

# parse command line arguments
parser = argparse.ArgumentParser(description='Convert a BIDS-structured database to the nn-Unet format.')
parser.add_argument('-i', '--input_directory', help='Path to BIDS structured database.', required=True)
parser.add_argument('-o', '--output_directory', help='Path to output directory.', required=True)
parser.add_argument('-tn', '--taskname', help='Specify the task name, e.g. Hippocampus', default='MSBrainLesion', type=str)
parser.add_argument('-tno','--tasknumber', help='Specify the task number, has to be greater than 500', default=500,type=int)

args = parser.parse_args()

path_in = Path(args.input_directory)
path_out = Path(os.path.join(os.path.abspath(args.output_directory), f'Task{args.tasknumber}_{args.taskname}'))
path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

test_image_t1w = []
test_image_flair = []
test_image_labels = []

if __name__ == '__main__':

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    conversion_dict = {}
    dirs = sorted(list(path_in.glob('*')))
    scan_cnt = 0

    for dir in dirs:  # iterate over subdirs
        # glob the session directories
        subdirs = sorted(list(dir.glob('*')))
        for subdir in subdirs:
            
            scan_cnt+= 1
            print(subdir)
            # obtain the filenames
            flair_file = sorted(list(subdir.rglob('*space-mni_flair.nii.gz')))[0]
            t1w_file = sorted(list(subdir.rglob('*space-mni_t1.nii.gz')))[0]
            # seg_file = sorted(list(subdir.rglob('*space-mni_seg.nii.gz')))[0]
            
            # create the new convention names
            t1w_file_nnunet = os.path.join(path_out_imagesTs,f'{args.taskname}_{scan_cnt:03d}_0000.nii.gz')
            flair_file_nnunet = os.path.join(path_out_imagesTs,f'{args.taskname}_{scan_cnt:03d}_0001.nii.gz')
            # seg_file_nnunet = os.path.join(path_out_labelsTs,f'{args.taskname}_{scan_cnt:03d}.nii.gz')

            test_image_t1w.append(str(t1w_file_nnunet))
            test_image_flair.append(str(flair_file_nnunet))
            # test_image_labels.append(str(seg_file_nnunet))

            print(t1w_file_nnunet)

            # copy the files to new structure
            shutil.copyfile(t1w_file, t1w_file_nnunet)
            shutil.copyfile(flair_file, flair_file_nnunet)
            # shutil.copyfile(seg_file, seg_file_nnunet)

            conversion_dict[str(os.path.abspath(t1w_file))] = t1w_file_nnunet
            conversion_dict[str(os.path.abspath(flair_file))] = flair_file_nnunet
            # conversion_dict[str(seg_file)] = seg_file_nnunet


    # create dataset_description.json
    json_object = json.dumps(conversion_dict, indent=4)
    # write to dataset description
    with open(os.path.join(path_out,"conversion_dict_test.json"), "w") as outfile:
        outfile.write(json_object)


    # c.f. dataset json generation
    # general info : https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/utils.py
    # example: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task055_SegTHOR.py

    json_dict = OrderedDict()
    json_dict['name'] = args.taskname
    json_dict['description'] = args.taskname
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "TBD"
    json_dict['licence'] = "TBD"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T1w",
        "1": "Flair",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "lesion",

   }
    json_dict['numTraining'] = 0
    json_dict['numTest'] = scan_cnt
    json_dict['test'] = [{'image': str(t1w_file_nnunet[i])} #.replace("labelsTr", "imagesTr")} #, "label": test_image_labels[i] }
                            for i in range(len(t1w_file_nnunet))]
    json_dict['train'] = []

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    with open(os.path.join(path_out,"dataset_test.json"), "w") as outfile:
        outfile.write(json_object)



