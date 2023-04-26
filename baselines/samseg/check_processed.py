import argparse
import os
from pathlib import Path
import pandas as pd
import re
import numpy as np

from utils import getSegList, getT1wList, getSessionID, getSubjectID

####################################################
# main script

parser = argparse.ArgumentParser(description='Check for misisng segmentation files (e.g., when processing did not work).')
parser.add_argument('-i', '--input_directory', help='Folder of BIDS database.', required=True)
parser.add_argument('-o', '--output_directory', help='Destination folder for the output file.', required=True)

# read the arguments
args = parser.parse_args()

# define path of the derivatives folder
derivatives_dir = os.path.join(args.input_directory, "derivatives/samseg-7.3.2")

# get a list with all subject folders in the database
T1w_list = getT1wList(args.input_directory)
# get a list with the paths of the _seg.mgz files 
seg_list = getSegList(derivatives_dir)

# initialize empty dataframe into which we will write the IDs of missing segmentations
df_seg_missing = pd.DataFrame(columns=['subject-ID', 'session-ID'])
for i in range(len(T1w_list)):
    # get subject and session ID
    subjectID = getSubjectID(T1w_list[i])
    sessionID = getSessionID(T1w_list[i])
    # define path of segmentation file
    seg_path = Path(os.path.join(derivatives_dir, f'sub-{subjectID}', f'ses-{sessionID}', 'anat', f'sub-{subjectID}_ses-{sessionID}_seg.mgz'))
    # check if segmentation file ist in seg_list (if not, then the segmentation must have failed and we want to write the IDs into df_seg_missing)
    if seg_path in seg_list:
        print(f'{subjectID}/{sessionID}: processed successfully!')
    else:
        # write stats in the final dataframe
        dict_miss = {df_seg_missing.columns[0]:[subjectID], df_seg_missing.columns[1]:[sessionID]}
        df_miss = pd.DataFrame(data=dict_miss)
        df_seg_missing = pd.concat([df_seg_missing, df_miss])
        print(f'{subjectID}/{sessionID}: segmentation failed!')

# write dataframe with missing files as .csv file in chosen output directory
df_seg_missing.to_csv(os.path.join(args.output_directory, "samseg_cross_missing.csv"), index=False)
