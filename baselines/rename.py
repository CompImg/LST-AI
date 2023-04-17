import json
import pandas
import os
from pathlib import Path

import argparse
# parse command line arguments
parser = argparse.ArgumentParser(description='Rename nn-unet segmentation masks to original filenames again.')
parser.add_argument('-i', '--input_directory', help='Path to nn-unet segmentation masks.')#, required=True)
parser.add_argument('-c', '--conversion_dict', help='Path to nn-unet segmentation masks.', default='conversion_dict_test.json', type=str)

args = parser.parse_args()

with open(args.conversion_dict) as f:
    d = json.load(f)

# swap keys and vals
d = dict((v,k) for k,v in d.items())

res = dict()
for key, val in d.items():
   if 't1' in val:
        new_key = os.path.basename(key).replace('.gz','')
        new_val = os.path.basename(val).replace('.gz','')
        new_val = new_val.replace("t1","mask")
        new_key = new_key.replace("_0000","")
        res[new_key] = new_val

# swap keys and vals
res = dict((v,k) for k,v in res.items())

for key, val in res.items():
    print(f'Renaming {val} to {key}')
    src = os.path.join(os.path.abspath(args.input_directory), val)
    dest = os.path.join(os.path.abspath(args.input_directory), key)
    assert os.path.isfile(src) == True, 'File does not exist! Exception occured.'
    os.rename(src, dest )