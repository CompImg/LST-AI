import os
import shutil
from pathlib import Path
import re
from bs4 import BeautifulSoup
import numpy as np

# bids helpers
def getSubjectID(path):
    """
    :param path: path to data file
    :return: return the BIDS-compliant subject ID
    """
    stringList = str(path).split("/")
    indices = [i for i, s in enumerate(stringList) if 'sub-' in s]
    text = stringList[indices[0]]
    try:
        found = re.search(r'sub-([a-zA-Z0-9]+)', text).group(1)
    except AttributeError:
        found = ''
    return found

def getSessionID(path):
    """
    :param path: path to data file
    :return: return the BIDS-compliant session ID
    """
    stringList = str(path).split("/")
    indices = [i for i, s in enumerate(stringList) if '_ses-' in s]
    text = stringList[indices[0]]
    try:
        found = re.search(r'ses-([a-zA-Z0-9]+)', text).group(1)
    except AttributeError:
        found = ''
    return found

# multiprocessing helpers
def split_list(alist, splits=1):
    length = len(alist)
    return [alist[i * length // splits: (i + 1) * length // splits]
            for i in range(splits)]


# path helpers
def MoveandCheck(orig, target):
    '''
    This function tries to move a file with current path "orig" to a target path "target" and 
    checks if the orig file exists and if it was copied successfully to target
    :param orig: full path of original location (with (original) filename in the path)
    :param target: full path of target location (with (new) filename in the path)
    '''
    if os.path.exists(orig):
        shutil.move(orig, target)
        if not os.path.exists(target):
            raise ValueError(f'failed to move {orig}')
        else:
            print(f'successfully copied {os.path.basename(orig)} to {os.path.basename(target)} target location')
    else:
        raise Warning(f'file {os.path.basename(orig)} does not exist in original folder!')


def getSegList(path):
    '''
    This function lists all "*_seg.mgz"-files that are in the given path. 

    :param path: path to BIDS derviatives database
    :return: return the lists of "*_seg.mgz"-files.
    '''
    seg_ls = sorted(list(Path(path).rglob('*_seg.mgz')))
    return seg_ls

def getT1wList(path):
    '''
    This function lists all "*_T1w.mgz"-files that are in the given path. 

    :param path: path to BIDS derviatives database
    :return: return the lists of "*_seg.mgz"-files.
    '''
    T1w_ls = sorted(list(Path(path).rglob('*_T1w.*')))
    return T1w_ls
