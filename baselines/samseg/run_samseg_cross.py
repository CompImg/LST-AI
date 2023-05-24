import argparse
import os
import shutil
import logging
import datetime
from pathlib import Path
import multiprocessing
from utils import getSessionID, getSubjectID, MoveandCheck, split_list

def process_samseg(dirs, derivatives_dir, freesurfer_path, remove_temp=False, coregister=False):
    
    # initialize logging file
    logging.basicConfig(filename = os.path.join(derivatives_dir, "samseg_logfile.log"), level = logging.INFO)

    # iterate through all subject folders
    for dir in dirs:

        ### assemble T1w and FLAIR file lists
        t1w = sorted(list(Path(dir).rglob('*T1w.nii.gz')))
        flair = sorted(list(Path(dir).rglob('*FLAIR.nii.gz')))
        
        # convert entries to string
        t1w = [str(x) for x in t1w]
        flair = [str(x) for x in flair]
        
        # get subject ID of current subject
        loop_subID = getSubjectID(t1w[0])

        try:
            # check if more than 1 scans are available and if there are as many t1 as flair images
            if (len(t1w) != len(flair)) or (len(t1w) < 1) or (len(flair) < 1):
                    # instead of using assert we use this mechanism due to parallel processing
                    # assert len(t1w) == len(flair), 'Mismatch T1w/FLAIR number'
                    # we do not check for file corresondance as lists are sorted anyway
                    print(f"Fatal Error for {dir}")
                    logging.error(f"{datetime.datetime.now()} {dir}: Conflict with number of T1w or FLAIR images.")
                    break

            for i in range(len(t1w)):
                loop_sesID_t1 = getSessionID(t1w[i])
                loop_sesID_flair = getSessionID(flair[i])
                # check if subID and sesID are the same for T1w and FLAIR images
                if (loop_sesID_t1 != loop_sesID_flair):
                    print(f"Fatal Error for {dir}, different session IDs for T1w and FLAIR!")
                    logging.error(f"{datetime.datetime.now()} {dir}: T1w and FLAIR image session-ID are not the same: T1w->{loop_sesID_t1}, FLAIR->{loop_sesID_flair}")
                    break

                ### perform coregistration of FLAIR to T1w
                # do the registration in a template folder and distribute its results to BIDS conform output directories later

                # create template folder
                #print(t1w)
                temp_dir = os.path.join(derivatives_dir, f'sub-{loop_subID}', f'ses-{loop_sesID_t1}', 'temp')
                print(temp_dir)
                temp_dir_output = os.path.join(temp_dir, "output")
                print(temp_dir_output)
                Path(temp_dir_output).mkdir(parents=True, exist_ok=True)

                # generate a derivative folder for each session (BIDS)
                deriv_ses = os.path.join(derivatives_dir, f'sub-{loop_subID}', f'ses-{loop_sesID_t1}', 'anat')
                Path(deriv_ses).mkdir(parents=True, exist_ok=True)

                # co-register FLAIR to T1w if necessary
                if coregister:
                    # pre-define paths of registered image(s) 
                    flair_reg_field = str(Path(flair[i]).name).replace("FLAIR.nii.gz", "space-T1w_FLAIR.lta")
                    flair_reg = str(Path(flair[i]).name).replace("FLAIR.nii.gz", "space-T1w_FLAIR.mgz")

                    # get transformation
                    os.system(f'export FREESURFER_HOME={freesurfer_path} ; \
                                cd {temp_dir}; \
                                mri_coreg --mov {flair[i]} --ref {t1w[i]} --reg {flair_reg_field};\
                                ')

                    # apply transformation
                    os.system(f'export FREESURFER_HOME={freesurfer_path} ; \
                                cd {temp_dir}; \
                                mri_vol2vol --mov {flair[i]} --reg {flair_reg_field} --o {flair_reg} --targ {t1w[i]};\
                                ')
                    
                    # copy the FLAIR transformation file from template folder to derivatives session folder
                    reg_field_temp_location = os.path.join(temp_dir, flair_reg_field)
                    reg_field_target_location = os.path.join(deriv_ses, flair_reg_field)
                    #print(reg_field_temp_location)
                    #print(reg_field_target_location)
                    MoveandCheck(reg_field_temp_location, reg_field_target_location)
                    # copy the registered FLAIR image file from template fodler to derivatives session folder
                    flair_temp_location = os.path.join(temp_dir, flair_reg)
                    flair_target_location = os.path.join(deriv_ses, flair_reg)
                    #print(flair_temp_location)
                    #print(flair_target_location)
                    MoveandCheck(flair_temp_location, flair_target_location)
                else:
                    flair_reg = flair[i]

                ### run SAMSEG cross sectional segmentation 
                os.system(f'export FREESURFER_HOME={freesurfer_path} ; \
                            cd {temp_dir}; \
                            run_samseg --input {t1w[i]} {flair_reg} --threads 4 --pallidum-separate --lesion --lesion-mask-pattern 0 1 -o output/\
                            ')

                ### aggregate the samseg output files and move to appropriate directories
                # write template file paths into a list 
                output_files = os.listdir(temp_dir_output)
                # check if fodler is empty or not
                if ('seg.mgz' in output_files):
                    # iterate over all output files and copy them to derivatives anat folder
                    for filename in output_files:
                        # rename to BIDS
                        file_bids = f'sub-{loop_subID}_ses-{loop_sesID_t1}_' + filename

                        # define location of file in temporary fodler
                        filename_temp_location = os.path.join(temp_dir_output, filename)
                        # define target location of file
                        filename_target_location = os.path.join(deriv_ses, file_bids)

                        # copy file
                        MoveandCheck(filename_temp_location, filename_target_location)
                else:
                    print(f'{dir}: failed to generate segmentation!')
                    logging.error(f"{datetime.datetime.now()} {dir}: seg.mgz file was not created!")

                if remove_temp:
                    # delete the temp folder
                    shutil.rmtree(temp_dir)
                    if os.path.exists(temp_dir):
                        raise ValueError(f'failed to delete the template folder: {temp_dir}')
                    else:
                        print(f'successfully deleted the template folder: {temp_dir}')
        except:
            print("Error occured during processing, proceeding with next subject.")
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run SAMSEG cross sectional Pipeline on cohort.')
    parser.add_argument('-i', '--input_directory', help='Folder of derivatives in BIDS database.', required=True)
    parser.add_argument('-n', '--number_of_workers', help='Number of parallel processing cores.', type=int, default=os.cpu_count()-1)
    parser.add_argument('-f', '--freesurfer_path', help='Path to freesurfer binaries.', default='/home/twiltgen/Tun_software/Freesurfer/FS_7.3.2/freesurfer')
    parser.add_argument('--coregister', action='store_true')
    parser.add_argument('--remove_temp', action='store_true')

    # read the arguments
    args = parser.parse_args()

    if args.remove_temp:
        remove_temp = True
    else:
        remove_temp = False
    
    if args.coregister:
        coregister = True
    else:
        coregister = False

    # generate derivatives/labels/
    derivatives_dir = os.path.join(args.input_directory, "derivatives/samseg-7.3.2")
    Path(derivatives_dir).mkdir(parents=True, exist_ok=True)
    data_root = Path(os.path.join(args.input_directory))

    dirs = sorted(list(data_root.glob('*')))
    dirs = [str(x) for x in dirs]
    dirs = [x for x in dirs if "sub-" in x]
    files = split_list(dirs, args.number_of_workers)

    # initialize multithreading
    pool = multiprocessing.Pool(processes=args.number_of_workers)
    # creation, initialisation and launch of the different processes
    for x in range(0, args.number_of_workers):
        pool.apply_async(process_samseg, args=(files[x], derivatives_dir, args.freesurfer_path, remove_temp, coregister))

    pool.close()
    pool.join()

