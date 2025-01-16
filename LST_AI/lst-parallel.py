import os
import glob
import pandas as pd # Required. Not installed via pip install -e .
import time
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run lst command in parallel for multiple T1 and FLAIR images.")
    
    # Required arguments for lst-parallel
    parser.add_argument('--t1',
                        dest='t1',
                        help='Path to T1 folder',
                        type=str,
                        required=True)
    parser.add_argument('--flair',
                        dest='flair',
                        help='Path to FLAIR folder',
                        type=str,
                        required=True)
    parser.add_argument('--output',
                        dest='output',
                        help='Path to Output folder',
                        type=str,
                        required=True)
    parser.add_argument('--max_workers',
                        required=True,
                        dest='max_workers',
                        help='Number of parallel processes to start. Default is 5.',
                        type=int)

    # Optional arguments which will be passed to lst command
    parser.add_argument('--save_temp',
                        dest='save_temp',
                        action='store_true',
                        help='Store temp files')
    parser.add_argument('--segment_only',
                        action='store_true',
                        dest='segment_only',
                        help='Only perform the segmentation, and skip lesion annotation.')
    parser.add_argument('--stripped',
                        action='store_true',
                        dest='stripped',
                        help='Images are already skull stripped. Skip skull-stripping.')
    parser.add_argument('--threshold',
                        dest='threshold',
                        help='Threshold for binarizing the joint segmentation (default: 0.5)',
                        type=float,
                        default=0.5)
    parser.add_argument('--lesion_threshold',
                        dest='lesion_threshold',
                        help='minimum lesion size',
                        type=int,
                        default=0)
    parser.add_argument('--clipping',
                        dest='clipping',
                        help='Clipping (min & max) for standardization of image intensities (default: 0.5 99.5).',
                        nargs='+',
                        type=str,
                        default=('0.5','99.5'))
    parser.add_argument('--fast-mode',
                        action='store_true',
                        dest='fast',
                        help='Only use one model for hd-bet.')
    parser.add_argument('--probability_map',
                        action='store_true',
                        dest='probability_map',
                        help='Additionally store the probability maps of the three models and of the ensemble network.')

    return parser.parse_args()

def get_sorted_image_lists(t1_dir, flair_dir):
    """Retrieve and sort T1 and FLAIR image lists."""
    t1_images = sorted([f for f in os.listdir(t1_dir) if f.endswith('.nii.gz')])
    flair_images = sorted([f for f in os.listdir(flair_dir) if f.endswith('.nii.gz')])
    
    if len(t1_images) != len(flair_images):
        raise ValueError("The number of T1 images does not match the number of FLAIR images.")
    
    return t1_images, flair_images

def compile_stats(output_dir: str):
    """Compile the annotated stats file for all the subjects."""
    lesion_stats  = sorted(glob.glob(output_dir + "/*/*_lesion_*"))

    data = pd.DataFrame()
    for k in range(len(lesion_stats)):
        temp = pd.read_csv(lesion_stats[k])
        temp['Subject'] = lesion_stats[k].split("/")[-2]
        data = pd.concat([data, temp], ignore_index=True)

    pivoted_data = data.pivot_table(
        index = 'Subject',
        columns = 'Region',
        values = 'Lesion_Volume',
        aggfunc = 'first'
    )
    
    pivoted_data.reset_index(inplace=True)
    pivoted_data.columns = ['Subject'] + [f"{col}_Volume" for col in pivoted_data.columns[1:]]

    return pivoted_data

def add_optional_args(command, args):
    """Add optional arguments to the command if they are specified."""
    if args.segment_only:
        command.append('--segment_only')
    if args.stripped:
        command.append('--stripped')
    if args.threshold is not None:
        command.extend(['--threshold', str(args.threshold)])
    if args.lesion_threshold is not None:
        command.extend(['--lesion_threshold', str(args.lesion_threshold)])
    if args.clipping is not None:
        command.extend(['--clipping'] + [str(float(x)) for x in args.clipping])
    if args.fast:
        command.append('--fast-mode')
    if args.probability_map:
        command.append('--probability_map')

def process_image_pair(t1_image, flair_image, t1_dir, flair_dir, output_dir, save_temp, log_dir, args):
    """Process a single pair of T1 and FLAIR images."""
    t1_path = os.path.join(t1_dir, t1_image)
    flair_path = os.path.join(flair_dir, flair_image)
    subject_name = os.path.basename(t1_path).split(".")[0]

    # Paths for output, temp (optional), and logs
    subject_output = os.path.join(output_dir, "Output", subject_name)
    log_file = os.path.join(log_dir, f"{subject_name}_processing.log")

    # Base command configuration
    command = [
        'lst',
        '--t1', t1_path,
        '--flair', flair_path,
        '--output', subject_output,
        '--device', 'cpu',  # DO NOT CHANGE
        '--threads', '1'    # DO NOT CHANGE
    ]

    # If --save_temp is used, add the --temp flag to the lst command
    if save_temp:
        temp_dir = os.path.join(output_dir, "Temp", subject_name)
        os.makedirs(temp_dir, exist_ok=True)  # Ensure the directory exists
        command.extend(['--temp', temp_dir])

    # Add optional arguments
    add_optional_args(command, args)

    print(f"Running command for {subject_name}: {' '.join(command)}")
    start_time = time.time()

    with open(log_file, "w") as log:
        result = subprocess.run(command, stdout=log, stderr=log, text=True)

    time_taken = time.time() - start_time
    print(f"Time taken to process {subject_name}: {time_taken:.2f} seconds")

    # Write completion details to log
    with open(log_file, "a") as log:
        if result.returncode == 0:
            log.write(f"\nSuccessfully processed T1: {t1_image} and FLAIR: {flair_image}\n")
        else:
            log.write(f"\nError processing T1: {t1_image} and FLAIR: {flair_image}\n")
        log.write(f"Time taken: {time_taken:.2f} seconds\n")

def main():
    start_time = time.time()
    args = parse_arguments()
    print(args)

    # Ensure output and log directories exist
    output_dir = os.path.join(args.output, "Output")
    log_dir = os.path.join(args.output, "Logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Get sorted lists of images
    t1_images, flair_images = get_sorted_image_lists(args.t1, args.flair)

    # Parallel processing
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        executor.map(
            process_image_pair,
            t1_images, flair_images,
            [args.t1] * len(t1_images),
            [args.flair] * len(flair_images),
            [args.output] * len(t1_images),
            [args.save_temp] * len(t1_images),
            [log_dir] * len(t1_images),
            [args] * len(t1_images)
        )

    end_time = time.time()  # End timing
    print(f"Processing complete. Logs are saved in: {log_dir}")
    
    # Compile the stats file for all the subjects
    stats_df = compile_stats(output_dir)
    stats_df.to_csv(os.path.join(output_dir, 'lst_stats.csv'))
    print(f"Processing complete. Stats are saved in: {output_dir}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
