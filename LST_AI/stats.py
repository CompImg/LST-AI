import nibabel as nib
import numpy as np
import csv
import argparse
from scipy.ndimage import label

def compute_stats(mask_file, output_file, multi_class):
    """
    Compute statistics from a lesion mask and save the results to a CSV file.

    Parameters:
    mask_file (str): Path to the input mask file in NIfTI format.
    output_file (str): Path to the output CSV file where results will be saved.
    multi_class (bool): Flag indicating whether the mask contains multiple classes (True) or is binary (False).

    This function calculates the number of lesions, the number of voxels in lesions, and the total lesion volume.
    If `multi_class` is True, these statistics are calculated for each lesion class separately.
    """
    # Load the mask file
    mask = nib.load(mask_file)
    mask_data = mask.get_fdata()

    # Voxel dimensions to calculate volume
    voxel_dims = mask.header.get_zooms()

    results = []

    if multi_class:
        # Multi-class processing
        lesion_labels = [1, 2, 3, 4]
        label_names = {
            1: 'Periventricular',
            2: 'Juxtacortical',
            3: 'Subcortical',
            4: 'Infratentorial'
        }

        for lesion_label in lesion_labels:
            class_mask = mask_data == lesion_label

            # Count lesions (connected components) for each class
            _ , num_lesions = label(class_mask)

            voxel_count = np.count_nonzero(class_mask)
            volume = voxel_count * np.prod(voxel_dims)

            results.append({
                'Region': label_names[lesion_label],
                'Num_Lesions': num_lesions,
                'Num_Vox': voxel_count,
                'Lesion_Volume': volume
            })

    else:
        # Binary mask processing
        # Assert that only two unique values are present (0 and 1)
        unique_values = np.unique(mask_data)
        assert len(unique_values) <= 2, "Binary mask must contain no more than two unique values."

        # Count lesions (connected components) in binary mask
        _, num_lesions = label(mask_data > 0)

        voxel_count = np.count_nonzero(mask_data)
        volume = voxel_count * np.prod(voxel_dims)

        results.append({
            'Num_Lesions': num_lesions,
            'Num_Vox': voxel_count,
            'Lesion_Volume': volume
        })

    # Save results to CSV
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        if multi_class:
            writer.writerow(['Region', 'Num_Lesions', 'Num_Vox', 'Lesion_Volume'])
            for result in results:
                writer.writerow([result['Region'], result['Num_Lesions'], result['Num_Vox'], result['Lesion_Volume']])
        else:
            writer.writerow(['Num_Lesions', 'Num_Vox', 'Lesion_Volume'])
            for result in results:
                writer.writerow([result['Num_Lesions'], result['Num_Vox'], result['Lesion_Volume']])

if __name__ == "__main__":
    """
    Main entry point of the script. Parses command-line arguments and calls the compute_stats function.
    """
    parser = argparse.ArgumentParser(description='Process a lesion mask file.')
    parser.add_argument('--in', dest='input_file', required=True, help='Input mask file path')
    parser.add_argument('--out', dest='output_file', required=True, help='Output CSV file path')
    parser.add_argument('--multi-class', dest='multi_class', action='store_true', help='Flag for multi-class processing')

    args = parser.parse_args()

    compute_stats(args.input_file, args.output_file, args.multi_class)
