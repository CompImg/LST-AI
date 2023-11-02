import os
import subprocess
import shlex

def mni_registration(atlas_t1, path_org_t1, path_org_flair,
                     path_mni_t1, path_mni_flair, path_t1_affine, path_flair_affine):
    """
    Perform registration of T1 and FLAIR images into MNI space.

    Parameters:
    ----------
    atlas_t1 : str
        Path to T1 atlas image.
    path_org_t1 : str
        Path to original T1 image.
    path_org_flair : str
        Path to original FLAIR image.
    path_mni_t1 : str
        Path to MNI T1 image.
    path_mni_flair : str
        Path to MNI FLAIR image.
    path_t1_affine : str
        Path to save affine for T1 registration.
    path_flair_affine : str
        Path to save affine for FLAIR registration.

    Returns:
    --------
    None

    """

    # Register T1 -> Atlas_T1 (6 DOF)
    rigid_call = (f"greedy-d 3 -a -dof 6 -m NMI -ia-image-centers "
                  f"-n 100x50x10 -i {atlas_t1} {path_org_t1} -o {path_t1_affine}")
    subprocess.run(shlex.split(rigid_call), check=True)

    warp_call = (f"greedy-d 3 -rf {atlas_t1} -rm {path_org_t1} "
                 f"{path_mni_t1} -r {path_t1_affine}")
    subprocess.run(shlex.split(warp_call), check=True)

    # Register FLAIR -> T1 in atlas space and save the affine for later
    rigid_call = (f"greedy-d 3 -a -dof 6 -m NMI -ia-image-centers "
                  f"-n 100x50x10 -i {path_mni_t1} {path_org_flair} -o {path_flair_affine}")
    subprocess.run(shlex.split(rigid_call), check=True)

    warp_call = (f"greedy-d 3 -rf {atlas_t1} -rm {path_org_flair} "
                 f"{path_mni_flair} -r {path_flair_affine}")
    subprocess.run(shlex.split(warp_call), check=True)



def warp_mni_to_orig(image_org_space, image_to_mni_affine, seg_mni, seg_orig):
    """Warps an image from MNI space to its original space."""
    warp_back_call = (
        f"greedy-d 3 -rf {image_org_space} -ri LABEL 0.2vox -rm {seg_mni} "
        f"{seg_orig} -r {image_to_mni_affine},-1"
    )
    subprocess.run(shlex.split(warp_back_call), check=True)


def warp_orig_to_mni(image_org_space, image_to_mni_affine, seg_orig, seg_mni):
    """Warps an image from its original space to MNI space."""
    warp_back_call = (
        f"greedy-d 3 -rf {image_org_space} -ri LABEL 0.2vox -rm {seg_orig} "
        f"{seg_mni} -r {image_to_mni_affine}"
    )
    subprocess.run(shlex.split(warp_back_call), check=True)



if __name__ == "__main__":

    # Working directory
    script_dir = os.getcwd()
    parent_directory = os.path.dirname(script_dir)

    atlas_t1_path = os.path.join(parent_directory, "atlas", "sub-mni152_space-mni_t1.nii.gz")
    atlas_mask_path = os.path.join(parent_directory, "atlas", "sub-mni152_space-mni_msmask.nii.gz")

    # Example data
    path_org_t1_path = os.path.join(parent_directory, "testing", "segmentation", "T1W.nii.gz")
    path_org_flair_path = os.path.join(parent_directory, "testing", "segmentation", "FLAIR.nii.gz")

    # Temp directory
    temp_path = os.path.join(parent_directory, "temp")
    # If temp directory does not exist, make it
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    # Where to store the results
    path_mni_t1_path = os.path.join(temp_path, "mni_t1.nii.gz")
    path_mni_flair_path = os.path.join(temp_path, "mni_flair.nii.gz")
    path_t1_affine_path = os.path.join(temp_path, "t1_mni_affine.mat")
    path_flair_affine_path = os.path.join(temp_path, "flair_mni_affine.mat")

    mni_registration(
        atlas_t1=atlas_t1_path,
        path_org_t1=path_org_t1_path,
        path_org_flair=path_org_flair_path,
        path_mni_t1=path_mni_t1_path,
        path_mni_flair=path_mni_flair_path,
        path_t1_affine=path_t1_affine_path,
        path_flair_affine=path_flair_affine_path
    )
    print("Finished!")
