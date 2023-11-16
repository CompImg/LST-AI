import os
import subprocess
import shlex

def mni_registration(atlas_t1, path_org_t1, path_org_flair,
                     path_mni_t1, path_mni_flair,
                     path_t1_affine, path_flair_affine,
                     n_threads=1):
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
    threads : int, optional
        Number of threads to use for registration. Default is 1.

    Returns:
    --------
    None

    """

    # Register T1 -> Atlas_T1 (6 DOF)
    rigid_call = (f"greedy -d 3 -a -dof 6 -m NMI -ia-image-centers "
                  f"-n 100x50x10 -i {atlas_t1} {path_org_t1} -o {path_t1_affine} -threads {n_threads}")
    subprocess.run(shlex.split(rigid_call), check=True)

    warp_call = (f"greedy -d 3 -rf {atlas_t1} -rm {path_org_t1} "
                 f"{path_mni_t1} -r {path_t1_affine} -threads {n_threads}")
    subprocess.run(shlex.split(warp_call), check=True)

    # Register FLAIR -> T1 in atlas space and save the affine for later
    rigid_call = (f"greedy -d 3 -a -dof 6 -m NMI -ia-image-centers "
                  f"-n 100x50x10 -i {path_mni_t1} {path_org_flair} -o {path_flair_affine} -threads {n_threads}")
    subprocess.run(shlex.split(rigid_call), check=True)

    warp_call = (f"greedy -d 3 -rf {atlas_t1} -rm {path_org_flair} "
                 f"{path_mni_flair} -r {path_flair_affine} -threads {n_threads}")
    subprocess.run(shlex.split(warp_call), check=True)

def rigid_reg(moving, fixed, affine, destination, n_threads):
    """
    Perform rigid registration.

    Parameters:
    ----------
    moving : str
        Path to moving image.
    fixed : str
        Path to fixed image.
    affine : str
        Path to affine.
    destination : str
        Path to registered image.
    n_threads : int
        Number of threads used for registration.
    """
    # Register moving -> fixed (6 DOF)
    rigid_call = (f"greedy -d 3 -a -dof 6 -m NMI -ia-image-centers "
                  f"-n 100x50x10 -i {fixed} {moving} -o {affine} -threads {n_threads}")
    subprocess.run(shlex.split(rigid_call), check=True)

    warp_call = (f"greedy -d 3 -rf {fixed} -rm {moving} "
                 f"{destination} -r {affine} -threads {n_threads}")
    subprocess.run(shlex.split(warp_call), check=True)


def apply_warp(image_org_space, affine, origin, target, reverse=False, n_threads=1):
    """
    Warps an image between its original space and target space.

    Parameters:
    image_org_space: str - The image in its original space.
    affine: str - The affine transformation file.
    origin: str - The origin image file.
    target: str - The target image file.
    reverse: bool - If True, warps from target space to original space; otherwise,
            from original space to target space.
    threads : int, optional
        Number of threads to use for registration. Default is 1.
    """
    if reverse:
        warp_call = (
            f"greedy -threads {n_threads} -d 3 -rf {image_org_space} -ri LABEL 0.2vox -rm {origin} "
            f"{target} -r {affine},-1"
        )
    else:
        warp_call = (
            f"greedy -threads {n_threads} -d 3 -rf {image_org_space} -ri LABEL 0.2vox -rm {origin} "
            f"{target} -r {affine}"
        )

    subprocess.run(shlex.split(warp_call), check=True)



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
