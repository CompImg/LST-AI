import requests
from setuptools import setup, find_packages
import os
from io import open
import zipfile

def download_model_weights():
    model_url = "https://syncandshare.lrz.de/dl/fiPiTmWKv5Ga4S7YA7TNks/.dir"
    target_path = "data.zip"
    extract_path = "./"  # This is the base directory.

    atlas_path = os.path.join(extract_path, 'atlas')
    model_path = os.path.join(extract_path, 'model')  
    testing_path = os.path.join(extract_path, 'testing')

    paths_to_check = [atlas_path, model_path, testing_path]

    # Check if all paths exist.
    if not all(os.path.exists(path) for path in paths_to_check):

        # Download the zip file if it doesn't exist.
        if not os.path.exists(target_path):
            response = requests.get(model_url, stream=True)
            if response.status_code == 200:
                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
            else:
                raise ValueError(f"Failed to download lst.ai data. HTTP Status Code: {response.status_code}")

        # Unzip the file to the base directory.
        with zipfile.ZipFile(target_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # Remove the ZIP file after extracting its contents.
        os.remove(target_path)


# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

package_data_list = ['model', 'atlas', 'testing']

setup(name='lst.ai',
      version='1.0.0',
      description='Lesion Segmentation Toolbox AI',
      url='https://github.com/Comp.Img/lst.ai',
      author='LST AI Team',
      author_email='benedict.wiestler@tum.de',
      keywords=['lesion_segmentation', 'ms', 'lst', 'ai'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires = [
        'torch>=1.6.0',
        'numpy>=1.16.0',
        'tqdm>=4.29.0',
        'scikit-learn>=0.20.0',
        'pandas>=0.24.0',
        'six>=1.12.0',
        'urllib3>=1.24.0',
        'outdated>=0.2.0'
      ],
      license='MIT',
      # packages=find_packages(exclude=['dataset', 'examples', 'docs']),
      # package_data={'lst.ai': package_data_list},
      # include_package_data=True,
      classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
    ],
)

download_model_weights()


