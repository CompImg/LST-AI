from setuptools import setup, find_packages

setup(
    name='LST_AI',
    version='1.1.0',
    description='Lesion Segmentation Toolbox AI',
    url='https://github.com/CompImg/LST-AI',
    author='LST-AI Team',
    author_email=[
        'julian.mcginnis@tum.de',
        'tun.wiltgen@tum.de',
        'mark.muehlau@tum.de',
        'benedict.wiestler@tum.de'
    ],
    keywords=['lesion_segmentation', 'ms', 'lst', 'ai'],
    python_requires='>=3.8,<3.11',
    install_requires=[
        'numpy<1.24.4',
        'pillow',
        'scipy>=1.9.0',
        'scikit-image>=0.21.0',
        'tensorflow>=2.13,<2.16',
        'torch<=2.1.0',
        'nibabel',
        'requests'
    ],
    scripts=['LST_AI/lst'],
    license='MIT',
    packages=find_packages(include=['LST_AI']),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: Unix'
    ],
)
