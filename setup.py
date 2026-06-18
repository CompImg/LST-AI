from setuptools import setup, find_packages

setup(
    name='LST_AI',
    version='1.3.0',
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
    # Modernised, pip-only stack (no TensorFlow, no compiled greedy, no git HD-BET):
    #   - inference: ONNX Runtime (segment.py --backend onnx); the .h5->.onnx
    #     conversion lives in scripts/tf_to_onnx.py and needs TF separately.
    #   - registration: picsl-greedy (Python API, same greedy engine).
    #   - brain extraction: HD-BET v2 (PyPI), which sets the python>=3.10 floor.
    # For GPU, install onnxruntime-gpu in place of onnxruntime in the deployment image.
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'pillow',
        'scipy>=1.9.0',
        'scikit-image>=0.21.0',
        'nibabel',
        'requests',
        'onnxruntime',
        'picsl-greedy',
        'hd-bet>=2.0.1',
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
