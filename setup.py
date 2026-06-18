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
    #
    # The ONNX runtime is an EXTRA, not a base dependency, because 'onnxruntime'
    # (CPU) and 'onnxruntime-gpu' (CUDA) install into the same import namespace and
    # cannot coexist — so the backend is chosen explicitly at install time:
    #   pip install "lst-ai[cpu]"   # portable CPU wheel (x86_64 / aarch64 / macOS)
    #   pip install "lst-ai[gpu]"   # NVIDIA CUDA (onnxruntime-gpu)
    # The 'gpu' extra is deliberately unversioned: the CUDA generation is a property
    # of the deployment (the host/container's CUDA + cuDNN), not of LST-AI, so the
    # deployment image pins onnxruntime-gpu to match its CUDA (see medmcp-neuro-ms).
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'pillow',
        'scipy>=1.9.0',
        'scikit-image>=0.21.0',
        'nibabel',
        'requests',
        'picsl-greedy',
        'hd-bet>=2.0.1',
    ],
    extras_require={
        'cpu': ['onnxruntime'],
        'gpu': ['onnxruntime-gpu'],
    },
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
