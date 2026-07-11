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
    #   - inference: PyTorch via onnx2torch (loads the ONNX UNet3D ensemble as an
    #     nn.Module). The .h5->.onnx conversion lives in scripts/tf_to_onnx.py (TF
    #     needed separately, conversion-time only). PyTorch replaced ONNX Runtime as
    #     the backend: the ORT CUDA arena transiently grabbed ~40 GB at session init
    #     and OOM'd when sharing a GPU; torch's allocator keeps it to a few GB.
    #   - registration: picsl-greedy (Python API, same greedy engine).
    #   - brain extraction: HD-BET v2 (PyPI), which sets the python>=3.10 floor.
    #
    # torch is pulled transitively (HD-BET, onnx2torch). CPU vs CUDA is a property of
    # the deployment's torch wheel (the host/container's CUDA), not of LST-AI, so no
    # per-backend extra is needed — the deployment selects the wheel (see medmcp-neuro-ms,
    # which installs the cu128 torch build).
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
        'onnx',
        'onnx2torch',
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
