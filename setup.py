import importlib.util
import os
import sys
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

_HERE = Path(__file__).parent

# Without these two, PyPI renders an empty project page and twine warns on every upload.
_README = (_HERE / 'README.md').read_text(encoding='utf-8')


def _load_fastsurfer_module():
    """Load lst_ai/fastsurfer.py by path, rather than by `import lst_ai.fastsurfer`.

    setup.py runs before the package is installed, and cannot count on the source tree
    being importable: PEP 517 backends execute it from a directory of their own choosing.
    Reading the one file directly sidesteps sys.path entirely -- and that module is
    stdlib-only precisely so it can be read here, with none of LST-AI's own dependencies
    installed yet.
    """
    spec = importlib.util.spec_from_file_location(
        '_lst_ai_fastsurfer', _HERE / 'lst_ai' / 'fastsurfer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fastsurfer = _load_fastsurfer_module()


class BuildPyWithFastSurfer(build_py):
    """Fetch FastSurfer as part of the build, so a pip install ships it like the rest.

    Lesion annotation *is* a FastSurfer segmentation of the T1, and it runs in every mode
    except --segment_only, so FastSurfer belongs with greedy and HD-BET rather than in a
    list of manual follow-up steps. Its Python dependencies are ordinary wheels and ride
    along in install_requires below; only the source tree has to be fetched here, since
    FastSurfer is not published on PyPI (see lst_ai/fastsurfer.py for the full reasoning,
    including why its own metadata cannot be used).

    build_py is the hook that runs for `pip install .` *and* `pip install -e .`, on both
    architectures; `install` would be skipped by every wheel-based install. The tree is
    written outside the build directory, so it is never baked into the wheel -- a wheel
    built here stays a normal small wheel, and lst_ai.fastsurfer.ensure_fastsurfer()
    fetches on first use for whoever installs it.
    """

    def run(self):
        if os.environ.get('LST_AI_SKIP_FASTSURFER'):
            print('LST_AI_SKIP_FASTSURFER set: not installing FastSurfer.')
        else:
            try:
                # find first: an existing FASTSURFER_HOME or a run_fastsurfer.sh on PATH
                # is the installation this machine already has, and it wins.
                home = (_fastsurfer.find_fastsurfer()
                        or _fastsurfer.install_fastsurfer())
                print(f'FastSurfer {_fastsurfer.FASTSURFER_REF} ready in {home}')
            except Exception as exc:
                # Never fail the install over this. A machine that cannot reach GitHub
                # right now still gets a working --segment_only, and the first run that
                # needs an annotation retries the download itself.
                print(f'WARNING: could not install FastSurfer: {exc}\n'
                      f'         LST-AI will retry on the first run that annotates '
                      f'lesions; --segment_only is unaffected.', file=sys.stderr)
        super().run()

setup(
    # Distribution name lst-ai (what you pip install), import package lst_ai (what you
    # import) -- the PEP 503/PEP 8 pairing.
    name='lst-ai',
    version='2.0.0',
    description='Lesion Segmentation Toolbox AI',
    long_description=_README,
    long_description_content_type='text/markdown',
    url='https://github.com/CompImg/LST-AI',
    author='LST-AI Team',
    # A single RFC-822 style string -- a Python list here serializes as its repr and
    # renders garbage on the PyPI page.
    author_email=('julian.mcginnis@tum.de, tun.wiltgen@tum.de, '
                  'mark.muehlau@tum.de, b.wiestler@tum.de'),
    keywords=['lesion_segmentation', 'ms', 'lst', 'ai'],
    # Pip-only stack, PyTorch throughout (no TensorFlow, no ONNX Runtime, no onnx,
    # no onnx2torch, no compiled greedy, no git HD-BET):
    #   - inference: native PyTorch (lst_ai/model.py). Since v2.0.0 the released
    #     weights ship as .pt, so nothing outside torch is needed to read them.
    #     Running under torch rather than ONNX Runtime also bounds GPU memory: the
    #     ORT CUDA arena transiently grabbed ~40 GB at session init and OOM'd when
    #     sharing a GPU, where torch's caching allocator stays at a few GB.
    #   - registration: picsl-greedy (Python API, same greedy engine).
    #   - brain extraction: brainles_hd_bet, a pinned HD-BET v1 fork -- the version the
    #     released weights were validated against, and the only one with an arm64 wheel.
    #   - anatomical annotation: FastSurfer. Its Python dependencies are the wheels in
    #     FASTSURFER_REQUIRES; the source tree itself is fetched by the build_py hook
    #     above, because FastSurfer is not on PyPI.
    #
    # CPU vs CUDA is a property of the deployment's torch wheel (the host/container's
    # CUDA), not of LST-AI, so no per-backend extra is needed.
    python_requires='>=3.10',
    # The floors on numpy, scipy, nibabel, h5py and requests are FastSurfer's, not
    # LST-AI's own -- FastSurfer now runs out of the same environment, so its
    # requirements on the packages the two share are requirements of this install.
    # scikit-image's floor is already above the >=0.19.3 it asks for.
    install_requires=[
        'numpy>=1.25',
        'pillow',
        'scipy>=1.10.1,!=1.13.0',
        'scikit-image>=0.21.0',
        'nibabel>=5.4.0',
        'requests>=2.31.0',
        'torch',
        'h5py>=3.7',
        # >=1.4.0.1: first release with linux/aarch64 wheels; older versions would fall
        # back to an sdist build (compiling VTK) on arm64.
        'picsl-greedy>=1.4.0.1',
        'brainles_hd_bet',
    ] + _fastsurfer.FASTSURFER_REQUIRES,
    extras_require={
        # Only for reading a legacy v1.3.0 .onnx bundle, or re-running the .onnx -> .pt
        # export in lst_ai/weights.py. Not needed to run inference.
        'onnx': ['onnx'],
    },
    scripts=['lst_ai/lst'],
    license='MIT',
    packages=find_packages(include=['lst_ai']),
    cmdclass={'build_py': BuildPyWithFastSurfer},
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: Unix'
    ],
)
