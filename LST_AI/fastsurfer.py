"""Install, and later find, the FastSurfer tree LST-AI annotates lesions with.

Every mode except ``--segment_only`` assigns lesions to anatomical regions from a
FastSurfer segmentation of the T1 (LST_AI/annotate.py shells out to
``run_fastsurfer.sh``), so FastSurfer is a hard requirement of a normal run rather than
an opt-in extra. ``setup.py`` therefore fetches it at install time, next to greedy and
HD-BET; this module is both the code that does the fetching and the lookup that finds
the result at run time.

FastSurfer cannot simply be named in ``install_requires``. It is not published on PyPI,
and its own metadata could not be honoured anyway: it pins ``torch==2.7.*``, which has no
aarch64 CUDA wheel at all -- obeying it would silently swap an arm64 GPU install for a
CPU build -- and it pulls ``meshpy``, an x86_64-only wheel, for the corpus-callosum
module LST-AI switches off with ``--no_cc``. So the split is:

  * the Python dependencies of the one code path LST-AI drives (``--seg_only``, i.e.
    ``run_prediction.py`` and ``reduce_to_aseg.py``) go into setup.py's
    ``install_requires`` as :data:`FASTSURFER_REQUIRES` -- ordinary wheels, resolving the
    same way on x86_64 and on arm64/aarch64;
  * the source tree itself is unpacked here from the pinned release tarball.

Nothing is imported *from* FastSurfer, so the tree deliberately does not go into
site-packages: ``run_fastsurfer.sh`` puts ``$FASTSURFER_HOME`` at the front of
``PYTHONPATH`` itself, and a second copy under site-packages would never be the one that
runs. The Docker image installs it through this same module, for the same reason it is
pinned -- one FastSurfer, in one place, whichever way LST-AI was installed.

The copy installed here is the only one LST-AI will use. Whatever FastSurfer a machine
already has -- ``FASTSURFER_HOME``, something on ``PATH`` -- is ignored, and there is no
switch to prefer it; see :func:`find_fastsurfer`.

The VINN checkpoints (~65 MB) are not fetched here. FastSurfer downloads any it is
missing on its first segmentation, which is the same "fetch on first use" behaviour as
LST-AI's own model bundle (LST_AI/utils.py). Pre-fetch them for an offline machine with::

    python -m LST_AI.fastsurfer --checkpoints
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib import request

# The FastSurfer release LST-AI annotates with. Keep in sync with docker/Dockerfile's
# FASTSURFER_REF -- the container and a pip install must run the same segmentation.
FASTSURFER_REF = 'v2.5.4'

# The release tarball rather than a `git clone`: one stdlib download of ~17 MB, no git
# required on the machine, and the same files the image clones from that tag. GitHub's
# tarballs preserve the executable bit, which run_fastsurfer.sh needs.
FASTSURFER_URL = (
    f'https://github.com/Deep-MI/FastSurfer/archive/refs/tags/{FASTSURFER_REF}.tar.gz'
)

RUN_SCRIPT = 'run_fastsurfer.sh'

# FastSurfer's dependencies for the segmentation path LST-AI drives, at upstream's
# version floors, minus everything LST-AI already requires in its own right (nibabel,
# numpy, scipy, scikit-image, h5py, requests -- setup.py carries FastSurfer's floors for
# those) and minus the two entries in its pyproject.toml that this package cannot take:
# the torch pin and meshpy, for the reasons in the module docstring.
#
# torchvision is deliberately left unpinned where upstream writes `>=0.22.1,<0.23`. That
# bound exists only to match its torch pin, and applying it here would fight the wheels
# the Docker image installs from the CUDA index (torch and torchvision together, from
# one index, so the pair matches the image's CUDA build).
#
# When bumping FASTSURFER_REF, diff this against upstream's pyproject.toml
# `dependencies`; docker/Dockerfile carries the same list and must move with it.
FASTSURFER_REQUIRES = [
    'lapy>=1.5.0',
    'neuroreg>=0.6.2',
    'pandas>=1.5.3',
    'pyyaml>=6.0',
    'torchio>=0.18.83',
    'torchvision',
    'tqdm>=4.65',
    'yacs>=0.1.8',
]


def _managed_dirs():
    """Candidate locations for a FastSurfer tree installed by LST-AI, best first.

    The environment's own ``share/`` first, so that a shared virtualenv serves every user
    of it and the tree is discarded with the environment; the per-user data directory as
    the fallback for installs whose prefix is not writable (``pip install --user``, a
    system Python). Both are probed on the way back out, so it does not matter which one
    a given install landed in.

    The reference is part of the directory name, so bumping FASTSURFER_REF installs
    beside the old tree rather than into it, and an LST-AI that is still pinned to the
    old one keeps finding its own.
    """
    name = f'FastSurfer-{FASTSURFER_REF}'
    user_data = os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share'
    return [
        Path(sys.prefix) / 'share' / 'lst-ai' / name,
        Path(user_data) / 'lst-ai' / name,
    ]


def find_fastsurfer():
    """Return FASTSURFER_HOME for the FastSurfer LST-AI installed, or None if absent.

    Only the copy LST-AI installs is ever considered. A FastSurfer that happens to be on
    the machine already -- ``FASTSURFER_HOME`` in the environment, a ``run_fastsurfer.sh``
    on ``PATH`` -- is deliberately *not* picked up, and there is no way to point LST-AI at
    one. Annotation is part of LST-AI's output, so which FastSurfer produced it cannot
    depend on what a given machine has lying around: the pinned version below is the one
    the label mapping and the flags in LST_AI/annotate.py were written against, and the
    only one whose results are comparable across users.
    """
    for candidate in _managed_dirs():
        if (candidate / RUN_SCRIPT).is_file():
            return candidate

    return None


def _first_writable(candidates):
    """First candidate whose parent directory we can actually create and write to."""
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(candidate.parent, os.W_OK):
            return candidate
    raise OSError(
        'no writable location for FastSurfer; tried '
        + ', '.join(str(c) for c in candidates)
    )


def install_fastsurfer(force=False):
    """Unpack the pinned FastSurfer release and return its FASTSURFER_HOME.

    The destination is not a parameter: it has to be one of the directories
    :func:`find_fastsurfer` looks in, or LST-AI would not use what was just installed.

    Unpacked into a sibling temporary directory and moved into place only once it is
    complete, so an interrupted download cannot leave a half-extracted tree that later
    looks installed.
    """
    dest = _first_writable(_managed_dirs())

    if (dest / RUN_SCRIPT).is_file() and not force:
        return dest

    print(f'Downloading FastSurfer {FASTSURFER_REF} (~17 MB) to {dest} ...')
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.fastsurfer-', dir=dest.parent))
    try:
        tarball = staging / 'fastsurfer.tar.gz'
        with request.urlopen(FASTSURFER_URL) as response, open(tarball, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

        with tarfile.open(tarball) as archive:
            # The 'data' filter refuses absolute paths, symlinks escaping the archive and
            # anything that is not a plain file or directory, while keeping the
            # executable bit run_fastsurfer.sh depends on. It is the default from Python
            # 3.14; hasattr covers the 3.10 and 3.11 point releases that predate it.
            safe = {'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}
            archive.extractall(staging, **safe)

        # A GitHub tag tarball holds exactly one top-level directory, 'FastSurfer-2.5.4'
        # for tag v2.5.4 -- found rather than spelled out, so a tag whose name is not
        # simply 'v' + version still works.
        roots = [p for p in staging.iterdir() if (p / RUN_SCRIPT).is_file()]
        if len(roots) != 1:
            raise RuntimeError(
                f'unexpected FastSurfer tarball layout: no single directory containing '
                f'{RUN_SCRIPT} in {sorted(p.name for p in staging.iterdir())}'
            )

        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(roots[0]), str(dest))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f'FastSurfer {FASTSURFER_REF} installed in {dest}')
    return dest


def ensure_fastsurfer():
    """Return FASTSURFER_HOME, installing FastSurfer first if it is not there yet.

    The install-time fetch in setup.py is best-effort -- it must not turn a network
    hiccup into a failed ``pip install`` -- and it does not run at all when LST-AI is
    installed from a prebuilt wheel, since wheels have no install hook. This is what
    closes both gaps, on the first run that actually needs an annotation.
    """
    home = find_fastsurfer()
    if home is not None:
        return home

    try:
        return install_fastsurfer()
    except Exception as exc:
        raise RuntimeError(
            f'FastSurfer {FASTSURFER_REF} is needed to annotate lesions but is not '
            f'installed, and fetching it failed: {exc}\n'
            f'Install it with `python -m LST_AI.fastsurfer` once the machine can reach '
            f'GitHub, or run with --segment_only, which does not annotate.'
        ) from exc


def download_checkpoints(home=None):
    """Pre-fetch the three VINN checkpoints, using FastSurfer's own downloader.

    Only the checkpoints of the ``--seg_only`` path: CerebNet, HypVINN and the
    corpus-callosum weights belong to the modules LST-AI switches off.
    """
    home = Path(home) if home is not None else ensure_fastsurfer()
    subprocess.run(
        [sys.executable, str(home / 'FastSurferCNN' / 'download_checkpoints.py'), '--vinn'],
        check=True,
        cwd=str(home),
        env={**os.environ, 'PYTHONPATH': str(home)},
    )
    return home


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f'Install FastSurfer {FASTSURFER_REF} for LST-AI, into '
                    f'{_managed_dirs()[0]} (or {_managed_dirs()[1]} if that is not '
                    f'writable). Normally done for you at `pip install` time; run this by '
                    f'hand to repair the installation, or to pre-fetch the model '
                    f'checkpoints on a machine that will later be offline.')
    parser.add_argument('--force',
                        action='store_true',
                        help='Re-download even if it is already installed.')
    parser.add_argument('--checkpoints',
                        action='store_true',
                        help='Also download the VINN checkpoints (~65 MB), which '
                             'FastSurfer would otherwise fetch on its first run.')
    args = parser.parse_args()

    # Without --force this reports the tree a run would actually use, installing one only
    # if there is none -- so `--checkpoints` fills in the installation that is already
    # there rather than replacing it.
    fastsurfer_home = install_fastsurfer(force=True) if args.force else ensure_fastsurfer()
    print(f'FASTSURFER_HOME={fastsurfer_home}')
    if args.checkpoints:
        download_checkpoints(fastsurfer_home)
