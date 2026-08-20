import os
import stat
import tempfile

import nibabel as nib
import numpy as np

from lst_ai.utils import harmonize_affines, resolve_data_dir


def _write(path, qform, sform):
    img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), affine=np.eye(4))
    img.set_qform(qform, code=1)
    img.set_sform(sform, code=2)
    nib.save(img, path)


def test_harmonize_affines_resolves_qform_sform_mismatch():
    # Reproduces CompImg/LST-AI#44: an SPM-style co-registration updates only the
    # sform, so qform != sform. After harmonization both must equal nibabel's
    # resolved affine (the sform), so greedy (qform) and nibabel (sform) agree.
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.nii.gz"), os.path.join(d, "out.nii.gz")
        sform = np.eye(4)
        sform[:3, 3] = [10, -5, 3]
        _write(src, qform=np.eye(4), sform=sform)

        resolved = nib.load(src).affine  # nibabel prefers the sform
        harmonize_affines(src, dst)
        out = nib.load(dst)

        assert np.allclose(out.affine, resolved)
        assert np.allclose(out.get_qform(), resolved)
        assert np.allclose(out.get_sform(), resolved)
        assert np.allclose(out.get_qform(), out.get_sform())


def test_harmonize_affines_noop_when_consistent():
    # When qform == sform the data and affine are unchanged.
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.nii.gz"), os.path.join(d, "out.nii.gz")
        affine = np.eye(4)
        affine[:3, 3] = [1, 2, 3]
        _write(src, qform=affine, sform=affine)

        harmonize_affines(src, dst)
        out = nib.load(dst)

        assert np.allclose(out.affine, affine)
        assert np.allclose(out.get_qform(), out.get_sform())
        assert np.allclose(out.get_fdata(), nib.load(src).get_fdata())


def test_resolve_data_dir_env_override_wins():
    with tempfile.TemporaryDirectory() as d:
        assert resolve_data_dir(package_dir=d, environ={"LST_AI_DATA_DIR": "/opt/lst"}) == "/opt/lst"


def test_resolve_data_dir_prefers_existing_bundle():
    # A read-only package dir that already holds the bundle (a baked Docker image)
    # must still be chosen: presence beats writability.
    with tempfile.TemporaryDirectory() as d:
        os.mkdir(os.path.join(d, "atlas"))
        os.mkdir(os.path.join(d, "model"))
        os.chmod(d, stat.S_IRUSR | stat.S_IXUSR)
        try:
            assert resolve_data_dir(package_dir=d, environ={}) == d
        finally:
            os.chmod(d, stat.S_IRWXU)


def test_resolve_data_dir_uses_writable_package_dir():
    with tempfile.TemporaryDirectory() as d:
        assert resolve_data_dir(package_dir=d, environ={}) == d


def test_resolve_data_dir_falls_back_to_cache_when_unwritable():
    with tempfile.TemporaryDirectory() as d:
        pkg = os.path.join(d, "site-packages")
        os.mkdir(pkg)
        os.chmod(pkg, stat.S_IRUSR | stat.S_IXUSR)
        try:
            got = resolve_data_dir(package_dir=pkg, environ={"XDG_CACHE_HOME": os.path.join(d, "cache")})
            assert got == os.path.join(d, "cache", "lst_ai")
        finally:
            os.chmod(pkg, stat.S_IRWXU)
