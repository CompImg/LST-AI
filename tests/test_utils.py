import os
import tempfile

import nibabel as nib
import numpy as np

from LST_AI.utils import harmonize_affines


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
