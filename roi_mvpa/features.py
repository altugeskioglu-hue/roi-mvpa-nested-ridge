"""Extract unthresholded voxel values inside a binary ROI mask."""

import numpy as np


def load_binary_mask(mask_path):
    """Load a mask and require it to be strictly binary (0/1).

    Values above 1 usually mean masks were added with fslmaths without a
    final -bin, which would double-count overlapping voxels.
    """
    import nibabel as nib

    mask_img = nib.load(str(mask_path))
    data = mask_img.get_fdata(dtype=np.float64)
    values = np.unique(data)
    if not np.all(np.isin(values, [0.0, 1.0])):
        raise RuntimeError("Mask is not binary: {0}".format(values.tolist()))
    return mask_img, data > 0


def extract_features(mask_path, image_paths, expected_voxels=None):
    """Return an (n_images x n_voxels) matrix of in-mask values.

    Boolean indexing gives every image the same voxel ordering. Shape and
    affine checks ensure that a column refers to the same physical location
    in every participant. No outcome-informed voxel selection is performed.
    """
    import nibabel as nib

    mask_img, mask = load_binary_mask(mask_path)
    n_voxels = int(mask.sum())
    if expected_voxels is not None and n_voxels != expected_voxels:
        raise RuntimeError("Expected {0} mask voxels, found {1}".format(
            expected_voxels, n_voxels))

    rows = []
    for path in image_paths:
        img = nib.load(str(path))
        if img.shape[:3] != mask_img.shape[:3]:
            raise RuntimeError("{0}: image/mask shape mismatch".format(path))
        # Matching array shapes are not enough: the voxel-to-world mapping
        # must also agree.
        if not np.allclose(img.affine, mask_img.affine, rtol=0, atol=1e-5):
            raise RuntimeError("{0}: image/mask affine mismatch".format(path))
        values = img.get_fdata(dtype=np.float64)[mask]
        if not np.isfinite(values).all():
            raise RuntimeError("{0}: NaN or infinite values".format(path))
        rows.append(values)

    return np.vstack(rows), n_voxels
