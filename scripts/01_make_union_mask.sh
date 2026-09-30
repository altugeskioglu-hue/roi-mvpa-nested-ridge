#!/usr/bin/env bash
# Build a binary union ("composite") mask from several ROI masks with FSL.
#
# Usage:
#   scripts/01_make_union_mask.sh OUTPUT.nii.gz roi1.nii.gz roi2.nii.gz [...]
#
# All inputs must already be binary and on the same standard-space grid
# (e.g. 2-mm MNI152). No resampling is done here.
#
# Adding the masks lets overlapping voxels temporarily reach 2 or more; the
# final -bin sets every positive voxel back to 1, so overlapping voxels enter
# the feature space only once.

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 OUTPUT.nii.gz roi1.nii.gz roi2.nii.gz [...]" >&2
    exit 1
fi

output="$1"
shift

cmd=(fslmaths "$1")
shift
for mask in "$@"; do
    cmd+=(-add "$mask")
done
cmd+=(-bin "$output")
"${cmd[@]}"

# QC 1: a binary mask has intensity range 0 1.
echo "Intensity range (expected: 0 1):"
fslstats "$output" -R

# QC 2: voxel count and physical volume (mm^3). Record these and pass the
# voxel count to 02_extract_features.py --expected-voxels.
echo "Voxel count and volume (mm^3):"
fslstats "$output" -V

# Optional visual check:
# fsleyes "${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz" "$output" &
