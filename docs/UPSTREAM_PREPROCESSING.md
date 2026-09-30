# Upstream steps: from raw scans to COPE images and ROI masks

This repository starts from first-level contrast images (COPEs) and binary ROI
masks. This page documents how those inputs were produced in the original
analysis, so the full pipeline is transparent and reproducible in principle.

The original upstream scripts ran on an institutional server and are not
included. The commands shown below are **illustrative FSL equivalents** of
each step. They are not copies of the original scripts.

## 1. DICOM to NIfTI

Raw DICOM data were unpacked and converted to NIfTI before preprocessing.
The first dummy volumes were discarded to allow for T1 equilibration.

```bash
# illustrative
dcm2niix -z y -o nifti/ dicom/sub-01/
```

## 2. Preprocessing (FSL FEAT 6.0)

| Component | Setting |
|---|---|
| Motion correction | MCFLIRT |
| Slice-timing correction | Regular ascending |
| Brain extraction | BET |
| Spatial smoothing | 2 mm FWHM Gaussian kernel |
| Temporal filtering | High-pass, 100 s cutoff |
| Intensity normalisation | Not applied |
| Functional → structural | Linear registration |
| Structural → standard | Nonlinear registration to MNI152 2 mm |

Preprocessed images, brain extraction and registrations were visually
inspected for artefacts, incomplete coverage and misregistration before
modelling.

A 2 mm kernel is smaller than typical univariate choices (5–8 mm). It reduces
noise while keeping the fine-grained spatial patterns that multivoxel analysis
relies on.

## 3. First-level GLM (FSL FEAT)

**EV files.** One explanatory variable (EV) file per condition, in FSL
three-column format (onset in seconds, duration, amplitude), generated from
the behavioural logs:

```
# encoding.txt   onset  duration  weight
12.5   3   1
17.5   3   1
...
```

- Onsets were shifted to account for the discarded dummy volumes, so that
  events align with the analysed time series.
- Events were modelled with 3 s duration and amplitude 1.
- For the recall condition, only correctly recalled trials entered the EV;
  incorrect trials were excluded.

**Contrasts.** With EVs ordered [task A, task B, baseline]:

| COPE | Contrast | Vector |
|---|---|---|
| `cope1` | Task A > Baseline | [1, 0, −1] |
| `cope2` | Task B > Baseline | [0, 1, −1] |

**Standard space.** COPEs were transformed into MNI152 2 mm space with each
run's FEAT-derived functional-to-standard transformation, using trilinear
interpolation for these continuous images:

```bash
# illustrative
applywarp --ref="${FSLDIR}/data/standard/MNI152_T1_2mm" \
          --in=run.feat/stats/cope1 \
          --warp=run.feat/reg/example_func2standard_warp \
          --out=run.feat/stats/cope1_standard --interp=trilinear
```

No statistical thresholding was applied. Unthresholded COPEs keep the
continuous voxel-wise variation that multivoxel prediction uses, rather than
keeping only voxels that pass a univariate significance test.

## 4. ROI masks

- Cortical and subcortical ROIs came from the Harvard-Oxford structural
  atlases distributed with FSL.
- A brainstem ROI came from the Harvard Ascending Arousal Network atlas.
- The atlas files were provided by the supervisory team. I prepared them for
  analysis: where needed, I resampled each binary mask to the COPE grid, then
  checked its dimensions and alignment.

```bash
# illustrative: binarise, then resample to the 2 mm MNI grid.
# Nearest-neighbour interpolation keeps the mask binary.
fslmaths roi_atlas.nii.gz -bin roi_bin.nii.gz
flirt -in roi_bin.nii.gz -ref "${FSLDIR}/data/standard/MNI152_T1_2mm" \
      -applyxfm -usesqform -interp nearestneighbour -out roi_2mm.nii.gz

# QC: dimensions and voxel size must match the COPE images (91 x 109 x 91, 2 mm)
fslinfo roi_2mm.nii.gz
fslstats roi_2mm.nii.gz -R -V
```

Each mask was visually checked against the MNI152 template. The masks were
then combined with `scripts/01_make_union_mask.sh`, which is where this
repository begins.
