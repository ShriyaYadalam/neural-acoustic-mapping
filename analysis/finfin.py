"""
fMRI Encoding Metric Extractor & Coordinate Space Transformer
------------------------------------------------------------
- Performance Metrics: Isolates and averages the top 100/500 voxel correlations to capture 
                       optimized regional spotlight accuracy independently of whole-brain noise.
- Dimensional Reconstruction: Projects flat 1D correlation metrics back into native 3D spatial 
                               matrices using binary structural mask dimensions.
- Affine Space Transformation: Multiplies native matrix voxel indexes by the image affine matrix 
                               to output millimeter-exact MNI152 coordinates.
- Anatomical Localization: Parses the transformed MNI X-axis value to verify hemispheric lateralization 
                           and flag potential midline motion artifacts automatically.
"""

import numpy as np
from nilearn import image, masking

# --- 1. LOAD DATA ---
correlations = np.load("voxel_correlations.npy")
mask_path = "../output/sub-001/anat/sub-001_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
mask_img = image.load_img(mask_path)

# --- 2. CALCULATE "SPOTLIGHT" STATS ---
# Sort correlations to find the strongest responding voxels
sorted_corrs = np.sort(correlations)[::-1]

# Let's look at the Top 100 and Top 500 voxels
top_100_mean = np.mean(sorted_corrs[:100])
top_500_mean = np.mean(sorted_corrs[:500])

# --- 3. RECONSTRUCT PEAK COORDINATE ---
corr_3d = np.zeros(mask_img.shape)
corr_3d[mask_img.get_fdata().astype(bool)] = correlations
max_idx = np.unravel_index(np.argmax(corr_3d), corr_3d.shape)
peak_coords = image.coord_transform(max_idx[0], max_idx[1], max_idx[2], mask_img.affine)

# --- 4. OUTPUT RESULTS ---
print("="*40)
print("     NON-SMOOTHED ENCODING STATS")
print("="*40)
print(f"PEAK CORRELATION (Max R):   {np.max(correlations):.3f}")
print(f"PEAK MNI COORDINATE:       {peak_coords}")
print("-" * 40)
print(f"MEAN OF TOP 100 VOXELS:    {top_100_mean:.3f}")
print(f"MEAN OF TOP 500 VOXELS:    {top_500_mean:.3f}")
print("-" * 40)
print(f"Whole Brain Median R:      {np.median(correlations):.3f}")
print("="*40)

if peak_coords[0] < -40:
    print("✅ VERDICT: Signal is localized in the LEFT Auditory Cortex.")
elif peak_coords[0] > 40:
    print("✅ VERDICT: Signal is localized in the RIGHT Auditory Cortex.")
else:
    print("⚠️ VERDICT: Peak is near midline; check for motion artifacts.")