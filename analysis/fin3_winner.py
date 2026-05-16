"""
Voxel-Wise Categorical Mapping: Optimized Winner-Take-All (WTA) Parcellation
----------------------------------------------------------------------------
- Feature Scaling: Applies standard z-score normalization (StandardScaler) across 
                     the design matrix to ensure category-neutral regularization.
- High-Sensitivity Ridge: Fits L2-regularized models (alpha=10.0) in chunk-isolated loops 
                          to extract high-resolution, un-smoothed neural tuning weights.
- Localized Normalization: Abandons global scaling for a voxel-wise L1 relative-proportion 
                           norm, mitigating outlier distortion across distant regions.
- Competitive Assignment: Stacks relative weights and executes a Winner-Take-All decision 
                           rule (np.argmax) to map core auditory attributes (1: Timbre, 
                           2: Harmony, 3: Rhythm).
- Volumetric Packaging: Re-maps 1D categorical masks back to 3D MNI152 coordinates, 
                        exporting standard compliant NIfTI files (.nii.gz).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from nilearn import image
import gc
from pathlib import Path

# --- 1. CONFIG & LOADING ---
print("🚀 Initializing Mathematically Optimized WTA Encoding Model...")
SPACE = "space-MNI152NLin2009cAsym_res-2"
BASE_DIR = Path("../output/sub-001")
mask_path = BASE_DIR / "anat" / f"sub-001_{SPACE}_desc-brain_mask.nii.gz"

bold_all = np.load("bold_all.npy", mmap_mode='r') 
run_info = np.load("run_info.npy", allow_pickle=True)
clip_features = np.load("clip_features.npy", allow_pickle=True).item()
events_all = pd.read_csv("events_all.csv")
mask_img = image.load_img(str(mask_path))

# --- 2. CONSTRUCT & SCALE DESIGN MATRIX (X) ---
n_trs, n_voxels = bold_all.shape
sample_key = next(iter(clip_features))
n_features = clip_features[sample_key].shape[1] 

X_raw = np.zeros((n_trs, n_features))
for _, row in events_all.iterrows():
    key = (row['genre'], int(row['track']), round(row['start'], 1), round(row['end'], 1))
    if key in clip_features:
        start_tr = int(row['global_onset_tr'])
        X_raw[start_tr : start_tr + 10, :] = clip_features[key]

# 🌟 CRITICAL CORRECTION 1: Balance feature magnitude variances before regularization
print("⚖️ Applying Z-Score Feature Scaling across Design Matrix...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Partition timelines via pre-calculated run indices
train_idx = [i for r in run_info if r['task'] == "Training" for i in range(r['start_tr'], r['end_tr'])]
test_idx = [i for r in run_info if r['task'] == "Test" for i in range(r['start_tr'], r['end_tr'])]
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]

# --- 3. CHUNKED RIDGE & TUNING ---
chunk_size = 5000
all_corrs = []
t_raw, h_raw, r_raw = [], [], []

print(f"Processing {n_voxels} voxels with Sensitive Regularization (Alpha=10.0)...")

for i in range(0, n_voxels, chunk_size):
    end = min(i + chunk_size, n_voxels)
    Y_train = np.array(bold_all[train_idx, i:end])
    Y_test = np.array(bold_all[test_idx, i:end])
    
    # 🌟 CRITICAL CORRECTION 2: Low-alpha setting prevents over-smoothing of tuning differences
    model = Ridge(alpha=10.0)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    # Calculate predictive accuracy
    for v in range(Y_test.shape[1]):
        r = pearsonr(Y_test[:, v], Y_pred[:, v])[0] if np.std(Y_test[:, v]) > 1e-6 else 0
        all_corrs.append(r)

    # Deconstruct weights across time-delay index groups
    weights = model.coef_
    timbre_idx = [f + (d * 27) for d in range(6) for f in range(0, 13)]
    harmony_idx = [f + (d * 27) for d in range(6) for f in range(13, 25)]
    rhythm_idx = [f + (d * 27) for d in range(6) for f in [25, 26]]

    t_raw.extend(np.mean(np.abs(weights[:, timbre_idx]), axis=1))
    h_raw.extend(np.mean(np.abs(weights[:, harmony_idx]), axis=1))
    r_raw.extend(np.mean(np.abs(weights[:, rhythm_idx]), axis=1))
    
    del Y_train, Y_test, Y_pred; gc.collect()

# --- 4. OPTIMIZED LOCAL WINNER-TAKE-ALL ---
print("\n⚔️ Executing Localized Voxel-Wise Competition...")
t_arr = np.array(t_raw)
h_arr = np.array(h_raw)
r_arr = np.array(r_raw)

# 🌟 CRITICAL CORRECTION 3: Voxel-wise L1 relative normalization replaces flawed global Min-Max
# This isolates relative sensory preference within each voxel independently of whole-brain maximums.
eps = 1e-8
voxel_totals = t_arr + h_arr + r_arr + eps
t_norm = t_arr / voxel_totals
h_norm = h_arr / voxel_totals
r_norm = r_arr / voxel_totals

# Multi-class competitive alignment 
# Index Codes: 1 = Timbre (Blueish), 2 = Harmony (Reddish), 3 = Rhythm (Greenish)
stacked = np.vstack([t_norm, h_norm, r_norm]).T
winner_indices = np.argmax(stacked, axis=1) + 1

# Mask out areas failing statistical validation (r <= 0.15)
r_mask = (np.array(all_corrs) > 0.15).astype(float)
winner_map = winner_indices * r_mask

# --- 5. SAVE COMPLIANT NIFTI VOLUMES ---
def save_as_nifti(data_array, filename):
    vol = np.zeros(mask_img.shape)
    vol[mask_img.get_fdata().astype(bool)] = np.array(data_array)
    img = image.new_img_like(mask_img, vol)
    img.to_filename(filename)

save_as_nifti(all_corrs, "map_correlation.nii.gz")
save_as_nifti(winner_map, "map_WINNERS_OPTIMIZED.nii.gz")

print("✅ OPTIMIZED MAPS EXPORTED successfully.")
print(f"Summary Stats within Mask:")
print(f" -> Timbre Dominant Voxels:  {np.sum(winner_map==1)}")
print(f" -> Harmony Dominant Voxels: {np.sum(winner_map==2)}")
print(f" -> Rhythm Dominant Voxels:  {np.sum(winner_map==3)}")