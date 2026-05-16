"""
Voxel-Wise Encoding Model: Ridge Regression & Spatial Feature Mapping
---------------------------------------------------------------------
- Matrix Assembly: Constructs the global design matrix (X) by mapping delay-embedded 
                     acoustic feature sets into their specific fMRI onset timeframes.
- Feature Scaling: Applies standard z-score normalization (StandardScaler) across features 
                     to balance variances and ensure stable regularization penalties.
- Data Partitioning: Splits datasets strictly into "Training" and "Test" blocks using 
                     pre-calculated temporal indices to protect validation integrity.
- Memory & Modeling: Fits L2-regularized models (Ridge, alpha=10.0) across voxels in memory-safe 
                     chunks (5000 voxels/loop), generating continuous target predictions.
- Metric Extraction: Calculates Pearson correlation coefficients (r) to measure prediction 
                     accuracy, and partitions model beta weights into distinct architectural 
                     profiles (Timbre, Harmony, Rhythm).
- Volumetric Export: Unflattens 1D results back into 3D MNI152 space, exporting analytical 
                     brain maps as native NIfTI volumes (.nii.gz).
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
print("🚀 RESCUING THE MODEL: Initializing with Scaling...")
SPACE = "space-MNI152NLin2009cAsym_res-2"
BASE_DIR = Path("../output/sub-001")
mask_path = BASE_DIR / "anat" / f"sub-001_{SPACE}_desc-brain_mask.nii.gz"

bold_all = np.load("bold_all.npy", mmap_mode='r') 
run_info = np.load("run_info.npy", allow_pickle=True)
clip_features = np.load("clip_features.npy", allow_pickle=True).item()
events_all = pd.read_csv("events_all.csv")
mask_img = image.load_img(str(mask_path))

# --- 2. CONSTRUCT DESIGN MATRIX (X) ---
n_trs, n_voxels = bold_all.shape
sample_key = next(iter(clip_features))
n_features = clip_features[sample_key].shape[1] 

X_raw = np.zeros((n_trs, n_features))
for _, row in events_all.iterrows():
    key = (row['genre'], int(row['track']), round(row['start'], 1), round(row['end'], 1))
    if key in clip_features:
        start_tr = int(row['global_onset_tr'])
        X_raw[start_tr : start_tr + 10, :] = clip_features[key]

# --- 🌟 CRITICAL FIX: FEATURE SCALING ---
print("⚖️ Scaling Music Features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Split into train/test
train_idx = [i for r in run_info if r['task'] == "Training" for i in range(r['start_tr'], r['end_tr'])]
test_idx = [i for r in run_info if r['task'] == "Test" for i in range(r['start_tr'], r['end_tr'])]
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]

# --- 3. CHUNKED RIDGE WITH TUNED ALPHA ---
chunk_size = 5000
all_corrs = []
t_raw, h_raw, r_raw = [], [], []

print(f"Processing {n_voxels} voxels with Alpha=10...")

for i in range(0, n_voxels, chunk_size):
    end = min(i + chunk_size, n_voxels)
    Y_train = np.array(bold_all[train_idx, i:end])
    Y_test = np.array(bold_all[test_idx, i:end])
    
    # 🌟 CRITICAL FIX: Lower Alpha for better sensitivity
    model = Ridge(alpha=10.0) 
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    # Correlation Math
    for v in range(Y_test.shape[1]):
        r = pearsonr(Y_test[:, v], Y_pred[:, v])[0] if np.std(Y_test[:, v]) > 1e-6 else 0
        all_corrs.append(r)

    # Tuning Math
    weights = model.coef_
    timbre_idx = [f + (d * 27) for d in range(6) for f in range(0, 13)]
    harmony_idx = [f + (d * 27) for d in range(6) for f in range(13, 25)]
    rhythm_idx = [f + (d * 27) for d in range(6) for f in [25, 26]]

    t_raw.extend(np.mean(np.abs(weights[:, timbre_idx]), axis=1))
    h_raw.extend(np.mean(np.abs(weights[:, harmony_idx]), axis=1))
    r_raw.extend(np.mean(np.abs(weights[:, rhythm_idx]), axis=1))
    
    del Y_train, Y_test, Y_pred; gc.collect()

# --- 4. SAVE RESULTS ---
def save_as_nifti(data_array, filename):
    vol = np.zeros(mask_img.shape)
    vol[mask_img.get_fdata().astype(bool)] = np.array(data_array)
    img = image.new_img_like(mask_img, vol)
    img.to_filename(filename)

save_as_nifti(all_corrs, "map_correlation.nii.gz")
print("✅ FIXED MAPS SAVED. Now re-run genre_wise.py and roi_analysis.py!")