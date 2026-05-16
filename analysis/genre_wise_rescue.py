"""
Voxel-Wise Genre Specific Encoding & Cross-Validation Matrix
------------------------------------------------------------
- Feature Normalization: Incorporates standard z-score scaling (StandardScaler) across 
                         the unified design matrix to maintain stable ridge evaluations.
- Master Model Assembly: Fits a high-sensitivity L2-regularized Ridge regression (alpha=10.0) 
                         on the full combined multi-genre Training dataset in 5,000-voxel blocks, 
                         caching the comprehensive weight matrix.
- Out-of-Sample Sub-Targeting: Isolates separate genre-specific time windows within the Test dataset 
                               (e.g., Hip-Hop, Classical) to perform precise out-of-sample testing.
- Target Cross-Prediction: Generates synthetic voxel-wise BOLD predictions via matrix multiplication 
                           (X_test @ Weights.T) and evaluates prediction performance using local 
                           Pearson correlation coefficients (r).
- Individual Atlas Generation: Saves continuous, un-thresholded prediction accuracy volumes 
                               (.nii.gz) for every single available musical genre.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler # NEW
from scipy.stats import pearsonr
from nilearn import image
import gc
from pathlib import Path

# --- 1. CONFIG ---
print("🎸 RESCUING GENRE MAPS: Scaling + Alpha=10...")
SPACE = "space-MNI152NLin2009cAsym_res-2"
BASE_DIR = Path("../output/sub-001")
mask_path = BASE_DIR / "anat" / f"sub-001_{SPACE}_desc-brain_mask.nii.gz"

bold_all = np.load("bold_all.npy", mmap_mode='r') 
clip_features = np.load("clip_features.npy", allow_pickle=True).item()
events_all = pd.read_csv("events_all.csv")
mask_img = image.load_img(str(mask_path))

# --- 2. DESIGN MATRIX + SCALING ---
n_trs, n_voxels = bold_all.shape
sample_key = next(iter(clip_features))
n_features = clip_features[sample_key].shape[1] 

X_raw = np.zeros((n_trs, n_features))
for _, row in events_all.iterrows():
    key = (row['genre'], int(row['track']), round(row['start'], 1), round(row['end'], 1))
    if key in clip_features:
        start_tr = int(row['global_onset_tr'])
        X_raw[start_tr : start_tr + 10, :] = clip_features[key]

# 🌟 CRITICAL FIX: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# --- 3. TRAIN MASTER MODEL ---
train_mask = (events_all['task'] == "Training")
train_idx = []
for _, row in events_all[train_mask].iterrows():
    onset = int(row['global_onset_tr'])
    train_idx.extend(range(onset, onset + 10))

X_train = X_scaled[train_idx]
all_weights = np.zeros((n_voxels, n_features))

chunk_size = 5000
for i in range(0, n_voxels, chunk_size):
    end = min(i + chunk_size, n_voxels)
    Y_train = np.array(bold_all[train_idx, i:end])
    
    # 🌟 CRITICAL FIX: Alpha=10.0
    model = Ridge(alpha=10.0)
    model.fit(X_train, Y_train)
    all_weights[i:end, :] = model.coef_
    del Y_train; gc.collect()

# --- 4. GENRE-SPECIFIC LOOP ---
available_genres = events_all[events_all['task'] == "Test"]['genre'].unique()

for g in available_genres:
    print(f"📊 Processing: {g}")
    genre_test_mask = (events_all['task'] == "Test") & (events_all['genre'] == g)
    genre_test_trs = []
    for _, row in events_all[genre_test_mask].iterrows():
        onset = int(row['global_onset_tr'])
        genre_test_trs.extend(range(onset, onset + 10))
    
    if not genre_test_trs: continue

    X_test_genre = X_scaled[genre_test_trs]
    genre_corrs = np.zeros(n_voxels)
    
    for i in range(0, n_voxels, chunk_size):
        end = min(i + chunk_size, n_voxels)
        Y_test_genre = np.array(bold_all[genre_test_trs, i:end])
        Y_pred_genre = X_test_genre @ all_weights[i:end, :].T
        
        for v_idx in range(Y_test_genre.shape[1]):
            if np.std(Y_test_genre[:, v_idx]) > 1e-6:
                r, _ = pearsonr(Y_test_genre[:, v_idx], Y_pred_genre[:, v_idx])
                genre_corrs[i + v_idx] = r

    # Save Accuracy Maps
    # We save the RAW correlations; the Plotting script will handle the thresholding
    def save_as_nifti(data_array, filename):
        vol = np.zeros(mask_img.shape)
        vol[mask_img.get_fdata().astype(bool)] = np.array(data_array)
        img = image.new_img_like(mask_img, vol)
        img.to_filename(filename)

    save_as_nifti(genre_corrs, f"map_accuracy_{g}.nii.gz")

print("✅ NEW ACCURACY MAPS SAVED.")