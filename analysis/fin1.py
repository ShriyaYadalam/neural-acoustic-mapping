"""
fMRI Data Preprocessing & Aggregation Pipeline
----------------------------------------------
- Denoising: Regresses out 9 standard fMRIPrep noise channels (6 motion, CSF, WM, global signal)
             with simultaneous temporal detrending and sample-wise z-scoring.
- Masking:   Applies an MNI152 structural brain mask to flatten 4D volumes into a 2D matrix (TRs x Voxels).
- Memory:    Uses on-disk memory mapping (np.memmap) and proactive garbage collection to safely 
             aggregate 12 Training and 6 Test runs without causing Out-of-Memory (OOM) crashes.
- Outputs:   Saves clean, consolidated arrays ('bold_all.npy', 'run_info.npy') ready for encoding models.
"""

import pandas as pd
import numpy as np
import gc
import os
from pathlib import Path
from nilearn import masking, image
 
SPACE = "space-MNI152NLin2009cAsym_res-2"
BASE_DIR = Path("../output/sub-001") 
FUNC_DIR = BASE_DIR / "func"

# Updated to include all 12 Training runs and all 6 Test runs
RUNS = [("Training", f"{i:02d}") for i in range(1, 13)] + \
       [("Test", f"{i:02d}") for i in range(1, 7)]

mask_path = BASE_DIR / "anat" / f"sub-001_{SPACE}_desc-brain_mask.nii.gz"
mask_img = image.load_img(str(mask_path))
n_voxels = int(np.sum(mask_img.get_fdata()))
total_trs = 0
valid_runs = []

print("Scanning runs to calculate total size...")
for task, run_id in RUNS:
    path = FUNC_DIR / f"sub-001_task-{task}_run-{run_id}_{SPACE}_desc-preproc_bold.nii.gz"
    if path.exists():
        total_trs += image.load_img(str(path)).shape[3]
        valid_runs.append((task, run_id))
    else:
        print(f"⚠️ Warning: {task} run {run_id} not found in {FUNC_DIR}. Skipping.")

print(f"Total TRs: {total_trs}, Voxels: {n_voxels}")

bold_all = np.memmap('bold_all.dat', dtype='float32', mode='w+', shape=(total_trs, n_voxels))

run_info = []
current_tr = 0

for task, run_id in valid_runs:
    bold_path = FUNC_DIR / f"sub-001_task-{task}_run-{run_id}_{SPACE}_desc-preproc_bold.nii.gz"
    conf_path = FUNC_DIR / f"sub-001_task-{task}_run-{run_id}_desc-confounds_timeseries.tsv"

    print(f"Denoising {task} run {run_id}...")
    confounds_df = pd.read_csv(conf_path, sep='\t')
    
    # Standard fMRIPrep confound columns
    cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "global_signal", "csf", "white_matter"]
    conf_vals = confounds_df[cols].fillna(0).values

    cleaned_img = image.clean_img(bold_path, confounds=conf_vals, detrend=True, standardize='zscore_sample')
    
    bold_data = masking.apply_mask(cleaned_img, mask_img).astype('float32')
    
    n_trs = bold_data.shape[0]
    bold_all[current_tr : current_tr + n_trs, :] = bold_data
    
    run_info.append({
        "task": task, "run": run_id, "n_trs": n_trs,
        "start_tr": current_tr, "end_tr": current_tr + n_trs
    })
    
    current_tr += n_trs
    
    # Force RAM cleanup to avoid "Out of Memory" errors on your LOQ
    del cleaned_img, bold_data
    gc.collect()

# Finalize: Convert the .dat file to a standard .npy file
print("Finalizing file...")
final_array = np.array(bold_all)
np.save("bold_all.npy", final_array)
np.save("run_info.npy", np.array(run_info, dtype=object))

# Clean up temporary .dat file
if os.path.exists('bold_all.dat'):
    os.remove('bold_all.dat')

print(f"✅ Step 1 Done. Total BOLD shape: {final_array.shape}")