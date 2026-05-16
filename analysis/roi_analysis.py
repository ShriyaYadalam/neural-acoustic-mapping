"""
Anatomical Region of Interest (ROI) Analysis & Multi-Genre Benchmarking
----------------------------------------------------------------------
- Atlas Acquisition: Fetches and mounts the standard Destrieux 2009 cortical atlas 
                       to isolate precise coordinate label boundaries.
- Image Resampling: Standardizes structural spatial matrices using nearest-neighbor 
                     interpolation to match custom functional data resolution dimensions.
- Target Metric Masking: Extracts localized voxel values across 6 specific auditory regions, 
                          applying an r > 0.05 baseline mask to isolate active signal boundaries.
- Cross-Category Data Aggregation: Compiles, pivots, and structures high-dimensional categorical 
                                   metric dictionaries into clean dataframes for visualization.
- Vector Export: Render a high-resolution comparative bar plot (.png) featuring a research 
                 significance indicator threshold at r = 0.15.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from nilearn import datasets, image
from pathlib import Path

# --- 1. SETUP PATHS ---
file_list = glob.glob("map_accuracy_*.nii.gz")

if not file_list:
    print("❌ ERROR: No 'map_accuracy_*.nii.gz' files found.")
    exit()

# --- 2. FETCH ANATOMICAL ATLAS ---
print(f"📥 Found {len(file_list)} genre maps. Loading Destrieux Atlas...")
destrieux = datasets.fetch_atlas_destrieux_2009()
atlas_img = image.load_img(destrieux['maps'])

# Target IDs for Auditory Regions
auditory_rois = {
    'L-Heschl': 75, 'R-Heschl': 76,
    'L-STG': 73, 'R-STG': 74,
    'L-Planum': 65, 'R-Planum': 66
}

# --- 3. AUTOMATED EXTRACTION ---
results = []

# We resample the atlas once using the first genre map as a template
print("🔄 Aligning Atlas to your brain data resolution...")
template_img = image.load_img(file_list[0])
resampled_atlas = image.resample_to_img(atlas_img, template_img, interpolation='nearest')
atlas_data = resampled_atlas.get_fdata()

for map_path in file_list:
    current_genre = map_path.replace("map_accuracy_", "").replace(".nii.gz", "")
    print(f"📊 Analyzing ROI accuracy for: {current_genre}...")
    
    img_data = image.load_img(map_path).get_fdata()
    
    for name, label_id in auditory_rois.items():
        # Masking voxels
        roi_mask = (atlas_data == label_id)
        roi_values = img_data[roi_mask]
        
        # Filter for significant voxels (> 0.05) to show signal over noise
        significant_voxels = roi_values[roi_values > 0.05] 
        
        if len(significant_voxels) > 0:
            mean_r = np.mean(significant_voxels)
        else:
            mean_r = 0
            
        results.append({'Genre': current_genre, 'Region': name, 'r': mean_r})

# --- 4. DATA PROCESSING & PLOTTING ---
df = pd.DataFrame(results)
pivot_df = df.pivot(index='Region', columns='Genre', values='r')

# --- 5. THE PLOT ---
ax = pivot_df.plot(kind='bar', figsize=(15, 8), width=0.85, colormap='tab10')

plt.title('Thresholded Prediction Accuracy (r > 0.05) per Auditory ROI', fontsize=16, pad=20)
plt.ylabel('Mean Pearson Correlation (r)', fontsize=14)
plt.xlabel('Brain Region (ROI)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Significance line
plt.axhline(0.15, color='red', linestyle='--', linewidth=1.5, label='Research Significance (r=0.15)')

plt.legend(title='Musical Genres', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('roi_accuracy_plot_FINAL.png', dpi=300)
print("\n✅ SUCCESS! Graph saved as 'roi_accuracy_plot_FINAL.png'")
plt.show()