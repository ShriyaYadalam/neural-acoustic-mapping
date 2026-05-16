"""
Neuroimaging Parcellation Visualization: Discrete Categorical Mapping
---------------------------------------------------------------------
- Functional Atlas Projection: Imports the 3D indexed Winner-Take-All volume 
                               (map_WINNERS.nii.gz) containing discrete region tokens.
- Orthographic Glass Brain: Uses nilearn.plotting.plot_glass_brain ('lyrz' layout) 
                             to map discrete integer nodes transparently across brain hemispheres.
- Qualitative Color Boundary: Enforces a non-continuous color palette ('Paired') and hard thresholding 
                               (>0.5) to keep categorical boundary edges clean and un-blended.
- Anatomical Co-Registration: Standardizes structural background references to high-resolution 
                              T1w standard spaces for clean alignment cross-checks.
"""

import numpy as np
from nilearn import plotting, image
from pathlib import Path

# --- 1. CONFIG ---
print("🚀 Loading Winner-Take-All map...")
SPACE = "space-MNI152NLin2009cAsym_res-2"
BASE_DIR = Path("../output/sub-001")
t1_path = BASE_DIR / "anat" / f"sub-001_{SPACE}_desc-preproc_T1w.nii.gz"

# Load the winner map
winner_img = image.load_img("map_WINNERS.nii.gz")

# --- 2. GLASS BRAIN VIEW ---
print("🎨 Rendering Functional Parcellation...")
plotting.plot_glass_brain(
    winner_img, 
    display_mode='lyrz', 
    colorbar=True, 
    threshold=0.5, # Shows 1, 2, and 3
    title='Functional Specialization: Timbre (1), Harmony (2), Rhythm (3)',
    cmap='Paired', # This gives sharp, distinct colors
    plot_abs=False
)

# --- 3. INSTRUCTIONS FOR MRICROGL ---
print("\n--- 🖱️ MRICROGL STEPS FOR BEST VISUAL ---")
print("1. Open MRIcroGL and Load T1w background.")
print("2. Add 'map_WINNERS.nii.gz' as an overlay.")
print("3. Change 'Color Map' to 'Qualitative' or 'NIH'.")
print("4. Set Darkest: 0.5 | Brightest: 3.5")
print("5. Observe the boundaries between the 3 colors in the Temporal Lobe!")

plotting.show()