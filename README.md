fin1.py: fMRI Denoising & Memory-Efficient Aggregation
Purpose
Cleans, masks, and consolidates 12 training and 6 testing 4D BOLD fMRI runs into a unified 2D NumPy array ($TRs \times Voxels$) ready for encoding analysis.
Key Features
•	Nuisance Regression: Uses nilearn to regress out 9 standard fMRIPrep confound channels (6 motion vectors, global signal, CSF, and white matter) while applying temporal detrending and z-score standardization.
•	Dimensionality Reduction: Applies a binary MNI152 structural brain mask to extract active neural voxels and discard non-brain signal elements.
•	OOM Protection: Pre-allocates an on-disk buffer via np.memmap and forces progressive loop-level garbage collection (gc.collect()), allowing massive multi-run datasets to be aggregated safely on consumer hardware without running out of RAM.
Outputs
•	bold_all.npy: The continuous, denoised 2D BOLD data array.
•	run_info.npy: A metadata tracking index mapping the precise starting and ending TR boundaries for each run.

______________________________________________________________________ 

fin2.py: Feature Extraction & Design Matrix Alignment
Purpose
Extracts multi-layered acoustic features from stimulus audio clips, aligns them to the absolute fMRI acquisition timeline, and constructs a delay-embedded design matrix for voxel-wise ridge regression encoding models.
Key Features
•	Temporal Event Registration: Syncs discrete audio track events (onsets/offsets) across run fragments using run_info.npy to create a continuous global TR axis.
•	Hierarchical Audio Profiling: Leverages librosa to break down audio clips into synchronous 1.5-second windows (1 TR). For each window, it extracts a 27-dimensional feature vector balancing timbral texture (13 MFCCs), harmonic/spectral identity (12 Chromas), global loudness (1 RMS), and brightness (1 Centroid).
•	Hemodynamic Response Embedding: Constructs 5 sequential temporal delays for every feature vector. This expanding step structure compensates for the human brain's natural blood-oxygen lag, allowing linear regression models to accurately map fast auditory transitions to late-surging BOLD fluctuations.
Outputs
•	events_all.csv: A unified spreadsheet mapping all stimulus tracks to their global fMRI scan timelines.
•	clip_features.npy: A localized lookup dictionary containing the structured ($10 \text{ TRs} \times 162 \text{ Features}$) design matrices for every unique audio stimulus.

______________________________________________________________________ 


fin3.py: Voxel-Wise Ridge Regression Encoding Model
Purpose
Executes a high-dimensional linear encoding analysis that maps acoustic features directly to BOLD neural responses, validating predictive performance and isolating sensory feature preferences across individual brain voxels.
Key Features
•	Acoustic-Neural Synchronization: Maps multi-delay auditory feature sets into a continuous global time tracking timeline, establishing a structured ($TRs \times 162\text{ Features}$) design matrix ($X$).
•	Z-Score Feature Normalization: Standardizes acoustic parameters to uniform variance scales, preventing baseline amplitude biases from disproportionately distorting regularized model coefficients.
•	Chunked L2 Ridge Processing: Iterates over thousands of brain voxels in strict data blocks of 5,000 to manage local host RAM profiles. Applies tuned regularized Ridge equations ($\alpha = 10.0$) to stabilize predictions against multi-collinear musical inputs.
•	Hierarchical Weight Partitioning: Dissects trained beta parameter matrices into separate functional weights representing core acoustic dimensions: Timbre (MFCCs), Harmony (Chroma), and Rhythm/Dynamics (RMS + Centroid).
Outputs

______________________________________________________________________ 

fin3_winner.py: 
Purpose
Generates an optimized, mathematically balanced 3D Winner-Take-All (WTA) categorical functional parcellation map (map_WINNERS_OPTIMIZED.nii.gz) that labels each cortical voxel with an integer corresponding to its dominant sound processing preference (1: Timbre, 2: Harmony, 3: Rhythm).
Key Features
•	Variance-Agnostic Regularization: Combines StandardScaler transformations with a sensitive low-alpha parameter (alpha=10.0) to avoid scaling bias and keep multi-run voxel coefficients un-smoothed.
•	Voxel-Wise L1 Normalization Architecture: Scales feature groups locally within each individual voxel boundary rather than globally across the whole dataset, mitigating outlier bias from hyper-responsive primary auditory nodes.
•	Competitive Label Assembly: Processes normalized weight structures through a competitive np.argmax selection matrix, isolating distinct functional clusters across superior temporal structural anatomy while filtering out uninformative noise using a predictive threshold ($r > 0.15$).
Outputs
•	map_WINNERS_OPTIMIZED.nii.gz: A single discrete indexed NIfTI atlas customized for rendering layered functional zoning maps within visualization toolkits like MRIcroGL.

•	map_correlation.nii.gz: A 3D spatial accuracy map scoring where the acoustic feature model successfully predicts localized blood-oxygen changes.

______________________________________________________________________

genre_wise.py: Genre-Specific Prediction Accuracy Mapping
Purpose
Evaluates how effectively an acoustically trained Master Encoding Model can cross-predict independent neural responses to specific musical genres, enabling the spatial mapping of distinct spectro-temporal properties across the auditory cortex.
Key Features
•	Unified Master Model Optimization: Corrects variance and penalty scaling issues by standardizing features and setting a high-sensitivity alpha=10.0. It fits a robust multi-genre coefficient baseline (all_weights) across the entire training set.
•	Sub-Genre Target Partitioning: Dynamically filters testing timelines to isolate individual genre presentations, bypassing monolithic categorical testing in favor of targeted feature-based decoding.
•	High-Speed Matrix-Dot Prediction: Leverages optimized dot-product arithmetic (X_test_genre @ all_weights.T) on chunked data blocks to generate rapid, memory-safe synthetic BOLD timeseries without overflowing system memory.
•	Raw Statistical Atlas Export: Outputs independent, continuous spatial accuracy maps for every genre found in the test run. It deliberately bypasses hard-coded internal thresholding filters to preserve raw statistical sensitivity for downstream visualization tools.
Outputs
•	map_accuracy_[genre].nii.gz: Individual 3D brain maps showing exactly where the acoustic model accurately tracks neural fluctuations for that specific genre (e.g., map_accuracy_hiphop.nii.gz, map_accuracy_classical.nii.gz).

______________________________________________________________________

finfin.py - 
Purpose
Parses raw 1D voxel encoding correlation outputs to calculate peak regional prediction accuracies, global performance medians, and transform peak localized coordinates into millimeter-precise standard MNI152 space for paper reporting.
Key Features
•	Auditory Spotlight Profiling: Extracts upper-tail distribution metrics (Mean of Top 100/500 voxels) to document regional encoding capacity while filtering out unrelated whole-brain voxel noise.
•	MNI Spatial Translation: Utilizes affine coordinate transforms (nilearn.image.coord_transform) to map localized voxel array coordinates into universal standard MNI space millimeters.
•	Automated Hemispheric Audit: Evaluates structural axis to mathematically determine functional tracking dominance across the left and right temporal planes.
Outputs
•	A clean terminal readout providing the specific numeric values (Max R, Top-Voxel Means, Median R, and MNI Coordinates) required to complete the Results and Conclusion blocks of your scientific Abstract.

______________________________________________________________________

fin4.py - Plotting
Purpose
Translates localized mathematical model outputs into high-resolution, separate 3D orthographic and interactive visuals to contrast localized auditory cortex specializations across distinct acoustic dimensions (Timbre, Harmony, and Rhythm).
Key Features
•	Independent Spatial Group Multi-Plotting: Loops over pre-masked functional regions to render separate, customized glass brain profiles (lyrz layout) for Timbre (Blues), Harmony (Reds), and Rhythm (Greens).
•	Publication-Grade Export: Automatically saves individual multi-planar figures directly into raw .png images configured at 300 DPI with tight layout margins, making them instantly ready for manuscript sections or presentation slides.
•	Interactive 3D Profiler: Compiles a standalone interactive HTML mesh (subject001_correlation_map.html) to allow live spatial rotation and real-time slice inspections of underlying core validation data.
Outputs
•	subject001_correlation_map.html: Interactive 3D spatial accuracy profile.
•	glass_brain_timbre.png: Standalone orthographic panel tracking timbral textures.
•	glass_brain_harmony.png: Standalone orthographic panel tracking pitch structures.
•	glass_brain_rhythm.png: Standalone orthographic panel tracking transient pulses.

______________________________________________________________________

fin4_winner.py –
Purpose
Visualizes the completed Winner-Take-All (WTA) categorical functional parcellation map, displaying the discrete spatial boundaries where different regions of the auditory cortex specialize in Timbre, Harmony, or Rhythm.
Key Features
•	Discrete Categorical Rendering: Implements flat, multi-angle glass brain profiles (lyrz layout) using a qualitative color scheme (cmap='Paired') to ensure integer labels (1, 2, and 3) stay sharp, solid, and structurally distinct.
•	Anatomical Boundary Auditing: Configures background thresholds to display exact functional borders across the lateral temporal lobe and Heschl’s Gyrus without over-smoothing regional tissue transitions.
•	MRIcroGL Render Optimization: Hardcodes precise layer display thresholds (Darkest: 0.5, Brightest: 3.5) to prepare the index file for rapid, multi-color qualitative rendering over structural T1w anatomy.
Outputs
•	Direct orthographic glass-brain figures separating the temporal cortex into distinct, colorful functional zones based on competitive acoustic processing profiles.

______________________________________________________________________

roi_analysis.py:  
Purpose Parses all genre-specific NIfTI correlation volumes to extract, cross-reference, and chart the average encoding predictive performance across structurally defined anatomical regions of the human auditory cortex.
Key Features * Automated Atlas Coregistration: Imports the Destrieux structural atlas and uses nilearn.image.resample_to_img to automatically match voxel grids with your functional data resolution.
•	Localized Signal Isolation: Extracts voxel data from 6 critical temporal regions (Heschl’s Gyrus, Planum Temporale, and Superior Temporal Gyrus across both hemispheres) and shields the math from inactive tissue noise by applying a baseline filter threshold ($r > 0.05$).
•	Comparative Presentation Rendering: Automates multi-categorical matrix grouping to generate a high-impact presentation figure (roi_accuracy_plot_FINAL.png), evaluating different genres across anatomical boundaries alongside a fixed $r = 0.15$ significance line.
Outputs * roi_accuracy_plot_FINAL.png: A high-resolution comparative bar chart plotting localized feature-encoding metrics across standard anatomical brain segments.

______________________________________________________________________
