"""
Acoustic Feature Extraction & Hemodynamic Alignment Pipeline
-----------------------------------------------------------
- Event Alignment: Maps local fMRI audio event onsets (seconds) to a continuous global timeline 
                   (in Repetition Times, TR = 1.5s) across all 12 Training and 6 Test runs.
- Multi-Feature Extraction: Slices corresponding raw WAV audio clips (GTZAN dataset) into exact 
                            1.5-second TR segments. Extracts 27 total acoustic features per segment:
                            * Timbre: Mel-Frequency Cepstral Coefficients (13 MFCCs)
                            * Harmony: Chromagram vectors representing 12 semitones
                            * Dynamics: Root-Mean-Square (RMS) Energy (1 feature)
                            * Spectral Brightness: Spectral Centroid (1 feature)
- Lag Compensation: Implements a 5-step shifting matrix (Temporal Delays) to expand the feature 
                    dimension to 162 columns per TR, preparing the design matrix to align with 
                    delayed blood-oxygen-level-dependent (BOLD) neural responses.
"""

import pandas as pd
import numpy as np
import librosa
import os
import gc
from pathlib import Path

# --- CONFIG ---
GTZAN_DIR = Path("../genres_original")
RAW_DIR = Path("../ds003720/sub-001/func") 
run_info = np.load("run_info.npy", allow_pickle=True)

TR = 1.5
SR = 22050
N_TRS_PER_CLIP = 10
N_DELAYS = 5 

# --- STEP 3: STACK EVENTS ---
print("Aligning event timings...")
all_events = []
for r in run_info:
    ev_path = RAW_DIR / f"sub-001_task-{r['task']}_run-{r['run']}_events.tsv"
    if not ev_path.exists():
        continue
    
    ev = pd.read_csv(ev_path, sep='\t')
    ev["genre"] = ev["genre"].str.replace("'", "").str.strip()
    ev["task"], ev["run"] = r["task"], r["run"]
    ev["global_onset_tr"] = (ev["onset"] / TR).astype(int) + r["start_tr"]
    all_events.append(ev)

events_all = pd.concat(all_events, ignore_index=True)
events_all.to_csv("events_all.csv", index=False)

# --- STEP 4: MULTI-FEATURE EXTRACTION ---
print("\nExtracting Hierarchical Music Features...")
clip_features = {}
unique_clips = events_all[["genre", "track", "start", "end"]].drop_duplicates()

for _, row in unique_clips.iterrows():
    genre, track = row["genre"], int(row["track"])
    s_start, s_end = round(row["start"], 1), round(row["end"], 1)
    
    g_folder = genre.replace('-', '')
    a_path = GTZAN_DIR / g_folder / f"{g_folder}.{track-1:05d}.wav"
    
    if not a_path.exists(): continue

    try:
        audio, _ = librosa.load(str(a_path), sr=SR, offset=s_start, duration=TR * N_TRS_PER_CLIP)
        
        feats = []
        samples_per_tr = int(TR * SR)
        for i in range(N_TRS_PER_CLIP):
            seg = audio[i*samples_per_tr : (i+1)*samples_per_tr]
            if len(seg) < samples_per_tr:
                seg = np.pad(seg, (0, samples_per_tr - len(seg)))
            
            # 1. TIMBRE: MFCC (13 features)
            mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=13).mean(axis=1)
            
            # 2. HARMONY: Chroma (12 features - one per semitone)
            chroma = librosa.feature.chroma_stft(y=seg, sr=SR).mean(axis=1)
            
            # 3. DYNAMICS/RHYTHM: RMS Energy (1 feature)
            rms = librosa.feature.rms(y=seg).mean(axis=1)
            
            # 4. SPECTRAL BRIGHTNESS: Centroid (1 feature)
            centroid = librosa.feature.spectral_centroid(y=seg, sr=SR).mean(axis=1)
            
            # Combine into a single vector for this TR (Total: 27 features)
            combined_tr = np.concatenate([mfcc, chroma, rms, centroid])
            feats.append(combined_tr)
        
        feats = np.array(feats) # Shape (10, 27)
        
        # Add Temporal Delays (Account for Hemodynamic Lag)
        delayed = [feats]
        for d in range(1, N_DELAYS + 1):
            shft = np.zeros_like(feats)
            shft[d:] = feats[:-d]
            delayed.append(shft)
        
        # Final Vector: 10 TRs x (27 features * 6 delays) = 162 total features per TR
        clip_features[(genre, track, s_start, s_end)] = np.hstack(delayed)
        
    except Exception as e:
        print(f"⚠️ Error in {a_path.name}: {e}")
        continue

np.save("clip_features.npy", clip_features)
print(f"✅ Extracted {len(clip_features)} feature sets (27 base features per TR).")