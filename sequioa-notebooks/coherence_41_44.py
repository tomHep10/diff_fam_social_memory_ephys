#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import warnings
from tqdm import TqdmWarning
warnings.filterwarnings("ignore", category=TqdmWarning)


# Check current sys.path
print("Current sys.path:")
for p in sys.path:
    print("  ", p)

# === Add the ephys_analysis path ===
# This points to Thomas's shared repo where LFPCollection and coherence functions live
ephys_repo_path = "/blue/npadillacoreano/t.heeps/rehouse_code/ephys_analysis"

if ephys_repo_path not in sys.path:
    sys.path.append(ephys_repo_path)
    print(f"Added to sys.path: {ephys_repo_path}")
else:
    print(f"Path already exists in sys.path: {ephys_repo_path}")

# === Import LFPCollection from the LFP module ===
from LFP.lfp_collection import LFPCollection
print("Successfully imported LFPCollection.")


# In[7]:


# Correct absolute path
json_path = "/blue/npadillacoreano/sequioasmith/rehouse_code/lfp_collections/aligned_lfpcollection_all_subj_d0_d7.json"

# Load the collection
lfp_collection = LFPCollection.load_collection(json_path)

# Confirm successful load
print(f"Loaded {len(lfp_collection.recordings)} recordings")
print("Brain regions:", list(lfp_collection.brain_region_dict.keys()))


# In[8]:


lfp_collection.preprocess()


# In[9]:


lfp_collection.calculate_coherence()


# In[10]:


for rec in lfp_collection.recordings:
    print(f"Recording: {rec.name} (Subject: {rec.subject}) — Regions: {list(rec.brain_region_dict.keys())}")


# ### Shape of Coherence:
# coh.shape → (T, F, R, R):
# 
# T: time bins (based on window + step)
# 
# F: frequencies (based on multitaper settings)
# 
# R: brain regions (length of brain_region_dict)

# 
# This shape tells us what dimensions our coherence data has:
# 
# | Dimension | Meaning | Example in our data |
# |------------|----------|--------------------|
# | **T = 3600** | Number of time bins | The 30-minute recording was divided into small overlapping windows. Each window is 1 second long, and the analysis moves forward by 0.5 seconds each step. This means every second of the 30-minute recording is covered twice — once starting at the current time, and again starting 0.5 seconds later — giving **3600 small time segments**. |
# | **F = 500** | Number of frequency points | The analysis looks at 500 frequency values between about 0.5 Hz and 300 Hz. This gives a smooth curve for coherence across the frequency spectrum. |
# | **R × R = 5 × 5** | Brain region pairs | Coherence is measured between all pairs of regions. Since we have 5 regions (mPFC, MD, NAc, BLA, vHPC), the output is a 5×5 matrix showing coherence between every pair. The diagonal shows self-coherence (always 1.0). |
# 
# **In short:**  
# Each 30-minute recording is broken into 3600 overlapping 1-second chunks, and for each chunk, the coherence between every pair of 5 regions is computed across 500 frequencies.  
# 
# So the data at: coh[t, f, i, j]
# represents **how synchronized** region *i* and region *j* were at time *t* and frequency *f*.
# 

# In[13]:


d0_44_coh = lfp_collection.recordings[0].coherence
print(d0_44_coh.shape)


# In[12]:


print(list(lfp_collection.brain_region_dict.keys()))


# ## Coherence now made | Loading behaviors for analysis

# In[169]:


rec = lfp_collection.recordings[0]
print(rec.name)
print(rec.event_dict.keys())  # List of behavior types (e.g., 'sniffing', 'fighting', etc.)


# ### Creating time vector to allow us to put an event behavior mask over coherence

# ### We have one value per time window .shape[0], but we don't have a timestamp attached to thosse windows, we create it here with the timestep taking the time at the center of each window

# In[170]:


def create_time_vector(rec):
    T = rec.coherence.shape[0]
    step = rec.timestep  # this is usually 0.1 or 0.5s
    tvec = np.arange(T) * step
    return tvec


# In[171]:


tvec = create_time_vector(rec)
print("tvec shape:", tvec.shape)
print("Start:", tvec[0], "End:", tvec[-1])


# In[172]:


def make_event_mask(tvec, event_ranges):
    """
    Create a boolean mask over `tvec` for all event time ranges.
    """
    mask = np.zeros_like(tvec, dtype=bool)
    for start, stop in event_ranges:
        mask |= (tvec >= start) & (tvec <= stop)
    return mask


# In[173]:


for rec in lfp_collection.recordings:
    rec.tvec = create_time_vector(rec)
    print(rec.tvec)


# In[174]:


print(f"{rec.name}:")
print(f"  - first_timestamp: {rec.first_timestamp}")
print(f"  - timestep: {rec.timestep}")
print(f"  - coherence.shape[0] = {rec.coherence.shape[0]}")
print(f"  - tvec[0:5] = {tvec[:5]}")
print(f"  - tvec[-5:] = {tvec[-5:]}")


# In[229]:


print(rec.name)


# In[230]:


from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_event_coherence(
    lfp_collection, event='sniffing object', region_from='mPFC', region_to='NAc', 
    freq_limit=100, title=None, save_path=None
):
    curves = []

    for rec in lfp_collection.recordings:
        # Skip recordings without the event
        if event not in rec.event_dict:
            continue

        region_dict = rec.brain_region_dict
        if not all(region in region_dict for region in [region_from, region_to]):
            continue

        # mask
        mask = make_event_mask(rec.tvec, rec.event_dict[event])
        if not np.any(mask):
            continue

        # Filter coherence during event
        coh_event = rec.coherence[mask]  # (T_event, F, R, R)
        avg_coh = np.nanmean(coh_event, axis=0)  # (F, R, R)

        from_idx = region_dict[region_from]
        to_idx = region_dict[region_to]
        F, R, _ = avg_coh.shape
        if from_idx >= R or to_idx >= R:
            continue

        coh_curve = avg_coh[:, from_idx, to_idx]
        freqs = rec.frequencies
        freq_mask = freqs <= freq_limit
        freqs_plot = freqs[freq_mask]
        coh_plot = coh_curve[freq_mask]

        subj = rec.subject if hasattr(rec, "subject") else rec.name.split('_')[0]
        day = 'd0' if 'd0' in rec.name else 'd7'  # optionally improve this too

        # Detect day and familiarity
        if 'd0' in rec.name.lower():
            fam = 'low fam'
            fam_short = 'low fam'
        elif 'd7' in rec.name.lower():
            fam = 'high fam'
            fam_short = 'high fam'
        else:
            fam = 'unknown fam'
            fam_short = 'unknown'

        label = f"subj {subj} - {fam_short}"
        curves.append((label, freqs_plot, coh_plot, subj, fam, rec.name))


    if not curves:
        print(f"⚠️ No valid data to plot for event '{event}' and region {region_from} - {region_to}")
        return

    subjects = sorted(set(subj for _, _, _, subj, _, _ in curves))
    cmap = plt.get_cmap('tab10', len(subjects))
    # or
    # cmap = matplotlib.colormaps['tab10']

    subject_colors = {subj: cmap(i) for i, subj in enumerate(subjects)}

    plt.figure(figsize=(10, 6))
    for label, freqs, coh, subj, fam, rec_name in curves:
        if fam!='high fam' and fam!='low fam':
            print("error in fam")
            break
        linestyle = '--' if fam == 'high fam' else '-'
        color = subject_colors[subj]
        plt.plot(freqs, coh, label=label, linewidth=1.5, linestyle=linestyle, color=color)

        print(f"➡️  {rec_name}: Subject={subj}, Day={fam}, Freq Shape={freqs.shape}, Coherence Mean={np.nanmean(coh):.3f}")




    plt.title(title if title else f"{region_from} - {region_to} Coherence during '{event}'", fontsize=20)
    plt.xlabel("Frequency (Hz)", fontsize=16)
    plt.ylabel("Coherence", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(False)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Save or show
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{region_from}_to_{region_to}_{event.replace(' ', '_')}.png"
        full_path = save_dir / filename
        plt.savefig(full_path)
        plt.close()
        print(f"✅ Saved: {full_path}")
    else:
        plt.show()


# In[231]:


plot_event_coherence(
    lfp_collection,
    event='facial sniffing',
    region_from='mPFC',
    region_to='NAc',
    title="mPFC - NAc during 'facial sniffing'",
    save_path="/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/"
)

plot_event_coherence(
    lfp_collection,
    event='facial sniffing',
    region_from='mPFC',
    region_to='MD',
    title="mPFC - MD during 'facial sniffing'",
    save_path=r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/'
)


# In[217]:


plot_event_coherence(
    lfp_collection,
    event='anogenital sniffing',
    region_from='mPFC',
    region_to='NAc',
    title="mPFC - NAc during 'anogenital sniffing'",
    save_path="/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/"
)

plot_event_coherence(
    lfp_collection,
    event='anogenital sniffing',
    region_from='mPFC',
    region_to='MD',
    title="mPFC - MD during 'anogenital sniffing'",
    save_path=r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/'
)


# In[218]:


plot_event_coherence(
    lfp_collection,
    event='chasing',
    region_from='mPFC',
    region_to='NAc',
    title="mPFC - NAc during 'fighting'",
    # save_path="/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/"
)

plot_event_coherence(
    lfp_collection,
    event='chasing',
    region_from='mPFC',
    region_to='MD',
    title="mPFC - MD during 'fighting'",
    # save_path=r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/'
)


# In[219]:


plot_event_coherence(
    lfp_collection,
    event='fighting',
    region_from='mPFC',
    region_to='NAc',
    title="mPFC - NAc during 'fighting'",
    # save_path="/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/"
)

plot_event_coherence(
    lfp_collection,
    event='fighting',
    region_from='mPFC',
    region_to='MD',
    title="mPFC - MD during 'fighting'",
    # save_path=r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/'
)


# In[220]:


plot_event_coherence(
    lfp_collection,
    event='sniffing object',
    region_from='mPFC',
    region_to='NAc',
    title="mPFC - NAc during 'sniffing object'",
    save_path="/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/"
)

plot_event_coherence(
    lfp_collection,
    event='sniffing object',
    region_from='mPFC',
    region_to='MD',
    title="mPFC - MD during 'sniffing object'",
    save_path=r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/'
)


# In[251]:


def plot_event_coherence(
    lfp_collection, events='sniffing object', region_from='mPFC', region_to='NAc', 
    freq_limit=100, title=None, save_path=None
):
    if isinstance(events, str):
        events = [events]  # convert to list if single string

    curves = []

    for rec in lfp_collection.recordings:
        region_dict = rec.brain_region_dict
        if not all(region in region_dict for region in [region_from, region_to]):
            continue

        # Combine masks for all requested events
        tvec = create_time_vector(rec)
        combined_mask = np.zeros_like(tvec, dtype=bool)
        valid = False

        for event in events:
            if event in rec.event_dict:
                mask = make_event_mask(tvec, rec.event_dict[event])
                combined_mask |= mask  # logical OR
                valid = True

        if not valid or not np.any(combined_mask):
            continue

        coh_event = rec.coherence[combined_mask]  # (T_event, F, R, R)
        avg_coh = np.nanmean(coh_event, axis=0)  # (F, R, R)

        from_idx = region_dict[region_from]
        to_idx = region_dict[region_to]
        F, R, _ = avg_coh.shape
        if from_idx >= R or to_idx >= R:
            continue

        coh_curve = avg_coh[:, from_idx, to_idx]
        freqs = rec.frequencies
        freq_mask = freqs <= freq_limit
        freqs_plot = freqs[freq_mask]
        coh_plot = coh_curve[freq_mask]

        subj = rec.subject if hasattr(rec, "subject") else rec.name.split('_')[0]

        if 'd0' in rec.name.lower():
            fam = 'low fam'
        elif 'd7' in rec.name.lower():
            fam = 'high fam'
        else:
            fam = 'unknown fam'

        label = f"subj {subj} - {fam}"
        curves.append((label, freqs_plot, coh_plot, subj, fam, rec.name))

    if not curves:
        print(f"⚠️ No valid data to plot for events {events} and region {region_from} - {region_to}")
        return

    # Define hardcoded RGB colors (0–1 scale)
    low_fam_color = (1/255, 138/255, 126/255)   # d0
    high_fam_color = (100/255, 5/255, 49/255)   # d7

    plt.figure(figsize=(5.5, 5.5))
    for label, freqs, coh, subj, fam, rec_name in curves:
        if fam == 'low fam':
            color = low_fam_color
            linestyle = '-'
        elif fam == 'high fam':
            color = high_fam_color
            linestyle = '--'
        else:
            color = 'gray'
            linestyle = ':'

        plt.plot(freqs, coh, label=label, linewidth=2.5, linestyle=linestyle, color=color)
        print(f"➡️  {rec_name}: Subject={subj}, Day={fam}, Freq Shape={freqs.shape}, Coherence Mean={np.nanmean(coh):.3f}")


    event_str = ', '.join(events)
    plt.title(title if title else f"{region_from} - {region_to} Coherence during {event_str}", fontsize=18)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Coherence", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(False)
    plt.ylim(0.35, 1)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=14)

    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{region_from}_to_{region_to}_{'_'.join(e.replace(' ', '_') for e in events)}.png"
        full_path = save_dir / filename
        plt.savefig(full_path)
        plt.close()
        print(f"✅ Saved: {full_path}")
    else:
        plt.show()


# In[252]:


plot_event_coherence(
    lfp_collection,
    events=['facial sniffing', 'anogenital sniffing'],
    region_from='mPFC',
    region_to='NAc',
    title="mPFC - NAc during Social Investigation",
    save_path=r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/'

)
plot_event_coherence(
    lfp_collection,
    events=['facial sniffing', 'anogenital sniffing'],
    region_from='mPFC',
    region_to='MD',
    title="mPFC - MD during Social Investigation",
    save_path=r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/'

)


# In[260]:


def plot_event_coherence_avg_fam(
    lfp_collection, events='sniffing object', region_from='mPFC', region_to='NAc', 
    freq_limit=100, title=None, save_path=None
):
    if isinstance(events, str):
        events = [events]

    # Gather all coherence curves for each fam across subjects
    subj_data = defaultdict(list)
    freqs_plot = None  # For use in plotting

    for rec in lfp_collection.recordings:
        region_dict = rec.brain_region_dict
        if not all(region in region_dict for region in [region_from, region_to]):
            continue

        tvec = create_time_vector(rec)
        combined_mask = np.zeros_like(tvec, dtype=bool)
        valid = False

        for event in events:
            if event in rec.event_dict:
                mask = make_event_mask(tvec, rec.event_dict[event])
                combined_mask |= mask
                valid = True

        if not valid or not np.any(combined_mask):
            continue

        coh_event = rec.coherence[combined_mask]  # (T_event, F, R, R)
        avg_coh = np.nanmean(coh_event, axis=0)   # (F, R, R)

        from_idx = region_dict[region_from]
        to_idx = region_dict[region_to]
        F, R, _ = avg_coh.shape
        if from_idx >= R or to_idx >= R:
            continue

        coh_curve = avg_coh[:, from_idx, to_idx]
        freqs = rec.frequencies
        freq_mask = freqs <= freq_limit
        freqs_plot = freqs[freq_mask]
        coh_plot = coh_curve[freq_mask]

        # Figure out familiarity condition
        if 'd0' in rec.name.lower():
            fam = 'low fam'
        elif 'd7' in rec.name.lower():
            fam = 'high fam'
        else:
            fam = 'unknown fam'

        subj_data[fam].append(coh_plot)

    if not subj_data:
        print(f"⚠️ No valid data to plot for events {events} and region {region_from} - {region_to}")
        return

    # --- Define colors ---
    low_fam_color = (1/255, 138/255, 126/255)
    high_fam_color = (100/255, 5/255, 49/255)

    plt.figure(figsize=(5.5, 5.5))

    # --- Plot two lines: avg low fam, avg high fam ---
    for fam, curves in subj_data.items():
        coh_array = np.stack(curves, axis=0)  # (N_total_rec, F)
        mean_coh = np.nanmean(coh_array, axis=0)

        if fam == 'low fam':
            color = low_fam_color
            linestyle = '-'
        elif fam == 'high fam':
            color = high_fam_color
            linestyle = '-'
        else:
            color = 'gray'
            linestyle = ':'

        label = f"{fam}"
        plt.plot(freqs_plot, mean_coh, label=label, linewidth=2.5, linestyle=linestyle, color=color)
        print(f"📊 Averaged {fam}: {coh_array.shape[0]} recordings, Mean coherence = {np.nanmean(mean_coh):.3f}")

    # --- Pretty up plot ---
    event_str = ', '.join(events)
    plt.title(title if title else f"{region_from} - {region_to} Coherence during {event_str}", fontsize=18)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Coherence", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(False)
    plt.ylim(0.35, 1)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=14)

    # --- Save or show ---
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{region_from}_to_{region_to}_avgfam_{'_'.join(e.replace(' ', '_') for e in events)}.png"
        full_path = save_dir / filename
        plt.savefig(full_path)
        plt.close()
        print(f"✅ Saved: {full_path}")
    else:
        plt.show()


# In[262]:


plot_event_coherence_avg_fam(
    lfp_collection,
    events=['facial sniffing', 'anogenital sniffing'],
    region_from='mPFC',
    region_to='NAc',
    title="mPFC - NAc during Social Investigation",
    save_path=r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/'

)
plot_event_coherence_avg_fam(
    lfp_collection,
    events=['facial sniffing', 'anogenital sniffing'],
    region_from='mPFC',
    region_to='MD',
    title="mPFC - MD during Social Investigation",
    save_path=r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/'

)


# In[103]:


get_ipython().system('jupyter nbconvert --to script coherence_41_44.ipynb')


# ### Directionality plots for ***mpfc*** -> ***nac*** and ***mpfc*** -> ***md***

# In[50]:


def coh_plot(rec, freqs, mpfc_to_nac, mpfc_to_md, save_dir=None):
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, mpfc_to_nac, label='mPFC → NAc', color='blue')
    plt.plot(freqs, mpfc_to_md, label='mPFC → MD', color='green')
    plt.title(f"Average Coherence Directionality — {rec.name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir is not None:
        # Safe file name
        fname = f"coh_{rec.name.replace('.rec','')}.png"
        plt.savefig(os.path.join(save_dir, fname))
        plt.close()
    else:
        plt.show()


# In[51]:


# Pick a recording to visualize
d0_44 = lfp_collection.recordings[0]


# In[52]:


# Get region indices
region_dict = d0_44.brain_region_dict
mpfc_idx = region_dict['mPFC']
nac_idx = region_dict['NAc']
md_idx = region_dict['MD']


# ### Get Average, spectra for each direction, slice freqs 

# In[53]:


# Average over time (T axis)
avg_coh = np.nanmean(d0_44.coherence, axis=0)  # shape: (F, R, R)

# Get coherence spectra for each direction
mpfc_to_nac = avg_coh[:, mpfc_idx, nac_idx]
mpfc_to_md = avg_coh[:, mpfc_idx, md_idx]

# Slice to 0–100 Hz only
freqs = d0_44.frequencies  # full frequency vector, e.g., 0–500 Hz
freq_mask = freqs <= 100
freqs = freqs[freq_mask]
mpfc_to_nac = mpfc_to_nac[freq_mask]
mpfc_to_md = mpfc_to_md[freq_mask]


# In[54]:


print("avg_coh shape:", avg_coh.shape)
print("brain regions in recording:", rec.brain_region_dict)
print("mpfc_idx:", mpfc_idx, "nac_idx:", nac_idx, "bla_idx:", rec.brain_region_dict.get("BLA", "not found"))


# In[55]:


# Plot
coh_plot(d0_44, freqs, mpfc_to_nac, mpfc_to_md)


# ### Plots of 4.1 and 4.4 coherence directionality mpfc -> nac and mpfc -> MD

# In[56]:


for rec in lfp_collection.recordings:
    print(f"Processing {rec.name}")
    region_dict = rec.brain_region_dict

    # Skip if any region is missing (shouldn't happen with your recordings)
    if not all(region in region_dict for region in ['mPFC', 'NAc', 'MD']):
        print(f"Skipping {rec.name}: required regions not found.")
        continue

    mpfc_idx = region_dict['mPFC']
    nac_idx = region_dict['NAc']
    md_idx = region_dict['MD']

    avg_coh = np.nanmean(rec.coherence, axis=0)  # (F, R, R)
    mpfc_to_nac = avg_coh[:, mpfc_idx, nac_idx]
    mpfc_to_md = avg_coh[:, mpfc_idx, md_idx]

    # Frequency restriction
    freqs = rec.frequencies
    freq_mask = freqs <= 100
    freqs_plot = freqs[freq_mask]
    mpfc_to_nac_plot = mpfc_to_nac[freq_mask]
    mpfc_to_md_plot = mpfc_to_md[freq_mask]

    # Plot (and/or save)
    coh_plot(
        rec, 
        freqs_plot, 
        mpfc_to_nac_plot, 
        mpfc_to_md_plot,
        save_dir='/home/t.heeps/blue_npadillacoreano/rehouse_code/coherence_plots/plots',
    )


# In[57]:


# Collect all mPFC-NAc curves for each (subject, day)
curves = []  # Each item: (label, freqs, coherence)

for rec in lfp_collection.recordings:
    region_dict = rec.brain_region_dict
    if not all(region in region_dict for region in ['mPFC', 'NAc']):
        continue

    # Subject and day — adjust as needed based on your actual attributes
    subj = rec.subject if hasattr(rec, "subject") else rec.name.split('_')[0]
    day = 'd0' if 'd0' in rec.name else 'd7' if 'd7' in rec.name else 'UNK'

    mpfc_idx = region_dict['mPFC']
    nac_idx = region_dict['NAc']
    avg_coh = np.nanmean(rec.coherence, axis=0)  # (F, R, R)
    mpfc_to_nac = avg_coh[:, mpfc_idx, nac_idx]

    freqs = rec.frequencies
    freq_mask = freqs <= 100
    freqs_plot = freqs[freq_mask]
    mpfc_to_nac_plot = mpfc_to_nac[freq_mask]

    label = f"subj {subj}, {day}"
    curves.append((label, freqs_plot, mpfc_to_nac_plot))

# Plot all on one figure
plt.figure(figsize=(10, 6))
for label, freqs, coh in curves:
    plt.plot(freqs, coh, label=label)
plt.title("mPFC → NAc Coherence — All Subjects and Days")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Coherence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Optionally save:
# plt.savefig('mpfc_nac_all_subjects_days.png')
# plt.close()


# In[58]:


for rec in lfp_collection.recordings:
    print(f"Recording: {rec.name} (Subject: {rec.subject}) — Regions: {list(rec.brain_region_dict.keys())}")


# In[ ]:


print("🔍 Verifying brain region integrity for each recording...\n")
for rec in lfp_collection.recordings:
    region_dict = rec.brain_region_dict
    region_names = list(region_dict.keys())

    coh_shape = rec.coherence.shape  # (T, F, R, R)
    T, F, R, _ = coh_shape

    # Reverse mapping: index to region (e.g., 0 → 'mPFC')
    reverse_map = {v: k for k, v in region_dict.items()}
    region_list_from_indices = [reverse_map.get(i, 'MISSING') for i in range(R)]

    print(f"📁 {rec.name} (Subject: {rec.subject})")
    print(f"  - Regions in dict:      {region_names}")
    print(f"  - Coherence shape:      {coh_shape}")
    print(f"  - Region count (R):     {R}")
    print(f"  - Regions by index map: {region_list_from_indices}")
    print(f"  - Missing regions:      {[r for r in ['mPFC','NAc','MD','vHPC','BLA'] if r not in region_names]}")
    print()


# In[65]:


for recs in lfp_collection.recordings:
    print(f"{recs.name}, {recs.event_dict.keys()}")


# ### Plotting d0 and d7 rec together comparing coherence difference

# In[32]:


from pathlib import Path

def plot_coherence_between_regions(
    lfp_collection, region_from='mPFC', region_to='NAc', 
    freq_limit=100, title=None, save_path=None
):
    curves = []
    for rec in lfp_collection.recordings:
        region_dict = rec.brain_region_dict

        # Skip if either region missing in dictionary
        if not all(region in region_dict for region in [region_from, region_to]):
            continue

        from_idx = region_dict[region_from]
        to_idx = region_dict[region_to]

        avg_coh = np.nanmean(rec.coherence, axis=0)  # (F, R, R)
        _, R, _ = avg_coh.shape
        if from_idx >= R or to_idx >= R:
            print(f"⚠️ Skipping {rec.name}: index ({from_idx}, {to_idx}) out of bounds for R={R}")
            continue

        subj = rec.subject if hasattr(rec, "subject") else rec.name.split('_')[0]
        day = 'd0' if 'd0' in rec.name else 'd7' if 'd7' in rec.name else 'UNK'

        coh_curve = avg_coh[:, from_idx, to_idx]

        freqs = rec.frequencies
        freq_mask = freqs <= freq_limit
        freqs_plot = freqs[freq_mask]
        coh_plot = coh_curve[freq_mask]

        label = f"subj {subj}, {day}"
        curves.append((label, freqs_plot, coh_plot))

    if not curves:
        print(f"⚠️ No valid data to plot for {region_from} - {region_to}")
        return

    # Plot
    plt.figure(figsize=(10, 6))
    for label, freqs, coh in curves:
        plt.plot(freqs, coh, label=label)
    plt.title(title if title else f"{region_from} - {region_to} Coherence — All Subjects and Days")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to file
    if save_path:
        # Ensure directory exists
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{region_from}_to_{region_to}.png"
        full_path = save_dir / filename
        plt.savefig(full_path)
        plt.close()
        print(f"✅ Saved: {full_path}")
    else:
        plt.show()


# Example usage
plot_coherence_between_regions(
    lfp_collection, region_from='mPFC', region_to='NAc', save_path=r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/'
)
plot_coherence_between_regions(
    lfp_collection, region_from='mPFC', region_to='MD', save_path=r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/'
)
plot_coherence_between_regions(
    lfp_collection, region_from='mPFC', region_to='vHPC', title="mPFC → vHPC Coherence"
)
plot_coherence_between_regions(
    lfp_collection, region_from='mPFC', region_to='BLA', title="mPFC → BLA Coherence"
)


# In[40]:


from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict

def plot_coherence_between_regions(
    lfp_collection, region_from='mPFC', region_to='NAc', 
    freq_limit=100, title=None, save_path=None
):
    curves = []
    for rec in lfp_collection.recordings:
        region_dict = rec.brain_region_dict

        # Skip if either region missing in dictionary
        if not all(region in region_dict for region in [region_from, region_to]):
            continue

        from_idx = region_dict[region_from]
        to_idx = region_dict[region_to]

        avg_coh = np.nanmean(rec.coherence, axis=0)  # (F, R, R)
        _, R, _ = avg_coh.shape
        if from_idx >= R or to_idx >= R:
            print(f"⚠️ Skipping {rec.name}: index ({from_idx}, {to_idx}) out of bounds for R={R}")
            continue

        subj = rec.subject if hasattr(rec, "subject") else rec.name.split('_')[0]
        day = 'd0' if 'd0' in rec.name else 'd7' if 'd7' in rec.name else 'UNK'

        coh_curve = avg_coh[:, from_idx, to_idx]

        freqs = rec.frequencies
        freq_mask = freqs <= freq_limit
        freqs_plot = freqs[freq_mask]
        coh_plot = coh_curve[freq_mask]

        label = f"subj {subj}, {day}"
        curves.append((label, freqs_plot, coh_plot))

    if not curves:
        print(f"⚠️ No valid data to plot for {region_from} - {region_to}")
        return


    # Extract unique subject IDs
    subjects = sorted(set(label.split(',')[0].split()[-1] for label, _, _ in curves))
    
    # Generate color map
    cmap = cm.get_cmap('tab10', len(subjects))  # or 'Set1', 'tab20', etc.
    subject_colors = {subj: cmap(i) for i, subj in enumerate(subjects)}

    plt.figure(figsize=(10, 6))

    for label, freqs, coh in curves:
        subj = label.split(',')[0].split()[-1]
        day = label.split(',')[1].strip()

        linestyle = '--' if day == 'd7' else '-'
        color = subject_colors[subj]

        plt.plot(freqs, coh, label=label, linewidth=1.1, linestyle=linestyle, color=color)

    plt.title(title if title else f"{region_from} - {region_to} Coherence — All Subjects and Days")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{region_from}_to_{region_to}_color_same.png"
        full_path = save_dir / filename
        plt.savefig(full_path)
        plt.close()
        print(f"✅ Saved: {full_path}")
    else:
        plt.show()


# Example usage
plot_coherence_between_regions(
    lfp_collection, region_from='mPFC', region_to='NAc', save_path=r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/'
)
plot_coherence_between_regions(
    lfp_collection, region_from='mPFC', region_to='MD', save_path=r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/npadillacoreano/t.heeps/rehouse_code/coherence_plots/plots/'
)
plot_coherence_between_regions(
    lfp_collection, region_from='mPFC', region_to='vHPC', title="mPFC → vHPC Coherence"
)


# ### Filtering coherence data to only include coherence during the tone

# In[22]:


get_ipython().system('jupyter nbconvert --to script coherence_41_44.ipynb')


# In[54]:


print(dir(d0_44))


# In[55]:


d0_44.event_dict


# In[48]:


get_ipython().system('jupyter nbconvert --to script coherence_41_44.ipynb')

