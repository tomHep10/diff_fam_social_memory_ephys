#!/usr/bin/env python
# coding: utf-8

# In[42]:


import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
print(sys.path)

# Add the full absolute path to the ephys_analysis repo
sys.path.append('/blue/npadillacoreano/t.heeps/rehouse_code/ephys_analysis')

from LFP.lfp_collection import LFPCollection


# In[43]:


# Path to your saved JSON
json_path = "/home/t.heeps/blue_npadillacoreano/npadillacoreano/share/rehouse_data/lfp_collections/lfp_collection.json"

# Load the collection
lfp_collection = LFPCollection.load_collection(json_path)

# Confirm successful load
print(f"✅ Loaded {len(lfp_collection.recordings)} recordings")
print("Brain regions:", list(lfp_collection.brain_region_dict.keys()))


# In[44]:


lfp_collection.preprocess()


# In[46]:


lfp_collection.calculate_coherence()


# In[47]:


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

# In[48]:


d0_44_coh = lfp_collection.recordings[0].coherence
print(d0_44_coh.shape)


# ## Coherence now made | Loading behaviors for analysis

# In[72]:


rec = lfp_collection.recordings[0]
print(rec.name)
print(rec.event_dict.keys())  # List of behavior types (e.g., 'sniffing', 'fighting', etc.)


# In[73]:


print(dir(rec))


# ### Creating time vector to allow us to put an event behavior mask over coherence

# In[77]:


def create_time_vector(rec):
    T = rec.coherence.shape[0]         # number of time windows
    step = rec.timestep                # coherence step size (in seconds)
    start = rec.first_timestamp        # time offset of the first window

    tvec = np.arange(T) * step + start
    return tvec


# In[78]:


tvec = create_time_vector(rec)
print("tvec shape:", tvec.shape)
print("Start:", tvec[0], "End:", tvec[-1])


# In[81]:


def make_event_mask(tvec, event_ranges):
    """
    Create a boolean mask over `tvec` for all event time ranges.
    """
    mask = np.zeros_like(tvec, dtype=bool)
    for start, stop in event_ranges:
        mask |= (tvec >= start) & (tvec <= stop)
    return mask


# In[82]:


for rec in lfp_collection.recordings:
    rec.tvec = create_time_vector(rec)
    print(rec.tvec)


# In[84]:


for rec in lfp_collection.recordings:
    rec.coherence = rec.coherence[make_event_mask(rec.tvec, rec.event_dict['sniffing object'])]


# In[86]:


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
    lfp_collection, region_from='mPFC', region_to='NAc', title="mPFC - NAc"
)
plot_coherence_between_regions(
    lfp_collection, region_from='mPFC', region_to='MD', title="mPFC - MD"
)
plot_coherence_between_regions(
    lfp_collection, region_from='mPFC', region_to='vHPC', title="mPFC → vHPC Coherence"
)
plot_coherence_between_regions(
    lfp_collection, region_from='mPFC', region_to='BLA', title="mPFC → BLA Coherence"
)


# In[ ]:


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

