#!/usr/bin/env python
# coding: utf-8

# In[255]:


import sys, importlib, os, inspect

NEW_ROOT = "/blue_npadillacoreano/rehouse_code/diff_fam_social_memory_ephys/thomas-social-memory"
OLD_ROOT = "/blue/npadillacoreano/t.heeps/rehouse_code/ephys_analysis"

os.chdir('/blue/npadillacoreano/t.heeps/rehouse_code/diff_fam_social_memory_ephys') # seems to be the only real fix

import importlib
import pickle

import behavior.boris_extraction as boris
from spike.spike_analysis.spike_collection import SpikeCollection
import trodes.read_exported as tr

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path


print("behavior package from:", boris.__file__)
print("boris_extraction from:", inspect.getfile(boris))
print("trodes read_exported from: ", inspect.getfile(tr))

rec_subj_dict_path = "/home/t.heeps/blue_npadillacoreano/npadillacoreano/share/rehouse_data/d0_d7_rec_subj_dict"
event_dict_path = "/home/t.heeps/blue_npadillacoreano/npadillacoreano/share/rehouse_data/allsubjs_dict_d0_d7.pkl"
phy_recs = "/home/t.heeps/blue_npadillacoreano/npadillacoreano/share/rehouse_data/data_rehouse/phy_recs"


# In[256]:


import spike.spike_analysis.spike_recording as sr
import spike.spike_analysis.firing_rate_calculations as fr
import spike.spike_analysis.normalization as norm
import spike.spike_analysis.single_cell as single_cell
import behavior.behavioral_epoch_tools as betools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import behavior.boris_extraction as boris
import matplotlib.pyplot as plt
import pickle
import re
import seaborn as sns

# Set global seaborn style
sns.set_context("notebook", font_scale=1.9)

# Define color palette for Day 0 (Teal) and Day 7 (Magenta)
COLOR_DAY0 = '#53A1A9'  # Teal
COLOR_DAY7 = '#A95376'  # Magenta


# In[257]:


pd.set_option('display.max_colwidth', 0)  # 0 means unlimited in newer pandas versions

# Show all rows
pd.set_option("display.max_rows", None)

# Show all columns
pd.set_option("display.max_columns", None)

# Don’t truncate column contents
pd.set_option("display.max_colwidth", None)

# Expand the display to the full width of the screen
pd.set_option("display.width", 0)


# In[258]:


try:
    with open(rec_subj_dict_path, 'rb') as file:
        rec_subj_dict = pickle.load(file)
            
except FileNotFoundError:
    print("filenotfound")
rec_subj_dict


# In[259]:


try:
    with open(event_dict_path, 'rb') as file:
        event_dict = pickle.load(file)
            
except FileNotFoundError:
    print("filenotfound")
print(event_dict.keys())


# In[260]:


event_dict['22_rehouse_d0_merged.rec'].keys()


# In[261]:


for rec_name, ev_dict in event_dict.items():
    for ev_key, arr in ev_dict.items():
        if not isinstance(arr, np.ndarray):
            print(rec_name, ev_key, type(arr))
        elif arr.ndim != 2 or arr.shape[1] != 2:
            print(rec_name, ev_key, arr.shape)


# In[262]:


import os

phy_recs = "/blue/npadillacoreano/t.heeps/npadillacoreano/share/rehouse_data/phy_recs"
for root, dirs, files in os.walk(phy_recs):
    print("ROOT:", root)
    print("DIRS:", dirs)
    break


# In[263]:


for event in event_dict['22_rehouse_d0_merged.rec']:
    print(event)


# # Replacing event dict with combined social sniffing and object sniffing by familiarity

# In[264]:


# --- Loop through recordings ---
for rec_name, events in event_dict.items():
    # Check if relevant events exist
    has_facial = "facial sniffing" in events
    has_ano = "anogenital sniffing" in events

    # Combine the two if both (or even one) exist
    if has_facial or has_ano:
        combined_sniffing = []
        if has_facial:
            combined_sniffing.extend(events["facial sniffing"])
        if has_ano:
            combined_sniffing.extend(events["anogenital sniffing"])

        # Sort by time if your events are time intervals [(start, stop), ...]
        combined_sniffing = sorted(combined_sniffing, key=lambda x: x[0])

        # --- 🟩 Here's the key part ---
        # Unified event names so PCA sees the same structure in all recordings
        event_dict[rec_name] = {
            "social sniffing": np.array(combined_sniffing),
            "object sniffing": events.get("sniffing object"),
        }

        # --- 🟦 Add a label field for later plotting ---
        # (You can also store this in a parallel dict if you prefer)
        if "d0" in rec_name:
            event_dict[rec_name]["label"] = "novel"
        elif "d7" in rec_name:
            event_dict[rec_name]["label"] = "fam"


# In[265]:


sc = SpikeCollection(phy_recs, event_dict, rec_subj_dict)


# In[229]:


sc.analyze(timebin=100, ignore_freq=0.5, smoothing_window = 500)


# In[230]:


# ============================================================================
# FILTER RECORDINGS: Keep only subjects with BOTH d0 AND d7
# ============================================================================
print("="*80)
print("FILTERING RECORDINGS TO ENSURE PAIRED d0/d7 DATA")
print("="*80)

# Extract subject IDs and days from all recordings
subjects_by_day = {'d0': set(), 'd7': set()}

for rec in sc.recordings:
    rec_name = rec.name
    # Extract subject number (e.g., "22" from "22_rehouse_d0_merged.rec")
    subject_match = re.match(r'(\d+)_', rec_name)
    if subject_match:
        subject_id = subject_match.group(1)
        
        # Determine day
        if '_d0_' in rec_name or rec_name.endswith('_d0.rec'):
            subjects_by_day['d0'].add(subject_id)
        elif '_d7_' in rec_name or rec_name.endswith('_d7.rec'):
            subjects_by_day['d7'].add(subject_id)

print(f"\nSubjects with d0 recordings: {sorted(subjects_by_day['d0'])}")
print(f"Subjects with d7 recordings: {sorted(subjects_by_day['d7'])}")

# Find subjects that have BOTH d0 and d7
subjects_with_both = subjects_by_day['d0'] & subjects_by_day['d7']
subjects_missing_d0 = subjects_by_day['d7'] - subjects_by_day['d0']
subjects_missing_d7 = subjects_by_day['d0'] - subjects_by_day['d7']

print(f"\n✓ Subjects with BOTH d0 and d7: {sorted(subjects_with_both)}")
if subjects_missing_d0:
    print(f"⚠️ Subjects missing d0 (will be EXCLUDED): {sorted(subjects_missing_d0)}")
if subjects_missing_d7:
    print(f"⚠️ Subjects missing d7 (will be EXCLUDED): {sorted(subjects_missing_d7)}")

# Filter recordings to keep only those from subjects with both days
original_count = len(sc.recordings)
filtered_recordings = []
excluded_recordings = []

for rec in sc.recordings:
    rec_name = rec.name
    subject_match = re.match(r'(\d+)_', rec_name)
    if subject_match:
        subject_id = subject_match.group(1)
        if subject_id in subjects_with_both:
            filtered_recordings.append(rec)
        else:
            excluded_recordings.append(rec_name)
    else:
        # Keep recordings without clear subject ID pattern (shouldn't happen)
        filtered_recordings.append(rec)

# Update the SpikeCollection with filtered recordings
sc.recordings = filtered_recordings

if excluded_recordings:
    print(f"\nExcluded recordings:")
    for rec_name in excluded_recordings:
        print(f"  - {rec_name}")

print(f"\n✓ Analysis will proceed with {len(subjects_with_both)} subjects (paired d0/d7)")
print("="*80)


# In[231]:


sc.recordings


# In[232]:


all_durations = []
for rec in sc.recordings:
    for ev, arr in rec.event_dict.items():
        if ev == "social sniffing":  # or whichever
            all_durations.extend(arr[:,1] - arr[:,0])
plt.hist(np.array(all_durations)/1000, bins=30)
plt.xticks(np.arange(0, max(all_durations)/1000, 2))
plt.xlabel("Bout duration (s)")
plt.ylabel("Count")


# In[233]:


all_durations = []
for rec in sc.recordings:
    for ev, arr in rec.event_dict.items():
        if ev == "object sniffing":  # or whichever
            all_durations.extend(arr[:,1] - arr[:,0])
plt.hist(np.array(all_durations)/1000, bins=30)
plt.xticks(np.arange(0, max(all_durations)/1000, 2))
plt.xlabel("Bout duration (s)")
plt.ylabel("Count")


# # Preprocessing Done, Organizing for PCA/Decoding Analysis

# ### organize novel and familiar recordings novel -> recording name -> recording object

# In[234]:


sc.recordings[0].event_dict.keys()


# In[235]:


import spike.spike_analysis.decoders as dec
import spike.spike_analysis.pca_trajectories as pca_traj


# In[236]:


for rec in sc.recordings:
    for event, val in rec.event_dict.items():
        print(f"{rec.name}: {event}, {len(val)}")


# In[240]:


condition_dict = {
    "novel": [rec_name for rec_name in event_dict.keys() if "d0" in rec_name],
    "fam": [rec_name for rec_name in event_dict.keys() if "d7" in rec_name],
}

events = ["social sniffing", "object sniffing"]

pca2_result = pca_traj.condition_pca(
    sc,
    condition_dict=condition_dict,
    event_length=3,
    pre_window=1,
    min_neurons=5,
    events=events,
    d=3,
)


# In[247]:


print(pca2_result.transformed_data.keys())


# In[250]:


pca2_result.transformed_data['novel'].shape


# In[249]:


pca2_result.transformed_data['fam'].shape


# In[241]:


# pca2_result = pca_traj.avg_trajectories_pca(sc, 3, 1, min_neurons=5, events=events)


# In[242]:


# ac_trajectories = pca_traj.trial_trajectories_pca(sc, 3, 1, min_neurons = 5, events = events)


# In[ ]:


def plot_pca_results_3d(pca_result, title, colors, azim, elev, save = False):
    event_lengths = int(
            (pca_result.event_length + pca_result.pre_window + pca_result.post_window) * 1000 / pca_result.timebin
        )
    event_end = int((pca_result.event_length + pca_result.pre_window) * 1000 / pca_result.timebin)
    pre_window = pca_result.pre_window * 1000 / pca_result.timebin
    post_window = pca_result.post_window * 1000 / pca_result.timebin
    pc_var = pca_result.explained_variance
    PCA_key = pca_result.labels
    PCA_matrix = pca_result.transformed_data
    col_counter = 0
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection="3d")
    #plt.subplots_adjust(left=0.3, right=0.99, bottom=0.1, top=0.9)
    for i in range(0, len(PCA_key), event_lengths):
        event_label = PCA_key[i]
        onset = int(i + pre_window - 1)
        end = int(i + event_end - 1)
        post = int(i + event_lengths - 1)
        ax.plot3D(
            PCA_matrix[i : i + event_lengths, 0],
            PCA_matrix[i : i + event_lengths, 1],
            PCA_matrix[i : i + event_lengths, 2],
            label=event_label,
            color=colors[col_counter],
            linewidth = 5,
            alpha = 0.8
        )
        ax.scatter(
            PCA_matrix[i, 0],
            PCA_matrix[i, 1],
            PCA_matrix[i, 2],
            marker="s",
            s=100,
            c="w",
            edgecolors=colors[col_counter],
        )
        ax.scatter(
            PCA_matrix[onset, 0],
            PCA_matrix[onset, 1],
            PCA_matrix[onset, 2],
            marker="^",
            s=300,
            c="w",
            edgecolors=colors[col_counter],
        )
        ax.scatter(
            PCA_matrix[end, 0],
            PCA_matrix[end, 1],
            PCA_matrix[end, 2],
            marker="o",
            s=200,
            c="w",
            edgecolors=colors[col_counter],
        )
        if post_window != 0:
            print("woo")
            ax.scatter(
                PCA_matrix[post, 0],
                PCA_matrix[post, 1],
                PCA_matrix[post, 2],
                marker="D",
                s=200,
                c="w",
                edgecolors=colors[col_counter],
            )
        col_counter += 1
    ax.legend(loc="upper left", bbox_to_anchor=(.9,1), frameon = False, fontsize = 14)
    # ax.set_xlim(-20, 45)
    # ax.set_ylim(-5, 25)
    # ax.set_zlim(-20, 30)
    ax.view_init(azim = azim, elev =elev)
    ax.set_title(f"{title}", fontsize = 24, y = 1)
    ax.set_xlabel(f"PC1 ({pc_var[0]*100:.1f}% variance)", fontsize = 16, labelpad = -10)
    ax.set_ylabel(f"PC2 ({pc_var[1]*100:.1f}% variance)", fontsize = 16, labelpad = -10)
    ax.set_zlabel(f"PC3 ({pc_var[2]*100:.1f}% variance)", fontsize = 16, labelpad = -10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.yaxis.pane.set_alpha(0.9)
    ax.xaxis.pane.set_alpha(0.9)
    ax.zaxis.pane.set_alpha(0.9)
    plt.tight_layout()
    if save:
        plt.savefig(f'{title}.png', dpi = 600, transparent = True,bbox_inches='tight' )
    plt.show()


def plot_pca_results_2d(pca_result, title, colors, legend_spot, save=False):
    event_lengths = int(
        (pca_result.event_length + pca_result.pre_window + pca_result.post_window) * 1000 / pca_result.timebin
    )
    
    event_end = int((pca_result.event_length + pca_result.pre_window) * 1000 / pca_result.timebin)
    pre_window = pca_result.pre_window * 1000 / pca_result.timebin
    post_window = pca_result.post_window * 1000 / pca_result.timebin
    pc_var = pca_result.explained_variance
    PCA_key = pca_result.labels
    PCA_matrix = pca_result.transformed_data
    col_counter = 0
    
    # Create figure with updated size
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    for i in range(0, len(PCA_key), event_lengths):
        event_label = PCA_key[i]
        onset = int(i + pre_window - 1)
        end = int(i + event_end - 1)
        post = int(i + event_lengths - 1)
        
        # Plot the continuous line with updated styling
        ax.plot(
            PCA_matrix[i:i + event_lengths, 0],
            PCA_matrix[i:i + event_lengths, 1],
            label=event_label,
            color=colors[col_counter],
            linewidth=5,
            alpha=0.8,
            zorder = 1
        )
        
        # Add markers with updated sizes
        ax.scatter(
            PCA_matrix[i, 0],
            PCA_matrix[i, 1],
            marker="s",
            s=200,
            c="w",
            edgecolors=colors[col_counter],
            zorder = 2
        )
        ax.scatter(
            PCA_matrix[onset, 0],
            PCA_matrix[onset, 1],
            marker="^",
            s=300,
            c="w",
            edgecolors=colors[col_counter],
            zorder = 3
        )
        ax.scatter(
            PCA_matrix[end, 0],
            PCA_matrix[end, 1],
            marker="o",
            s=200,
            c="w",
            edgecolors=colors[col_counter],
            zorder = 4
        )
        if post_window != 0:
            ax.scatter(
                PCA_matrix[post, 0],
                PCA_matrix[post, 1],
                marker="D",
                s=200,
                c="w",
                edgecolors=colors[col_counter],
                zorder =5
            )
        col_counter += 1
    
    # Updated legend formatting
    ax.legend(loc="upper left", bbox_to_anchor=legend_spot, frameon=False, fontsize=14)
    
    # ax.set_xlim(-20, 45)
    # ax.set_ylim(-5, 25)
    
    # Updated title and label formatting
    ax.set_title(f"{title}", fontsize=24, y=1.01)
    ax.set_xlabel(f"PC1 ({pc_var[0]*100:.1f}% variance)", fontsize=16)
    ax.set_ylabel(f"PC2 ({pc_var[1]*100:.1f}% variance)", fontsize=16)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    
    # Add tight layout
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{title}.png', dpi=600, transparent=True, bbox_inches='tight')
    plt.show()


# In[ ]:


def hex_2_rgb(hex_color): # Orange color
    rgb_color = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5))
    return rgb_color


# In[246]:


color_id_dict = {'social sniffing': hex_2_rgb('#15616F'), 
                'familiar': (1.0, 0.6862745098039216, 0.0),
                  'cagemate': hex_2_rgb('#792910')}
plot_pca_results_2d(pca_result=pca_result, title = 'Cups', colors = [hex_2_rgb('#792910'),
                                                             (1.0, 0.6862745098039216, 0.0),
                                                             hex_2_rgb('#15616F'),
                                                             'black'], legend_spot = (0.67, 1))
#plot_pca_results_3d(pca_result=pca_result, title = 'Cups', colors = [hex_2_rgb('#792910'),
#                                                              (1.0, 0.6862745098039216, 0.0),
#                                                              hex_2_rgb('#15616F'),
#                                                              'black'], azim = 60, elev = 30)  


# In[217]:


for rec in sc.recordings:
    if rec.good_neurons < 5:
        continue
    available = [ev for ev in rec.event_dict.keys() if len(rec.event_dict[ev]) > 0]
    print(f"{rec.name}: {available}")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script /blue/npadillacoreano/t.heeps/rehouse_code/diff_fam_social_memory_ephys/thomas-social-memory/single_cell/pca_der.ipynb')

