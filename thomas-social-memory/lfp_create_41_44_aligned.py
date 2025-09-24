#!/usr/bin/env python
# coding: utf-8

# In[31]:


import sys, importlib, os, inspect

NEW_ROOT = "/blue_npadillacoreano/rehouse_code/diff_fam_social_memory_ephys/thomas-social-memory"
OLD_ROOT = "/blue/npadillacoreano/t.heeps/rehouse_code/ephys_analysis"

# 1) Put the desired repo first on sys.path
if NEW_ROOT not in sys.path:
    sys.path.insert(0, NEW_ROOT)

# 2) Optionally remove the stale path from this session
if OLD_ROOT in sys.path:
    sys.path.remove(OLD_ROOT)

# 3) Purge cached modules under ambiguous names
for m in list(sys.modules):
    if m == "behavior" or m.startswith("behavior."):
        del sys.modules[m]
    if m == "LFP" or m.startswith("LFP."):
        del sys.modules[m]

importlib.invalidate_caches()

# 4) Re-import and verify
import behavior.boris_extraction as boris
from LFP.lfp_collection import LFPCollection
import trodes.read_exported as tr
print("NOW boris_extraction from:", inspect.getfile(boris))

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

#os.chdir('/blue/npadillacoreano/t.heeps/rehouse_code/diff_fam_social_memory_ephys')

# Printing the search path to see what it's looking at
# print(sys.path)

# importing ephys analysis code
# import trodes.read_exported as tr
# import behavior.boris_extraction as boris

print("behavior package from:", boris.__file__)
print("boris_extraction from  :", inspect.getfile(boris))

# from LFP.lfp_collection import LFPCollection


# ### D7 is OM while D0 is MO?

# In[32]:


merge_to_video = {
    "22_rehouse_d0_merged.time": "22_23_rehouse_d0.2.videoTimeStamps",
    "23_rehouse_d0_merged.time": "22_23_rehouse_d0.1.videoTimeStamps",
    "22_rehouse_d1_merged.time": "22_23_rehouse_d1.1.videoTimeStamps",
    "23_rehouse_d1_merged.time": "22_23_rehouse_d1.2.videoTimeStamps",
    "22_rehouse_d3_merged.time": "22_23_rehouse_d3.2.videoTimeStamps",
    "23_rehouse_d3_merged.time": "22_23_rehouse_d3.1.videoTimeStamps",
    "22_rehouse_d4_merged.time": "22_23_rehouse_d4.2.videoTimeStamps",
    "23_rehouse_d4_merged.time": "22_23_rehouse_d4.1.videoTimeStamps",
    "22_rehouse_d5_merged.time": "22_23_rehouse_d5.2.videoTimeStamps",
    "23_rehouse_d5_merged.time": "22_23_rehouse_d5.1.videoTimeStamps",
    "22_rehouse_d6_merged.time": "22_23_rehouse_d6.2.videoTimeStamps",
    "23_rehouse_d6_merged.time": "22_23_rehouse_d6.1.videoTimeStamps",
    "22_rehouse_d7_merged.time": "22_23_rehouse_d7.2.videoTimeStamps",
    "23_rehouse_d7_merged.time": "22_23_rehouse_d7.1.videoTimeStamps",
    "31_rehouse_d0_merged.time": "31_32_rehouse_d0.1.videoTimeStamps",
    "32_rehouse_d0_merged.time": "31_32_rehouse_d0.2.videoTimeStamps",
    "31_rehouse_d1_merged.time": "31_32_rehouse_d1.2.videoTimeStamps",
    "32_rehouse_d1_merged.time": "31_32_rehouse_d1.1.videoTimeStamps",
    "31_rehouse_d2_merged.time": "31_32_rehouse_d2.2.videoTimeStamps",
    "32_rehouse_d2_merged.time": "31_32_rehouse_d2.1.videoTimeStamps",
    "31_rehouse_d3_merged.time": "31_32_rehouse_d3.2.videoTimeStamps",
    "32_rehouse_d3_merged.time": "31_32_rehouse_d3.1.videoTimeStamps",
    "31_rehouse_d4_merged.time": "31_32_rehouse_d4.2.videoTimeStamps",
    "32_rehouse_d4_merged.time": "31_32_rehouse_d4.1.videoTimeStamps",
    "31_rehouse_d5_merged.time": "31_32_rehouse_d5.2.videoTimeStamps",
    "32_rehouse_d5_merged.time": "31_32_rehouse_d5.1.videoTimeStamps",
    "31_rehouse_d6_merged.time": "31_32_rehouse_d6.2.videoTimeStamps",
    "32_rehouse_d6_merged.time": "31_32_rehouse_d6.1.videoTimeStamps",
    "31_rehouse_d7_merged.time": "31_32_rehouse_d7.2.videoTimeStamps",
    "32_rehouse_d7_merged.time": "31_32_rehouse_d7.1.videoTimeStamps",
    "41_rehouse_d0_merged.time": "41_44_rehouse_d0.2.videoTimeStamps",
    "44_rehouse_d0_merged.time": "41_44_rehouse_d0.1.videoTimeStamps",
    "41_rehouse_d1_merged.time": "41_44_rehouse_d1.2.videoTimeStamps",
    "44_rehouse_d1_merged.time": "41_44_rehouse_d1.1.videoTimeStamps",
    "41_rehouse_d2_merged.time": "41_44_rehouse_d2.2.videoTimeStamps",
    "44_rehouse_d2_merged.time": "41_44_rehouse_d2.1.videoTimeStamps",
    "41_rehouse_d3_merged.time": "41_44_rehouse_d3.1.videoTimeStamps",
    "44_rehouse_d3_merged.time": "41_44_rehouse_d3.2.videoTimeStamps",
    "41_rehouse_d4_merged.time": "41_44_rehouse_d4.2.videoTimeStamps",
    "44_rehouse_d4_merged.time": "41_44_rehouse_d4.1.videoTimeStamps",
    "41_rehouse_d5_merged.time": "41_44_rehouse_d5.2.videoTimeStamps",
    "44_rehouse_d5_merged.time": "41_44_rehouse_d5.1.videoTimeStamps",
    "41_rehouse_d6_merged.time": "41_44_rehouse_d6.2.videoTimeStamps",
    "44_rehouse_d6_merged.time": "41_44_rehouse_d6.1.videoTimeStamps",
    "41_rehouse_d7_merged.time": "41_44_rehouse_d7.2.videoTimeStamps",
    "44_rehouse_d7_merged.time": "41_44_rehouse_d7.1.videoTimeStamps"
}


# In[33]:


rec_to_timefile = {
    "41_rehouse_d0_merged.rec": "41_rehouse_d0_merged.time",
    "41_rehouse_d7_merged.rec": "41_rehouse_d7_merged.time",
    "44_rehouse_d0_merged.rec": "44_rehouse_d0_merged.time",
    "44_rehouse_d7_merged.rec": "44_rehouse_d7_merged.time",
}


# In[34]:


d0_41_beh_path = r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/share/rehouse_data/behavior_csvs/aggregated_csvs/41_rehouse_d0_OM.csv'
d7_41_beh_path = r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/share/rehouse_data/behavior_csvs/aggregated_csvs/41_rehouse_d7_MO.csv'

d0_44_beh_path = r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/share/rehouse_data/behavior_csvs/aggregated_csvs/44_rehouse_d0_MO.csv'
d7_44_beh_path = r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/share/rehouse_data/behavior_csvs/aggregated_csvs/44_rehouse_d7_OM.csv'


# ### data_path to folder with all raw lfp recs, used in lfp collection creation later 

# In[35]:


data_path = r'/home/t.heeps/blue_npadillacoreano/npadillacoreano/share/rehouse_data/raw_lfp_data'


# In[36]:


# 41 behaviors
d0_41_beh = pd.read_csv(d0_41_beh_path)
d7_41_beh = pd.read_csv(d7_41_beh_path)

# 44 behaviors
d0_44_beh = pd.read_csv(d0_44_beh_path)
d7_44_beh = pd.read_csv(d7_44_beh_path)

d0_41_beh.head()


# # Creating Behavior Dicts for LFP Collection

# ### Aligning Boris to Neural Data

# In[37]:


d0_44_beh.columns


# In[38]:


d0_44_bheaviors = d0_44_beh['Behavior'].unique()
print(d0_44_bheaviors)


# In[39]:


d0_44_beh['Subject'].unique()


# In[40]:


# Helper functions to get first timestamps Credit Meghan Cum

first_timestamp_dict = {}

# Step 1: extract first_timestamp from any *.dat file
def build_first_ts_dict(data_path):
    for dirpath, dirnames, filenames in os.walk(data_path):
        if os.path.basename(dirpath).endswith('.time'):
            for file in filenames:
                if file.endswith(".dat"):
                    ts_file = os.path.join(dirpath, file)
                    ts_dict = tr.read_trodes_extracted_data_file(ts_file)
                    first_timestamp = ts_dict['first_timestamp']

                    # key is .time folder name
                    time_folder_name = os.path.basename(dirpath)
                    first_timestamp_dict[time_folder_name] = int(first_timestamp)
                
    return first_timestamp_dict

                
# Step 2: build play_indexed_dict from each .videoTimeStamps, which holds a play indexed videotimestamped array
play_indexed_dict = {}

def build_play_indexed_ts_dict(data_path, first_timestamp_dict):
    # data_path = /home/t.heeps/blue_npadillacoreano/npadillacoreano/share/rehouse_data/raw_lfp_data
    for dirpath, dirnames, filenames in os.walk(data_path):
        
        # Check if current directory name ends with ".rec"
        if os.path.basename(dirpath).endswith(".rec"): 
            
            # directly accessing .videotimestamps using the mapping
            for time_folder in first_timestamp_dict.keys():
                if time_folder in dirnames:
                    
                    if time_folder not in merge_to_video:
                        print(f"no mapping for {time_folder}")
                        continue
                        
                    videotsfile = os.path.join(dirpath, merge_to_video[time_folder])
                    if not os.path.exists(videotsfile):
                        print(f"videotimestamps file does not exist or mapping incorrect: {videotsfile}\n")
                        continue
                    
                    videotsarray = tr.readCameraModuleTimeStamps(videotsfile)
                    first_ts = first_timestamp_dict[time_folder]
                    
                    videotsarray_play_indexed = videotsarray - first_ts/20000
                    if videotsarray_play_indexed[0] < 0:
                        if abs(videotsarray_play_indexed[0]) < .001:
                            videotsarray_play_indexed[0] = 0
                        else:
                            print(f'negative first timestamp: {videotsarray_play_indexed[0]} in {merge_to_video[time_folder]}')

                    play_indexed_dict[merge_to_video[time_folder]] = {
                        'play_indexed_array': videotsarray_play_indexed,
                        'stream_indexed_array': videotsarray,
                        'first_timestamp': first_ts,
                    }
    
    return play_indexed_dict


# ### Building dict with videotimestamp array and first timestamp from .dat

# In[41]:


# first .dat timestamp for each subject 
dat_first_timestamps = build_first_ts_dict(data_path)
#print(dat_first_timestamps)


# In[42]:


# dict with .videotimestamps array and first timestamp of .videotimestamps
timestamps_dict = build_play_indexed_ts_dict(data_path, dat_first_timestamps)
# print(timestamps_dict)


# In[43]:


print(timestamps_dict.keys())


# In[44]:


# simple helpers for getting Frames or FPS
def _has_frame_cols(df):
    return "Image index start" in df.columns

def _has_fps(df):
    return any("FPS" in c for c in df.columns)


# In[45]:


def extract_bouts_for_recording(rec_time_file, boris_df, subject, behavior):
    """
    Given a merged .time filename and its boris dataframe, returns bout array
    using frame-based or fps-based extraction depending on the BORIS file.
    
    rec_time_file: merged.time filename, used to find the corresponding .videotimestamps
    """
    # Locating corresponding .videotimestamps of merged.time folder, in other words corresponding subject camera instance
    videotimestamps_filename = merge_to_video[rec_time_file]
    cameratimestamps = timestamps_dict[videotimestamps_filename]['stream_indexed_array'] # stream indexed or unaligned camera timestamps
    
    first_timestamp = dat_first_timestamps[rec_time_file] # .dat first timestamp

    # choose extraction function
    if _has_frame_cols(boris_df):  # not MAC scored
        bouts = boris.get_behavior_bouts_frame(
            boris_df,
            cameratimestamps,
            first_timestamp,
            subject,
            behavior,
        )

    elif _has_fps(boris_df):  # MAC scored
        bouts = boris.get_behavior_bouts_fps(
            boris_df,
            cameratimestamps,
            first_timestamp,
            subject,
            behavior,
        )
    else:
        print(f"No frames or fps for {rec_time_file}")
        bouts = boris.get_behavior_bouts(
            boris_df,
            subject,
            behavior,
        )
    return bouts


# In[46]:


d0_44_behaviors = d0_44_beh['Behavior'].unique()
subjects = ['subject', 'social_agent']
print(d0_44_bheaviors)


# In[47]:


subj_41_d0_bouts = extract_bouts_for_recording('44_rehouse_d0_merged.time', d0_44_beh, subjects, d0_44_behaviors)


# In[48]:


# print(subj_41_d0_bouts)


# In[50]:


print(os.getcwd())


# In[ ]:


NOTEBOOK = "/blue/npadillacoreano/t.heeps/rehouse_code/diff_fam_social_memory_ephys/thomas-social-memory/lfp_create_41_44_aligned.ipynb"
get_ipython().system('jupyter nbconvert --to script "{NOTEBOOK}"')


# In[49]:


# !jupyter nbconvert --to script lfp_create_41_44_aligned.ipynb


# In[25]:


# takes behavior df and iterates through rows appending behaviors and their corresponding start stop times 
def build_event_dict(behavior_df):
    event_dict = defaultdict(list)
    for _, row in behavior_df.iterrows(): # loops through rows
        if not np.isnan(row['Start (s)']) and not np.isnan(row['Stop (s)']): # skips NaN start stops
            event_dict[row['Behavior']].append((row['Start (s)'], row['Stop (s)']))
    return {behavior: np.array(intervals) for behavior, intervals in event_dict.items()}

# Create recording_to_event_dict using build_event_dict
recording_to_event_dict = {
    "41_rehouse_d0_merged.rec": build_event_dict(d0_41_beh),
    "41_rehouse_d7_merged.rec": build_event_dict(d7_41_beh),
    "44_rehouse_d0_merged.rec": build_event_dict(d0_44_beh),
    "44_rehouse_d7_merged.rec": build_event_dict(d7_44_beh),
}

# Brief summary of recording_to_event_dict
for rec, events in recording_to_event_dict.items():
    print(f"\nEvents for {rec}:")
    for behavior, times in events.items():
        print(f"  {behavior}: {len(times)} occurrences")


# ### Creating subject_to_channel_dict using https://uflorida-my.sharepoint.com/:x:/g/personal/mcum_ufl_edu/EWN3ExBZMiJKkuqtl9b7yo4Bz1URBoukFjwLUwv4kTIzag?wdOrigin=TEAMS-MAGLEV.p2p_ns.rwc&wdExp=TEAMS-TREATMENT&wdhostclicktime=1754166004476&web=1

# ### Subject 44 NAc is bad, consider exclusion 

# In[26]:


subject_to_channel_dict = {
    "41": {
        "mPFC": 26,
        "vHPC": 31, # moved from 31 -> 30 since it seemed cleaner in diagnostic
        "BLA": 30,
        "NAc": 28,
        "MD": 29
    },
    "44": {
        "mPFC": 25, # moved from 26 -> 25 since it seemed cleaner in diagnostic
        "vHPC": 31,
        "BLA": 30,
        "NAc": 28,
        "MD": 29
    }
}


# ### recording_to_subject_dict creation

# In[27]:


recording_to_subject_dict = {
    "41_rehouse_d0_merged.rec": "41",
    "44_rehouse_d0_merged.rec": "44",
    "41_rehouse_d7_merged.rec": "41",
    "44_rehouse_d7_merged.rec": "44",
}


# In[28]:


from importlib import reload
import LFP.lfp_recording
import LFP.lfp_collection

reload(LFP.lfp_recording)      # <- reload this first!
reload(LFP.lfp_collection)

from LFP.lfp_collection import LFPCollection  # <- re-import class afterward


# In[30]:


get_ipython().system('jupyter nbconvert --to script lfp_create_41_44_aligned.ipynb')


# In[134]:


data_path = "/home/t.heeps/blue_npadillacoreano/npadillacoreano/share/rehouse_data/raw_lfp_data/just_41_44_d0_d7"

lfp_collection = LFPCollection(
    subject_to_channel_dict=subject_to_channel_dict,
    data_path=data_path,
    recording_to_subject_dict=recording_to_subject_dict,
    recording_to_event_dict=shifted_recording_to_event_dict,
    threshold=5, # leaving None for now, experiement later to see what's best
    trodes_directory=data_path,
)


# ### Link to good brain regions labels https://uflorida-my.sharepoint.com/:x:/g/personal/mcum_ufl_edu/EXgHRUX0XrpFkn-VUiKbVT0BucWNPtqQZvTRxIPhwALz8Q?wdOrigin=TEAMS-MAGLEV.p2p_ns.rwc&wdExp=TEAMS-TREATMENT&wdhostclicktime=1754168097667&web=1

# In[136]:


lfp_collection.diagnostic_plots(threshold=5)


# In[137]:


lfp_collection.diagnostic_plots_channel_finder()


# ### Saving collection to json+h5

# In[140]:


from importlib import reload
import LFP.lfp_recording
import LFP.lfp_collection

reload(LFP.lfp_recording)      # <- reload this first!
reload(LFP.lfp_collection)

from LFP.lfp_collection import LFPCollection  # <- re-import class afterward


# In[141]:


import json
import os

output_path = "/home/t.heeps/blue_npadillacoreano/npadillacoreano/share/rehouse_data/lfp_collections"

notes = "Rehouse data - d0/d7 for subjs 41/44 - NAc is supposed to be bad for subject 44"

# creating output dir
os.makedirs(output_path, exist_ok=True)

# setting frequencies none so save works, consider changing class later for this to always be true
lfp_collection.frequencies = None


# === Save the collection ===
print(f"Saving LFPCollection to {output_path}...")
LFPCollection.save_to_json(
    lfp_collection,
    output_path=output_path,
    notes=notes,
    filename="shifted_events_44_41"
)
print("✅ LFPCollection saved successfully to JSON + HDF5")


# In[1]:


get_ipython().system('jupyter nbconvert --to script lfp_creation.ipynb')

