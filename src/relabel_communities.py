# src/relabel_communities.py
# Purpose:
# This script processes a series of network analysis snapshot files (CSV format) generated at
# different timesteps of a simulation (e.g., T=0, T=5, T=10...). Each snapshot contains
# Louvain community assignments for students, where the community IDs are arbitrarily
# assigned by the Louvain algorithm at each independent run and are not persistent across time.
#
# What it does:
# 1. Reads all 'analysis_results_Txx.csv' files from the 'analysis_snapshots/' directory,
#    sorted chronologically.
# 2. For the first timestep (T=0), it establishes the original Louvain community IDs as the
#    initial 'persistentLouvainId's.
# 3. For each subsequent timestep, it attempts to match the Louvain communities found in that
#    snapshot to the persistent communities from the *previous* timestep. This matching is
#    done by calculating the Jaccard Index (member overlap) between each current community
#    and all persistent communities from the prior step.
# 4. A current community is assigned the 'persistentLouvainId' of the previous community
#    with which it shares the highest Jaccard Index, provided this index meets a
#    predefined `JACCARD_THRESHOLD`. This ensures that only sufficiently similar communities
#    are considered continuations. A greedy approach is used, prioritizing best matches first.
# 5. If a current community cannot be matched with sufficient overlap to any existing
#    persistent community, or if its best match has already been claimed by another
#    current community with a stronger overlap, it is assigned a new, unique 'persistentLouvainId'.
# 6. The script outputs new CSV files (e.g., 'analysis_results_Txx_relabelled.csv') into an
#    'analysis_snapshots_relabelled/' directory. These new files contain an additional
#    'persistentLouvainId' column, which allows for more meaningful longitudinal analysis
#    of community evolution, as the IDs now attempt to track the same underlying community
#    (or its descendants) across different time points.
#
# This heuristic approach helps to address the challenge of arbitrary community labeling by
# Louvain when analyzing dynamic networks, enabling a more consistent view of how specific
# groups of students evolve over the simulation.

import pandas as pd
import os
import glob
import json
import logging
from collections import defaultdict

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
JACCARD_THRESHOLD = 0.1  # Minimum Jaccard index to consider communities a match (tune this!)
SNAPSHOTS_DIR = "analysis_snapshots"  # Directory where your Txx.csv files are
OUTPUT_DIR_RELABELED = "analysis_snapshots_relabelled" # New directory for output

def get_snapshot_files(directory):
    """Gets all analysis_results_Txx.csv files, sorted by timestep."""
    files = glob.glob(os.path.join(directory, "analysis_results_T*.csv"))
    # Extract timestep from filename for sorting
    # Assumes format like analysis_results_T00.csv, analysis_results_T05.csv, ... T100.csv
    def sort_key(filepath):
        filename = os.path.basename(filepath)
        try:
            # Extract numbers after 'T' and before '.csv'
            return int(filename.split('T')[1].split('.')[0])
        except:
            return -1 # Should not happen with correct naming
    files.sort(key=sort_key)
    return files

def calculate_jaccard_index(set1, set2):
    """Calculates the Jaccard Index between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def main():
    if not os.path.exists(OUTPUT_DIR_RELABELED):
        os.makedirs(OUTPUT_DIR_RELABELED)
        logger.info(f"Created output directory: {OUTPUT_DIR_RELABELED}")

    snapshot_files = get_snapshot_files(SNAPSHOTS_DIR)
    if not snapshot_files:
        logger.error(f"No analysis snapshot files found in {SNAPSHOTS_DIR}. Exiting.")
        return

    # To store the mapping from (original_timestep_louvain_id) to persistent_id for the PREVIOUS step
    previous_step_persistent_ids_members = {} # persistent_id -> set_of_student_ids
    next_available_persistent_id = 0
    all_relabelled_dfs = []

    for file_idx, filepath in enumerate(snapshot_files):
        logger.info(f"Processing file: {filepath}")
        current_df = pd.read_csv(filepath)
        current_df['persistentLouvainId'] = -1 # Initialize new column

        # Get student members for each original Louvain community in the current dataframe
        current_original_communities_members = defaultdict(set)
        for _, row in current_df.iterrows():
            current_original_communities_members[row['louvainCommunity']].add(row['studentId'])

        current_step_persistent_ids_map = {} # Maps current original_id -> assigned persistent_id
        assigned_persistent_ids_this_step = set() # Tracks which persistent IDs are already used in this step

        if file_idx == 0: # First file, establish baseline persistent IDs
            logger.info("Establishing baseline persistent IDs for T00.")
            for original_id, members in current_original_communities_members.items():
                current_df.loc[current_df['louvainCommunity'] == original_id, 'persistentLouvainId'] = next_available_persistent_id
                current_step_persistent_ids_map[original_id] = next_available_persistent_id
                previous_step_persistent_ids_members[next_available_persistent_id] = members # Store for next iteration
                next_available_persistent_id += 1
        else:
            # Match current communities to previous step's persistent communities
            potential_matches = []
            for current_orig_id, current_members in current_original_communities_members.items():
                best_match_score = -1
                best_prev_persistent_id = -1
                for prev_persist_id, prev_members in previous_step_persistent_ids_members.items():
                    jaccard = calculate_jaccard_index(current_members, prev_members)
                    if jaccard > best_match_score:
                        best_match_score = jaccard
                        best_prev_persistent_id = prev_persist_id
                
                if best_prev_persistent_id != -1 and best_match_score >= JACCARD_THRESHOLD:
                    potential_matches.append({
                        'current_orig_id': current_orig_id,
                        'prev_persist_id': best_prev_persistent_id,
                        'jaccard': best_match_score
                    })
            
            # Sort potential matches by Jaccard score (descending) to prioritize best matches
            potential_matches.sort(key=lambda x: x['jaccard'], reverse=True)

            new_persistent_ids_members_for_next_step = {}

            for match in potential_matches:
                current_orig_id = match['current_orig_id']
                prev_persist_id_candidate = match['prev_persist_id']

                # If this current community hasn't been assigned a persistent ID yet,
                # AND the candidate persistent ID from previous step hasn't been taken by a better match in *this* step
                if current_step_persistent_ids_map.get(current_orig_id) is None and \
                   prev_persist_id_candidate not in assigned_persistent_ids_this_step:
                    
                    current_df.loc[current_df['louvainCommunity'] == current_orig_id, 'persistentLouvainId'] = prev_persist_id_candidate
                    current_step_persistent_ids_map[current_orig_id] = prev_persist_id_candidate
                    assigned_persistent_ids_this_step.add(prev_persist_id_candidate)
                    # Update members for this persistent ID for the *next* iteration
                    new_persistent_ids_members_for_next_step[prev_persist_id_candidate] = current_original_communities_members[current_orig_id]

            # Assign new persistent IDs to current communities that didn't match any previous ones
            for original_id, members in current_original_communities_members.items():
                if current_step_persistent_ids_map.get(original_id) is None: # Not yet assigned
                    current_df.loc[current_df['louvainCommunity'] == original_id, 'persistentLouvainId'] = next_available_persistent_id
                    current_step_persistent_ids_map[original_id] = next_available_persistent_id
                    new_persistent_ids_members_for_next_step[next_available_persistent_id] = members
                    next_available_persistent_id += 1
            
            previous_step_persistent_ids_members = new_persistent_ids_members_for_next_step

        all_relabelled_dfs.append(current_df)
        
        # Save the relabelled DataFrame
        output_filename = os.path.join(OUTPUT_DIR_RELABELED, os.path.basename(filepath))
        current_df.to_csv(output_filename, index=False)
        logger.info(f"Saved relabelled data to: {output_filename}")

    logger.info("Community relabeling process complete.")

if __name__ == "__main__":
    main()