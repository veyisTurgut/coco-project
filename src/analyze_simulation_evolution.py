# src/analyze_simulation_evolution.py
# Purpose:
# This script performs a longitudinal analysis of simulation outputs to understand how student
# network metrics and community structures evolve over time. It consolidates analysis results
# from multiple simulation timesteps, visualizes trends in key metrics (Shannon entropy,
# betweenness, PageRank), identifies potential echo chamber formations, and facilitates
# qualitative case studies by linking final-state metrics with student profiles and interaction logs.
#
# What it does:
# 1.  Configuration & Initialization:
#     - Sets up logging.
#     - Defines paths and patterns for input files: analysis snapshot CSVs (e.g.,
#       `analysis_snapshots/analysis_results_T{timestep}.csv`), student profile CSV
#       (`synthetic_data/students.csv`), and simulation log CSVs (`consumed_in_simulation.csv`,
#       `llm_recommendations_log.csv` from `simulation_outputs/`).
#     - Specifies the list of `TIME_STEPS` for which analysis snapshots are available.
#     - Defines a `LOW_ENTROPY_THRESHOLD` used for identifying potential echo chamber behavior.
#
# 2.  Data Consolidation & Preparation (`load_and_consolidate_analysis_data()`):
#     - Reads multiple analysis snapshot CSV files, each corresponding to a specific timestep
#       of the simulation.
#     - Adds a `TimeStep` column to each DataFrame before concatenating them into a single
#       master DataFrame (`all_results_df`).
#
# 3.  Quantitative Analysis of Metric Evolution:
#     - `plot_metric_distribution_over_time()`: Creates and saves boxplots to visualize the
#       distribution of Shannon entropy, betweenness centrality, and PageRank across all students
#       at each `TimeStep`.
#     - `plot_percentage_below_threshold_over_time()`: Creates and saves line plots showing the
#       percentage of students whose Shannon entropy is below `LOW_ENTROPY_THRESHOLD` over time.
#       It also plots the percentage of students with near-zero betweenness centrality over time.
#     - Community Dynamics Analysis: Calculates the average Shannon entropy within each Louvain
#       community at each `TimeStep`. It then plots the evolution of this average intra-community
#       entropy for each community over time.
#
# 4.  Qualitative Analysis & Case Studies (Focus on Final Timestep):
#     - Loads student profile data from the `students.csv` file.
#     - Merges the analysis results from the final `TimeStep` with the student profile data.
#     - Identifies and prints detailed information (including profile attributes like name,
#       learning style, loved/disliked topics) for:
#       - Low-Entropy Students: Students whose Shannon entropy at the final timestep is below
#         the `LOW_ENTROPY_THRESHOLD` (potential echo chamber candidates).
#       - High-Betweenness Students: Students whose betweenness centrality at the final timestep
#         is in the top quantile (potential bridge nodes).
#
# 5.  Simulation Log Deep Dive (Example Analysis):
#     - Loads the `consumed_in_simulation.csv` and `llm_recommendations_log.csv` files.
#     - For an example student identified as low-entropy, it retrieves and prints their recent
#       consumption history and a sample of recommendations they received during the simulation.
#       This serves as a template for more detailed investigation into individual student journeys.
#
# 6.  Output:
#     - Saves all generated plots (metric evolutions, threshold percentages, community entropy)
#       to a `figures/` directory.
#     - Prints summaries of quantitative analyses and detailed lists for case studies
#       (low-entropy students, bridge nodes) to the console.
#
# Key Libraries Used:
# - pandas: For data loading, manipulation, consolidation, and aggregation.
# - numpy: Used implicitly by pandas, potentially for numerical operations.
# - matplotlib.pyplot: For creating static, animated, and interactive visualizations.
# - seaborn: For making statistical graphics; used here for boxplots and lineplots.
# - json: For parsing JSON strings in CSV files (e.g., `lovedTopicIds`).
# - os: For path manipulation.
# - logging: For tracking script execution and potential issues.
# src/analyze_simulation_evolution.py
# Purpose:
# This script performs a longitudinal analysis of simulation outputs to understand how student
# network metrics and community structures evolve over time. It consolidates analysis results
# from multiple simulation timesteps, visualizes trends in key metrics (Shannon entropy,
# betweenness, PageRank), identifies potential echo chamber formations, and facilitates
# qualitative case studies by linking final-state metrics with student profiles and interaction logs.
#
# What it does:
# (Detailed comments from your provided version remain largely applicable here)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
import random # Used for sampling in individual plot if too many lines

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Constants ---
ANALYSIS_FILES_PATTERN = "analysis_snapshots/analysis_results_T{}.csv"
# Ensure TIME_STEPS match the files you have in analysis_snapshots
TIME_STEPS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
STUDENT_PROFILE_FILE = "synthetic_data/students.csv"
CONSUMED_SIM_LOG_FILE = "simulation_outputs/consumed_in_simulation.csv"
RECOMMENDATION_SIM_LOG_FILE = "simulation_outputs/llm_recommendations_log.csv"
LOW_ENTROPY_THRESHOLD = 2.0 # Used for % plot and for filtering individual trajectories
FIGURES_DIR = "figures"
COMMUNITY_ID_COLUMN = 'persistentLouvainId' 
CASE_STUDY_OUTPUT_DIR = "case_study_outputs" # Directory for case study CSVs

# --- Helper Functions ---
def create_output_directory_if_not_exists(directory_name):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        logger.info(f"Created output directory: {directory_name}")

# --- New Helper Function to Save DataFrame as Image ---
def save_df_as_table_image(df, fig_filepath, title=""):
    """Saves a pandas DataFrame as a table image using matplotlib."""
    if df.empty:
        logger.warning(f"DataFrame is empty. Skipping saving table image for: {title if title else fig_filepath}")
        return
    # Adjust figsize: width based on num_cols, height based on num_rows + title
    # Approx 1.5 units width per column, 0.3 units height per row
    fig_width = max(8, 1 + df.shape[1] * 1.5) 
    fig_height = max(3, 1 + df.shape[0] * 0.5 + (1 if title else 0))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    the_table = ax.table(cellText=df.values, 
                         colLabels=df.columns, 
                         rowLabels=df.index, 
                         cellLoc = 'center', 
                         loc='center',
                         colWidths=[0.2]*len(df.columns) if len(df.columns) > 0 else None) # Adjust colWidths if needed
                         
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    
    # Adjust cell padding and scale based on typical content
    the_table.scale(1.2, 1.2) # General scale factor

    if title:
        plt.title(title, fontsize=12, pad=20) # Add padding for title

    try:
        plt.savefig(fig_filepath, bbox_inches='tight', dpi=150)
        logger.info(f"Saved table image: {fig_filepath}")
    except Exception as e:
        logger.error(f"Failed to save table image {fig_filepath}: {e}")
    finally:
        plt.close(fig)

# --- 1. Data Consolidation & Preparation ---
def load_and_consolidate_analysis_data(pattern, time_steps_list):
    """Loads multiple analysis CSVs and concatenates them with a TimeStep column."""
    all_dfs = []
    for ts in time_steps_list:
        # Format timestep with leading zero for T00-T05, T10 etc.
        # T100 will be handled correctly as is by string formatting.
        filename_ts_str = f"{ts:02d}" if ts < 100 else str(ts)
        filename = pattern.format(filename_ts_str)
        try:
            df = pd.read_csv(filename)
            df['TimeStep'] = ts
            all_dfs.append(df)
            logger.info(f"Loaded and processed {filename}")
        except FileNotFoundError:
            logger.warning(f"File not found: {filename}. Skipping this timestep.")
    if not all_dfs:
        logger.error("No analysis files were loaded.")
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

# --- 2. Quantitative Analysis of Evolution ---
def plot_metric_distribution_over_time(df, metric_column, title, output_dir_val):
    """Plots the distribution of a metric over time steps using boxplots."""
    plt.figure(figsize=(14, 7)) # Wider for more timesteps
    sns.boxplot(x='TimeStep', y=metric_column, data=df)
    plt.title(title)
    plt.xlabel("Simulation Turn (Time Step)")
    plt.ylabel(metric_column)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_val, f"{metric_column}_evolution.png"))
    logger.info(f"Saved plot: {output_dir_val}/{metric_column}_evolution.png")
    plt.close()

def plot_percentage_below_threshold_over_time(df, metric_column, threshold, title, output_dir_val, lower_is_problem=True):
    """Plots the percentage of students meeting a threshold condition over time."""
    summary_data = []
    for ts in sorted(df['TimeStep'].unique()): # Ensure timesteps are sorted for line plot
        subset_df = df[df['TimeStep'] == ts]
        if lower_is_problem:
            count_problem = len(subset_df[subset_df[metric_column] < threshold])
        else:
            count_problem = len(subset_df[subset_df[metric_column] > threshold])
        total_students = len(subset_df)
        percentage = (count_problem / total_students) * 100 if total_students > 0 else 0
        summary_data.append({'TimeStep': ts, 'Percentage': percentage})

    summary_df = pd.DataFrame(summary_data)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='TimeStep', y='Percentage', data=summary_df, marker='o')
    plt.title(title)
    plt.xlabel("Simulation Turn (Time Step)")
    plt.ylabel(f"Percentage of Students")
    plt.ylim(0, 100) # Keep y-axis fixed for better comparison if re-running
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_val, f"{metric_column}_percentage_threshold.png"))
    logger.info(f"Saved plot: {output_dir_val}/{metric_column}_percentage_threshold.png")
    plt.close()

def plot_community_entropy_evolution(df, community_col_name, output_dir_val):
    """Plots the evolution of average Shannon entropy per specified community ID column."""
    community_entropy_data = []
    if community_col_name not in df.columns:
        logger.warning(f"Community column '{community_col_name}' not found. Skipping community entropy plot.")
        return

    for ts in sorted(df['TimeStep'].unique()):
        ts_df = df[df['TimeStep'] == ts].copy()
        ts_df[community_col_name] = ts_df[community_col_name].astype(str)
        valid_communities = ts_df[ts_df[community_col_name] != '-1'] # Assuming -1 is unassigned
        if not valid_communities.empty:
            avg_comm_entropy = valid_communities.groupby(community_col_name)['shannonEntropy'].mean().reset_index()
            avg_comm_entropy['TimeStep'] = ts
            community_entropy_data.append(avg_comm_entropy)

    if community_entropy_data:
        community_entropy_df = pd.concat(community_entropy_data)
        plt.figure(figsize=(14, 8)) # Wider for more communities/timesteps
        sns.lineplot(x='TimeStep', y='shannonEntropy', hue=community_col_name, data=community_entropy_df, marker='o', palette='tab10')
        plt.title(f'Evolution of Average Shannon Entropy per {community_col_name.replace("persistentL","L")}')
        plt.xlabel("Simulation Turn (Time Step)")
        plt.ylabel("Average Shannon Entropy")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title=community_col_name.replace("persistentL","L"), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
        plt.savefig(os.path.join(output_dir_val, f"community_entropy_evolution_{community_col_name}.png"))
        logger.info(f"Saved plot: {output_dir_val}/community_entropy_evolution_{community_col_name}.png")
        plt.close()
    else:
        logger.warning(f"No community data to plot average entropy for column {community_col_name}.")

def plot_individual_student_entropy_over_time(df, filter_threshold, output_dir_val, max_students_to_plot=30):
    """Plots Shannon Entropy for students who ever dropped below filter_threshold."""
    if not all(col in df.columns for col in ['studentId', 'TimeStep', 'shannonEntropy']):
        logger.warning("Missing required columns for individual entropy plot.")
        return

    df_sorted = df.sort_values(by=['studentId', 'TimeStep'])
    
    # Identify students who at ANY point had entropy < filter_threshold
    students_ever_low_entropy = df_sorted[df_sorted['shannonEntropy'] < filter_threshold]['studentId'].unique()

    if len(students_ever_low_entropy) == 0:
        logger.info(f"No students found with entropy ever below {filter_threshold}. Skipping individual entropy plot.")
        return

    df_filtered = df_sorted[df_sorted['studentId'].isin(students_ever_low_entropy)]
    num_students_plotted = df_filtered['studentId'].nunique()
    
    logger.info(f"Plotting individual entropy for {num_students_plotted} students (those with at least one reading < {filter_threshold}).")

    # If still too many students after filtering, take a sample
    if num_students_plotted > max_students_to_plot:
        sampled_student_ids = random.sample(list(students_ever_low_entropy), k=max_students_to_plot)
        df_filtered = df_filtered[df_filtered['studentId'].isin(sampled_student_ids)]
        num_students_plotted = df_filtered['studentId'].nunique() # Update count
        logger.info(f"  -> Sampled down to {num_students_plotted} students for clarity.")


    plt.figure(figsize=(18, 10))
    sns.lineplot(x='TimeStep', y='shannonEntropy', hue='studentId', data=df_filtered,
                 legend=(num_students_plotted <= 20), # Only show legend if few lines
                 alpha=0.7, linewidth=1.2)
    plt.title(f'Evolution of Shannon Entropy (Students with at least one reading < {filter_threshold})')
    plt.xlabel("Simulation Turn (Time Step)")
    plt.ylabel("Shannon Entropy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)
    if num_students_plotted <= 20:
        plt.legend(title='Student ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.tight_layout()
    plot_filename = os.path.join(output_dir_val, "individual_shannonEntropy_evolution_filtered.png")
    plt.savefig(plot_filename)
    logger.info(f"Saved plot: {plot_filename}")
    plt.close()

def run_quantitative_evolution_analysis(all_results_df_val, low_entropy_val, output_dir_val, community_col):
    logger.info("--- Running Quantitative Analysis of Metric Evolution ---")
    plot_metric_distribution_over_time(all_results_df_val, 'shannonEntropy', 'Evolution of Shannon Entropy Distribution', output_dir_val)
    plot_percentage_below_threshold_over_time(all_results_df_val, 'shannonEntropy', low_entropy_val,
                                               f'Percentage of Students with Entropy < {low_entropy_val} (Echo Chamber Risk)', output_dir_val)
    plot_metric_distribution_over_time(all_results_df_val, 'betweenness', 'Evolution of Betweenness Centrality Distribution', output_dir_val)
    plot_percentage_below_threshold_over_time(all_results_df_val, 'betweenness', 0.001, # Near-zero threshold
                                               'Percentage of Students with Near-Zero Betweenness', output_dir_val, lower_is_problem=True)
    plot_metric_distribution_over_time(all_results_df_val, 'pageRank', 'Evolution of PageRank Distribution', output_dir_val)
    plot_community_entropy_evolution(all_results_df_val, community_col, output_dir_val)
    plot_individual_student_entropy_over_time(all_results_df_val, low_entropy_val, output_dir_val)

# --- 3. Qualitative Analysis & Case Studies ---

def run_qualitative_case_studies(all_results_df_val, student_profiles_df_path, low_entropy_threshold_val, output_dir_val):
    logger.info("--- Running Qualitative Analysis & Case Studies ---")
    
    # Initialize return DataFrames
    low_entropy_cases_final_df = pd.DataFrame()
    bridge_node_cases_final_df = pd.DataFrame()

    try:
        student_profiles_df = pd.read_csv(student_profiles_df_path)
        student_profiles_df['studentId'] = student_profiles_df['studentId'].astype(str)
        
        # Ensure socialEngagementScore is numeric, handle potential load issues
        if 'socialEngagementScore' in student_profiles_df.columns:
            student_profiles_df['socialEngagementScore'] = pd.to_numeric(student_profiles_df['socialEngagementScore'], errors='coerce').fillna(0.5) # Default to 0.5 if unparseable
        else:
            logger.warning("socialEngagementScore column not found in student profiles. Will use default 0.5.")
            student_profiles_df['socialEngagementScore'] = 0.5


        # Parse JSON strings for loved/disliked topics and count them in the initial profiles
        for col in ['lovedTopicIds', 'dislikedTopicIds']:
            if col in student_profiles_df.columns:
                def safe_json_loads(x):
                    if pd.isna(x) or not isinstance(x, str): return []
                    try: return json.loads(x)
                    except json.JSONDecodeError: logger.debug(f"Failed to parse JSON for {col}: {x}"); return []
                student_profiles_df[col + '_list'] = student_profiles_df[col].apply(safe_json_loads)
                student_profiles_df['num_' + col.replace("TopicIds", "")] = student_profiles_df[col + '_list'].apply(len)
            else:
                logger.warning(f"Column {col} not found in student_profiles_df. Counts will be 0.")
                student_profiles_df['num_' + col.replace("TopicIds", "")] = 0
        
        # --- Data for T=0 (Initial State) ---
        initial_timestep_df = all_results_df_val[all_results_df_val['TimeStep'] == 0].copy()
        initial_timestep_df['studentId'] = initial_timestep_df['studentId'].astype(str)
        initial_data_merged_df = pd.merge(initial_timestep_df, student_profiles_df, on='studentId', how='left')

        # --- Data for T=final (Final State) ---
        final_timestep = all_results_df_val['TimeStep'].max()
        final_results_df = all_results_df_val[all_results_df_val['TimeStep'] == final_timestep].copy()
        final_results_df['studentId'] = final_results_df['studentId'].astype(str)
        final_data_merged_df = pd.merge(final_results_df, student_profiles_df, on='studentId', how='left')

        # --- Define columns to display for case studies ---
        # Use COMMUNITY_ID_COLUMN if defined globally, otherwise default
        community_col = COMMUNITY_ID_COLUMN if 'COMMUNITY_ID_COLUMN' in globals() else 'louvainCommunity'
        profile_cols_for_display = ['name', 'learningStyle', 'socialEngagementScore', 'num_loved', 'num_disliked']
        metric_cols_for_display = ['shannonEntropy', community_col, 'betweenness', 'pageRank']
        
        # Ensure all desired columns exist before trying to select them
        def filter_existing_cols(df, desired_cols):
            return [col for col in desired_cols if col in df.columns]

        display_cols_initial = ['studentId'] + filter_existing_cols(initial_data_merged_df, profile_cols_for_display + metric_cols_for_display)
        display_cols_final = ['studentId'] + filter_existing_cols(final_data_merged_df, profile_cols_for_display + metric_cols_for_display)


        num_case_studies = 7 # Number of students to show for each case

        # --- 1. Final State Analysis (Echo Chamber Candidates & Bridges) ---
        logger.info(f"\n--- Profiles of Top {num_case_studies} Students with LOWEST Shannon Entropy (T={final_timestep}) ---")
        low_entropy_cases_final_df = final_data_merged_df.nsmallest(num_case_studies, 'shannonEntropy')
        # print(low_entropy_cases_final_df[display_cols_final].to_string()) # Keep for debug if needed
        low_entropy_filename = os.path.join(output_dir_val, f"low_entropy_students_T{final_timestep}.csv")
        low_entropy_cases_final_df[display_cols_final].to_csv(low_entropy_filename, index=False)
        logger.info(f"Saved low-entropy student data to {low_entropy_filename}")

        logger.info(f"\n--- Profiles of Top {num_case_studies} Students with HIGHEST Shannon Entropy (T={final_timestep}) ---")
        highest_entropy_cases_final_df = final_data_merged_df.nlargest(num_case_studies, 'shannonEntropy')
        # print(highest_entropy_cases_final_df[display_cols_final].to_string())
        high_entropy_filename = os.path.join(output_dir_val, f"high_entropy_students_T{final_timestep}.csv")
        highest_entropy_cases_final_df[display_cols_final].to_csv(high_entropy_filename, index=False)
        logger.info(f"Saved high-entropy student data to {high_entropy_filename}")

        logger.info(f"\n--- Profiles of Top {num_case_studies} Students with HIGHEST Betweenness Centrality (Bridge Nodes at T={final_timestep}) ---")
        betweenness_threshold_final = final_data_merged_df['betweenness'].quantile(0.90) # Top 10%
        bridge_node_cases_final_df = final_data_merged_df[final_data_merged_df['betweenness'] >= betweenness_threshold_final].nlargest(num_case_studies, 'betweenness')
        # print(bridge_node_cases_final_df[display_cols_final].to_string())
        high_betweenness_filename = os.path.join(output_dir_val, f"high_betweenness_students_T{final_timestep}.csv")
        bridge_node_cases_final_df[display_cols_final].to_csv(high_betweenness_filename, index=False)
        logger.info(f"Saved high-betweenness student data to {high_betweenness_filename}")

        # --- 2. Evolution Based on Initial Profiles ---
        logger.info(f"\n--- Final Status (T={final_timestep}) of Initially LOWEST Social Engagement Students ---")
        initial_lowest_engagement = initial_data_merged_df.nsmallest(num_case_studies, 'socialEngagementScore')
        final_status_low_engage = pd.merge(initial_lowest_engagement[['studentId']], final_data_merged_df, on='studentId', how='left')
        # print(final_status_low_engage[display_cols_final].to_string())
        filename_fs_low_engage = os.path.join(output_dir_val, f"final_status_initially_lowest_engagement_T{final_timestep}.csv")
        final_status_low_engage[display_cols_final].to_csv(filename_fs_low_engage, index=False)
        logger.info(f"Saved final status of initially lowest engagement students to {filename_fs_low_engage}")

        logger.info(f"\n--- Final Status (T={final_timestep}) of Initially HIGHEST Social Engagement Students ---")
        initial_highest_engagement = initial_data_merged_df.nlargest(num_case_studies, 'socialEngagementScore')
        final_status_high_engage = pd.merge(initial_highest_engagement[['studentId']], final_data_merged_df, on='studentId', how='left')
        # print(final_status_high_engage[display_cols_final].to_string())
        filename_fs_high_engage = os.path.join(output_dir_val, f"final_status_initially_highest_engagement_T{final_timestep}.csv")
        final_status_high_engage[display_cols_final].to_csv(filename_fs_high_engage, index=False)
        logger.info(f"Saved final status of initially highest engagement students to {filename_fs_high_engage}")

        if 'num_loved' in initial_data_merged_df.columns:
            logger.info(f"\n--- Final Status (T={final_timestep}) of Students with MOST Initial Loved Topics ---")
            initial_most_loved = initial_data_merged_df.nlargest(num_case_studies, 'num_loved')
            final_status_most_loved = pd.merge(initial_most_loved[['studentId']], final_data_merged_df, on='studentId', how='left')
            # print(final_status_most_loved[display_cols_final].to_string())
            filename_fs_most_loved = os.path.join(output_dir_val, f"final_status_initially_most_loved_T{final_timestep}.csv")
            final_status_most_loved[display_cols_final].to_csv(filename_fs_most_loved, index=False)
            logger.info(f"Saved final status of students with most initial loved topics to {filename_fs_most_loved}")

        if 'num_disliked' in initial_data_merged_df.columns:
            logger.info(f"\n--- Final Status (T={final_timestep}) of Students with MOST Initial Disliked Topics ---")
            initial_most_disliked = initial_data_merged_df.nlargest(num_case_studies, 'num_disliked')
            final_status_most_disliked = pd.merge(initial_most_disliked[['studentId']], final_data_merged_df, on='studentId', how='left')
            # print(final_status_most_disliked[display_cols_final].to_string())
            filename_fs_most_disliked = os.path.join(output_dir_val, f"final_status_initially_most_disliked_T{final_timestep}.csv")
            final_status_most_disliked[display_cols_final].to_csv(filename_fs_most_disliked, index=False)
            logger.info(f"Saved final status of students with most initial disliked topics to {filename_fs_most_disliked}")

        # --- 3. Initial Status of Key Final Groups ---
        logger.info(f"\n--- Initial Status (T=0) of Students who became FINAL Bridge Nodes (Highest Betweenness at T={final_timestep}) ---")
        if not bridge_node_cases_final_df.empty:
            initial_status_of_final_bridges = pd.merge(bridge_node_cases_final_df[['studentId']], initial_data_merged_df, on='studentId', how='left')
            # print(initial_status_of_final_bridges[display_cols_initial].to_string())
            filename_is_final_bridges = os.path.join(output_dir_val, f"initial_status_of_final_bridge_nodes_T0_vs_T{final_timestep}.csv")
            initial_status_of_final_bridges[display_cols_initial].to_csv(filename_is_final_bridges, index=False)
            logger.info(f"Saved initial status of final bridge nodes to {filename_is_final_bridges}")

        logger.info(f"\n--- Initial Status (T=0) of Students who became FINAL Lowest Entropy (T={final_timestep}) ---")
        if not low_entropy_cases_final_df.empty:
            initial_status_of_final_low_entropy = pd.merge(low_entropy_cases_final_df[['studentId']], initial_data_merged_df, on='studentId', how='left')
            # print(initial_status_of_final_low_entropy[display_cols_initial].to_string())
            filename_is_final_low_entropy = os.path.join(output_dir_val, f"initial_status_of_final_low_entropy_T0_vs_T{final_timestep}.csv")
            initial_status_of_final_low_entropy[display_cols_initial].to_csv(filename_is_final_low_entropy, index=False)
            logger.info(f"Saved initial status of final low entropy students to {filename_is_final_low_entropy}")

        # --- 4. Aggregate Profile Stats for Key Final Groups ---
        logger.info(f"\n--- Aggregate Profile Stats for FINAL Low Entropy Students (T={final_timestep}) ---")
        if not low_entropy_cases_final_df.empty:
            agg_stats_low_entropy = low_entropy_cases_final_df[['socialEngagementScore', 'num_loved', 'num_disliked']].agg(['mean', 'median', 'std']).round(2)
            filename_agg_low_entropy_txt = os.path.join(output_dir_val, f"agg_stats_final_low_entropy_T{final_timestep}.txt")
            with open(filename_agg_low_entropy_txt, 'w') as f:
                f.write(agg_stats_low_entropy.to_string())
            logger.info(f"Saved aggregate stats for final low entropy students to {filename_agg_low_entropy_txt}")
            filename_agg_low_entropy_png = os.path.join(output_dir_val, f"agg_stats_final_low_entropy_T{final_timestep}.png")
            save_df_as_table_image(agg_stats_low_entropy, filename_agg_low_entropy_png, title=f"Aggregate Stats: Final Low Entropy (T={final_timestep})")

        logger.info(f"\n--- Aggregate Profile Stats for FINAL Bridge Nodes (T={final_timestep}) ---")
        if not bridge_node_cases_final_df.empty:
            agg_stats_bridge_nodes = bridge_node_cases_final_df[['socialEngagementScore', 'num_loved', 'num_disliked']].agg(['mean', 'median', 'std']).round(2)
            filename_agg_bridge_nodes_txt = os.path.join(output_dir_val, f"agg_stats_final_bridge_nodes_T{final_timestep}.txt")
            with open(filename_agg_bridge_nodes_txt, 'w') as f:
                f.write(agg_stats_bridge_nodes.to_string())
            logger.info(f"Saved aggregate stats for final bridge nodes to {filename_agg_bridge_nodes_txt}")
            filename_agg_bridge_nodes_png = os.path.join(output_dir_val, f"agg_stats_final_bridge_nodes_T{final_timestep}.png")
            save_df_as_table_image(agg_stats_bridge_nodes, filename_agg_bridge_nodes_png, title=f"Aggregate Stats: Final Bridge Nodes (T={final_timestep})")
            
        logger.info(f"\n--- Aggregate Profile Stats for FINAL High Entropy Students (T={final_timestep}) ---")
        if not highest_entropy_cases_final_df.empty: # Ensure this df is defined
            agg_stats_high_entropy = highest_entropy_cases_final_df[['socialEngagementScore', 'num_loved', 'num_disliked']].agg(['mean', 'median', 'std']).round(2)
            filename_agg_high_entropy_txt = os.path.join(output_dir_val, f"agg_stats_final_high_entropy_T{final_timestep}.txt")
            with open(filename_agg_high_entropy_txt, 'w') as f:
                f.write(agg_stats_high_entropy.to_string())
            logger.info(f"Saved aggregate stats for final high entropy students to {filename_agg_high_entropy_txt}")
            filename_agg_high_entropy_png = os.path.join(output_dir_val, f"agg_stats_final_high_entropy_T{final_timestep}.png")
            save_df_as_table_image(agg_stats_high_entropy, filename_agg_high_entropy_png, title=f"Aggregate Stats: Final High Entropy (T={final_timestep})")


    except FileNotFoundError:
        logger.error(f"Student profile file not found: {student_profiles_df_path}. Skipping case study.")
    except Exception as e:
        logger.exception(f"Error during expanded qualitative case study preparation: {e}")
    
    # Return the key DFs for potential use in log deep dive
    return low_entropy_cases_final_df, bridge_node_cases_final_df
# --- End Qualitative Analysis ---


# --- 4. Simulation Log Deep Dive ---
def run_simulation_log_deep_dive(consumed_log_file, recs_log_file, low_entropy_cases_df_val, final_ts):
    logger.info("--- Running Simulation Log Analysis (Example Queries) ---")
    try:
        consumed_log_df = pd.read_csv(consumed_log_file)
        recs_log_df = pd.read_csv(recs_log_file)
        logger.info(f"Loaded {len(consumed_log_df)} consumed logs and {len(recs_log_df)} recommendation logs.")

        if not low_entropy_cases_df_val.empty:
            example_low_entropy_student = low_entropy_cases_df_val.iloc[0]['studentId']
            logger.info(f"\n--- Example Log Analysis for Low-Entropy Student (from T={final_ts}): {example_low_entropy_student} ---")
            
            student_consumed_log = consumed_log_df[consumed_log_df['studentId'] == example_low_entropy_student]
            student_recs_log = recs_log_df[recs_log_df['studentId'] == example_low_entropy_student]

            print(f"\nConsumed by {example_low_entropy_student} (last 10 turns of activity):")
            print(student_consumed_log[['turn', 'resourceId', 'rating', 'comment']].tail(10).to_string())

            print(f"\nRecommendations received by {example_low_entropy_student} (last 10 unique in last ~20 turns):")
            # Filter for recent turns to make it more manageable
            recent_recs_log = student_recs_log[student_recs_log['turn'] > (final_ts - 20)]
            print(recent_recs_log[['turn', 'resourceId', 'reason']].drop_duplicates(subset=['resourceId']).tail(10).to_string())
        else:
            logger.info("No low-entropy students identified from final timestep case studies for detailed log analysis example.")
    except FileNotFoundError:
        logger.error(f"Simulation log files ({consumed_log_file}, {recs_log_file}) not found. Skipping.")
    except Exception as e:
        logger.exception(f"Error during simulation log analysis: {e}")


# --- Main Execution ---
def main():
    """Main function to orchestrate the simulation evolution analysis."""
    logger.info("--- Starting Analysis of Simulation Evolution ---")
    create_output_directory_if_not_exists(FIGURES_DIR)
    create_output_directory_if_not_exists(CASE_STUDY_OUTPUT_DIR) # Create directory for case study outputs

    all_results_df = load_and_consolidate_analysis_data(ANALYSIS_FILES_PATTERN, TIME_STEPS)

    if all_results_df.empty:
        logger.error("Exiting script: No analysis data loaded.")
        return

    # Run quantitative analysis (plots)
    run_quantitative_evolution_analysis(all_results_df, LOW_ENTROPY_THRESHOLD, FIGURES_DIR, COMMUNITY_ID_COLUMN)
    
    # Run qualitative analysis and get the list of low-entropy students from the final timestep
    final_timestep_val = all_results_df['TimeStep'].max()
    low_entropy_students_final, bridge_students_final = run_qualitative_case_studies(all_results_df, STUDENT_PROFILE_FILE, LOW_ENTROPY_THRESHOLD, CASE_STUDY_OUTPUT_DIR)
    
    # Run log deep dive using the identified low-entropy students
    run_simulation_log_deep_dive(CONSUMED_SIM_LOG_FILE, RECOMMENDATION_SIM_LOG_FILE, low_entropy_students_final, final_timestep_val)

    logger.info("--- Analysis Script Finished ---")

if __name__ == "__main__":
    main()