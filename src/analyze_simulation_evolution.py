import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Define the paths to your analysis result files and other data
ANALYSIS_FILES_PATTERN = "analysis_snapshots/analysis_results_T{}.csv" # e.g., analysis_results_T00.csv
TIME_STEPS = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
STUDENT_PROFILE_FILE = "synthetic_data/students.csv" # Path to your student profiles
CONSUMED_SIM_LOG_FILE = "simulation_outputs/consumed_in_simulation.csv"
RECOMMENDATION_SIM_LOG_FILE = "simulation_outputs/llm_recommendations_log.csv"

# Echo chamber definition threshold
LOW_ENTROPY_THRESHOLD = 2.0 # Adjust as needed

# --- 1. Data Consolidation & Preparation ---
def load_and_consolidate_analysis_data(pattern, time_steps):
    """Loads multiple analysis CSVs and concatenates them with a TimeStep column."""
    all_dfs = []
    for ts in time_steps:
        filename = pattern.format(f"{ts:02d}") # Ensures T00, T20, etc.
        try:
            df = pd.read_csv(filename)
            df['TimeStep'] = ts
            all_dfs.append(df)
            logger.info(f"Loaded and processed {filename}")
        except FileNotFoundError:
            logger.warning(f"File not found: {filename}. Skipping this timestep.")
    if not all_dfs:
        logger.error("No analysis files were loaded. Exiting.")
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

logger.info("--- 1. Consolidating Analysis Data ---")
all_results_df = load_and_consolidate_analysis_data(ANALYSIS_FILES_PATTERN, TIME_STEPS)

if all_results_df.empty:
    exit()

# --- 2. Quantitative Analysis of Evolution ---
logger.info("\n--- 2. Quantitative Analysis of Metric Evolution ---")

def plot_metric_distribution_over_time(df, metric_column, title):
    """Plots the distribution of a metric over time steps using boxplots."""
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='TimeStep', y=metric_column, data=df)
    plt.title(title)
    plt.xlabel("Simulation Turn (Time Step)")
    plt.ylabel(metric_column)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{metric_column}_evolution.png") # Save the plot
    logger.info(f"Saved plot: {metric_column}_evolution.png")
    plt.show()

def plot_percentage_below_threshold_over_time(df, metric_column, threshold, title, lower_is_problem=True):
    """Plots the percentage of students meeting a threshold condition over time."""
    summary_data = []
    for ts in df['TimeStep'].unique():
        subset_df = df[df['TimeStep'] == ts]
        if lower_is_problem:
            count_problem = len(subset_df[subset_df[metric_column] < threshold])
        else: # Higher is problem (not typical for entropy/betweenness in this context)
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
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{metric_column}_percentage_threshold.png")
    logger.info(f"Saved plot: {metric_column}_percentage_threshold.png")
    plt.show()

# Analyze Shannon Entropy
plot_metric_distribution_over_time(all_results_df, 'shannonEntropy', 'Evolution of Shannon Entropy Distribution')
plot_percentage_below_threshold_over_time(all_results_df, 'shannonEntropy', LOW_ENTROPY_THRESHOLD,
                                           f'Percentage of Students with Entropy < {LOW_ENTROPY_THRESHOLD} (Echo Chamber Risk)')

# Analyze Betweenness Centrality
plot_metric_distribution_over_time(all_results_df, 'betweenness', 'Evolution of Betweenness Centrality Distribution')
# To show concentration, we could plot top N or % with zero betweenness
plot_percentage_below_threshold_over_time(all_results_df, 'betweenness', 0.001, # Near zero
                                           'Percentage of Students with Near-Zero Betweenness',
                                           lower_is_problem=True)


# Analyze PageRank
plot_metric_distribution_over_time(all_results_df, 'pageRank', 'Evolution of PageRank Distribution')

# Community Dynamics (Average Intra-Community Entropy)
logger.info("\nCalculating Average Intra-Community Entropy...")
community_entropy_data = []
for ts in all_results_df['TimeStep'].unique():
    ts_df = all_results_df[all_results_df['TimeStep'] == ts]
    # Ensure community is treated as categorical for grouping
    ts_df['louvainCommunity'] = ts_df['louvainCommunity'].astype(str)
    valid_communities = ts_df[ts_df['louvainCommunity'] != '-1'] # Filter unassigned if any
    if not valid_communities.empty:
        avg_comm_entropy = valid_communities.groupby('louvainCommunity')['shannonEntropy'].mean().reset_index()
        avg_comm_entropy['TimeStep'] = ts
        community_entropy_data.append(avg_comm_entropy)

if community_entropy_data:
    community_entropy_df = pd.concat(community_entropy_data)
    plt.figure(figsize=(12, 7))
    sns.lineplot(x='TimeStep', y='shannonEntropy', hue='louvainCommunity', data=community_entropy_df, marker='o', palette='tab10')
    plt.title('Evolution of Average Shannon Entropy per Louvain Community')
    plt.xlabel("Simulation Turn (Time Step)")
    plt.ylabel("Average Shannon Entropy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Louvain Community')
    plt.savefig("community_entropy_evolution.png")
    logger.info("Saved plot: community_entropy_evolution.png")
    plt.show()
else:
    logger.warning("No community data to plot average entropy.")


# --- 3. Qualitative Analysis & Case Studies ---
logger.info("\n--- 3. Preparing for Qualitative Analysis & Case Studies ---")

try:
    student_profiles_df = pd.read_csv(STUDENT_PROFILE_FILE)
    # Merge with final timestep data for case studies
    final_timestep = all_results_df['TimeStep'].max()
    final_results_df = all_results_df[all_results_df['TimeStep'] == final_timestep]
    case_study_df = pd.merge(final_results_df, student_profiles_df, on='studentId', how='left')

    # Parse JSON strings for loved/disliked topics
    for col in ['lovedTopicIds', 'dislikedTopicIds']:
        if col in case_study_df.columns:
            case_study_df[col] = case_study_df[col].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else [])

    logger.info(f"\n--- Low-Entropy Students (Echo Chamber Candidates at T={final_timestep}) ---")
    low_entropy_cases = case_study_df[case_study_df['shannonEntropy'] < LOW_ENTROPY_THRESHOLD].sort_values('shannonEntropy')
    print(low_entropy_cases[['studentId', 'name', 'learningStyle', 'shannonEntropy', 'louvainCommunity', 'betweenness', 'lovedTopicIds', 'dislikedTopicIds']].to_string())

    logger.info(f"\n--- High-Betweenness Students (Bridge Nodes at T={final_timestep}) ---")
    # Define high betweenness (e.g., top 10% or fixed threshold)
    betweenness_threshold = case_study_df['betweenness'].quantile(0.90)
    bridge_node_cases = case_study_df[case_study_df['betweenness'] >= betweenness_threshold].sort_values('betweenness', ascending=False)
    print(bridge_node_cases[['studentId', 'name', 'learningStyle', 'shannonEntropy', 'louvainCommunity', 'betweenness', 'lovedTopicIds', 'dislikedTopicIds']].to_string())

except FileNotFoundError:
    logger.error(f"Student profile file not found: {STUDENT_PROFILE_FILE}. Skipping case study profile integration.")
except Exception as e:
    logger.error(f"Error during case study preparation: {e}")


# --- 4. Simulation Log Deep Dive (Example Snippets) ---
logger.info("\n--- 4. Preparing for Simulation Log Analysis (Example Queries) ---")

try:
    consumed_log_df = pd.read_csv(CONSUMED_SIM_LOG_FILE)
    recs_log_df = pd.read_csv(RECOMMENDATION_SIM_LOG_FILE)
    logger.info(f"Loaded {len(consumed_log_df)} consumed simulation logs and {len(recs_log_df)} recommendation logs.")

    # Example: Analyze a specific low-entropy student from 'low_entropy_cases'
    if not low_entropy_cases.empty:
        example_low_entropy_student = low_entropy_cases.iloc[0]['studentId']
        logger.info(f"\n--- Example Log Analysis for Low-Entropy Student: {example_low_entropy_student} ---")

        student_consumed_log = consumed_log_df[consumed_log_df['studentId'] == example_low_entropy_student]
        student_recs_log = recs_log_df[recs_log_df['studentId'] == example_low_entropy_student]

        print(f"\nConsumed by {example_low_entropy_student} (last 10):")
        print(student_consumed_log[['turn', 'resourceId', 'rating', 'comment']].tail(10).to_string())

        print(f"\nRecommendations received by {example_low_entropy_student} (last 10 unique):")
        # Showing unique recommendations to see what was offered
        print(student_recs_log[['turn', 'resourceId', 'reason']].drop_duplicates(subset=['resourceId']).tail(10).to_string())

        # To check if adaptive recommender triggered for this student:
        # You'd need to join this with the 'needs_adapt' flag logged during simulation.py (if you logged it)
        # Or, re-calculate entropy per turn for this student from consumed_log_df and see when it dropped.
    else:
        logger.info("No low-entropy students identified for detailed log analysis example.")

except FileNotFoundError:
    logger.error("Simulation log files not found. Skipping log analysis examples.")
except Exception as e:
    logger.error(f"Error during simulation log analysis preparation: {e}")

logger.info("\n--- Analysis Script Finished ---")