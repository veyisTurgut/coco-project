# src/echo_chamber_detection.py
# Purpose:
# This script analyzes network metrics from a pre-computed CSV file (typically an output
# from `analyze_graph.py` and potentially processed by `relabel_communities.py`)
# to identify potential echo chambers and key structural nodes within a student interaction network.
# It focuses on community-level diversity and individual student characteristics.
#
# What it does:
# 1.  Data Loading:
#     - Reads a CSV file (e.g., `analysis_snapshots_relabelled/analysis_results_T<timestep>.csv`)
#       into a pandas DataFrame. This file is expected to contain columns such as
#       `studentId`, `louvainCommunity`, `shannonEntropy`, and `betweenness`.
#
# 2.  Community-Level Metrics Calculation:
#     - Filters out nodes that are not assigned to a valid community (if an identifier like '-1' is used).
#     - Groups the data by the `louvainCommunity` identifier.
#     - For each community, it calculates and prints:
#       - `avg_entropy`: The average Shannon entropy of students within that community, indicating
#         the mean topic diversity.
#       - `std_entropy`: The standard deviation of Shannon entropy, showing variability in diversity.
#       - `avg_betweenness`: The average betweenness centrality of students in the community.
#       - `size`: The number of students in the community.
#
# 3.  Potential Echo Chamber Identification (Community-Level):
#     - Based on the calculated community metrics, it identifies communities where the
#       `avg_entropy` falls below a predefined `low_entropy_threshold`.
#     - Prints these communities, as they might represent echo chambers due to low overall
#       topic diversity among their members.
#
# 4.  Low Topic Diversity Identification (Individual-Level):
#     - Filters and prints individual students whose `shannonEntropy` is below the
#       `low_entropy_threshold`, highlighting students with narrow topic engagement regardless
#       of their community's average.
#
# 5.  Bridge Node Analysis:
#     - Identifies potential "bridge nodes" by selecting students whose `betweenness` centrality
#       is above a certain threshold (e.g., the 90th percentile of all betweenness scores).
#     - Prints these high-betweenness students, as they may play a crucial role in connecting
#       different parts of the network or different communities.
#     - Further analyzes if any of the previously identified Louvain communities lack such
#       high-betweenness bridge nodes, potentially indicating isolated communities.
#
# 6.  Output:
#     - All analyses and identified lists (community metrics, potential echo chambers,
#       low-entropy students, bridge nodes, communities lacking bridges) are printed to the console.
#
# Key Libraries Used:
# - pandas: For data loading, manipulation, aggregation, and filtering.

import pandas as pd

results_df = pd.read_csv('analysis_snapshots_relabelled/analysis_results_T100.csv')

# Ensure community is treated as categorical/string for grouping if needed
results_df['louvainCommunity'] = results_df['louvainCommunity'].astype(str)

# Filter out nodes not assigned to a community if using -1
valid_communities = results_df[results_df['louvainCommunity'] != '-1']

if not valid_communities.empty:
    community_metrics = valid_communities.groupby('louvainCommunity').agg(
        avg_entropy=('shannonEntropy', 'mean'),
        std_entropy=('shannonEntropy', 'std'),
        avg_betweenness=('betweenness', 'mean'),
        size=('studentId', 'count')
    ).round(3)

    print("\n--- Community Level Metrics ---")
    print(community_metrics)

    # Identify potential echo chambers (example threshold)
    low_entropy_threshold = 2.0 # Adjust this threshold based on your data range
    potential_echo_chambers = community_metrics[community_metrics['avg_entropy'] < low_entropy_threshold]
    if not potential_echo_chambers.empty:
        print(f"\nPotential Echo Chamber Communities (Avg Entropy < {low_entropy_threshold}):")
        print(potential_echo_chambers)
    else:
        print(f"\nNo communities found with average entropy below {low_entropy_threshold}.")
    print("-------------------------------\n")

else:
    print("No valid community assignments found for analysis.")
    
print()
print()
print()
print("-"*50)
print("Individual Low Entropy Nodes")
    

low_entropy_students = results_df[results_df['shannonEntropy'] < low_entropy_threshold] # Use same threshold or a different one

if not low_entropy_students.empty:
    print(f"\n--- Students with Low Topic Diversity (Entropy < {low_entropy_threshold}) ---")
    # Sort by entropy and show key metrics
    print(low_entropy_students[['studentId', 'shannonEntropy', 'louvainCommunity', 'betweenness']].sort_values('shannonEntropy'))
    print(f"Found {len(low_entropy_students)} students with low entropy.")
    print("---------------------------------------------------------\n")
else:
    print(f"No students found with entropy below {low_entropy_threshold}.")
        
print()
print()
print()
print("-"*50)
print("Bridge Node Analysis")
    
high_betweenness_threshold = results_df['betweenness'].quantile(0.90) # Example: Top 10%
bridge_nodes = results_df[results_df['betweenness'] >= high_betweenness_threshold]

if not bridge_nodes.empty:
    print("\n--- Potential Bridge Nodes (High Betweenness Centrality) ---")
    print(bridge_nodes[['studentId', 'betweenness', 'louvainCommunity', 'shannonEntropy']].sort_values('betweenness', ascending=False))
    print("-----------------------------------------------------------\n")

    # Check if any communities LACK bridge nodes
    if 'community_metrics' in locals(): # Check if previous analysis ran
         communities_with_bridges = set(bridge_nodes['louvainCommunity'].astype(str).unique())
         all_communities = set(community_metrics.index.astype(str).unique())
         communities_without_bridges = all_communities - communities_with_bridges
         if communities_without_bridges:
             print(f"Communities potentially lacking strong bridges: {communities_without_bridges}")

else:
    print("No nodes found with high betweenness centrality based on threshold.")
    

"""
first results:


--- Community Level Metrics ---
                  avg_entropy  std_entropy  avg_betweenness  size
louvainCommunity                                                 
0                       2.641        0.422            0.022    16
1                       2.809        0.466            0.010    15
2                       2.660        0.391            0.016    19

No communities found with average entropy below 2.0.
-------------------------------




--------------------------------------------------
Individual Low Entropy Nodes

--- Students with Low Topic Diversity (Entropy < 2.0) ---
   studentId  shannonEntropy louvainCommunity  betweenness
3      S0034        1.918296                0     0.052723
43     S0024        1.918296                2     0.036149
49     S0030        1.918296                1     0.021785
2      S0033        1.921928                2     0.048125
27     S0008        1.921928                0     0.072699
25     S0006        1.950212                2     0.029665
Found 6 students with low entropy.
---------------------------------------------------------




--------------------------------------------------
Bridge Node Analysis

--- Potential Bridge Nodes (High Betweenness Centrality) ---
   studentId  betweenness louvainCommunity  shannonEntropy
27     S0008     0.072699                0        1.921928
3      S0034     0.052723                0        1.918296
2      S0033     0.048125                2        1.921928
32     S0013     0.045921                0        2.500000
5      S0036     0.037831                1        2.321928
-----------------------------------------------------------

"""




"""
t =20
--- Community Level Metrics ---
                  avg_entropy  std_entropy  avg_betweenness  size
louvainCommunity                                                 
0                       3.569        0.231            0.014    16
1                       3.520        0.254            0.011    17
2                       3.337        0.478            0.038    17

No communities found with average entropy below 2.0.
-------------------------------




--------------------------------------------------
Individual Low Entropy Nodes
No students found with entropy below 2.0.



--------------------------------------------------
Bridge Node Analysis

--- Potential Bridge Nodes (High Betweenness Centrality) ---
   studentId  betweenness louvainCommunity  shannonEntropy
41     S0042     0.150489                2        2.251629
48     S0049     0.124542                2        3.000000
49     S0050     0.102791                2        3.095795
12     S0013     0.081273                2        3.459432
9      S0010     0.072633                2        2.321928
-----------------------------------------------------------

Communities potentially lacking strong bridges: {'1', '0'}
"""