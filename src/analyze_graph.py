import pandas as pd
import networkx as nx
from py2neo import Graph
import community as community_louvain # Louvain algorithm
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity # Potentially useful for affinity matrix
import numpy as np
from scipy.stats import entropy as shannon_entropy
import collections
import time # To time computations

# --- Configuration ---
NEO4J_URI = "bolt://gcloud.madlen.io:7687" # Or your AuraDB URI
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "madlen-is-the-future" # Use environment variables or a config file in practice!

OUTPUT_RESULTS_FILE = "analysis_results.csv"
UPDATE_NEO4J = True # Flag to control writing results back to Neo4j

# --- Connect to Neo4j ---
try:
    graph_db = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # Test connection
    graph_db.run("MATCH (n) RETURN count(n) AS count")
    print("Successfully connected to Neo4j.")
except Exception as e:
    print(f"Failed to connect to Neo4j: {e}")
    exit()
    
    
    
def get_student_interaction_graph(graph_db):
    """
    Extracts student nodes and creates a weighted graph where edge weights
    represent shared interactions (e.g., consuming same resource or participating in same topic).
    """
    print("Extracting student interaction data from Neo4j...")

    # Get all student nodes
    students_query = "MATCH (s:Student) RETURN s.studentId AS studentId"
    students_df = graph_db.run(students_query).to_data_frame()
    if students_df.empty:
        print("No student nodes found.")
        return None
    student_nodes = students_df['studentId'].tolist()
    print(f"Found {len(student_nodes)} students.")

    # Query for shared interactions (this can be computationally intensive)
    # Option 1: Shared resource consumption
    shared_resource_query = """
    MATCH (s1:Student)-[:CONSUMED]->(r:Resource)<-[:CONSUMED]-(s2:Student)
    WHERE id(s1) < id(s2) // Avoid self-loops and duplicate pairs
    RETURN s1.studentId AS student1, s2.studentId AS student2, count(r) AS weight
    """
    # Option 2: Shared topic participation
    shared_topic_query = """
    MATCH (s1:Student)-[:PARTICIPATED_IN]->(t:Topic)<-[:PARTICIPATED_IN]-(s2:Student)
    WHERE id(s1) < id(s2)
    RETURN s1.studentId AS student1, s2.studentId AS student2, count(t) AS weight
    """
    # Option 3: Combined - Interacted with same topic (via consumption or participation)
    # This requires linking resources back to topics first.
    shared_topic_interaction_query = """
    MATCH (s1:Student)-[:CONSUMED|PARTICIPATED_IN]->(item)
    WHERE (item:Resource OR item:Topic) // Ensure item is Resource or Topic
    // Find the topic associated with the item
    WITH s1, item
    OPTIONAL MATCH (item)-[:ABOUT_TOPIC]->(t_res:Topic) // For resources
    WITH s1, COALESCE(t_res, item) AS topicNode // Use resource's topic or the topic itself
    WHERE topicNode:Topic // Ensure we have a topic node
    MATCH (topicNode)<-[:ABOUT_TOPIC|PARTICIPATED_IN]-(item2)<-[:CONSUMED|PARTICIPATED_IN]-(s2:Student)
    WHERE id(s1) < id(s2) AND item2 = topicNode // Ensure interaction is with the same topic
    RETURN s1.studentId AS student1, s2.studentId AS student2, count(DISTINCT topicNode) AS weight // Count shared topics
    """
    # Let's use Option 3 (Combined Interaction on Topic) as it's more comprehensive

    start_time = time.time()
    print("Executing shared interaction query (this might take time)...")
    # edges_df = graph_db.run(shared_resource_query).to_data_frame() # Use if only resource-based
    # edges_df = graph_db.run(shared_topic_query).to_data_frame() # Use if only topic-participation-based
    edges_df = graph_db.run(shared_topic_interaction_query).to_data_frame()
    print(f"Query took {time.time() - start_time:.2f} seconds.")

    # Build NetworkX Graph
    G = nx.Graph()
    G.add_nodes_from(student_nodes) # Add all students, even if isolated

    if not edges_df.empty:
        print(f"Adding {len(edges_df)} edges to NetworkX graph...")
        for _, row in edges_df.iterrows():
            # Ensure nodes exist before adding edge (should be guaranteed by query)
            if row['student1'] in student_nodes and row['student2'] in student_nodes:
                 G.add_edge(row['student1'], row['student2'], weight=row['weight'])
            else:
                 print(f"Warning: Skipping edge with unknown node: {row['student1']} or {row['student2']}")

    print(f"Built NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

     # Remove isolates if they are not meaningful for community detection (optional)
    # isolates = list(nx.isolates(G))
    # G.remove_nodes_from(isolates)
    # print(f"Removed {len(isolates)} isolated nodes. New size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # Ensure graph is connected for Spectral Clustering (or handle components separately)
    if not nx.is_connected(G):
        print("Warning: Graph is not connected. Spectral clustering might behave unexpectedly or apply to the largest component.")
        # Option: work on the largest connected component
        # largest_cc = max(nx.connected_components(G), key=len)
        # G = G.subgraph(largest_cc).copy()
        # print(f"Using largest connected component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    return G

# --- Main Execution Area ---
G_interaction = get_student_interaction_graph(graph_db)

if G_interaction is None or G_interaction.number_of_nodes() == 0:
    print("Exiting: No graph data to analyze.")
    exit()

results = {} # Dictionary to store results: {studentId: {metric1: val1, ...}}
for node in G_interaction.nodes():
    results[node] = {}
    
    

# --- Community Detection ---

# Louvain
print("Running Louvain community detection...")
start_time = time.time()
# Ensure the graph is not empty and has edges for Louvain
if G_interaction.number_of_edges() > 0:
    partition_louvain = community_louvain.best_partition(G_interaction, weight='weight', random_state=42)
    num_louvain_communities = len(set(partition_louvain.values()))
    print(f"Louvain found {num_louvain_communities} communities in {time.time() - start_time:.2f} seconds.")
    for node, comm_id in partition_louvain.items():
        if node in results:
             results[node]['louvainCommunity'] = comm_id
        else:
             print(f"Warning: Node {node} from Louvain not in initial results dict.") # Should not happen if G nodes match results keys
else:
    print("Skipping Louvain: Graph has no edges.")
    num_louvain_communities = 0
    for node in results: # Assign default value if no communities detected
        results[node]['louvainCommunity'] = -1 # Or None or 0

# Spectral Clustering
# Needs number of clusters. Let's estimate based on Louvain or set a fixed number.
# Note: Spectral clustering works best on connected graphs. Need adjacency matrix.
N_CLUSTERS_SPECTRAL = max(2, min(num_louvain_communities, 10)) # Example: Use Louvain's count, capped at 10, minimum 2
print(f"Running Spectral Clustering (k={N_CLUSTERS_SPECTRAL})...")
start_time = time.time()

if G_interaction.number_of_nodes() > 1 and G_interaction.number_of_edges() > 0 :
    # Get adjacency matrix
    # Note: Ensure node order is consistent for matrix and results mapping
    node_list = list(G_interaction.nodes())
    adjacency_matrix = nx.to_numpy_array(G_interaction, nodelist=node_list, weight='weight')

    # Handle disconnected graph: compute affinity matrix (e.g., using cosine similarity of something, or just use adjacency)
    # Using 'precomputed' affinity allows handling non-connected graphs better sometimes.
    # affinity_matrix = cosine_similarity(adjacency_matrix) # Example if features were available
    affinity_matrix = 'precomputed' # Or 'rbf' if using features

    # If using adjacency directly and graph might be disconnected, SpectralClustering might only cluster the largest component.
    # Using affinity='nearest_neighbors' can sometimes handle disconnected better if connectivity is sparse.
    try:
        # Try with 'precomputed' if using adjacency directly on potentially disconnected graph
        sc = SpectralClustering(N_CLUSTERS_SPECTRAL,
                                affinity='precomputed', # Use adjacency directly as similarity
                                assign_labels='kmeans', # Common assignment strategy
                                random_state=42)
        # Pass the adjacency matrix directly when affinity='precomputed'
        spectral_labels = sc.fit_predict(adjacency_matrix)

        print(f"Spectral Clustering finished in {time.time() - start_time:.2f} seconds.")
        for i, node in enumerate(node_list):
            if node in results:
                 results[node]['spectralCommunity'] = int(spectral_labels[i]) # Ensure type is Python int
            else:
                 print(f"Warning: Node {node} from Spectral Clustering not in results dict.")

    except Exception as e:
         print(f"Spectral Clustering failed: {e}. Assigning default value.")
         for node in results:
             results[node]['spectralCommunity'] = -1 # Default value on failure
else:
    print("Skipping Spectral Clustering: Not enough nodes/edges or graph disconnected and unhandled.")
    for node in results:
        results[node]['spectralCommunity'] = -1 # Default value
        
        
        
        
        

# --- Calculate Metrics ---

# Centrality Metrics (on the NetworkX graph G_interaction)
print("Calculating Centrality metrics (PageRank, Betweenness)...")
start_time = time.time()
try:
    pagerank = nx.pagerank(G_interaction, weight='weight')
    print(f"  - PageRank calculated.")
except Exception as e:
    print(f"  - PageRank calculation failed: {e}")
    pagerank = {node: 0.0 for node in G_interaction.nodes()} # Default value

try:
    # Betweenness centrality can be slow on large graphs. Consider sampling (k=...) if needed.
    betweenness = nx.betweenness_centrality(G_interaction, weight='weight', normalized=True)
    print(f"  - Betweenness Centrality calculated.")
except Exception as e:
    print(f"  - Betweenness Centrality calculation failed: {e}")
    betweenness = {node: 0.0 for node in G_interaction.nodes()} # Default value

print(f"Centrality calculations took {time.time() - start_time:.2f} seconds.")

for node in G_interaction.nodes():
    if node in results:
        results[node]['pageRank'] = pagerank.get(node, 0.0)
        results[node]['betweenness'] = betweenness.get(node, 0.0)
    else:
         print(f"Warning: Node {node} from Centrality not in results dict.")


# Shannon Entropy (requires querying Neo4j per student for topic interactions)
print("Calculating Shannon Entropy for topic diversity per student...")
start_time = time.time()
entropy_query = """
MATCH (s:Student {studentId: $studentId})
// Path 1: Consumed Resource -> Topic
OPTIONAL MATCH (s)-[:CONSUMED]->(r:Resource)-[:ABOUT_TOPIC]->(t1:Topic)
WITH s, collect(DISTINCT t1.topicId) AS consumed_topics
// Path 2: Participated in Topic
OPTIONAL MATCH (s)-[:PARTICIPATED_IN]->(t2:Topic)
WITH s, consumed_topics, collect(DISTINCT t2.topicId) AS participated_topics
// Combine and unwind topics
WITH s, consumed_topics + participated_topics AS all_topic_ids
UNWIND all_topic_ids AS topicId
// Filter NULLs *before* the final aggregation
WITH topicId
WHERE topicId IS NOT NULL
// Aggregate counts
RETURN topicId, count(*) AS interaction_count
"""

entropies = {}
processed_students = 0
total_students = len(results)
for student_id in results.keys():
    topic_counts_df = graph_db.run(entropy_query, studentId=student_id).to_data_frame()

    if not topic_counts_df.empty:
        counts = topic_counts_df['interaction_count'].values
        # Calculate probability distribution
        probabilities = counts / counts.sum()
        # Calculate Shannon Entropy
        entropies[student_id] = shannon_entropy(probabilities, base=2) # Use base 2 typically
    else:
        entropies[student_id] = 0.0 # No interactions or no topics found -> zero entropy

    processed_students += 1
    if processed_students % (total_students // 10 + 1) == 0: # Print progress
         print(f"  - Calculated entropy for {processed_students}/{total_students} students...")


print(f"Shannon Entropy calculation took {time.time() - start_time:.2f} seconds.")

for student_id, entropy_value in entropies.items():
    if student_id in results:
        results[student_id]['shannonEntropy'] = entropy_value
    else:
        print(f"Warning: Student {student_id} from Entropy not in results dict.")





        
# --- Store Results ---

# Convert results dict to DataFrame for easier CSV export
results_list = []
for student_id, metrics in results.items():
    row = {'studentId': student_id}
    row.update(metrics)
    results_list.append(row)

results_df = pd.DataFrame(results_list)
# Reorder columns for clarity
cols_order = ['studentId', 'louvainCommunity', 'spectralCommunity', 'pageRank', 'betweenness', 'shannonEntropy']
# Ensure all expected columns exist, adding missing ones with None or default values
for col in cols_order:
    if col not in results_df.columns:
        results_df[col] = None # Or suitable default like -1 for communities, 0.0 for metrics

results_df = results_df[cols_order]


# Option 1: Save to CSV
print(f"Saving analysis results to {OUTPUT_RESULTS_FILE}...")
results_df.to_csv(OUTPUT_RESULTS_FILE, index=False)
print("Results saved.")

# Option 2: Update Neo4j Node Properties
if UPDATE_NEO4J:
    print("Updating Student nodes in Neo4j with computed metrics...")
    update_query = """
    UNWIND $results_list AS row
    MERGE (s:Student {studentId: row.studentId})
    SET s.louvainCommunity = toIntegerOrNull(row.louvainCommunity), // Ensure correct types
        s.spectralCommunity = toIntegerOrNull(row.spectralCommunity),
        s.pageRank = toFloatOrNull(row.pageRank),
        s.betweenness = toFloatOrNull(row.betweenness),
        s.shannonEntropy = toFloatOrNull(row.shannonEntropy)
    """
    # Define helper functions in Cypher if needed, or handle type conversion in Python
    # Simplified approach: ensure data types are correct before sending
    # Convert DataFrame to list of dictionaries for UNWIND
    results_dict_list = results_df.replace({np.nan: None}).to_dict('records') # Replace NaN with None for Neo4j compatibility

    # Define Cypher functions for safe type conversion (run these once or ensure they exist)
    try:
        graph_db.run("""
        CREATE FUNCTION toIntegerOrNull AS (input) -> 
          CASE 
            WHEN input IS NULL THEN null
            WHEN input = apoc.convert.MISSING THEN null 
            ELSE toInteger(input) 
          END
        """)
        graph_db.run("""
        CREATE FUNCTION toFloatOrNull AS (input) -> 
          CASE 
            WHEN input IS NULL THEN null 
            WHEN input = apoc.convert.MISSING THEN null
            ELSE toFloat(input) 
          END
        """)
        print("Helper functions toIntegerOrNull/toFloatOrNull created or already exist.")
    except Exception as e:
        # Ignore if functions already exist, handle other errors
        if "already exists" not in str(e):
             print(f"Could not create helper functions (APOC might be needed or permissions issue): {e}")
             print("Proceeding without helper functions, ensure data types are correct.")
             # If functions can't be created, ensure types in results_dict_list are correct
             for row in results_dict_list:
                 row['louvainCommunity'] = int(row['louvainCommunity']) if row['louvainCommunity'] is not None else None
                 row['spectralCommunity'] = int(row['spectralCommunity']) if row['spectralCommunity'] is not None else None
                 row['pageRank'] = float(row['pageRank']) if row['pageRank'] is not None else None
                 row['betweenness'] = float(row['betweenness']) if row['betweenness'] is not None else None
                 row['shannonEntropy'] = float(row['shannonEntropy']) if row['shannonEntropy'] is not None else None


    # Batch update for performance
    batch_size = 500
    start_idx = 0
    tx = None # Initialize transaction variable
    print(f"Updating {len(results_dict_list)} nodes in batches of {batch_size}...")
    try:
        while start_idx < len(results_dict_list):
            batch = results_dict_list[start_idx : start_idx + batch_size]
            if not tx: # Start transaction if not already started
                 tx = graph_db.begin()
            tx.run(update_query, results_list=batch)
            tx.process() # Send current batch operations
            start_idx += batch_size
            print(f"  Processed batch up to index {start_idx}")
        if tx: # Commit the final transaction
            tx.commit()
        print("Neo4j update complete.")
    except Exception as e:
        print(f"Error updating Neo4j: {e}")
        if tx: # Rollback on error
             try:
                 tx.rollback()
                 print("Transaction rolled back.")
             except: # Handle cases where rollback might fail
                  pass


print("\nPhase 2 Analysis Complete.")
# You can add basic analysis here, e.g., print average entropy per Louvain community
# community_entropy = results_df.groupby('louvainCommunity')['shannonEntropy'].mean()
# print("\nAverage Shannon Entropy per Louvain Community:")
# print(community_entropy)