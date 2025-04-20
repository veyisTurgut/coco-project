import pandas as pd
import networkx as nx
import community as community_louvain
import numpy as np
import time
import os
import json # For parsing profile lists
import logging
from dotenv import load_dotenv
from py2neo import Graph
from scipy.stats import entropy as shannon_entropy
from sklearn.cluster import SpectralClustering
# Optional: For sentiment analysis

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not found. Sentiment analysis will be skipped. (pip install textblob)")

load_dotenv()

# --- Logger Setup ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# --- Configuration ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://gcloud.madlen.io:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

OUTPUT_RESULTS_FILE = "analysis_results_rich.csv" # New output name
UPDATE_NEO4J = False # Keep False for now unless explicitly needed
RUN_PROFILE_ANALYSIS = True # Flag to run the new analyses
RUN_SENTIMENT_ANALYSIS = TEXTBLOB_AVAILABLE # Run only if library is available

# --- Neo4j Connection ---
graph_db = None
try:
    graph_db = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    graph_db.run("RETURN 1")
    logger.info(f"Successfully connected to Neo4j at {NEO4J_URI}")
except Exception as e:
    logger.exception(f"Failed to connect to Neo4j: {e}")
    exit()
# ---

def get_student_interaction_graph_corrected(graph_db):
    """
    Corrected query: Extracts student nodes and creates a weighted graph based on
    shared TOPIC interactions (via CONSUMED->Resource->Topic OR PARTICIPATED_IN->Topic).
    """
    logger.info("Extracting student interaction data from Neo4j (Corrected Query)...")
    students_query = "MATCH (s:Student) RETURN s.studentId AS studentId"
    students_df = graph_db.run(students_query).to_data_frame()
    if students_df.empty: logger.error("No student nodes found."); return None
    student_nodes = students_df['studentId'].tolist()
    logger.info(f"Found {len(student_nodes)} students.")

    # Corrected query combining consumption and participation paths to the same topic
    shared_topic_interaction_query = """
    MATCH (s1:Student)-[:CONSUMED|PARTICIPATED_IN]->(item1)
    // Get topic for item1
    WITH s1, CASE WHEN item1:Resource THEN [(item1)-[:ABOUT_TOPIC]->(t) | t][0] ELSE item1 END AS topic1
    WHERE topic1:Topic // Ensure we got a topic
    // Find s2 interacting with the same topic
    MATCH (s2:Student)-[:CONSUMED|PARTICIPATED_IN]->(item2) WHERE id(s1) < id(s2)
    // Get topic for item2
    WITH s1, topic1, s2, CASE WHEN item2:Resource THEN [(item2)-[:ABOUT_TOPIC]->(t) | t][0] ELSE item2 END AS topic2
    WHERE topic1 = topic2 // The core condition: same topic
    // Return pair and count of shared topics (which is just 1 per match pair here, aggregate later)
    RETURN s1.studentId AS student1, s2.studentId AS student2, topic1.topicId AS sharedTopicId
    """
    # We need to aggregate the weights in pandas/networkx after getting pairs
    start_time = time.time()
    logger.info("Executing corrected shared interaction query...")
    try:
        edges_data = graph_db.run(shared_topic_interaction_query).to_data_frame()
        logger.info(f"Query successful, took {time.time() - start_time:.2f} seconds. Found {len(edges_data)} interaction pairs.")
    except Exception as e:
         logger.error(f"Error running shared interaction query: {e}")
         return None # Cannot proceed if query fails

    # Aggregate weights (count how many topics each pair shares)
    if not edges_data.empty:
        # Group by student pair and count distinct shared topics
        edge_weights = edges_data.groupby(['student1', 'student2']).sharedTopicId.nunique().reset_index()
        edge_weights.rename(columns={'sharedTopicId': 'weight'}, inplace=True)
        logger.info(f"Aggregated weights for {len(edge_weights)} unique student pairs.")
    else:
        edge_weights = pd.DataFrame(columns=['student1', 'student2', 'weight']) # Empty DF if no edges
        logger.warning("No shared topic interactions found between students.")


    # Build NetworkX Graph
    G = nx.Graph()
    G.add_nodes_from(student_nodes) # Add all students

    if not edge_weights.empty:
        logger.info(f"Adding {len(edge_weights)} edges to NetworkX graph...")
        for _, row in edge_weights.iterrows():
            G.add_edge(row['student1'], row['student2'], weight=int(row['weight'])) # Ensure weight is int

    logger.info(f"Built NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    if G.number_of_edges() > 0 and not nx.is_connected(G):
        logger.warning("Graph is not connected. Analysis might focus on components.")
        # Optional: Use largest component?
        # largest_cc = max(nx.connected_components(G), key=len)
        # G = G.subgraph(largest_cc).copy()
        # logger.info(f"Using largest connected component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    return G

# --- Analysis Functions ---

def run_community_detection(G: nx.Graph, results: dict):
    """Performs Louvain and Spectral clustering."""
    num_louvain_communities = 0
    # Louvain
    logger.info("Running Louvain community detection...")
    start_time = time.time()
    if G.number_of_edges() > 0:
        partition_louvain = community_louvain.best_partition(G, weight='weight', random_state=42)
        num_louvain_communities = len(set(partition_louvain.values()))
        logger.info(f"Louvain found {num_louvain_communities} communities in {time.time() - start_time:.2f}s.")
        for node, comm_id in partition_louvain.items():
            if node in results: results[node]['louvainCommunity'] = comm_id
    else:
        logger.warning("Skipping Louvain: Graph has no edges.")
        for node in results: results[node]['louvainCommunity'] = -1

    # Spectral Clustering
    N_CLUSTERS_SPECTRAL = max(2, min(num_louvain_communities, 10)) if num_louvain_communities > 0 else 2
    logger.info(f"Running Spectral Clustering (k={N_CLUSTERS_SPECTRAL})...")
    start_time = time.time()
    if G.number_of_nodes() > 1 and G.number_of_edges() > 0 :
        try:
            node_list = list(G.nodes())
            adjacency_matrix = nx.to_numpy_array(G, nodelist=node_list, weight='weight')
            sc = SpectralClustering(N_CLUSTERS_SPECTRAL, affinity='precomputed', assign_labels='kmeans', random_state=42)
            spectral_labels = sc.fit_predict(adjacency_matrix)
            logger.info(f"Spectral Clustering finished in {time.time() - start_time:.2f}s.")
            for i, node in enumerate(node_list):
                if node in results: results[node]['spectralCommunity'] = int(spectral_labels[i])
        except Exception as e:
             logger.error(f"Spectral Clustering failed: {e}. Assigning default.")
             for node in results: results[node]['spectralCommunity'] = -1
    else:
        logger.warning("Skipping Spectral Clustering: Not enough nodes/edges.")
        for node in results: results[node]['spectralCommunity'] = -1


def run_centrality_metrics(G: nx.Graph, results: dict):
    """Calculates PageRank and Betweenness Centrality."""
    logger.info("Calculating Centrality metrics (PageRank, Betweenness)...")
    start_time = time.time()
    if G.number_of_edges() > 0: # Avoid errors on empty graph
        try: pagerank = nx.pagerank(G, weight='weight'); logger.info("  - PageRank calculated.")
        except Exception as e: logger.error(f"  - PageRank failed: {e}"); pagerank = {}
        try: betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True); logger.info("  - Betweenness calculated.")
        except Exception as e: logger.error(f"  - Betweenness failed: {e}"); betweenness = {}
    else:
        logger.warning("Skipping centrality: Graph has no edges.")
        pagerank = {}; betweenness = {}

    logger.info(f"Centrality calculations took {time.time() - start_time:.2f}s.")
    for node in results:
        results[node]['pageRank'] = pagerank.get(node, 0.0)
        results[node]['betweenness'] = betweenness.get(node, 0.0)


def run_shannon_entropy(graph_db: Graph, results: dict):
    """Calculates Shannon Entropy for topic diversity per student."""
    logger.info("Calculating Shannon Entropy for topic diversity per student...")
    start_time = time.time()
    # Combined query for both consumed and participated topics
    entropy_query = """
    MATCH (s:Student {studentId: $studentId})
    // Path 1: Consumed Resource -> Topic
    OPTIONAL MATCH (s)-[:CONSUMED]->(r:Resource)-[:ABOUT_TOPIC]->(t1:Topic)
    WITH s, collect(DISTINCT t1.topicId) AS consumed_topic_ids
    // Path 2: Participated in Topic
    OPTIONAL MATCH (s)-[:PARTICIPATED_IN]->(t2:Topic)
    WITH s, consumed_topic_ids, collect(DISTINCT t2.topicId) AS participated_topic_ids
    // Combine, unwind, filter nulls, count
    WITH consumed_topic_ids + participated_topic_ids AS all_topic_ids
    UNWIND all_topic_ids AS topicId
    WITH topicId WHERE topicId IS NOT NULL
    RETURN topicId, count(*) AS interaction_count
    """
    entropies = {}
    total_students = len(results)
    processed_students = 0
    for student_id in results.keys():
        try:
            topic_counts_df = graph_db.run(entropy_query, studentId=student_id).to_data_frame()
            if not topic_counts_df.empty and topic_counts_df['interaction_count'].sum() > 0:
                counts = topic_counts_df['interaction_count'].values
                probabilities = counts / counts.sum()
                entropies[student_id] = shannon_entropy(probabilities, base=2)
            else: entropies[student_id] = 0.0
        except Exception as e:
            logger.error(f"Error calculating entropy for {student_id}: {e}")
            entropies[student_id] = -1.0 # Error indicator

        processed_students += 1
        if processed_students % (max(1, total_students // 10)) == 0:
            logger.info(f"  - Calculated entropy for {processed_students}/{total_students} students...")

    logger.info(f"Shannon Entropy calculation took {time.time() - start_time:.2f}s.")
    for student_id, entropy_value in entropies.items():
        if student_id in results: results[student_id]['shannonEntropy'] = entropy_value


# --- NEW: Profile and Feedback Analysis Functions ---
def fetch_full_interaction_data(graph_db: Graph) -> pd.DataFrame:
    """Fetches detailed interaction data for analysis."""
    logger.info("Fetching detailed interaction data (students, resources, topics, consumed)...")
    query = """
    MATCH (s:Student)-[c:CONSUMED]->(r:Resource)
    OPTIONAL MATCH (r)-[:ABOUT_TOPIC]->(t:Topic)
    RETURN s.studentId AS studentId,
           s.learningStyle AS learningStyle,
           s.lovedTopicIds AS lovedTopicIds,    // Keep as JSON string for now
           s.dislikedTopicIds AS dislikedTopicIds, // Keep as JSON string for now
           r.resourceId AS resourceId,
           r.modality AS modality,
           t.topicId AS topicId,
           t.name AS topicName,
           c.rating AS rating,
           c.comment AS comment,
           c.timestamp AS timestamp,
           c.feedback_generated_by AS feedback_source // Include source if available
    ORDER BY s.studentId, c.timestamp
    """
    try:
        df = graph_db.run(query).to_data_frame()
        # Convert rating to numeric, coercing errors
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        # Parse JSON lists - handle potential errors
        for col in ['lovedTopicIds', 'dislikedTopicIds']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else [])
        logger.info(f"Fetched {len(df)} CONSUMED interactions with profile data.")
        return df
    except Exception as e:
        logger.error(f"Error fetching detailed interaction data: {e}")
        return pd.DataFrame() # Return empty DataFrame on error


def analyze_profile_correlations(interactions_df: pd.DataFrame):
    """Performs basic correlation analysis."""
    if interactions_df.empty or 'rating' not in interactions_df.columns:
        logger.warning("Skipping profile correlation analysis: No interaction data or ratings available.")
        return
    print(interactions_df)
    logger.info("\n--- Profile Correlation Analysis ---")

    # 1. Rating by Learning Style / Modality Match
    logger.info("Average Rating by Learning Style:")
    print(interactions_df.groupby('learningStyle')['rating'].agg(['mean', 'count']).round(2))

    # Create a modality match flag
    def modality_match(row):
        style = row['learningStyle']
        modality = row['modality']
        if not style or not modality or style == 'Unknown' or modality == 'Unknown' or style == 'Mixed':
            return 'Unknown/Mixed'
        elif style == modality:
            return 'Match'
        else:
            return 'Mismatch'
    interactions_df['modalityMatch'] = interactions_df.apply(modality_match, axis=1)
    logger.info("\nAverage Rating by Modality Match:")
    print(interactions_df.groupby('modalityMatch')['rating'].agg(['mean', 'count']).round(2))

    # 2. Rating by Topic Affinity
    def topic_affinity(row):
        topic = row['topicId']
        loved = row['lovedTopicIds']
        disliked = row['dislikedTopicIds']
        if not topic or not isinstance(loved, list) or not isinstance(disliked, list): return 'Unknown'
        if topic in loved: return 'Loved'
        if topic in disliked: return 'Disliked'
        return 'Neutral'
    interactions_df['topicAffinity'] = interactions_df.apply(topic_affinity, axis=1)
    logger.info("\nAverage Rating by Topic Affinity:")
    print(interactions_df.groupby('topicAffinity')['rating'].agg(['mean', 'count']).round(2))
    logger.info("----------------------------------\n")


def analyze_comment_sentiment(interactions_df: pd.DataFrame):
    """Performs basic sentiment analysis on comments."""
    if not RUN_SENTIMENT_ANALYSIS:
        logger.info("Skipping sentiment analysis (TextBlob not available or disabled).")
        return
    if interactions_df.empty or 'comment' not in interactions_df.columns:
        logger.warning("Skipping sentiment analysis: No interaction data or comments available.")
        return

    logger.info("\n--- Comment Sentiment Analysis (Basic) ---")
    interactions_df['comment_sentiment'] = interactions_df['comment'].apply(
        lambda x: TextBlob(x).sentiment.polarity if pd.notna(x) else None
    )
    # Filter out rows where sentiment couldn't be calculated
    valid_sentiment_df = interactions_df.dropna(subset=['comment_sentiment', 'rating'])

    if not valid_sentiment_df.empty:
        logger.info("Average Sentiment Polarity by Rating:")
        print(valid_sentiment_df.groupby('rating')['comment_sentiment'].agg(['mean', 'count']).round(3))

        # Correlation between rating and sentiment
        correlation = valid_sentiment_df['rating'].corr(valid_sentiment_df['comment_sentiment'])
        logger.info(f"\nCorrelation between Rating and Comment Sentiment Polarity: {correlation:.3f}")
    else:
        logger.warning("No valid comments with ratings found for sentiment analysis.")

    logger.info("----------------------------------------\n")

# --- Main Execution Area ---
if __name__ == "__main__":
    logger.info("Starting Rich Data Analysis...")
    analysis_start_time = time.time()

    # --- Graph Structure Analysis ---
    G_interaction = get_student_interaction_graph_corrected(graph_db)
    if G_interaction is None: exit()

    # Initialize results dict
    results = {node: {} for node in G_interaction.nodes()}
    run_community_detection(G_interaction, results)
    run_centrality_metrics(G_interaction, results)
    run_shannon_entropy(graph_db, results)
    # --- Fetch Detailed Data for Profile/Feedback Analysis ---
    if RUN_PROFILE_ANALYSIS or RUN_SENTIMENT_ANALYSIS:
        interactions_df = fetch_full_interaction_data(graph_db)
    else:
        interactions_df = pd.DataFrame() # Empty if not needed

    # --- Run New Analyses ---
    if RUN_PROFILE_ANALYSIS:
        analyze_profile_correlations(interactions_df)

    if RUN_SENTIMENT_ANALYSIS:
        analyze_comment_sentiment(interactions_df)

    # --- Store Student-Level Results ---
    logger.info("Preparing student-level results CSV...")
    results_list = []
    for student_id, metrics in results.items():
        row = {'studentId': student_id}
        row.update(metrics)
        results_list.append(row)
    results_df = pd.DataFrame(results_list)
    cols_order = ['studentId', 'louvainCommunity', 'spectralCommunity', 'pageRank', 'betweenness', 'shannonEntropy']
    for col in cols_order:
        if col not in results_df.columns: results_df[col] = None
    results_df = results_df[cols_order]

    logger.info(f"Saving student-level analysis results to {OUTPUT_RESULTS_FILE}...")
    results_df.to_csv(OUTPUT_RESULTS_FILE, index=False)
    logger.info("Results saved.")

    # --- (Neo4j Update Section Removed as UPDATE_NEO4J is False) ---

    logger.info(f"\nAnalysis Complete. Total time: {time.time() - analysis_start_time:.2f} seconds.")