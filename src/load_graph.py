# src/load_graph.py
# Purpose:
# This script is responsible for populating a Neo4j graph database with initial data
# from CSV files. It handles the setup of the graph structure, including nodes and
# relationships, which form the baseline for subsequent simulations and analyses.
#
# What it does:
# 1.  Initialization and Configuration:
#     - Loads environment variables for Neo4j connection (URI, user, password).
#     - Defines configuration parameters such as the data directory (`DATA_DIR`),
#       a flag to clear the database before loading (`CLEAR_DB_BEFORE_LOADING`),
#       and the batch size for database operations (`BATCH_SIZE`).
#
# 2.  File Checks and Pre-requisites:
#     - `check_files_exist()`: Verifies the presence of required CSV files (e.g.,
#       `students.csv`, `topics.csv`, `resources.csv`, `consumed_initially.csv`,
#       `resource_topic.csv`) and optional files (e.g., `participated.csv`) in the
#       specified `DATA_DIR`. Exits if required files are missing.
#
# 3.  Database Operations:
#     - Establishes a connection to the Neo4j database.
#     - `clear_database()`: If `CLEAR_DB_BEFORE_LOADING` is true, it deletes all existing
#       nodes and relationships from the database.
#     - `create_constraints()`: Creates uniqueness constraints on `studentId` for Student nodes,
#       `resourceId` for Resource nodes, and `topicId` for Topic nodes to ensure data
#       integrity and improve query performance.
#
# 4.  Node Loading (`load_nodes()`):
#     - Loads nodes for different labels (Topic, Resource, Student) from their respective CSV files.
#     - Reads CSV data using pandas.
#     - Uses `df_to_batches()` to process data in manageable chunks.
#     - Constructs and executes Cypher `MERGE` queries to create or update nodes, setting their
#       properties. This function handles various property types, including lists/JSON strings
#       stored in CSV columns (e.g., `lovedTopicIds` for Students).
#
# 5.  Relationship Loading (`load_relationships()`):
#     - Loads relationships between nodes from CSV files.
#     - Supports different relationship types (`ABOUT_TOPIC`, `CONSUMED`, `PARTICIPATED_IN`).
#     - Reads CSV data and processes it in batches.
#     - Constructs and executes Cypher `MERGE` queries to create relationships between
#       existing nodes, setting relationship properties from CSV columns.
#     - Allows adding `extra_rel_props` (e.g., `{"source": "initial"}`) to relationships,
#       which is used to distinguish initially loaded `CONSUMED` interactions.
#
# 6.  Specific Data Loading Steps (in `if __name__ == "__main__":`):
#     - Loads Topic nodes.
#     - Loads Resource nodes.
#     - Loads Student nodes (including properties like `learningStyle`, `lovedTopicIds`,
#       `dislikedTopicIds`, `socialEngagementScore`).
#     - Loads `ABOUT_TOPIC` relationships between Resources and Topics.
#     - Loads initial `CONSUMED` relationships between Students and Resources from
#       `consumed_initially.csv`, including properties like `timestamp`, `rating`, `comment`,
#       `feedback_generated_by`, and an extra `source: "initial"` property.
#     - Loads optional `PARTICIPATED_IN` relationships between Students and Topics from
#       `participated.csv`, including `timestamp` and `interactionType`.
#
# 7.  Verification and Completion:
#     - After loading, it queries the Neo4j database to get counts of loaded nodes (Student,
#       Topic, Resource) and relationships (CONSUMED, PARTICIPATED_IN, ABOUT_TOPIC).
#     - Prints these counts to the console for verification.
#     - Logs the total execution time of the script.
#
# Key Libraries Used:
# - pandas: For reading and processing CSV files in batches.
# - py2neo: For interacting with the Neo4j graph database (connecting, running queries,
#   managing transactions).
# - dotenv: For managing environment variables (Neo4j credentials).
# - os, traceback, time, math, json, numpy: Standard Python libraries for file operations,
#   error handling, timing, JSON processing, and numerical operations (handling NaNs).

import pandas as pd
from py2neo import Graph, Node, Relationship, Transaction, Subgraph
import os, traceback, time, math, json # Added json
import numpy as np # Added numpy
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://gcloud.madlen.io:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Directory containing the generated CSV files
DATA_DIR = "synthetic_data" # Make sure this matches generate_data output

CLEAR_DB_BEFORE_LOADING = True
BATCH_SIZE = 1000

# --- Helper Functions ---
def file_path(filename):
    return os.path.join(DATA_DIR, filename)

def df_to_batches(df, batch_size):
    """Yields successive batch_size chunks from DataFrame."""
    for i in range(0, len(df), batch_size):
        # Convert potential pandas Int64 types explicitly, handle NA/NaN
        yield df.iloc[i:i + batch_size].fillna(np.nan).replace([np.nan], [None]).astype(object).where(pd.notnull(df), None).to_dict('records')


# --- File Check ---
def check_files_exist():
    """Checks if all necessary CSV files exist."""
    # Core files for loading nodes and essential relationships
    required_files = [
        "students.csv", "topics.csv", "resources.csv",
        "consumed_initially.csv", # Use the initial consumption file
        "resource_topic.csv"
    ]
    # Optional files
    optional_files = [
         "participated.csv",
         # No longer loading recommended.csv here
    ]
    all_required_exist = True
    print("Checking for required CSV files...")
    for filename in required_files:
        path = file_path(filename)
        if not os.path.exists(path):
            print(f"  ERROR: Required file not found: {path}")
            all_required_exist = False
        else:
            print(f"  Found: {path}")

    print("\nChecking for optional CSV files...")
    for filename in optional_files:
         path = file_path(filename)
         if os.path.exists(path): print(f"  Found: {path}")
         else: print(f"  Optional file not found: {path} (Will be skipped)")

    if not all_required_exist:
        print("\nOne or more required CSV files are missing. Please run generate_data.py first.")
        exit()
    print("\nFile check complete.")
    return True

# --- Main Loading Functions ---

def clear_database(graph: Graph):
    """Deletes all nodes and relationships."""
    print("Clearing existing database...")
    start_time = time.time()
    graph.run("MATCH (n) DETACH DELETE n")
    print(f"Database cleared in {time.time() - start_time:.2f} seconds.")

def create_constraints(graph: Graph):
    """Creates uniqueness constraints for node IDs."""
    print("Creating constraints...")
    start_time = time.time()
    try:
        graph.run("CREATE CONSTRAINT unique_student_id IF NOT EXISTS FOR (s:Student) REQUIRE s.studentId IS UNIQUE")
        graph.run("CREATE CONSTRAINT unique_resource_id IF NOT EXISTS FOR (r:Resource) REQUIRE r.resourceId IS UNIQUE")
        graph.run("CREATE CONSTRAINT unique_topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.topicId IS UNIQUE")
        print("Constraints created or already exist.")
    except Exception as e:
        print(f"Error creating constraints (might be okay if they already exist): {e}")
    print(f"Constraint creation took {time.time() - start_time:.2f} seconds.")

def load_nodes(graph: Graph, label: str, filename: str, id_column: str, property_columns: list):
    """Loads nodes from a CSV file in batches."""
    filepath = file_path(filename)
    print(f"Loading nodes with label ':{label}' from {filename}...")
    start_time = time.time()
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"ERROR: File not found {filepath}. Skipping node type {label}.")
        return

    total_rows = len(df)
    print(f"Read {total_rows} rows from {filepath}.")

    load_count = 0
    prop_set_parts = [f"{col}: row.{col}" for col in property_columns if col != id_column]
    prop_set_string = "{" + ", ".join(prop_set_parts) + "}" if prop_set_parts else ""
    # Using SET n += ensures properties are updated if node already exists (e.g., learning style)
    # Use ON CREATE SET n += if you only want properties added when node is new
    set_clause = f" SET n += {prop_set_string}" if prop_set_string else ""

    query = f"""
    UNWIND $batch AS row
    MERGE (n:{label} {{ {id_column}: row.{id_column} }})
    {set_clause}
    """

    tx = None
    batch_num = 0
    try:
        for batch in df_to_batches(df, BATCH_SIZE):
            batch_num += 1
            if not tx: tx = graph.begin()
            tx.run(query, batch=batch)
            load_count += len(batch)
            print(f"  Processed batch {batch_num}/{math.ceil(total_rows/BATCH_SIZE)}, loaded {load_count}/{total_rows} nodes...")

            if batch_num % 10 == 0:
                 tx.commit()
                 print(f"    Committed transaction at batch {batch_num}.")
                 tx = None

        if tx:
            tx.commit()
            print("    Committed final transaction.")

    except Exception as e:
        print(f"\nError loading nodes from {filename}: {e}")
        traceback.print_exc()
        if tx:
            try: tx.rollback()
            except: pass
        raise

    print(f"Finished loading {label} nodes in {time.time() - start_time:.2f} seconds.\n")


def load_relationships(graph: Graph, start_node_label: str, start_id_col: str,
                       end_node_label: str, end_id_col: str,
                       rel_type: str, filename: str, property_columns: list = None,
                       extra_rel_props: dict = None):
    """Loads relationships from a CSV file in batches."""
    filepath = file_path(filename)
    if not os.path.exists(filepath):
        print(f"Skipping relationship '{rel_type}': File not found {filepath}.")
        return

    print(f"Loading ':{rel_type}' relationships from {filename}...")
    start_time = time.time()
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading CSV {filepath}: {e}")
        return

    total_rows = len(df)
    print(f"Read {total_rows} rows from {filepath}.")
    if total_rows == 0:
        print("Skipping relationship loading - file is empty.")
        return

    load_count = 0
    # Prepare property string for Cypher query from CSV columns
    prop_set_parts = [f"{col}: row.{col}" for col in property_columns] if property_columns else []

    # Add extra static properties
    if extra_rel_props:
        for key, value in extra_rel_props.items():
             prop_set_parts.append(f"{key}: {json.dumps(value)}")

    prop_set_string = "{" + ", ".join(prop_set_parts) + "}" if prop_set_parts else ""
    set_clause = f" SET rel += {prop_set_string}" if prop_set_string else ""

    # Using MERGE creates relationship if it doesn't exist based on start/end nodes.
    # If you want multiple identical relationship types possible (e.g., multiple CONSUMED between same nodes), use CREATE instead of MERGE.
    # However, MERGE is safer for preventing accidental duplicates during loading.
    query = f"""
    UNWIND $batch AS row
    MATCH (start:{start_node_label} {{ {start_id_col}: row.{start_id_col} }})
    MATCH (end:{end_node_label} {{ {end_id_col}: row.{end_id_col} }})
    MERGE (start)-[rel:{rel_type}]->(end)
    {set_clause}
    """

    tx = None
    batch_num = 0
    try:
        for batch in df_to_batches(df, BATCH_SIZE):
            batch_num += 1
            if not tx: tx = graph.begin()
            try:
                tx.run(query, batch=batch)
            except Exception as run_e:
                 print(f"\nError processing batch {batch_num} for {rel_type} from {filename}: {run_e}")
                 print(f"Failed Batch Data (first row): {batch[0] if batch else 'N/A'}")
                 continue # Skip this batch

            load_count += len(batch)
            print(f"  Processed batch {batch_num}/{math.ceil(total_rows/BATCH_SIZE)}, loaded {load_count}/{total_rows} relationships...")

            if batch_num % 10 == 0:
                 tx.commit()
                 print(f"    Committed transaction at batch {batch_num}.")
                 tx = None

        if tx:
            tx.commit()
            print("    Committed final transaction.")

    except Exception as e:
        print(f"\nUnhandled error loading relationships from {filename}: {e}")
        traceback.print_exc()
        if tx:
            try: tx.rollback()
            except: pass
        raise

    print(f"Finished loading {rel_type} relationships in {time.time() - start_time:.2f} seconds.\n")


# --- Main Execution ---
if __name__ == "__main__":
    start_total_time = time.time()
    if not check_files_exist():
        exit()

    print(f"\nConnecting to Neo4j at {NEO4J_URI}...")
    try:
        graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        graph.run("RETURN 1")
        print("Successfully connected.")
    except Exception as e:
        traceback.print_exc()
        print(f"Failed to connect: {e}")
        exit()

    if CLEAR_DB_BEFORE_LOADING:
        clear_database(graph)

    create_constraints(graph)

    # --- Load Nodes ---
    load_nodes(graph, "Topic", "topics.csv", "topicId", ["name"])
    load_nodes(graph, "Resource", "resources.csv", "resourceId", ["title", "type", "modality"])
    load_nodes(graph, "Student", "students.csv", "studentId", ["name", "learningStyle", "lovedTopicIds", "dislikedTopicIds", "socialEngagementScore"])

    # --- Load Relationships ---
    load_relationships(graph, "Resource", "resourceId", "Topic", "topicId",
                       "ABOUT_TOPIC", "resource_topic.csv")

    # Load INITIAL consumption data - MARKED AS INITIAL
    load_relationships(graph, "Student", "studentId", "Resource", "resourceId",
                       "CONSUMED", "consumed_initially.csv", # Use initial filename
                       ["timestamp", "rating", "comment", "feedback_generated_by"],
                       extra_rel_props={"source": "initial"}) # Mark source

    # Load OPTIONAL participation data
    load_relationships(graph, "Student", "studentId", "Topic", "topicId",
                       "PARTICIPATED_IN", "participated.csv", ["timestamp", "interactionType"])

    # NO LONGER LOADING recommended.csv

    print("\n--- Data Loading Complete ---")
    # --- Final Verification Counts ---
    print("Verifying counts in Neo4j:")
    try:
        counts = {}
        counts['Student'] = graph.run("MATCH (n:Student) RETURN count(n) AS c").evaluate()
        counts['Topic'] = graph.run("MATCH (n:Topic) RETURN count(n) AS c").evaluate()
        counts['Resource'] = graph.run("MATCH (n:Resource) RETURN count(n) AS c").evaluate()
        counts['CONSUMED'] = graph.run("MATCH ()-[r:CONSUMED]->() RETURN count(r) AS c").evaluate()
        counts['PARTICIPATED_IN'] = graph.run("MATCH ()-[r:PARTICIPATED_IN]->() RETURN count(r) AS c").evaluate()
        counts['ABOUT_TOPIC'] = graph.run("MATCH ()-[r:ABOUT_TOPIC]->() RETURN count(r) AS c").evaluate()
        # Add count for LLM_RECOMMENDED if needed for verification, but not loaded here
        # counts['LLM_RECOMMENDED'] = graph.run("MATCH ()-[r:LLM_RECOMMENDED]->() RETURN count(r) AS c").evaluate()

        print(f"  Nodes:")
        print(f"    Student: {counts.get('Student', 0)}")
        print(f"    Topic: {counts.get('Topic', 0)}")
        print(f"    Resource: {counts.get('Resource', 0)}")
        print(f"  Relationships:")
        print(f"    CONSUMED (Initial): {counts.get('CONSUMED', 0)}") # This count includes ONLY initial load
        print(f"    PARTICIPATED_IN (Initial): {counts.get('PARTICIPATED_IN', 0)}")
        print(f"    ABOUT_TOPIC: {counts.get('ABOUT_TOPIC', 0)}")
        # print(f"    LLM_RECOMMENDED (Simulated): {counts.get('LLM_RECOMMENDED', 0)}") # Will be 0 after initial load

    except Exception as e:
        print(f"Could not verify counts: {e}")

    print(f"\nTotal execution time: {time.time() - start_total_time:.2f} seconds.")