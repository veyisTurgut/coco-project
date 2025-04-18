import pandas as pd
from py2neo import Graph, Node, Relationship, Transaction, Subgraph
import os, traceback, time, math

# --- Configuration ---
# !!! UPDATE THESE DETAILS !!!
NEO4J_URI = "bolt://gcloud.madlen.io:7687"  # Your Neo4j server URI
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "madlen-is-the-future" # The password you set

# Directory containing the generated CSV files
DATA_DIR = "synthetic_data" # Or "synthetic_data" if you used the first script

CLEAR_DB_BEFORE_LOADING = True # Set to False if you want to add to existing data
BATCH_SIZE = 1000 # Process this many rows per transaction batch

# --- Helper Functions ---
def file_path(filename):
    return os.path.join(DATA_DIR, filename)

def check_files_exist():
    """Checks if all necessary CSV files exist."""
    required_files = [
        "students.csv", "topics.csv", "resources.csv",
        "consumed.csv", "resource_topic.csv"
    ]
    all_exist = True
    print("Checking for required CSV files...")
    for filename in required_files:
        path = file_path(filename)
        if not os.path.exists(path):
            print(f"  ERROR: File not found: {path}")
            all_exist = False
        else:
            print(f"  Found: {path}")
    if not all_exist:
        print("\nOne or more required CSV files are missing. Please run generate_data.py first.")
        exit()
    print("All required files found.")
    return True

def df_to_batches(df, batch_size):
    """Yields successive batch_size chunks from DataFrame."""
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size].replace({pd.NA: None, pd.NaT: None, float('nan'): None}).to_dict('records')


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
    df = pd.read_csv(filepath)
    total_rows = len(df)
    print(f"Read {total_rows} rows from {filepath}.")

    load_count = 0
    # Prepare property string for Cypher query
    # Example: ON CREATE SET n += { name: row.name, otherProp: row.otherProp }
    prop_set_parts = [f"{col}: row.{col}" for col in property_columns if col != id_column]
    prop_set_string = "{" + ", ".join(prop_set_parts) + "}" if prop_set_parts else ""
    if prop_set_string:
        prop_set_string = f" ON CREATE SET n += {prop_set_string}"

    query = f"""
    UNWIND $batch AS row
    MERGE (n:{label} {{ {id_column}: row.{id_column} }})
    {prop_set_string}
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

            # Commit periodically within large files to manage memory/transaction size
            if batch_num % 10 == 0: # Commit every 10 batches
                 tx.commit()
                 print(f"    Committed transaction at batch {batch_num}.")
                 tx = None # Start new transaction for next batch

        if tx: # Commit any remaining transaction
            tx.commit()
            print("    Committed final transaction.")

    except Exception as e:
        print(f"\nError loading nodes from {filename}: {e}")
        if tx:
            try: tx.rollback()
            except: pass # Ignore rollback errors
        raise # Re-raise the exception to stop execution

    print(f"Finished loading {label} nodes in {time.time() - start_time:.2f} seconds.\n")


def load_relationships(graph: Graph, start_node_label: str, start_id_col: str,
                       end_node_label: str, end_id_col: str,
                       rel_type: str, filename: str, property_columns: list = None):
    """Loads relationships from a CSV file in batches."""
    filepath = file_path(filename)
    print(f"Loading ':{rel_type}' relationships from {filename}...")
    start_time = time.time()
    df = pd.read_csv(filepath)
    total_rows = len(df)
    print(f"Read {total_rows} rows from {filepath}.")

    if total_rows == 0:
        print("Skipping relationship loading - file is empty.")
        return

    load_count = 0
    # Prepare property string for Cypher query
    prop_set_string = ""
    if property_columns:
        prop_set_parts = [f"{col}: row.{col}" for col in property_columns]
        if prop_set_parts:
             prop_set_string = "{" + ", ".join(prop_set_parts) + "}"
             prop_set_string = f" ON CREATE SET rel += {prop_set_string}"


    query = f"""
    UNWIND $batch AS row
    MATCH (start:{start_node_label} {{ {start_id_col}: row.{start_id_col} }})
    MATCH (end:{end_node_label} {{ {end_id_col}: row.{end_id_col} }})
    MERGE (start)-[rel:{rel_type}]->(end)
    {prop_set_string}
    """

    tx = None
    batch_num = 0
    try:
        for batch in df_to_batches(df, BATCH_SIZE):
            batch_num += 1
            if not tx: tx = graph.begin()
            tx.run(query, batch=batch)
            load_count += len(batch)
            print(f"  Processed batch {batch_num}/{math.ceil(total_rows/BATCH_SIZE)}, loaded {load_count}/{total_rows} relationships...")

            # Commit periodically
            if batch_num % 10 == 0:
                 tx.commit()
                 print(f"    Committed transaction at batch {batch_num}.")
                 tx = None

        if tx: # Commit any remaining transaction
            tx.commit()
            print("    Committed final transaction.")

    except Exception as e:
        print(f"\nError loading relationships from {filename}: {e}")
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
        # Test connection
        graph.run("RETURN 1")
        print("Successfully connected to Neo4j.")
    except Exception as e:
        traceback.print_exc()
        print(f"Failed to connect to Neo4j: {e}")
        exit()

    if CLEAR_DB_BEFORE_LOADING:
        clear_database(graph)

    create_constraints(graph)

    # --- Load Nodes ---
    load_nodes(graph, "Topic", "topics.csv", "topicId", ["name"])
    load_nodes(graph, "Resource", "resources.csv", "resourceId", ["title", "type"])
    load_nodes(graph, "Student", "students.csv", "studentId", ["name"])

    # --- Load Relationships ---
    # (Resource)-[:ABOUT_TOPIC]->(Topic)
    load_relationships(graph, "Resource", "resourceId", "Topic", "topicId",
                       "ABOUT_TOPIC", "resource_topic.csv")

    # (Student)-[:CONSUMED]->(Resource)
    load_relationships(graph, "Student", "studentId", "Resource", "resourceId",
                       "CONSUMED", "consumed.csv", ["timestamp"])


    print("\n--- Data Loading Complete ---")
    # Final Verification Counts
    print("Verifying counts in Neo4j:")
    try:
        student_count = graph.run("MATCH (n:Student) RETURN count(n) AS count").evaluate()
        topic_count = graph.run("MATCH (n:Topic) RETURN count(n) AS count").evaluate()
        resource_count = graph.run("MATCH (n:Resource) RETURN count(n) AS count").evaluate()
        consumed_count = graph.run("MATCH ()-[r:CONSUMED]->() RETURN count(r) AS count").evaluate()
        participated_count = graph.run("MATCH ()-[r:PARTICIPATED_IN]->() RETURN count(r) AS count").evaluate()

        print(f"  Student Nodes: {student_count}")
        print(f"  Topic Nodes: {topic_count}")
        print(f"  Resource Nodes: {resource_count}")
        print(f"  CONSUMED Relationships: {consumed_count}")
        print(f"  PARTICIPATED_IN Relationships: {participated_count}")
        # Add other counts if desired

    except Exception as e:
        print(f"Could not verify counts: {e}")

    print(f"\nTotal execution time: {time.time() - start_total_time:.2f} seconds.")