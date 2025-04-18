# Project Handover: AI-Powered Learning Recommendations

**Version:** 1.0

## 1. Project Overview

This project focuses on developing an **AI-powered recommendation engine** for a simulated 9th-grade learning environment. The core idea is to leverage rich student profiles and interaction histories to generate personalized learning resource recommendations using a Large Language Model (LLM).

The system involves:

1.  Generating synthetic data representing students with specific learning styles and topic affinities (likes/dislikes).
2.  Simulating initial student interactions with learning resources, including ratings and textual comments.
3.  Storing this rich data in a Neo4j graph database running via Docker.
4.  Providing a mechanism to format student profile information and interaction history into a detailed prompt suitable for an LLM.
5.  (Future Work) Integrating an LLM to process the prompt and generate tailored recommendations.

## 2. Technology Stack

*   **Language:** Python 3.10+ (Developed with 3.13)
*   **Graph Database:** Neo4j (Version 5.x recommended, running via Docker)
*   **Core Python Libraries:**
    *   `pandas`: Data manipulation and CSV handling.
    *   `numpy`: Numerical operations.
    *   `py2neo`: Python driver for interacting with Neo4j.
    *   `faker`: Generating synthetic names and comments.
    *   `json`: Handling list storage within CSVs for profile data.
    *   *(Anticipated for LLM Integration)*: `openai`, `anthropic`, `requests`, or other relevant LLM client libraries.
*   **Containerization:** Docker (for running Neo4j)
*   **Cloud Platform (Deployment):** Google Cloud Platform (GCP) VM (Ubuntu) used for hosting the Neo4j Docker container.

## 3. Environment Setup

### 3.1. Python Environment

1.  **Clone the Repository:** Get the project code.
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # Linux/macOS
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` includes `pandas`, `numpy`, `py2neo`, `faker`, `json`)*

### 3.2. Neo4j Setup (Docker on Server)

This project uses Neo4j running in a Docker container, typically hosted on a remote server (like GCP).

1.  **Ensure Docker is installed** on the server (`sudo apt install docker.io` on Ubuntu).
2.  **Create Host Directories (Recommended for Persistence):**
    ```bash
    # On the server hosting Docker
    mkdir -p $HOME/neo4j/data
    mkdir -p $HOME/neo4j/logs
    mkdir -p $HOME/neo4j/conf
    ```
3.  **Set Password:** Choose a strong password for the `neo4j` user.
    ```bash
    # On the server hosting Docker
    export NEO4J_PASSWORD="YOUR_CHOSEN_PASSWORD"
    ```
4.  **Run Neo4j Container:**
    ```bash
    # On the server hosting Docker
    docker run \
        --name neo4j-server \
        -d \
        -p 7474:7474 -p 7687:7687 \
        -v $HOME/neo4j/data:/data \
        -v $HOME/neo4j/logs:/logs \
        -v $HOME/neo4j/conf:/var/lib/neo4j/conf \
        -e NEO4J_AUTH=neo4j/${NEO4J_PASSWORD} \
        neo4j:latest # Or specify a version like neo4j:5.18.0
    ```
    *   **Password Note:** The `-e NEO4J_AUTH` flag only sets the *initial* password if the `/data` volume is empty. If data exists from previous runs, the password stored in the data volume is used. See Section 8 (Known Issues) if you encounter authentication problems.

### 3.3. Firewall Configuration (Remote Neo4j Server)

Ensure the firewall on the server hosting Neo4j (e.g., GCP firewall rules) allows incoming TCP traffic on:

*   **Port 7687:** For Bolt connections (used by `py2neo`).
*   **Port 7474:** For HTTP access to Neo4j Browser.

Restrict Source IP ranges (e.g., your IP `/32`) for better security.

## 4. Directory Structure (Example)

```
coco-project/
├── .venv/                  # Python virtual environment (ignored by git)
├── synthetic_data/    # Output of generate_data.py
│   ├── students.csv
│   ├── topics.csv
│   ├── resources.csv
│   ├── consumed.csv
│   └── resource_topic.csv
├── src/                    # Source code directory
│   ├── generate_data.py   # Rich profile data generation script
│   ├── load_graph.py         # Script to load CSV data into Neo4j
│   └── recommender_engine.py        # Contains recommend_to_student prompt generation function
│   └── analyze_graph.py        # analyzed graph data and runs algorithms
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 5. Workflow / How to Run

**Step 1: Generate Synthetic Data**

1.  Review/modify parameters in `src/generate_data.py` (e.g., `NUM_STUDENTS`, `NUM_RESOURCES`). Ensure `OUTPUT_DIR` is set to `synthetic_data`.
2.  Run the script:
    ```bash
    python src/generate_data.py
    ```
3.  CSVs will be generated in `synthetic_data/`.

**Step 2: Load Data into Neo4j**

1.  **Configure `src/load_graph.py`:**
    *   Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` to your Neo4j instance details.
    *   Ensure `DATA_DIR` points to `synthetic_data`.
    *   Set `CLEAR_DB_BEFORE_LOADING = True` for clean runs.
2.  **Run the script:**
    ```bash
    python src/load_graph.py
    ```
3.  **Verify:** Check script output. Connect via Neo4j Browser (`http://<neo4j_host>:7474`) to confirm data presence (e.g., `MATCH (s:Student) RETURN s LIMIT 5;`).

**Step 3: Prepare Data & Generate Recommender Prompt**

1.  **Understand `src/recommender_engine.py`:** It contains `recommend_to_student`, which formats data into an LLM prompt.
2.  **Integrate Data Fetching:** Before calling `recommend_to_student`, you need code to:
    *   Load resource/topic lookup maps (`resource_info_map`, `topic_id_map`) from CSVs or query Neo4j.
    *   For a target student ID, fetch their profile data (`learningStyle`, `lovedTopics`, `dislikedTopics`) and recent interaction `history` (consumed resources with ratings/comments) from Neo4j or the loaded DataFrames.
    *   Assemble this into the `student_data` dictionary structure expected by the function.
3.  **Generate Prompt:** Call `recommend_to_student` with the prepared `student_data` and lookup maps.
    ```python
    # Example (Conceptual - needs actual data fetching)
    # student_data = fetch_student_data_from_neo4j(student_id)
    # prompt = recommend_to_student(student_data, resource_info_map, topic_id_map)
    # print(prompt)
    ```
4.  *(Example)* Run the `if __name__ == "__main__":` block in `recommender_engine.py` (if present) to see a sample prompt generated using data loaded directly from CSVs.

**Step 4: (Future Work) LLM Integration**

1.  **Initialize LLM Client:** Set up your chosen LLM library (e.g., `openai`).
2.  **Send Prompt:** Pass the `prompt` generated in Step 3 to the LLM's API.
3.  **Parse Response:** Implement logic to parse the LLM's text response and extract the structured recommendations (Resource IDs, justifications).

## 6. Component Details

### 6.1. Data Generation (`generate_data.py`)

*   **Goal:** Create realistic synthetic data for powering the LLM recommender_engine.
*   **Features:**
    *   **Students:** Have `studentId`, `name`, `learningStyle` (e.g., 'Visual', 'Audio'), `lovedTopicIds`, `dislikedTopicIds` (stored as JSON lists in CSV).
    *   **Resources:** Have `resourceId`, `title`, `type`, `modality` (e.g., 'Visual', 'Audio', derived from type).
    *   **Topics:** Based on a 9th-grade curriculum structure (Physics, Bio, Chem, Math).
    *   **Interactions (`consumed.csv`):** Simulates initial `CONSUMED` events with `rating` (influenced by topic affinity) and `comment` (generated based on rating, style match, affinity).
*   **Output:** CSV files in `synthetic_data/`.

### 6.2. Data Loading (`load_graph.py`)

*   Connects to Neo4j using `py2neo`.
*   Optionally clears the database.
*   Creates uniqueness constraints (`studentId`, `resourceId`, `topicId`).
*   Loads nodes (`Student`, `Topic`, `Resource`) and relationships (`ABOUT_TOPIC`, `CONSUMED`) from the `synthetic_data` CSVs in batches using `UNWIND ... MERGE`.
*   **Note:** Ensure this script is updated to handle loading all properties from `generate_data.py` (e.g., `learningStyle` on Student, `modality` on Resource, `rating`/`comment` on `CONSUMED`).

### 6.3. Recommender Prompt Generation (`recommender_engine.py`)

*   **Function: `recommend_to_student`**
*   **Purpose:** Constructs a detailed, formatted prompt for an LLM based on student data.
*   **Inputs:** `student_data` dictionary (profile & history), `resource_info_map`, `topic_id_map`.
*   **Process:** Formats profile details (style, affinities) and recent interaction history (ratings, comments) into a clear summary. Embeds this within a larger prompt instructing the LLM to act as an AI Tutor and provide N recommendations with justifications, following specific guidelines (alignment, exploration, format).
*   **Output:** Returns the prompt string, ready to be sent to an LLM.

## 7. Configuration

*   **Neo4j Credentials:** Must be correctly set in `load_graph.py` (and any other script connecting to Neo4j). Use environment variables or a config file for better practice.
*   **Data Directories:** Paths in scripts (`DATA_DIR`) must point to the correct location (`synthetic_data`).
*   **Generation Parameters:** Tunable parameters are at the top of `generate_data.py`.

## 8. Current Status & Known Issues

*   **Data Generation:** `generate_data.py` produces rich profile/interaction data.
*   **Data Loading:** `load_graph.py` loads data into Neo4j. Ensure it handles all new properties from v2 data.
*   **Recommender:** Prompt generation logic exists in `recommend_to_student`. **LLM integration and response parsing are not implemented.**
*   **Password Management (Docker):** The `NEO4J_AUTH` environment variable only sets the password on the *first* run with an empty data volume. If the password is forgotten or mismatched later, it needs to be reset using `neo4j-admin` against the persistent volume (see Section 3.2 note and previous troubleshooting).

## 9. Next Steps / Future Work

1.  **Implement Data Fetching:** Write Python code (likely using `py2neo`) to query Neo4j and retrieve the necessary `student_data` (profile, history) for a given student ID to feed into the recommender function.
2.  **Implement LLM Recommender:**
    *   Integrate an LLM client (VERTEX AI).
    *   Send the generated prompt to the LLM.
    *   Parse the LLM's response to extract resource IDs and justifications.
3.  **Refine Recommender:** Iterate on the prompt and potentially the data fetching based on the quality and relevance of the LLM's recommendations.
4.  **Evaluation:** Define metrics or methods to evaluate the quality of the generated recommendations (e.g., relevance, diversity, alignment with profile).
5.  **(Optional) Explore Graph Features:** Consider if graph metrics (e.g., topic centrality, student interaction diversity/entropy calculated from Neo4j) could be added as further inputs to the recommender prompt or used in post-processing LLM results.