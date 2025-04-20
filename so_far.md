# Project Summary: AI-Powered Learning Recommendations & Simulation

**Version:** 2.0

## 1. Project Overview

This project implements and simulates an **AI-powered recommendation engine** for a personalized 9th-grade learning environment. It leverages rich student profiles, interaction histories stored in a Neo4j graph database, and Google's Vertex AI (Gemini) Large Language Model (LLM) to generate recommendations.

The system includes:

1.  **Data Generation:** Creates synthetic initial data for students (learning styles, topic affinities), resources (type, modality), and initial interactions (`CONSUMED` with LLM-generated feedback, `PARTICIPATED_IN`).
2.  **Graph Database:** Uses Neo4j (via Docker) to store student profiles, resources, topics, and evolving interaction relationships.
3.  **LLM Recommender Engine:** Fetches student data from Neo4j, constructs detailed prompts (including candidate resources to prevent hallucination), and uses an LLM to generate personalized resource recommendations.
4.  **Dynamic Simulation:** Simulates student interactions over time. Students receive LLM recommendations, make choices based on their profile, consume resources, provide LLM-generated feedback (ratings/comments), and potentially participate in related topics. This updates the Neo4j graph dynamically.
5.  **Adaptive Recommendation:** The simulation incorporates logic to detect low topic diversity (via Shannon Entropy) for a student and adapt the LLM prompt to encourage more exploratory recommendations.
6.  **Data Analysis:** Includes tools to analyze the graph structure (communities, centrality), individual topic diversity (entropy), profile-behavior correlations.

## 2. Technology Stack

*   **Language:** Python 3.10+
*   **Graph Database:** Neo4j (Version 5.x recommended, running via Docker)
*   **LLM Service:** Google Vertex AI (Gemini 2.5 Flash Preview via custom `curl` wrapper)
*   **Core Python Libraries:**
    *   `pandas`: Data manipulation and CSV handling/logging.
    *   `numpy`: Numerical operations.
    *   `py2neo`: Python driver for interacting with Neo4j.
    *   `faker`: Generating synthetic names.
    *   `json`: Handling list storage/parsing & LLM output.
    *   `python-dotenv`: Loading environment variables.
    *   `networkx`: Graph analysis (Communities, Centrality).
    *   `python-louvain` (community): Louvain community detection.
    *   `scikit-learn`: Spectral clustering.
    *   `scipy`: Shannon entropy calculation.
    *   *(Internal LLM Lib)*: Custom classes in `src/llm/` (e.g., `CurlVertex`, `LLMConfig`) for Vertex AI interaction.
*   **Containerization:** Docker (for Neo4j)
*   **Cloud Platform:** Google Cloud Platform (GCP) for hosting Neo4j and Vertex AI.

## 3. Environment Setup

### 3.1. Python Environment

1.  **Clone Repository.**
2.  **Create Virtual Environment:** `python -m venv .venv && source .venv/bin/activate` (or equivalent).
3.  **Install Dependencies:** `pip install -r requirements.txt`.

### 3.2. Neo4j Setup (Docker on Server)

1.  **Ensure Docker is installed** on the server.
2.  **Create Host Directories:** `mkdir -p $HOME/neo4j/{data,logs,conf}`.
3.  **Set Neo4j Password:** `export NEO4J_PASSWORD="YOUR_NEO4J_PASSWORD"`.
4.  **Run Container:**
    ```bash
    docker run --name neo4j-server -d -p 7474:7474 -p 7687:7687 \
        -v $HOME/neo4j/data:/data \
        -v $HOME/neo4j/logs:/logs \
        -v $HOME/neo4j/conf:/var/lib/neo4j/conf \
        -e NEO4J_AUTH=neo4j/${NEO4J_PASSWORD} \
        neo4j:latest
    ```
    *(See Section 8 for password reset if needed)*.
5.  **Configure Server Firewall:** Allow TCP traffic on ports 7474 (HTTP) and 7687 (Bolt).

### 3.3. GCP / Vertex AI Setup

1.  **GCP Project:** Ensure access to a GCP project with Vertex AI API enabled.
2.  **Authentication:** A service account key file `vertex_keys.json` (in project root) is expected by the LLM code (`src/llm/curl_vertex.py`).
3.  **Region:** LLM calls default to `us-central1`. Verify or update in `LLMConfig`.

## 4. Directory Structure

```
coco-project/
├── .venv/
├── src/
│   ├── llm/                  # LLM interaction code (curl_vertex.py, mylib.py)
│   ├── analyze_graph.py      # Graph/Profile/Sentiment Analysis script
│   ├── generate_data.py      # Generates initial rich data + participation
│   ├── load_graph.py         # Loads initial data into Neo4j
│   ├── recommender_engine.py # Core recommendation logic & LLM calls
│   └── simulation.py         # Runs the dynamic interaction simulation
├── synthetic_data/         # Output of generate_data.py
│   ├── consumed_initially.csv
│   ├── participated.csv
│   ├── resource_topic.csv
│   ├── resources.csv
│   ├── students.csv
│   └── topics.csv
├── simulation_outputs/     # Output logs from simulation.py
│   ├── llm_recommendations_log.csv
│   └── consumed_in_simulation.csv
├── .gitignore
├── analysis_results_rich.csv # Output of analyze_graph.py (student metrics)
├── analysis_results_initial.csv # Output of analyze_graph.py on initial data (before simulations)
├── proposal.md             # Original project proposal
├── readme.md               # This file
├── requirements.txt
└── vertex_keys.json        # GCP Service Account Key
```

## 5. Workflow / How to Run

**Step 1: Generate Initial Synthetic Data**

1.  Configure parameters (student count, etc.) in `src/generate_data.py`.
2.  Ensure Vertex AI auth (`vertex_keys.json`) is set up, as this script uses the LLM for initial feedback.
3.  Run: `python src/generate_data.py`. Output CSVs (including `consumed_initially.csv` and `participated.csv`) go to `synthetic_data/`. This step involves many LLM calls and may take time.

**Step 2: Load Initial Data into Neo4j**

1.  Configure Neo4j credentials in `.env` (used by `load_graph.py`).
2.  Ensure `DATA_DIR` in `load_graph.py` points to `synthetic_data`.
3.  Run: `python src/load_graph.py`. This populates Neo4j with the initial state.

**Step 3: Run the Dynamic Simulation**

1.  Configure simulation parameters (turns, students per turn, diversity threshold) in `src/simulation.py`.
2.  Ensure Neo4j and Vertex AI credentials are correctly set in `.env` / `vertex_keys.json`.
3.  Run: `python src/simulation.py`.
    *   This loop fetches student data from Neo4j.
    *   Calculates topic entropy.
    *   Generates recommendations (potentially adapted) via `recommender_engine.py`.
    *   Logs all LLM recommendations (`[:LLM_RECOMMENDED]` in Neo4j, `llm_recommendations_log.csv`).
    *   Simulates student choice.
    *   Generates feedback for chosen items using the LLM.
    *   Persists simulated consumption (`[:CONSUMED]` with `source='simulation'` in Neo4j, `consumed_in_simulation.csv`).
    *   Persists simulated participation (`[:PARTICIPATED_IN]` in Neo4j).
4.  Creates the `simulation_outputs/` directory for logs.

**Step 4: Analyze Data**

1.  Configure Neo4j credentials in `.env` (used by `analyze_graph.py`).
2.  Ensure Vertex AI auth is available if running sentiment analysis.
3.  Run: `python src/analyze_graph.py`.
    *   Builds student-student graph based on *all* `CONSUMED` and `PARTICIPATED_IN` data in Neo4j (initial + simulated).
    *   Calculates communities, centrality, and entropy per student.
    *   Fetches detailed interaction data.
    *   Performs profile correlation analysis (Rating vs Style/Affinity/Modality).
    *   Performs LLM-based sentiment analysis on comments.
    *   Saves student-level metrics to `analysis_results_rich.csv`.
    *   Prints correlation/sentiment analysis to console.

## 6. Component Details

*   **`src/generate_data.py`:** Creates the initial dataset (`synthetic_data/`). Uses LLM (via parallel batching) to generate realistic ratings/comments for `consumed_initially.csv`. Also probabilistically creates initial `participated.csv`.
*   **`src/load_graph.py`:** Loads all initial CSVs from `synthetic_data/` into Neo4j. Creates nodes (`Student`, `Resource`, `Topic`) and relationships (`ABOUT_TOPIC`, `CONSUMED` with `source='initial'`, `PARTICIPATED_IN`).
*   **`src/recommender_engine.py`:**
    *   Contains core functions: `connect_neo4j`, `fetch_lookup_maps_from_neo4j`, `fetch_student_data_from_neo4j`.
    *   `recommend_to_student`: Takes student data, builds a detailed prompt including candidate resources and optional adaptation instructions, calls `chat_with_llm`.
    *   `chat_with_llm`: Handles interaction with Vertex AI Gemini, requests and parses JSON output.
*   **`src/simulation.py`:**
    *   Orchestrates the multi-turn simulation loop.
    *   Uses `recommender_engine.py` for recommendations.
    *   Includes `calculate_topic_entropy`, `simulate_student_choice`.
    *   Uses `chat_with_llm` via `generate_student_feedback_llm` to simulate student feedback.
    *   Includes `add_consumed_interaction_to_neo4j`, `add_participation_to_neo4j`, `add_llm_recommendation_to_neo4j` to update the graph.
    *   Logs recommendations and simulated consumption to separate CSV files in `simulation_outputs/`.
*   **`src/analyze_graph.py`:**
    *   Performs analysis on the *current state* of the Neo4j graph (initial + simulated data).
    *   Builds student-student interaction graph based on shared topics.
    *   Calculates graph metrics (communities, centrality) and student topic entropy.
    *   Fetches detailed data to perform profile/rating correlations and LLM-based sentiment analysis on comments.
    *   Outputs student metrics CSV and prints other analyses.
*   **`src/llm/`:** Contains utility code for Vertex AI API interaction.

## 7. Configuration

*   **`.env` file:** Should define `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`. Used by `load_graph.py`, `simulation.py`, `analyze_graph.py`.
*   **`vertex_keys.json`:** GCP service account key for Vertex AI authentication. Used by any script calling `chat_with_llm`. **Add to `.gitignore`**.
*   **Script Parameters:** Tunable parameters (number of students/turns, probabilities, thresholds, batch sizes, worker counts) are located near the top of `generate_data.py`, `simulation.py`, and `analyze_graph.py`.

## 8. Current Status & Known Issues

*   **Core Loop:** Data generation, loading, simulation (including recommendation, choice, feedback, persistence, adaptation), and analysis components are implemented.
*   **LLM Integration:** Used for initial feedback generation, simulated feedback during the run, recommendations, and sentiment analysis. Quality depends on prompts and model capabilities.
*   **Performance:** Initial data generation (LLM feedback) and sentiment analysis (LLM) use parallel batching for efficiency but can still be time-consuming. Simulation loop processes students sequentially per turn.
*   **Password Management (Docker):** `NEO4J_AUTH` env var only sets initial password. Reset using `docker run --rm -v ... neo4j:latest neo4j-admin set-initial-password <newpass>` if needed.
*   **Model Tuning:** Student choice model (`simulate_student_choice`) is basic; could be refined. LLM prompts might require further tuning for optimal results.

## 9. Visualizations and Evaluation Metrics

* 


## 9. Next Steps / Future Work

1.  **Refine Models:** Improve the `simulate_student_choice` model. Tune LLM prompts for feedback and recommendations. Adjust `DIVERSITY_THRESHOLD` for adaptation.
2.  **Evaluation Metrics:** Define specific metrics to quantitatively evaluate the simulation (e.g., average entropy change, recommendation alignment score, echo chamber metrics).
3.  **Visualization:** Create visualizations (e.g., using Neo4j Bloom, Gephi, or Python libraries) of the graph structure, communities, and metric changes over simulation turns.


