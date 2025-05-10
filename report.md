## Project Report: Detecting Echo Chambers and Bias in AI-Powered Learning Communities

**Date:** 22 April 2025

**Prepared By:** Adalet Veyis Turgut

### 1. Introduction & Research Questions

**1.1. Overview:**
This project simulates an AI-driven personalized learning environment modeled as a dynamic computer network. Students, educational resources, and topics are represented as nodes, with interactions (consumption, participation, AI recommendations) forming relationships. The primary goal is to analyze the emergent social network structure among students based on their interactions, focusing on cooperation, competition (manifesting as potential information inequality), and the formation or prevention of echo chambers over time. An LLM-powered recommendation engine with an adaptive diversity mechanism is central to the simulation.

**1.2. Research Questions:**
This project wants to investigate:

1.  How does the **structure of the student interaction network** (communities, centrality distribution) evolve over time under the influence of an LLM-based recommender system?
2.  Does the simulated learning environment, incorporating personalized recommendations and student preferences, lead to the formation of **echo chambers**, characterized by low interaction diversity (Shannon Entropy) and network fragmentation?
3.  What is the role of **student profiles** (learning styles, initial topic affinities) in determining a student's position within the network structure and their topic diversity?
4.  How effective is an **adaptive recommendation mechanism**, triggered by low student diversity, in promoting broader topic engagement and mitigating potential echo chamber effects within the simulation?

### 2. Methodology Overview

The project employs a multi-stage approach combining synthetic data generation, graph database modeling, LLM integration, dynamic simulation, and social network analysis:

1.  **Synthetic Data Generation (`src/generate_data.py`):**
    *   Creates an initial population of students with assigned learning styles and topic affinities (loved/disliked subjects across STEM and Humanities).
    *   Generates resources covering various topics, tagged with modalities (Visual, Audio, etc.).
    *   Simulates initial student interactions: `CONSUMED` relationships (Student -> Resource) seeded with realistic ratings and comments generated via **parallelized batch LLM calls** based on profile/resource alignment.
    *   Probabilistically generates initial `PARTICIPATED_IN` relationships (Student -> Topic) based on consumption.
    *   Outputs data to CSV files in `synthetic_data/`.

2.  **Neo4j Data Model & Loading (`src/load_graph.py`):**
    *   Uses Neo4j (graph database) to store the network state.
    *   Nodes: `Student` (with profile properties), `Resource` (with modality), `Topic`.
    *   Relationships: `ABOUT_TOPIC` (Resource->Topic), `CONSUMED` (Student->Resource, properties: rating, comment, timestamp, source='initial'), `PARTICIPATED_IN` (Student->Topic, properties: timestamp, interactionType).
    *   Loads the initial state from CSVs into Neo4j.

3.  **LLM Recommender Engine (`src/recommender_engine.py`):**
    *   Fetches student profile and recent interaction history from Neo4j.
    *   Fetches current aggregate metrics for resources (`avgRating`, `ratingCount`, `recommendationCount`) from Neo4j.
    *   Constructs a detailed prompt for the LLM, including student context, a list of **candidate resources** (unconsumed items with their metrics) to prevent hallucination, and instructions considering profile alignment, resource metrics, and exploration.
    *   Includes an **adaptive mechanism**: If triggered (by low entropy signal from simulation), modifies the prompt to explicitly instruct the LLM to promote topic diversity.
    *   Calls the LLM via a custom wrapper (`src/llm/`) requesting structured JSON output.

4.  **Dynamic Simulation (`src/simulation.py`):**
    *   Runs for a set number of turns (`NUM_TURNS`).
    *   In each turn, for a sample of students:
        *   Fetches current state from Neo4j.
        *   Calculates current topic diversity (Shannon Entropy).
        *   Determines if adaptation is needed based on entropy threshold.
        *   Calls `recommender_engine.py` to get LLM recommendations (logging all suggestions to CSV and `[:LLM_RECOMMENDED]` in Neo4j).
        *   Simulates student choice based on recommendations and profile biases.
        *   If a resource is chosen:
            *   Generates student feedback (rating, comment) using an **LLM call** simulating the student's reaction.
            *   Persists the `[:CONSUMED]` interaction to Neo4j (with `source='simulation'`) and logs to CSV.
            *   Probabilistically persists a `[:PARTICIPATED_IN]` interaction to Neo4j.
    *   Uses **parallel processing** (`ThreadPoolExecutor`) for LLM recommendation and feedback calls across students within a turn for efficiency.

5.  **Network Analysis (`src/analyze_graph.py`):**
    *   Analyzes the state of the Neo4j graph *after* a simulation run (or on the initial data).
    *   Builds a weighted **student-student interaction graph** where edges represent shared topic engagement (via consumption or participation).
    *   Applies **Social Network Analysis (SNA)** algorithms:
        *   Community Detection: **Louvain Modularity** and **Spectral Clustering**.
        *   Centrality Measures: **PageRank** (influence) and **Betweenness Centrality** (bridging).
    *   Calculates **Shannon Entropy** per student (topic diversity).
    *   Performs correlation analysis between student profiles and interaction ratings.
    *   Outputs student-level metrics to CSV and prints aggregate analyses.

### 3. Simulation Setup (for 40-Turn Results Presented)

*   **Total Turns:** `NUM_TURNS = 40`
*   **Students Simulated Per Turn:** `STUDENTS_PER_TURN = 30` -- maybe too high
*   **Diversity Adaptation Threshold:** `DIVERSITY_THRESHOLD = 2.0` (Adaptation triggered if Shannon Entropy < 2.0)
*   **Participation Probability (Post-Consumption):** `PROBABILITY_PARTICIPATE_AFTER_CONSUME = 0.05`
*   **LLM Concurrency:** `MAX_WORKERS = 30`
*   **Initial Data:** Generated using `generate_data.py` with LLM feedback and initial participation probability (`PROBABILITY_INITIAL_PARTICIPATION = 0.05`).
*   **Curriculum:** Included PHY, BIO, CHM, MAT, HIS, GEO, LIT subjects (35 topics total). 210 resources total.

### 4. Network Analysis Results & Evolution (T=0 vs T=40)

*(Note: T=0 refers to analysis on initial data; T=40 refers to analysis after 40 simulation turns)*

1.  **Community Structure:**
    *   **T=0:** Both Louvain and Spectral methods identified 3-4 initial clusters, often showing disagreement, suggesting weak initial community boundaries based primarily on profile affinities expressed in limited initial interactions.
    *   **T=40:** A clearer structure with 4 distinct communities (per Louvain) emerged. Community membership changed significantly from T=0, indicating dynamic regrouping based on simulated interactions. Communities showed alignment with broad subject domains (e.g., Physical Sci/Math, Life Sci, Humanities/Geo).
    *   *Insight:* The simulation allowed communities based on interaction behavior to form and solidify over time, influenced by, but not identical to, initial profile preferences.

2.  **Interaction Diversity (Shannon Entropy):**
    *   **T=0:** Moderate entropy across students (range ~1.5-3.7), suggesting reasonable initial diversity from the generator.
    *   **T=40:** **Significant divergence.** While the *average* entropy increased (many students exploring more broadly, reaching values > 4.0), a notable subset of students developed **very low entropy** (min 0.91, several < 1.6).
    *   *Insight:* The simulation resulted in **individual echo chamber formation** for students likely starting with narrow preferences. However, the system *also* supported increased exploration for others, possibly due to the adaptive recommender or inherent network effects.

3.  **Betweenness Centrality (Brokerage):**
    *   **T=0:** Generally low betweenness values across the network.
    *   **T=40:** **Dramatic concentration.** A very small number of students (S0042, S0013, S0045, S0029, S0050) emerged as **critical bridge nodes** with extremely high betweenness (max 0.31), while many others had near-zero betweenness.
    *   *Insight:* The network evolved towards a structure reliant on a few key individuals to maintain connectivity between the emergent communities. This prevents complete fragmentation but creates potential information bottlenecks. The profiles of these brokers often revealed initial cross-domain interests or lack of strong dislikes.

4.  **PageRank (Influence):**
    *   Showed shifts over time, with different students becoming central within topic interaction neighborhoods at T=0 vs T=40. Influence within the network is dynamic.

5.  **Profile Correlations & Sentiment:**
    *   *(Analysis performed by `analyze_graph.py`)*: Indicated expected correlations (e.g., higher ratings for loved topics/matched modalities) and showed reasonable sentiment distribution based on ratings, adding credibility to the simulation's feedback loop.

### 5. Discussion & Interpretation

*   **Research Question 1 (Network Evolution):** The student interaction network is highly dynamic. Communities form, dissolve, and reform over time. Centrality measures show significant evolution, particularly the concentration of brokerage roles into a few key nodes.
*   **RQ2 (Echo Chambers):** Strong, isolated *group-level* echo chambers (entire communities with very low diversity and no external links) did *not* form within 40 turns. However, **clear individual echo chambers emerged**, evidenced by the significant number of students developing very low Shannon Entropy. The network avoided fragmentation due to the emergence of strong bridge nodes.
*   **RQ3 (Role of Profiles):** Initial profiles (especially narrow `lovedTopicIds` or broad `dislikedTopicIds`) were strong predictors of students developing low entropy (echo chambers). Conversely, students with initial cross-domain interests or fewer initial biases were more likely to become high-betweenness bridge nodes.
*   **RQ4 (Adaptive Mechanism):** The overall *increase* in average entropy for many students and the *prevention of complete fragmentation* suggests the adaptive mechanism (triggering exploration prompts) and/or other simulation dynamics (like participation, student choice allowing some randomness) are **partially effective** in promoting diversity. However, they were insufficient to prevent deep specialization among students with strong initial biases over 40 turns.
*   **Cooperation vs. Competition:** The simulation shows both. Students cooperate implicitly by interacting around shared topics. The adaptive AI cooperates by trying to broaden exposure. However, competition arises as information inequality grows – low-entropy students become disadvantaged, while high-betweenness nodes gain structural power (potential information gatekeepers). The system didn't collapse into fully competing, isolated silos but developed a potentially fragile core-periphery structure reliant on brokers.

### 6. Conclusion

This project modeled and simulated an AI-driven learning network, incorporating sophisticated elements like LLM-based recommendations, simulated student choice, LLM-generated feedback, and adaptive diversity mechanisms. The 40-turn simulation demonstrated that while the system fosters connectivity and increases overall interaction diversity for many, it also allows students with strong initial biases to form individual echo chambers. The network evolved a structure reliant on a few key bridge nodes rather than fragmenting completely. This highlights the complex interplay between personalization, student agency, and network structure, emphasizing the need for careful design and potentially stronger adaptive interventions to ensure equitable information access in AI learning environments.

### 7. Future Work

1.  **Tune Parameters:** Experiment with stronger biasing factors (affinity, choice model) or a higher `DIVERSITY_THRESHOLD` to study echo chamber formation more intensely.
2.  **Longer Simulations:** Run for significantly more turns (e.g., 100+) to observe long-term stability or fragmentation.

### Appendix A: Project Structure & Running

**Structure:**

```
coco-project/
├── .venv/
├── simulation_outputs/
│   ├── consumed_in_simulation.csv
│   └── llm_recommendations_log.csv
├── src/
│   ├── llm/
│   ├── analyze_graph.py
│   ├── echo_chamber_detection.py # (Note: Mention if this was used or integrated)
│   ├── generate_data.py
│   ├── load_graph.py
│   ├── recommender_engine.py
│   └── simulation.py
├── synthetic_data/
│   ├── consumed_initially.csv
│   ├── participated.csv
│   ├── resource_topic.csv
│   ├── resources.csv
│   ├── students.csv
│   └── topics.csv
├── .env
├── .gitignore
├── echo_chamber_analysis.md     # Supporting analysis notes
├── gephi_edges_00.csv           # Edges at T=0
├── gephi_edges_20.csv           # Edges at T=20
├── gephi_edges_40.csv           # Edges at T=40
├── gephi_nodes_00.csv           # Nodes w/ metrics at T=0
├── gephi_nodes_20.csv           # Nodes w/ metrics at T=20 
├── gephi_nodes_40.csv           # Nodes w/ metrics at T=40
├── proposal.md
├── requirements.txt
├── student_network.html         # Pyvis output
└── vertex_keys.json             # GCP Credentials (gitignore!)
```

**How to Run Workflow:**

1.  **Setup:** Clone repo, create venv (`python -m venv .venv`), activate, `pip install -r requirements.txt`. Configure Neo4j (Docker recommended) and GCP (`vertex_keys.json`). Create `.env` file with `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.
2.  **Generate Initial Data:** `python src/generate_data.py` (Outputs to `synthetic_data/`)
3.  **Load Initial Data:** `python src/load_graph.py` (Loads from `synthetic_data/` to Neo4j)
4.  **(Optional) Analyze Initial State:** Modify output filenames in `src/analyze_graph.py` (e.g., `_T00`) and run `python src/analyze_graph.py`. Save resulting CSVs.
5.  **Run Simulation:** Configure `NUM_TURNS` etc. in `src/simulation.py`. Run `python src/simulation.py`. (Outputs logs to `simulation_outputs/`, updates Neo4j).
6.  **Analyze Final State:** Ensure output filenames in `src/analyze_graph.py` are set for the final run (e.g., `analysis_results_rich.csv`, `gephi_edges_40.csv`). Run `python src/analyze_graph.py`.
