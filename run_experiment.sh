#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_EXECUTABLE="python3" # Or just "python" or the path to your .venv python
SRC_DIR="src"
ANALYSIS_SCRIPT="${SRC_DIR}/analyze_graph.py"
SIMULATION_SCRIPT="${SRC_DIR}/simulation.py"

# Simulation Parameters
TOTAL_SIMULATION_DURATION=100 # Total number of turns you want to simulate in the end
ANALYSIS_INTERVAL=5        # How often to run analysis_graph.py (every N simulation turns)
SIM_TURNS_PER_LEG=$ANALYSIS_INTERVAL # Number of turns simulation.py runs each time

# Output directory for analysis files
ANALYSIS_OUTPUT_DIR="analysis_snapshots"
GEPHI_OUTPUT_DIR="gephi_snapshots"

# Create output directories if they don't exist
mkdir -p $ANALYSIS_OUTPUT_DIR
mkdir -p $GEPHI_OUTPUT_DIR

# --- Initial State (T=0) ---
# This assumes your load_graph.py is run MANUALLY ONCE to set up the initial DB state
# Or, you can add a command here to clear and load DB if needed, e.g.:
echo "Clearing and loading initial data into Neo4j..."
$PYTHON_EXECUTABLE ${SRC_DIR}/load_graph.py # Ensure CLEAR_DB_BEFORE_LOADING=True in load_graph.py

echo "--- Analyzing Initial State (T=0) ---"
NODES_FILE_T0="${ANALYSIS_OUTPUT_DIR}/analysis_results_T00.csv"
EDGES_FILE_T0="${GEPHI_OUTPUT_DIR}/gephi_edges_T00.csv"
$PYTHON_EXECUTABLE $ANALYSIS_SCRIPT --output_nodes $NODES_FILE_T0 --output_edges $EDGES_FILE_T0
echo "Initial analysis complete. Results in $NODES_FILE_T0 and $EDGES_FILE_T0"
echo "-------------------------------------------"

# --- Simulation and Analysis Loop ---
current_total_turns=0
while [ $current_total_turns -lt $TOTAL_SIMULATION_DURATION ]; do
    echo ""
    echo "--- Running Simulation: Turns $(($current_total_turns + 1)) to $(($current_total_turns + $SIM_TURNS_PER_LEG)) ---"
    # Note: simulation.py needs to be aware it's continuing, not restarting from scratch.
    # Its internal state (Neo4j) is the continuation.
    $PYTHON_EXECUTABLE $SIMULATION_SCRIPT --num_turns $SIM_TURNS_PER_LEG
    current_total_turns=$(($current_total_turns + $SIM_TURNS_PER_LEG))
    echo "Simulation leg complete. Total turns simulated: $current_total_turns"
    echo "-------------------------------------------"

    echo ""
    echo "--- Analyzing State at T=$current_total_turns ---"
    # Format timestep with leading zero if needed (e.g., T05, T10)
    TIMESTEP_LABEL=$(printf "T%02d" $current_total_turns)
    if [ $current_total_turns -eq 100 ]; then # Handle T100 if it's 3 digits
        TIMESTEP_LABEL=$(printf "T%03d" $current_total_turns)
    fi


    NODES_FILE="${ANALYSIS_OUTPUT_DIR}/analysis_results_${TIMESTEP_LABEL}.csv"
    EDGES_FILE="${GEPHI_OUTPUT_DIR}/gephi_edges_${TIMESTEP_LABEL}.csv"
    $PYTHON_EXECUTABLE $ANALYSIS_SCRIPT --output_nodes $NODES_FILE --output_edges $EDGES_FILE
    echo "Analysis at T=$current_total_turns complete. Results in $NODES_FILE and $EDGES_FILE"
    echo "-------------------------------------------"
done

echo ""
echo "===== Full Experiment Run Complete ====="
echo "Total turns simulated: $TOTAL_SIMULATION_DURATION"
echo "Analysis snapshots saved in $ANALYSIS_OUTPUT_DIR"
echo "Gephi edge files saved in $GEPHI_OUTPUT_DIR"
