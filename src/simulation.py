# src/simulation.py
import logging, os, random, time, pandas as pd
from datetime import datetime
from py2neo import Graph
from scipy.stats import entropy as shannon_entropy
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed # For parallelism


# Import functions from the engine
from recommender_engine import (
    connect_neo4j,
    fetch_lookup_maps_from_neo4j,
    fetch_student_data_from_neo4j,
    recommend_to_student,
    chat_with_llm, # We need this for feedback generation
    fetch_resource_metrics_from_neo4j
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Simulation Parameters ---
MAX_WORKERS = 30 # Number of concurrent LLM calls (for recs AND feedback)
NUM_TURNS = 40
STUDENTS_PER_TURN = 30
DIVERSITY_THRESHOLD = 0.5 # Example: If Shannon entropy drops below this, trigger adaptation
HISTORY_LIMIT_FOR_ENTROPY = 10 # How far back to look for entropy calculation
HISTORY_LIMIT_FOR_PROMPT = 3 # How far back to show LLM in prompt
NUM_RECOMMENDATIONS_PER_STUDENT = 3
# Student Choice Model Params
BASE_SCORE = 1.0
LOVED_TOPIC_BONUS = 10.0
DISLIKED_TOPIC_PENALTY = -5.0
MODALITY_MATCH_BONUS = 4
PROBABILITY_CHOOSE_NOTHING = 0.2 # 20% chance student ignores recommendations
PROBABILITY_PARTICIPATE_AFTER_CONSUME = 0.05 # 5% chance to also participate in the topic


LLM_FEEDBACK_SYSTEM_PROMPT = """
You are simulating a 9th-grade student providing feedback on a learning resource they just consumed.
Base the feedback ONLY on the provided student profile and resource details.
Output ONLY the JSON object requested, with a rating between 1 and 5.
"""
LLM_FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "rating": {"type": "integer", "description": "A realistic integer rating from 1 to 5."},
        "comment": {"type": "string", "description": "A brief (1-2 sentence) comment reflecting the student's likely reaction."}
    },
    "required": ["rating", "comment"]
}

# --- NEW: Output CSV filenames ---
OUTPUT_DIR_SIM = "simulation_outputs" # Store simulation logs separately
RECOMMENDATION_LOG_CSV = os.path.join(OUTPUT_DIR_SIM, "llm_recommendations_log.csv")
CONSUMPTION_LOG_CSV = os.path.join(OUTPUT_DIR_SIM, "consumed_in_simulation.csv")


# --- Helper Functions ---

def calculate_topic_entropy(history: list, resource_info_map: dict, history_limit: int) -> float:
    """Calculates Shannon entropy based on topics in recent history."""
    if not history:
        return 0.0 # Max entropy? Or 0? Let's use 0 for no data.

    recent_history = history[:history_limit] # Assumes history is sorted descending by time
    topic_counts = {}
    valid_interactions = 0
    for item in recent_history:
        resource_id = item.get('resourceId')
        resource_info = resource_info_map.get(resource_id)
        if resource_info and resource_info.get('topicId'):
            topic_id = resource_info['topicId']
            topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
            valid_interactions += 1

    if not topic_counts or valid_interactions == 0:
        return 0.0

    counts = list(topic_counts.values())
    probabilities = [count / valid_interactions for count in counts]

    if not probabilities:
        return 0.0

    return shannon_entropy(probabilities, base=2)


def simulate_student_choice(student_profile: dict, recommendations: list, resource_info_map: dict, topic_id_map: dict) -> str | None:
    """Simulates which recommended resource the student chooses."""
    if not recommendations:
        return None # No recommendations to choose from

    # Add option to choose nothing
    choices = [None] + [rec['resource_id'] for rec in recommendations]
    weights = [PROBABILITY_CHOOSE_NOTHING]

    # Calculate scores for each actual recommendation
    for rec in recommendations:
        res_id = rec['resource_id']
        res_info = resource_info_map.get(res_id)
        if not res_info:
            weights.append(0) # Cannot score if info is missing
            continue

        score = BASE_SCORE
        res_topic_id = res_info.get('topicId')
        res_modality = res_info.get('modality')
        student_style = student_profile.get('learningStyle')

        # Topic Affinity
        if res_topic_id in student_profile.get('lovedTopics', []):
            score += LOVED_TOPIC_BONUS
        elif res_topic_id in student_profile.get('dislikedTopics', []):
            score += DISLIKED_TOPIC_PENALTY

        # Modality Match
        if res_modality and student_style and res_modality == student_style:
            score += MODALITY_MATCH_BONUS
        elif student_style == 'Mixed': # Small bonus for mixed learners?
             score += MODALITY_MATCH_BONUS * 0.5

        weights.append(max(0.1, score)) # Ensure weight is slightly positive

    # Normalize weights to sum to 1 (excluding the 'None' probability)
    total_rec_weight = sum(weights[1:])
    if total_rec_weight > 0:
         normalized_rec_weights = [(w / total_rec_weight) * (1.0 - PROBABILITY_CHOOSE_NOTHING) for w in weights[1:]]
         final_weights = [PROBABILITY_CHOOSE_NOTHING] + normalized_rec_weights
    else:
        # If all scores were somehow zero or negative, give equal chance among recs (excluding None initially)
        if len(recommendations) > 0:
            equal_prob = (1.0 - PROBABILITY_CHOOSE_NOTHING) / len(recommendations)
            final_weights = [PROBABILITY_CHOOSE_NOTHING] + [equal_prob] * len(recommendations)
        else: # Only None is possible
             final_weights = [1.0]


    # Make weighted choice
    chosen_id = random.choices(choices, weights=final_weights, k=1)[0]
    return chosen_id


def generate_student_feedback_llm(student_profile: dict, chosen_resource_info: dict, topic_id_map: dict) -> dict | None:
    """Generates simulated student rating and comment using LLM."""
    learning_style = student_profile.get('learningStyle', 'Unknown')
    loved_topics = [topic_id_map.get(tid, tid) for tid in student_profile.get('lovedTopics', [])]
    disliked_topics = [topic_id_map.get(tid, tid) for tid in student_profile.get('dislikedTopics', [])]
    loved_str = ", ".join(loved_topics) if loved_topics else "None specified"
    disliked_str = ", ".join(disliked_topics) if disliked_topics else "None specified"

    res_title = chosen_resource_info.get('title', 'N/A')
    res_topic_id = chosen_resource_info.get('topicId')
    res_topic_name = topic_id_map.get(res_topic_id, "N/A")
    res_modality = chosen_resource_info.get('modality', 'N/A')

    prompt = f"""Simulate feedback for the following scenario:

Student Profile:
- Preferred Learning Style: {learning_style}
- Enjoys Topics Like: {loved_str}
- Dislikes Topics Like: {disliked_str}

Consumed Resource:
- Title: '{res_title}'
- Topic: {res_topic_name}
- Modality: {res_modality}

Task:
Generate a realistic integer rating (1-5) and a brief comment (1-2 sentences) reflecting how this specific student likely felt about this resource. Consider their learning style match with the resource modality and their affinity for the resource's topic.

Output Format (JSON only):
"""
    # Using the specific schema defined above
    feedback_json = chat_with_llm(prompt, system_prompt=LLM_FEEDBACK_SYSTEM_PROMPT, schema=LLM_FEEDBACK_SCHEMA)

    # Validate the response structure
    if feedback_json and isinstance(feedback_json, dict) and "rating" in feedback_json and "comment" in feedback_json:
        # Further validation: ensure rating is within range
        try:
             rating = int(feedback_json["rating"])
             if 1 <= rating <= 5:
                  return {"rating": rating, "comment": str(feedback_json["comment"])}
             else:
                  logger.warning(f"LLM feedback rating out of range: {rating}. Discarding.")
                  return None
        except ValueError:
             logger.warning(f"LLM feedback rating not an integer: {feedback_json['rating']}. Discarding.")
             return None
    else:
        logger.warning(f"LLM feedback failed or returned invalid format: {feedback_json}")
        return None


# --- Persistence Functions (Updated CONSUMED, new LLM_RECOMMENDED, keep PARTICIPATED_IN) ---
def add_consumed_interaction_to_neo4j(graph: Graph, student_id: str, resource_id: str, rating: int, comment: str, timestamp: str, source: str = "simulation"):
    """Adds a new CONSUMED relationship to Neo4j, marking the source."""
    try:
        query = """
        MATCH (s:Student {studentId: $s_id})
        MATCH (r:Resource {resourceId: $r_id})
        CREATE (s)-[c:CONSUMED {
            rating: $rating, comment: $comment, timestamp: $ts,
            simulated: true, source: $source  // Added source property
        }]->(r)
        RETURN id(c) AS rel_id
        """
        result = graph.run(query, s_id=student_id, r_id=resource_id, rating=rating, comment=comment, ts=timestamp, source=source).data()
        if result and result[0]['rel_id'] is not None:
            logger.info(f"Persisted CONSUMED ({source}): {student_id} -> {resource_id} (Rating: {rating})")
            return True
        else: logger.warning(f"Failed to persist CONSUMED ({source}) for {student_id} -> {resource_id}"); return False
    except Exception as e: logger.error(f"Error adding CONSUMED ({source}) to Neo4j for {student_id} -> {resource_id}: {e}"); return False

def add_participation_to_neo4j(graph: Graph, student_id: str, topic_id: str, timestamp: str, interaction_type: str = "simulated_post"):
    # ...(Implementation unchanged)...
    if not topic_id: logger.warning(f"Cannot add participation for {student_id}: topic_id missing."); return False
    try:
        query = """ MATCH (s:Student {studentId: $s_id}) MATCH (t:Topic {topicId: $t_id})
        CREATE (s)-[p:PARTICIPATED_IN { interactionType: $type, timestamp: $ts, simulated: true }]->(t) RETURN id(p) AS rel_id """
        result = graph.run(query, s_id=student_id, t_id=topic_id, type=interaction_type, ts=timestamp).data()
        if result and result[0]['rel_id'] is not None: logger.info(f"Persisted PARTICIPATED_IN: {student_id} -> {topic_id}"); return True
        else: logger.warning(f"Failed to persist PARTICIPATED_IN for {student_id} -> {topic_id}"); return False
    except Exception as e: logger.error(f"Error adding PARTICIPATED_IN to Neo4j for {student_id} -> {topic_id}: {e}"); return False

# --- NEW: Function to add LLM_RECOMMENDED ---
def add_llm_recommendation_to_neo4j(graph: Graph, student_id: str, resource_id: str, reason: str, timestamp: str):
    """Adds a new LLM_RECOMMENDED relationship to Neo4j."""
    try:
        query = """
        MATCH (s:Student {studentId: $s_id})
        MATCH (r:Resource {resourceId: $r_id})
        CREATE (s)-[rec:LLM_RECOMMENDED {
            reason: $reason,
            timestamp: $ts,
            simulated: true
        }]->(r)
        RETURN id(rec) AS rel_id
        """
        result = graph.run(query, s_id=student_id, r_id=resource_id, reason=reason, ts=timestamp).data()
        if result and result[0]['rel_id'] is not None:
            logger.debug(f"Persisted LLM_RECOMMENDED: {student_id} -> {resource_id}") # Debug level might be better
            return True
        else:
            logger.warning(f"Failed to persist LLM_RECOMMENDED for {student_id} -> {resource_id}")
            return False
    except Exception as e:
        logger.error(f"Error adding LLM_RECOMMENDED to Neo4j for {student_id} -> {resource_id}: {e}")
        return False
# --- End Persistence ---


# --- NEW: Wrapper Functions for Parallel Execution ---
def process_student_recommendation(args):
    """Wrapper to generate recommendations for one student."""
    student_id, student_data, resource_map, topic_map, num_recs, needs_adapt, current_resource_metrics = args
    logger.debug(f"Thread {os.getpid()} starting recommendations for {student_id} (Adapt: {needs_adapt})...")
    try:
        recommendations = recommend_to_student(
            student_data,
            resource_map,
            topic_map,
            num_recommendations=num_recs,
            resource_metrics_map=current_resource_metrics,
            adapt_prompt=needs_adapt
        )
        logger.debug(f"Thread {os.getpid()} finished recommendations for {student_id}.")
        # Return student ID along with results for mapping later
        return student_id, recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations for {student_id} in thread {os.getpid()}: {e}")
        return student_id, None # Return None on error

def process_student_feedback(args):
    """Wrapper to generate feedback for one student's choice."""
    student_id, chosen_resource_id, student_profile, resource_info, topic_map = args
    logger.debug(f"Thread {os.getpid()} starting feedback for {student_id} on {chosen_resource_id}...")
    try:
        feedback = generate_student_feedback_llm(
            student_profile,
            resource_info,
            topic_map
        )
        logger.debug(f"Thread {os.getpid()} finished feedback for {student_id} on {chosen_resource_id}.")
        # Return identifiers along with result
        return student_id, chosen_resource_id, feedback
    except Exception as e:
        logger.error(f"Error generating feedback for {student_id}/{chosen_resource_id} in thread {os.getpid()}: {e}")
        return student_id, chosen_resource_id, None # Return None on error


# --- Main Simulation Loop (Parallelized) ---
if __name__ == "__main__":
    logger.info("Starting Parallel Simulation...")
    start_time = time.time()
    if not os.path.exists(OUTPUT_DIR_SIM): os.makedirs(OUTPUT_DIR_SIM); logger.info(f"Created dir: {OUTPUT_DIR_SIM}")
    all_llm_recommendations_log = []; all_simulated_consumptions_log = []

    try:
        db = connect_neo4j();
        if not db: exit()
        logger.info("Fetching initial lookup maps...")
        topic_map, resource_map = fetch_lookup_maps_from_neo4j(db)
        if not topic_map or not resource_map: exit()
        logger.info("Fetching list of all student IDs...")
        all_student_ids = [record['studentId'] for record in db.run("MATCH (s:Student) RETURN s.studentId AS studentId")]
        if not all_student_ids: exit()
        logger.info(f"Found {len(all_student_ids)} students.")

        # Create a ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # --- Simulation Loop ---
            for turn in range(1, NUM_TURNS + 1):
                logger.info(f"===== Starting Simulation Turn {turn}/{NUM_TURNS} =====")
                turn_start_time = time.time()
                students_this_turn = random.sample(all_student_ids, k=min(STUDENTS_PER_TURN, len(all_student_ids)))
                logger.info(f"Simulating for students: {students_this_turn}")

                # --- NEW: Fetch current resource metrics for this turn ---
                logger.info(f"Fetching current resource metrics for Turn {turn}...")
                current_resource_metrics = fetch_resource_metrics_from_neo4j(db)
                # --- End Fetch Metrics ---


                # --- Phase 1: Fetch Data, Calc Entropy, Prep Rec Tasks ---
                rec_tasks_args = []
                student_states = {} # Store fetched data and entropy results
                logger.info("Phase 1: Fetching data and preparing recommendation tasks...")
                for student_id in students_this_turn:
                    student_data = fetch_student_data_from_neo4j(db, student_id, history_limit=HISTORY_LIMIT_FOR_ENTROPY)
                    if not student_data: logger.warning(f"Skip {student_id}: Could not fetch data."); continue
                    current_entropy = calculate_topic_entropy(student_data['history'], resource_map, HISTORY_LIMIT_FOR_ENTROPY)
                    needs_adaptation = current_entropy < DIVERSITY_THRESHOLD
                    student_states[student_id] = {'data': student_data, 'entropy': current_entropy, 'needs_adapt': needs_adaptation}
                    rec_tasks_args.append((student_id, student_data, resource_map, topic_map, NUM_RECOMMENDATIONS_PER_STUDENT, needs_adaptation, current_resource_metrics))
                    logger.debug(f"Prepared rec task for {student_id} (Entropy: {current_entropy:.3f}, Adapt: {needs_adaptation})")

                # --- Phase 2: Parallel Recommendation Generation ---
                recommendation_results = {} # student_id -> list_of_recs or None
                logger.info(f"Phase 2: Submitting {len(rec_tasks_args)} recommendation tasks to {MAX_WORKERS} workers...")
                rec_futures = {executor.submit(process_student_recommendation, args): args[0] for args in rec_tasks_args} # map future to student_id
                for future in as_completed(rec_futures):
                    s_id = rec_futures[future]
                    try:
                        _, recs = future.result() # Unpack result
                        recommendation_results[s_id] = recs
                        logger.debug(f"Received recommendations for {s_id}")
                    except Exception as exc:
                        logger.error(f'Rec task for {s_id} generated an exception: {exc}')
                        recommendation_results[s_id] = None

                # --- Phase 3: Simulate Choice & Prep Feedback Tasks ---
                feedback_tasks_args = []
                student_choices = {} # student_id -> chosen_resource_id or None
                logger.info("Phase 3: Simulating choices and preparing feedback tasks...")
                recommendation_ts = datetime.now().isoformat() # Single timestamp for all recs in this turn/batch
                for student_id in students_this_turn:
                    if student_id not in student_states: continue # Skipped earlier
                    recs = recommendation_results.get(student_id)
                    # -- Log/Persist ALL Recommendations --
                    if recs:
                        logger.info(f"LLM Recommendations for {student_id}: {[r['resource_id'] for r in recs]}")
                        for rec in recs:
                            rec_id = rec.get('resource_id'); reason = rec.get('reason', '')
                            if rec_id:
                                add_llm_recommendation_to_neo4j(db, student_id, rec_id, reason, recommendation_ts) # Persist now
                                all_llm_recommendations_log.append({ "turn": turn, "studentId": student_id, "resourceId": rec_id, "reason": reason, "timestamp": recommendation_ts })
                    else:
                         logger.warning(f"No recommendations available for {student_id} to make a choice.")
                         student_choices[student_id] = None
                         continue # Cannot choose if no recs

                    # Simulate choice
                    chosen_resource_id = simulate_student_choice(student_states[student_id]['data']['profile'], recs, resource_map, topic_map)
                    student_choices[student_id] = chosen_resource_id

                    if chosen_resource_id:
                        logger.info(f"Student {student_id} chose: {chosen_resource_id}")
                        chosen_resource_info = resource_map.get(chosen_resource_id)
                        if chosen_resource_info:
                            # Prep feedback task arguments
                            feedback_tasks_args.append((student_id, chosen_resource_id, student_states[student_id]['data']['profile'], chosen_resource_info, topic_map))
                        else:
                            logger.warning(f"Info missing for chosen resource {chosen_resource_id} for {student_id}. Cannot generate feedback.")
                    else:
                        logger.info(f"Student {student_id} chose not to interact.")

                # --- Phase 4: Parallel Feedback Generation ---
                feedback_results = {} # (student_id, resource_id) -> feedback_dict or None
                logger.info(f"Phase 4: Submitting {len(feedback_tasks_args)} feedback tasks to {MAX_WORKERS} workers...")
                feedback_futures = {executor.submit(process_student_feedback, args): (args[0], args[1]) for args in feedback_tasks_args} # map future to (s_id, r_id)
                for future in as_completed(feedback_futures):
                    s_id, r_id = feedback_futures[future]
                    try:
                        _, _, feedback = future.result() # Unpack result
                        feedback_results[(s_id, r_id)] = feedback
                        logger.debug(f"Received feedback for {s_id} on {r_id}")
                    except Exception as exc:
                        logger.error(f'Feedback task for {s_id}/{r_id} generated an exception: {exc}')
                        feedback_results[(s_id, r_id)] = None

                # --- Phase 5: Persist Choices and Feedback ---
                logger.info("Phase 5: Persisting simulation results...")
                for student_id in students_this_turn:
                    if student_id not in student_choices: continue # Should not happen if handled above
                    chosen_resource_id = student_choices[student_id]

                    if chosen_resource_id:
                        feedback = feedback_results.get((student_id, chosen_resource_id))
                        consumption_ts = datetime.now().isoformat() # Timestamp for this specific consumption

                        if feedback:
                            consumed_persisted = add_consumed_interaction_to_neo4j(db, student_id, chosen_resource_id, feedback['rating'], feedback['comment'], consumption_ts, source="simulation")
                            if consumed_persisted:
                                all_simulated_consumptions_log.append({"turn": turn, "studentId": student_id, "resourceId": chosen_resource_id, "rating": feedback['rating'], "comment": feedback['comment'], "timestamp": consumption_ts, "feedback_generated_by": "llm"})
                                # Optional Participation
                                if random.random() < PROBABILITY_PARTICIPATE_AFTER_CONSUME:
                                    chosen_resource_info = resource_map.get(chosen_resource_id)
                                    topic_id = chosen_resource_info.get('topicId') if chosen_resource_info else None
                                    if topic_id: add_participation_to_neo4j(db, student_id, topic_id, consumption_ts)
                        else:
                             logger.warning(f"Feedback generation failed for {student_id} on {chosen_resource_id}. CONSUMED interaction not persisted.")
                    # No else needed here, already logged if student chose nothing

                logger.info(f"===== Finished Turn {turn} in {time.time() - turn_start_time:.2f}s =====")

        # End simulation loop
        total_time = time.time() - start_time
        logger.info(f"Simulation finished {NUM_TURNS} turns in {total_time:.2f} seconds.")

    except Exception as e:
        logger.exception("An critical error occurred during the simulation:")
    finally:
        # Write log files
        logger.info("Saving simulation log files...")
        try: pd.DataFrame(all_llm_recommendations_log).to_csv(RECOMMENDATION_LOG_CSV, index=False); logger.info(f"Saved recommendations log.")
        except Exception as e: logger.error(f"Failed to save recommendation log CSV: {e}")
        try: pd.DataFrame(all_simulated_consumptions_log).to_csv(CONSUMPTION_LOG_CSV, index=False); logger.info(f"Saved consumptions log.")
        except Exception as e: logger.error(f"Failed to save consumption log CSV: {e}")