# src/simulation.py
import logging, os, random, time, pandas as pd
from datetime import datetime
from py2neo import Graph
from scipy.stats import entropy as shannon_entropy
from dotenv import load_dotenv


# Import functions from the engine
from recommender_engine import (
    connect_neo4j,
    fetch_lookup_maps_from_neo4j,
    fetch_student_data_from_neo4j,
    recommend_to_student,
    chat_with_llm # We need this for feedback generation
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Simulation Parameters ---
NUM_TURNS = 10
STUDENTS_PER_TURN = 5
DIVERSITY_THRESHOLD = 2.0 # Example: If Shannon entropy drops below this, trigger adaptation
HISTORY_LIMIT_FOR_ENTROPY = 20 # How far back to look for entropy calculation
HISTORY_LIMIT_FOR_PROMPT = 5 # How far back to show LLM in prompt
NUM_RECOMMENDATIONS_PER_STUDENT = 3
# Student Choice Model Params
BASE_SCORE = 1.0
LOVED_TOPIC_BONUS = 1.5
DISLIKED_TOPIC_PENALTY = -1.0
MODALITY_MATCH_BONUS = 1.0
PROBABILITY_CHOOSE_NOTHING = 0.3 # 30% chance student ignores recommendations
PROBABILITY_PARTICIPATE_AFTER_CONSUME = 0.20 # 20% chance to also participate in the topic


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


# --- Main Simulation Loop ---
if __name__ == "__main__":
    logger.info("Starting Simulation...")
    start_time = time.time()

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR_SIM):
        os.makedirs(OUTPUT_DIR_SIM)
        logger.info(f"Created simulation output directory: {OUTPUT_DIR_SIM}")

    # --- Initialize log lists ---
    all_llm_recommendations_log = []
    all_simulated_consumptions_log = []

    try:
        db = connect_neo4j()
        if not db: exit()
        logger.info("Fetching initial lookup maps...")
        topic_map, resource_map = fetch_lookup_maps_from_neo4j(db)
        if not topic_map or not resource_map: exit()
        logger.info("Fetching list of all student IDs...")
        all_student_ids = [record['studentId'] for record in db.run("MATCH (s:Student) RETURN s.studentId AS studentId")]
        if not all_student_ids: exit()
        logger.info(f"Found {len(all_student_ids)} students.")

        # --- Simulation Loop ---
        for turn in range(1, NUM_TURNS + 1):
            logger.info(f"===== Starting Simulation Turn {turn}/{NUM_TURNS} =====")
            turn_start_time = time.time()
            students_this_turn = random.sample(all_student_ids, k=min(STUDENTS_PER_TURN, len(all_student_ids)))
            logger.info(f"Simulating for students: {students_this_turn}")

            for student_id in students_this_turn:
                logger.info(f"--- Simulating for Student: {student_id} ---")
                sim_step_start = time.time()
                recommendation_ts = datetime.now().isoformat() # Timestamp for recommendation event

                # 1. Fetch State
                student_data = fetch_student_data_from_neo4j(db, student_id, history_limit=HISTORY_LIMIT_FOR_ENTROPY)
                if not student_data: continue

                # 2. Calculate Diversity & Check Adaptation
                current_entropy = calculate_topic_entropy(student_data['history'], resource_map, HISTORY_LIMIT_FOR_ENTROPY)
                needs_adaptation = current_entropy < DIVERSITY_THRESHOLD
                logger.info(f"Student {student_id} entropy: {current_entropy:.3f} (Adapt: {needs_adaptation})")

                # 3. Generate Recommendations
                recommendations = recommend_to_student(student_data, resource_map, topic_map,
                                                       num_recommendations=NUM_RECOMMENDATIONS_PER_STUDENT,
                                                       adapt_prompt=needs_adaptation)

                # 4. --- NEW: Log & Persist *ALL* Recommendations ---
                if recommendations:
                    logger.info(f"LLM Recommendations for {student_id}: {[r['resource_id'] for r in recommendations]}")
                    for rec in recommendations:
                        rec_id = rec.get('resource_id')
                        reason = rec.get('reason', '')
                        if rec_id:
                            # Add to Neo4j
                            add_llm_recommendation_to_neo4j(db, student_id, rec_id, reason, recommendation_ts)
                            # Add to CSV log list
                            all_llm_recommendations_log.append({
                                "turn": turn,
                                "studentId": student_id,
                                "resourceId": rec_id,
                                "reason": reason,
                                "timestamp": recommendation_ts
                            })
                else:
                    logger.warning(f"No recommendations generated for {student_id}.")
                    continue # Skip choice/consumption if no recommendations

                # 5. Simulate Choice
                chosen_resource_id = simulate_student_choice(student_data['profile'], recommendations, resource_map, topic_map)

                # 6. If Choice Made
                if chosen_resource_id:
                    logger.info(f"Student {student_id} chose: {chosen_resource_id}")
                    chosen_resource_info = resource_map.get(chosen_resource_id)
                    consumption_ts = datetime.now().isoformat() # Timestamp for consumption event

                    if chosen_resource_info:
                        # 7. Generate Feedback (LLM)
                        feedback = generate_student_feedback_llm(student_data['profile'], chosen_resource_info, topic_map)

                        if feedback:
                            logger.info(f"Generated feedback for {student_id}: {feedback}")
                            # 8. Persist CONSUMED Interaction
                            consumed_persisted = add_consumed_interaction_to_neo4j(
                                db, student_id, chosen_resource_id,
                                feedback['rating'], feedback['comment'], consumption_ts,
                                source="simulation" # Mark as simulation-generated
                            )

                            # --- NEW: Add to simulated consumption log list ---
                            if consumed_persisted:
                                all_simulated_consumptions_log.append({
                                    "turn": turn,
                                    "studentId": student_id,
                                    "resourceId": chosen_resource_id,
                                    "rating": feedback['rating'],
                                    "comment": feedback['comment'],
                                    "timestamp": consumption_ts,
                                    "feedback_generated_by": "llm" # Assume LLM feedback succeeded if we got here
                                })

                            # 9. Optional Participation
                            if consumed_persisted and random.random() < PROBABILITY_PARTICIPATE_AFTER_CONSUME:
                                topic_id = chosen_resource_info.get('topicId')
                                if topic_id:
                                    logger.info(f"Student {student_id} will also participate in topic {topic_id}.")
                                    add_participation_to_neo4j(db, student_id, topic_id, consumption_ts) # Use consumption timestamp
                                # else: (logging handled in function)

                        else: logger.warning(f"Feedback generation failed for {student_id}. Consumed interaction not fully persisted.")
                    else: logger.warning(f"Info missing for chosen resource {chosen_resource_id}. Consumed interaction not persisted.")
                else:
                    logger.info(f"Student {student_id} chose not to interact.")

                logger.debug(f"Processing time for {student_id}: {time.time() - sim_step_start:.2f}s")

            logger.info(f"===== Finished Turn {turn} in {time.time() - turn_start_time:.2f}s =====")

        total_time = time.time() - start_time
        logger.info(f"Simulation finished {NUM_TURNS} turns in {total_time:.2f} seconds.")

    except Exception as e:
        logger.exception("An critical error occurred during the simulation:")
    finally:
        # --- NEW: Write log files at the end ---
        logger.info("Saving simulation log files...")
        try:
            recs_df = pd.DataFrame(all_llm_recommendations_log)
            recs_df.to_csv(RECOMMENDATION_LOG_CSV, index=False)
            logger.info(f"Saved {len(recs_df)} LLM recommendations to {RECOMMENDATION_LOG_CSV}")
        except Exception as e:
            logger.error(f"Failed to save recommendation log CSV: {e}")

        try:
            consumptions_df = pd.DataFrame(all_simulated_consumptions_log)
            consumptions_df.to_csv(CONSUMPTION_LOG_CSV, index=False)
            logger.info(f"Saved {len(consumptions_df)} simulated consumptions to {CONSUMPTION_LOG_CSV}")
        except Exception as e:
            logger.error(f"Failed to save consumption log CSV: {e}")
