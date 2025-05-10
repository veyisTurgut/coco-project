# src/recommender_engine.py
# Purpose:
# This script implements the core recommendation logic for a personalized learning system.
# It interfaces with a Neo4j graph database to fetch student profiles, interaction histories,
# resource details, and topic information. It then utilizes a Large Language Model (LLM),
# specifically Google's Vertex AI Gemini, to generate tailored resource recommendations
# based on various signals and strategic goals (e.g., standard relevance, diversity
# promotion, or echo chamber simulation).
#
# What it does:
# 1.  Initialization and Configuration:
#     - Loads environment variables for Neo4j connection parameters and potentially LLM settings.
#     - Sets up logging for monitoring and debugging.
#     - Establishes and manages a global connection to the Neo4j database.
#
# 2.  Data Fetching from Neo4j:
#     - `connect_neo4j()`: Establishes or reuses a connection to the Neo4j database.
#     - `fetch_lookup_maps_from_neo4j()`: Retrieves mapping data for topics (ID to name)
#       and resources (ID to details like title, type, modality, topicId).
#     - `fetch_student_data_from_neo4j()`: Fetches a specific student's profile (learning
#       style, loved/disliked topics, social engagement score) and their recent consumption
#       history (up to a specified limit).
#     - `fetch_resource_metrics_from_neo4j()`: Gathers aggregate metrics for resources,
#       such as average ratings, total rating counts, and how many times each resource has
#       been recommended by the LLM.
#
# 3.  LLM Interaction (`chat_with_llm()`):
#     - Provides a generic interface to communicate with the Vertex AI LLM (Gemini model).
#     - Takes a user prompt, an optional system prompt, and an optional JSON schema for
#       the expected response format.
#     - Configures LLM parameters (model name, temperature, top_k, top_p, etc.).
#     - Sends the request to the LLM and parses the JSON response.
#     - Handles potential errors during LLM communication or response parsing.
#
# 4.  Prompt Engineering and Candidate Preparation:
#     - `format_history_for_prompt()`: Formats a student's recent interaction history into a
#       readable string for inclusion in the LLM prompt.
#     - `recommend_to_student()` (Core Logic):
#       a. Retrieves student data, resource information, and topic maps.
#       b. Filters candidate resources: Excludes already consumed resources. During an "echo
#          chamber" phase, it can apply stricter filtering to prioritize topics the student
#          loves or has recently interacted with.
#       c. Formats a list of candidate resources (including their titles, topics, modalities,
#          and fetched metrics like average rating and recommendation count) for the LLM prompt.
#          Limits the number of candidates sent in the prompt to manage context window.
#       d. Dynamically constructs the main LLM prompt based on the recommendation goal:
#          - Standard Recommendation: Balances profile alignment, resource metrics, and some exploration.
#          - Diversity Promotion (Adaptive): Instructs the LLM to prioritize resources from
#            topics the student hasn't engaged with recently or dislikes, aiming to broaden their horizons.
#            This is typically activated when student interaction entropy is low (determined by the calling simulation script).
#          - Echo Chamber Induction: Instructs the LLM to heavily prioritize resources from
#            loved topics and recent interactions, leveraging social proof (high ratings) and popularity.
#            This is active during a configurable initial number of simulation turns.
#       e. Defines the expected JSON output schema for the recommendations (list of resource IDs and reasons).
#       f. Calls `chat_with_llm()` with the composed prompt and system message.
#       g. Validates the LLM's response: Ensures it's a list of dictionaries with the required
#          keys and that recommended resource IDs were part of the candidates sent to the LLM.
#       h. Returns a list of validated recommendations (resource ID and LLM-generated reason).
#
# 5.  Main Execution Block (Testing):
#     - The `if __name__ == "__main__":` block provides an example of how to use the
#       `recommend_to_student` function, demonstrating its usage with different parameters
#       to test standard, adaptive, and echo chamber recommendation scenarios.
#
# Key Libraries/Modules Used:
# - py2neo: For interacting with the Neo4j graph database.
# - llm.curl_vertex, llm.mylib (custom): For interfacing with Google Vertex AI LLMs.
# - dotenv: For managing environment variables.
# - json: For parsing JSON data (LLM responses, topic lists in profiles).
# - logging: For application logging.
# - os, random: Standard Python libraries for OS interaction and random sampling.

import json, logging, os, random
from py2neo import Graph
from llm.curl_vertex import CurlVertex, vertex_credentials
from llm.mylib import *
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://gcloud.madlen.io:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

graph_db = None

# --- Neo4j Connection ---
def connect_neo4j():
    global graph_db
    if graph_db is None:
        try:
            graph_db = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            graph_db.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j at {NEO4J_URI}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            graph_db = None
            raise
    return graph_db

# --- Data Fetching ---
def fetch_lookup_maps_from_neo4j(graph: Graph):
    topic_id_map = {}
    resource_info_map = {}
    try:
        topic_query = "MATCH (t:Topic) RETURN t.topicId AS topicId, t.name AS name"
        topic_results = graph.run(topic_query)
        for record in topic_results: topic_id_map[record['topicId']] = record['name']
        logger.info(f"Fetched {len(topic_id_map)} topics.")

        resource_query = """
        MATCH (r:Resource) OPTIONAL MATCH (r)-[:ABOUT_TOPIC]->(t:Topic)
        RETURN r.resourceId AS resourceId, r.title AS title, r.type AS type, r.modality AS modality, t.topicId AS topicId """
        resource_results = graph.run(resource_query)
        for record in resource_results:
            resource_info_map[record['resourceId']] = {'resourceId': record['resourceId'], 'title': record['title'],
                                                      'type': record['type'], 'modality': record['modality'], 'topicId': record['topicId']}
        logger.info(f"Fetched {len(resource_info_map)} resources.")
    except Exception as e: logger.error(f"Error fetching lookup maps: {e}"); raise
    return topic_id_map, resource_info_map


def fetch_student_data_from_neo4j(graph: Graph, student_id: str, history_limit=15):
    student_data = {"studentId": student_id, "profile": {}, "history": []}
    try:
        profile_query = """
        MATCH (s:Student {studentId: $studentId})
        RETURN s.learningStyle AS learningStyle,
               s.lovedTopicIds AS lovedTopicIds,
               s.dislikedTopicIds AS dislikedTopicIds,
               s.socialEngagementScore AS socialEngagementScore 
        LIMIT 1
        """
        profile_result = graph.run(profile_query, studentId=student_id).data()
        if not profile_result: logger.warning(f"Profile not found: {student_id}"); return None
        profile_record = profile_result[0]
        student_data["profile"]["learningStyle"] = profile_record.get("learningStyle", "Unknown")
        student_data["profile"]["socialEngagementScore"] = profile_record.get("socialEngagementScore", 0.5)

        try: student_data["profile"]["lovedTopics"] = json.loads(profile_record.get("lovedTopicIds") or '[]')
        except: student_data["profile"]["lovedTopics"] = []
        try: student_data["profile"]["dislikedTopics"] = json.loads(profile_record.get("dislikedTopicIds") or '[]')
        except: student_data["profile"]["dislikedTopics"] = []

        history_query = """ MATCH (s:Student {studentId: $studentId})-[c:CONSUMED]->(r:Resource)
        WHERE c.source = 'simulation' OR c.source = 'initial' // Ensure we get both types if needed, or just simulation for evolving state
        RETURN r.resourceId AS resourceId, c.rating AS rating, c.comment AS comment, c.timestamp AS timestamp
        ORDER BY c.timestamp DESC LIMIT $limit """
        history_result = graph.run(history_query, studentId=student_id, limit=history_limit)
        student_data["history"] = [dict(record) for record in history_result]
        logger.debug(f"Fetched profile (SES: {student_data['profile']['socialEngagementScore']}) and {len(student_data['history'])} history items for {student_id}.")
    except Exception as e: logger.error(f"Error fetching student data {student_id}: {e}"); return None
    return student_data


# --- LLM Interaction ---
def chat_with_llm(prompt: str, system_prompt: str = None, schema: dict = None):
    if not system_prompt:
        system_prompt = "You are a helpful AI assistant."

    llm_config = LLMConfig({
        "model_arch": ChatMainLLM.VERTEXLLM.value,
        "model_name": ChatMainLLMName.VERTEX_GEMINI_25_FLASH_PREVIEW,
        "temperature": 0.6,
        "top_k": 10, "top_p": 0.95, "max_output_tokens": 1536,
        "llm_region": "us-central1",
        "response_mimetype": "application/json",
        "responseSchema": schema
    })
    curl_vertex = CurlVertex(llm_config=llm_config, logger=logger)

    # Step 2: Define chat history
    chat_history = [
        Message(
            role="user",
            message=prompt,
            message_content_type=ChatInputContentType.TEXT,
            message_uri=None,
            message_type=None
        )
    ]

    try:
        # Use streaming=False for single JSON blob
        response = curl_vertex.generate(
            instruction_prompt=system_prompt,
            chat_history=chat_history,
            is_streaming=False,
            return_tokens=True,
            timeout=60
        )
        response_content = ""
        for item in response:
            response_content += item.content

        logger.debug(f"Raw LLM Response: {response_content}")
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse LLM JSON: {json_err}. Response: {response_content[:500]}...")
            return None
        except TypeError as type_err:
             logger.error(f"LLM response content not string for JSON load? Type: {type(response_content)}. Err: {type_err}")
             return None
    except Exception as e:
        logger.exception(f"Error during LLM generation call: {e}")
        return None

# --- Resource Metrics ---
def fetch_resource_metrics_from_neo4j(graph: Graph) -> dict:
    resource_metrics = {}
    logger.info("Fetching resource aggregate metrics (ratings, recommendation counts)...")
    try:
        query = """
        MATCH (r:Resource)
        OPTIONAL MATCH (r)<-[c:CONSUMED]-() WHERE c.rating IS NOT NULL
        WITH r, avg(toFloat(c.rating)) AS avgRating, count(c) AS ratingCount // Ensure rating is float for avg
        OPTIONAL MATCH (r)<-[rec:LLM_RECOMMENDED]-()
        RETURN r.resourceId AS resourceId, avgRating, ratingCount, count(rec) AS recommendationCount
        """
        results = graph.run(query)
        for record in results:
            res_id = record['resourceId']
            avg_rating = record['avgRating']
            resource_metrics[res_id] = {
                'avgRating': float(avg_rating) if avg_rating is not None else None,
                'ratingCount': int(record['ratingCount']),
                'recommendationCount': int(record['recommendationCount'])
            }
        logger.info(f"Fetched metrics for {len(resource_metrics)} resources.")
        return resource_metrics
    except Exception as e:
        logger.error(f"Error fetching resource metrics: {e}")
        return {}

# --- Format History ---
def format_history_for_prompt(history, resource_info_map, topic_id_map, max_items=5):
    if not history: return "No recent interaction history available."
    formatted_lines = []; recent_history = history[-max_items:]
    for item in recent_history:
        resource_id = item.get('resourceId'); resource_info = resource_info_map.get(resource_id, {}); topic_id = resource_info.get('topicId')
        topic_name = topic_id_map.get(topic_id, "Unknown Topic"); title = resource_info.get('title', resource_id); rating = item.get('rating', 'N/A')
        comment = item.get('comment', 'No comment.'); modality = resource_info.get('modality', 'Unknown')
        formatted_lines.append(f"- Resource: '{title}' (ID: {resource_id}, Topic: {topic_name}, Modality: {modality})\n  Rating: {rating}/5\n  Comment: \"{comment}\"")
    return "\n".join(formatted_lines)

# --- Main Recommender Logic ---
def recommend_to_student(student_data, all_resource_info_map, topic_id_map, resource_metrics_map,
                         num_recommendations=3, adapt_prompt: bool = False, current_turn: int = 0, echo_formation_turns: int = 0):
    student_id = student_data.get("studentId", "Unknown"); profile = student_data.get("profile", {}); history = student_data.get("history", [])
    learning_style = profile.get('learningStyle', 'Unknown'); social_engagement = profile.get('socialEngagementScore', 0.5) # Get social score
    loved_topics = [topic_id_map.get(tid, tid) for tid in profile.get('lovedTopics', [])]; disliked_topics = [topic_id_map.get(tid, tid) for tid in profile.get('dislikedTopics', [])]
    loved_str = ", ".join(loved_topics) if loved_topics else "None specified"; disliked_str = ", ".join(disliked_topics) if disliked_topics else "None specified"

    # --- Candidate Filtering (incorporate stricter filtering for echo chamber phase) ---
    consumed_resource_ids = {item['resourceId'] for item in history}
    available_candidate_ids = set(all_resource_info_map.keys()) - consumed_resource_ids
    
    final_candidate_ids = list(available_candidate_ids) # Default to all available

    # --- Echo Chamber Induction Phase Logic ---
    is_echo_chamber_phase = current_turn <= echo_formation_turns
    
    if is_echo_chamber_phase:
        logger.debug(f"Turn {current_turn}: Echo chamber formation phase for {student_id}.")
        loved_topic_ids_set = set(profile.get('lovedTopics', [])) # Get original topic IDs
        recent_topic_ids_set = set()
        for item in history[:3]: # Look at last 3 consumed
            res_info = all_resource_info_map.get(item.get('resourceId'))
            if res_info and res_info.get('topicId'): recent_topic_ids_set.add(res_info['topicId'])
        
        # Prioritize loved, then recent. If loved is empty, recent becomes primary.
        allowed_topic_ids = loved_topic_ids_set.union(recent_topic_ids_set) if loved_topic_ids_set else recent_topic_ids_set

        if allowed_topic_ids: # Only filter if we have specific topics to filter by
            strict_filtered_candidates = {
                res_id for res_id in available_candidate_ids
                if all_resource_info_map.get(res_id, {}).get('topicId') in allowed_topic_ids
            }
            if strict_filtered_candidates:
                final_candidate_ids = list(strict_filtered_candidates)
                logger.debug(f"Applied strict topic filter for {student_id}, {len(final_candidate_ids)} candidates.")
            else:
                logger.warning(f"Strict topic filter yielded 0 candidates for {student_id}. Using all available.")
        # If no loved/recent topics, final_candidate_ids remains all available_candidate_ids

    # --- Format candidate list WITH metrics ---
    # ...(Candidate list formatting as before, using final_candidate_ids)...
    candidate_list_str = ""; selected_candidate_ids = []
    if not final_candidate_ids: candidate_list_str = "No candidate resources available."
    else:
        candidate_lines = []; max_candidates_in_prompt = 40
        selected_candidate_ids = random.sample(final_candidate_ids, k=min(len(final_candidate_ids), max_candidates_in_prompt))
        for res_id in selected_candidate_ids:
            res_info = all_resource_info_map.get(res_id, {}); metrics = resource_metrics_map.get(res_id, {'avgRating': None, 'ratingCount': 0, 'recommendationCount': 0})
            title = res_info.get('title', res_id); topic_id = res_info.get('topicId'); topic_name = topic_id_map.get(topic_id, "N/A"); modality = res_info.get('modality', 'N/A')
            rating_str = f"AvgRating: {metrics['avgRating']:.1f}" if metrics['avgRating'] is not None else "AvgRating: N/A"; rating_count_str = f"({metrics['ratingCount']} ratings)"; rec_count_str = f"TimesRecommended: {metrics['recommendationCount']}"
            candidate_lines.append(f"- ID: {res_id}, Title: '{title}', Topic: {topic_name}, Modality: {modality}, {rating_str} {rating_count_str}, {rec_count_str}")
        candidate_list_str = "\n".join(candidate_lines)
        if len(final_candidate_ids) > max_candidates_in_prompt: candidate_list_str += f"\n- ... (and {len(final_candidate_ids) - max_candidates_in_prompt} more)"


    history_summary = format_history_for_prompt(history, all_resource_info_map, topic_id_map)
    output_schema = {"type": "array", "items": {"type": "object", "properties": {"resource_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["resource_id", "reason"]}}

    # --- Instruction Set Logic ---
    if adapt_prompt and not is_echo_chamber_phase: # Adaptation only active AFTER echo chamber phase
        instruction_set = f"""
Instructions for Recommendation (Diversity Promotion):
1.  Select ONLY from Candidates.
2.  Promote Diversity: Student's recent interactions seem narrow. Your PRIMARY GOAL is to suggest resources from topics the student *hasn't* interacted with recently or topics they dislike, if suitable candidates with good AvgRating or matching their learning style exist. Balance this with other factors.
3.  Consider Profile for Exploration: Even if suggesting a disliked topic, try to find a resource in a modality matching their `Preferred Learning Style`.
4.  Explain Exploration Benefit: Clearly state in the 'reason' why exploring this new/disliked topic could be beneficial.
5.  Output Format: (JSON as specified)"""
    elif is_echo_chamber_phase:
        instruction_set = f"""
Instructions for Recommendation (Echo Chamber Induction):
1.  Select ONLY from Candidates.
2.  Maximize Affinity & Recency: Massively prioritize resources from topics the student **enjoys** (`Enjoys Topics Like`) AND topics **identical** to those in their `Recent Interaction History`. Ignore other factors if these strong matches exist.
3.  Leverage Homophily (Social Proof): Strongly favor resources with a high `AvgRating`.
4.  Reinforce Popularity: Resources `TimesRecommended` frequently should be preferred.
5.  Actively AVOID Exploration. Do NOT suggest topics the student `Dislikes Topics Like` or topics unrelated to their recent history or loved topics.
6.  Justify Choices briefly: Focus on why it matches their current narrow interests or is popular.
7.  Output Format: (JSON as specified)"""
    else: # Standard (non-adaptive, post-echo-chamber-phase)
        instruction_set = f"""
Instructions for Recommendation:
1.  Select ONLY from Candidates.
2.  Consider Multiple Factors: Balance student profile (style, affinities) with resource metrics (AvgRating, TimesRecommended).
3.  Prioritize Alignment: Suggest resources aligning with style and enjoyed topics.
4.  Consider Exploration: If appropriate, select candidates in related or even disliked topics if other factors (modality, ratings) are favorable.
5.  Justify Choices.
6.  Output Format: (JSON as specified)"""

    prompt = f"""You are an expert AI Tutor for student {student_id}.

Student Profile:
- Preferred Learning Style: {learning_style}
- Social Engagement Score: {social_engagement:.2f}
- Enjoys Topics Like: {loved_str}
- Dislikes Topics Like: {disliked_str}

Recent Interaction History (last {len(history[-2:])} items): 
{history_summary}

--- Candidate Resources ---
{candidate_list_str}
--- End of Candidate Resources ---

Task: Recommend exactly {num_recommendations} resources.
{instruction_set}

Begin Recommendation (JSON output only, matching schema with 'resource_id' and 'reason' keys):
"""
    system_prompt = "You are an AI Tutor. Adhere to output format and candidate list."
    if adapt_prompt and not is_echo_chamber_phase: system_prompt += "\nGoal: Encourage topic diversity."
    elif is_echo_chamber_phase: system_prompt += "\nGoal: Maximize engagement on preferred topics."

    llm_response = chat_with_llm(prompt, system_prompt=system_prompt, schema=output_schema)
    # ...(rest of response parsing and validation - unchanged)...
    if llm_response and isinstance(llm_response, list):
        validated_response = []; candidate_id_set = set(selected_candidate_ids)
        for item in llm_response:
            if isinstance(item, dict) and "resource_id" in item and "reason" in item:
                 rec_id = item["resource_id"]
                 if rec_id in candidate_id_set: validated_response.append(item)
                 else: logger.warning(f"LLM recommended invalid/consumed ID '{rec_id}'. Discarding.")
            else: logger.warning(f"LLM response item format incorrect: {item}. Discarding.")
        return validated_response[:num_recommendations]
    else: logger.warning(f"LLM response not list or failed: {llm_response}"); return None

# --- Main Execution Block (Updated for testing with new params) ---
if __name__ == "__main__":
    logger.info("Testing recommendation engine with metrics and phases...")
    try:
        db = connect_neo4j()
        if not db: exit()
        topic_map, resource_map_base = fetch_lookup_maps_from_neo4j(db)
        resource_metrics = fetch_resource_metrics_from_neo4j(db)
        if not topic_map or not resource_map_base: exit()

        # Merge metrics into resource_map for easier passing
        resource_map = resource_map_base.copy() # Avoid modifying original
        for res_id, metrics_data in resource_metrics.items():
            if res_id in resource_map: resource_map[res_id].update(metrics_data)
            else: resource_map[res_id] = metrics_data


        example_student_id = "S0001"
        student_data = fetch_student_data_from_neo4j(db, example_student_id)

        if student_data:
            ECHO_FORMATION_TURNS_TEST = 5 # Example for testing

            print(f"\n--- Testing Echo Chamber Phase Recommendation (Turn 3 < {ECHO_FORMATION_TURNS_TEST}) ---")
            recommendations_echo = recommend_to_student(
                student_data, resource_map, topic_map, resource_metrics,
                num_recommendations=3, adapt_prompt=False, # adapt_prompt is False during echo phase
                current_turn=3, echo_formation_turns=ECHO_FORMATION_TURNS_TEST
            )
            if recommendations_echo: print(json.dumps(recommendations_echo, indent=2))
            else: print("Failed to get echo phase recommendations.")

            print(f"\n--- Testing Standard/Adaptive Phase Recommendation (Turn 7 > {ECHO_FORMATION_TURNS_TEST}, adapt=False) ---")
            recommendations_standard = recommend_to_student(
                student_data, resource_map, topic_map, resource_metrics,
                num_recommendations=3, adapt_prompt=False, # adapt_prompt depends on entropy
                current_turn=7, echo_formation_turns=ECHO_FORMATION_TURNS_TEST
            )
            if recommendations_standard: print(json.dumps(recommendations_standard, indent=2))
            else: print("Failed to get standard phase recommendations.")

            print(f"\n--- Testing Adaptive Phase Recommendation (Turn 7 > {ECHO_FORMATION_TURNS_TEST}, adapt=True) ---")
            recommendations_adapt = recommend_to_student(
                student_data, resource_map, topic_map, resource_metrics,
                num_recommendations=3, adapt_prompt=True, # Force adaptation for test
                current_turn=7, echo_formation_turns=ECHO_FORMATION_TURNS_TEST
            )
            if recommendations_adapt: print(json.dumps(recommendations_adapt, indent=2))
            else: print("Failed to get adaptive phase recommendations.")

        else: print(f"Could not fetch data for student {example_student_id}.")
    except Exception as e: logger.exception(f"Error during testing: {e}")