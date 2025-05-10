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

# --- Neo4j Connection (Keep connect_neo4j) ---
def connect_neo4j():
    """Establishes connection to Neo4j."""
    global graph_db
    if graph_db is None:
        # ...(rest of connect_neo4j as before)...
        try:
            graph_db = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            graph_db.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j at {NEO4J_URI}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            graph_db = None
            raise
    return graph_db

# --- Data Fetching (Keep fetch_lookup_maps_from_neo4j and fetch_student_data_from_neo4j) ---
def fetch_lookup_maps_from_neo4j(graph: Graph):
    # ...(rest of fetch_lookup_maps_from_neo4j as before)...
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
    # ...(rest of fetch_student_data_from_neo4j as before)...
    student_data = {"studentId": student_id, "profile": {}, "history": []}
    try:
        profile_query = """ MATCH (s:Student {studentId: $studentId})
        RETURN s.learningStyle AS learningStyle, s.lovedTopicIds AS lovedTopicIds, s.dislikedTopicIds AS dislikedTopicIds LIMIT 1 """
        profile_result = graph.run(profile_query, studentId=student_id).data()
        if not profile_result: logger.warning(f"Profile not found: {student_id}"); return None
        profile_record = profile_result[0]
        student_data["profile"]["learningStyle"] = profile_record.get("learningStyle", "Unknown")
        try: student_data["profile"]["lovedTopics"] = json.loads(profile_record.get("lovedTopicIds") or '[]')
        except: student_data["profile"]["lovedTopics"] = []
        try: student_data["profile"]["dislikedTopics"] = json.loads(profile_record.get("dislikedTopicIds") or '[]')
        except: student_data["profile"]["dislikedTopics"] = []

        history_query = """ MATCH (s:Student {studentId: $studentId})-[c:CONSUMED]->(r:Resource)
        RETURN r.resourceId AS resourceId, c.rating AS rating, c.comment AS comment, c.timestamp AS timestamp
        ORDER BY c.timestamp DESC LIMIT $limit """
        history_result = graph.run(history_query, studentId=student_id, limit=history_limit)
        student_data["history"] = [dict(record) for record in history_result]
        logger.debug(f"Fetched profile and {len(student_data['history'])} history items for {student_id}.")
    except Exception as e: logger.error(f"Error fetching student data {student_id}: {e}"); return None
    return student_data


# --- LLM Interaction (Updated for robustness) ---
def chat_with_llm(prompt: str, system_prompt: str = None, schema: dict = None): # Added type hint for schema
    if not system_prompt:
        system_prompt = "You are a helpful AI assistant." # Generic default

    vertex_credentials.initialize()

    # Configure LLM
    llm_config = LLMConfig({
        "model_arch": ChatMainLLM.VERTEXLLM.value,
        "model_name": ChatMainLLMName.VERTEX_GEMINI_25_FLASH_PREVIEW,
        "temperature": 0.6, # Adjusted temp
        "top_k": 10, "top_p": 0.95, "max_output_tokens": 4500,
        "llm_region": "us-central1",
        "response_mimetype": "application/json", # Expect JSON
        "responseSchema": schema # Pass the schema
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

        # Attempt to parse the JSON response
        try:
            parsed_json = json.loads(response_content)
            return parsed_json
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse LLM JSON response: {json_err}. Response: {response_content}")
            return None
        except TypeError as type_err:
             logger.error(f"LLM response was not a string or bytes-like object? Type: {type(response_content)}. Error: {type_err}")
             return None

    except Exception as e:
        logger.exception(f"Error during LLM generation call: {e}") # Use logger.exception to include traceback
        return None

# Add this function after fetch_student_data_from_neo4j

def fetch_resource_metrics_from_neo4j(graph: Graph) -> dict:
    """Fetches aggregate ratings and recommendation counts for all resources."""
    resource_metrics = {}
    logger.info("Fetching resource aggregate metrics (ratings, recommendation counts)...")
    try:
        # Query for ratings and recommendation counts
        # Using OPTIONAL MATCH allows resources with no ratings/recs to still be included
        query = """
        MATCH (r:Resource)
        OPTIONAL MATCH (r)<-[c:CONSUMED]-()
        WITH r, avg(c.rating) AS avgRating, count(c) AS ratingCount
        OPTIONAL MATCH (r)<-[rec:LLM_RECOMMENDED]-()
        RETURN r.resourceId AS resourceId,
               avgRating,
               ratingCount,
               count(rec) AS recommendationCount
        """
        results = graph.run(query)
        for record in results:
            res_id = record['resourceId']
            # Convert py2neo Float to Python float, handle None
            avg_rating = record['avgRating']
            resource_metrics[res_id] = {
                'avgRating': float(avg_rating) if avg_rating is not None else None,
                'ratingCount': int(record['ratingCount']), # Convert to int
                'recommendationCount': int(record['recommendationCount']) # Convert to int
            }
        logger.info(f"Fetched metrics for {len(resource_metrics)} resources.")
        return resource_metrics
    except Exception as e:
        logger.error(f"Error fetching resource metrics from Neo4j: {e}")
        return {} # Return empty dict on error

# --- Helper: Format History (unchanged) ---
def format_history_for_prompt(history, resource_info_map, topic_id_map, max_items=5):
    if not history: return "No recent interaction history available."
    formatted_lines = []
    recent_history = history[-max_items:]
    for item in recent_history:
        resource_id = item.get('resourceId')
        resource_info = resource_info_map.get(resource_id, {})
        topic_id = resource_info.get('topicId')
        topic_name = topic_id_map.get(topic_id, "Unknown Topic")
        title = resource_info.get('title', resource_id)
        rating = item.get('rating', 'N/A')
        comment = item.get('comment', 'No comment.')
        modality = resource_info.get('modality', 'Unknown')
        formatted_lines.append(f"- Resource: '{title}' (ID: {resource_id}, Topic: {topic_name}, Modality: {modality})\n"
                                f"  Rating: {rating}/5\n  Comment: \"{comment}\"")
    return "\n".join(formatted_lines)


# --- Main Recommender Logic ---
def recommend_to_student(student_data, all_resource_info_map, topic_id_map, resource_metrics_map, 
                         num_recommendations=3, adapt_prompt: bool = False):
    """Generates recommendations using an LLM, grounded by candidate resources and metrics, optionally adapting prompt."""
    student_id = student_data.get("studentId", "Unknown")
    profile = student_data.get("profile", {})
    history = student_data.get("history", [])

    learning_style = profile.get('learningStyle', 'Unknown')
    loved_topics = [topic_id_map.get(tid, tid) for tid in profile.get('lovedTopics', [])]
    disliked_topics = [topic_id_map.get(tid, tid) for tid in profile.get('dislikedTopics', [])]
    loved_str = ", ".join(loved_topics) if loved_topics else "None specified"
    disliked_str = ", ".join(disliked_topics) if disliked_topics else "None specified"

    # Filter candidate resources
    consumed_resource_ids = {item['resourceId'] for item in history}
    candidate_ids = set(all_resource_info_map.keys()) - consumed_resource_ids

    # --- Format candidate list WITH metrics ---
    candidate_list_str = ""
    selected_candidate_ids = [] # Keep track of IDs sent to LLM
    if not candidate_ids:
        candidate_list_str = "No candidate resources available."
    else:
        candidate_lines = []
        max_candidates_in_prompt = 40
        selected_candidate_ids = random.sample(list(candidate_ids), k=min(len(candidate_ids), max_candidates_in_prompt))

        for res_id in selected_candidate_ids:
            res_info = all_resource_info_map.get(res_id, {})
            metrics = resource_metrics_map.get(res_id, {'avgRating': None, 'ratingCount': 0, 'recommendationCount': 0}) # Get metrics

            title = res_info.get('title', res_id)
            topic_id = res_info.get('topicId')
            topic_name = topic_id_map.get(topic_id, "N/A")
            modality = res_info.get('modality', 'N/A')

            # Format metrics string
            rating_str = f"AvgRating: {metrics['avgRating']:.1f}" if metrics['avgRating'] is not None else "AvgRating: N/A"
            rating_count_str = f"({metrics['ratingCount']} ratings)"
            rec_count_str = f"TimesRecommended: {metrics['recommendationCount']}"

            candidate_lines.append(f"- ID: {res_id}, Title: '{title}', Topic: {topic_name}, Modality: {modality}, "
                                   f"{rating_str} {rating_count_str}, {rec_count_str}")

        candidate_list_str = "\n".join(candidate_lines)
        if len(candidate_ids) > max_candidates_in_prompt:
             candidate_list_str += f"\n- ... (and {len(candidate_ids) - max_candidates_in_prompt} more available resources)"

    history_summary = format_history_for_prompt(history, all_resource_info_map, topic_id_map)

    # Define the desired JSON output schema (using 'resource_id' as key)
    output_schema = {
        "type": "array",
        "description": f"A list of exactly {num_recommendations} resource recommendations.",
        "items": {
            "type": "object",
            "properties": {
                "resource_id": {"type": "string", "description": "The unique ID of the recommended resource, chosen ONLY from the candidate list provided."},
                "reason": {"type": "string", "description": "A brief (1 sentence) justification for recommending this resource to this specific student."}
            },
            "required": ["resource_id", "reason"]
        }
    }

    # --- Adaptation Instruction (same as before) ---
    adaptation_instruction = ""
    output_format_instruction = "" # Define base instruction
    if adapt_prompt:
        adaptation_instruction = ("\n4.  **Promote Diversity:** Student's recent interactions seem narrow. Prioritize suggesting candidate resources from topics the student *hasn't* interacted with recently, balancing this with profile alignment and resource metrics. Explain this exploration benefit.")
        output_format_instruction = ("\n5.  **Output Format:** Provide the recommendations as a JSON array of objects matching the schema. Keys must be 'resource_id' and 'reason'.")
    else:
        # Instruction 4 becomes Output Format if not adapting
        output_format_instruction = ("\n4.  **Output Format:** Provide the recommendations as a JSON array of objects matching the schema. Keys must be 'resource_id' and 'reason'. Justify based on profile/history and resource metrics.")


    # --- Construct the **UPDATED** prompt ---
    prompt = f"""You are an expert AI Tutor designing a personalized learning path for a 9th-grade student (ID: {student_id}).

Student Profile:
- Preferred Learning Style: {learning_style}
- Enjoys Topics Like: {loved_str}
- Dislikes Topics Like: {disliked_str}

Recent Interaction History (last {len(history[-5:])} items):
{history_summary}

--- Candidate Resources ---
Below is a list of available resources the student has NOT yet consumed, along with community feedback and system recommendation frequency. You MUST choose recommendations ONLY from this list.

{candidate_list_str}
--- End of Candidate Resources ---

Task:
Based *only* on the student's profile, interaction history, and the candidate resource list (including their metrics), recommend exactly {num_recommendations} specific resources for this student to engage with next.

Instructions for Recommendation:
1.  **Select ONLY from Candidates:** Choose 'ID' values strictly from the 'Candidate Resources' list.
2.  **Prioritize Strong Affinity & Recency:**
    *   Strongly prefer resources from topics the student **enjoys** (`Enjoys Topics Like`).
    *   Strongly prefer resources from topics **identical or very similar** to those in their `Recent Interaction History`.
3.  **Leverage Social Proof & Popularity (Engagement Signals):**
    *   Favor resources with a high `AvgRating`, especially if based on a good `ratingCount` (e.g., > 5). This indicates other students liked it.
    *   Consider resources that have been `TimesRecommended` before, as this might indicate system-identified relevance or general popularity.
4.  **Modality Match:** If possible, align with the student's `Preferred Learning Style`.
5.  **Minimize Exploration (Unless No Other Option):** De-prioritize topics the student `Dislikes Topics Like` or topics unrelated to their recent history or loved topics, unless there are no strong matches from preferred areas. If suggesting exploration, acknowledge it.
6.  **Justify Choices:** Briefly explain *why* each resource is a good engagement-focused recommendation for *this student* in the 'reason' field, referencing their profile, recent history, or the resource's popularity/ratings.

{adaptation_instruction}{output_format_instruction}

"""

    # System prompt (can remain the same or be tweaked)
    system_prompt = """
    You are an expert AI Tutor generating personalized resource recommendations considering student profiles and community feedback.
    You MUST strictly adhere to the output format specified and only use Resource IDs provided in the prompt's candidate list.
    """
    if adapt_prompt:
         system_prompt += "\nYour primary goal now is to encourage topic diversity and exploration."

    # --- Call LLM ---
    # Pass the schema defined above to ensure correct output structure
    llm_response = chat_with_llm(prompt, system_prompt=system_prompt, schema=output_schema)

    # --- Return parsed JSON or None (with validation using 'resource_id') ---
    if llm_response and isinstance(llm_response, list):
        validated_response = []
        candidate_id_set = set(selected_candidate_ids) # Use the IDs actually sent
        for item in llm_response:
            # Check for correct keys based on the schema requested
            if isinstance(item, dict) and "resource_id" in item and "reason" in item:
                 rec_id = item["resource_id"] # Use the key defined in the schema
                 if rec_id in candidate_id_set: validated_response.append(item)
                 else: logger.warning(f"LLM recommended invalid/consumed ID '{rec_id}'. Discarding.")
            else: logger.warning(f"LLM response item format incorrect (expected 'resource_id', 'reason'): {item}. Discarding.")
        return validated_response[:num_recommendations]
    else:
        logger.warning(f"LLM response was not a list or failed: {llm_response}")
        return None

# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("Testing recommendation engine functions with resource metrics...")
    try:
        db = connect_neo4j()
        if not db: exit()
        # Fetch ALL necessary data upfront for testing
        topic_map, resource_map = fetch_lookup_maps_from_neo4j(db)
        resource_metrics = fetch_resource_metrics_from_neo4j(db) # Fetch metrics
        if not topic_map or not resource_map: exit() # Metrics can be empty initially

        # Merge metrics into resource_map for easier passing
        for res_id, metrics in resource_metrics.items():
            if res_id in resource_map:
                resource_map[res_id].update(metrics)
            else:
                 # Should not happen if resource_map is comprehensive, but handle just in case
                 logger.warning(f"Metrics found for resource {res_id} not present in basic resource info map.")
                 # resource_map[res_id] = metrics # Option to add it if missing


        example_student_id = "S0001"
        student_data = fetch_student_data_from_neo4j(db, example_student_id)

        if student_data:
            print("\n--- Testing Standard Recommendation with Metrics ---")
            recommendations = recommend_to_student(
                student_data,
                resource_map, # Pass combined map
                topic_map,
                resource_metrics, # Also pass separately if needed elsewhere, but map is main carrier now
                num_recommendations=3,
                adapt_prompt=False
            )
            if recommendations: print(json.dumps(recommendations, indent=2))
            else: print("Failed to get standard recommendations.")

            print("\n--- Testing Adaptive Recommendation with Metrics ---")
            adapt_recommendations = recommend_to_student(
                student_data,
                resource_map, # Pass combined map
                topic_map,
                resource_metrics,
                num_recommendations=3,
                adapt_prompt=True
            )
            if adapt_recommendations: print(json.dumps(adapt_recommendations, indent=2))
            else: print("Failed to get adaptive recommendations.")

        else: print(f"Could not fetch data for student {example_student_id}.")
    except Exception as e: logger.exception(f"Error during testing: {e}")
    logger.info("Testing recommendation engine functions...")
    try:
        db = connect_neo4j()
        if not db: exit()
        topic_map, resource_map = fetch_lookup_maps_from_neo4j(db)
        if not topic_map or not resource_map: exit()

        example_student_id = "S0001"
        student_data = fetch_student_data_from_neo4j(db, example_student_id)

        if student_data:
            print("\n--- Testing Standard Recommendation ---")
            recommendations = recommend_to_student(student_data, resource_map, topic_map, resource_metrics, num_recommendations=3, adapt_prompt=False)
            if recommendations: print(json.dumps(recommendations, indent=2))
            else: print("Failed to get standard recommendations.")

            print("\n--- Testing Adaptive Recommendation ---")
            adapt_recommendations = recommend_to_student(student_data, resource_map, topic_map, resource_metrics, num_recommendations=3, adapt_prompt=True)
            if adapt_recommendations: print(json.dumps(adapt_recommendations, indent=2))
            else: print("Failed to get adaptive recommendations.")

        else: print(f"Could not fetch data for student {example_student_id}.")
    except Exception as e: logger.exception(f"Error during testing: {e}")