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
        "top_k": 10, "top_p": 0.95, "max_output_tokens": 500,
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


# --- Main Recommender Logic (Updated signature & prompt) ---
def recommend_to_student(student_data, all_resource_info_map, topic_id_map,
                         num_recommendations=3, adapt_prompt: bool = False): # Added adapt_prompt flag
    """Generates recommendations using an LLM, grounded by candidate resources, optionally adapting prompt."""
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

    candidate_list_str = ""
    selected_candidate_ids = [] # Keep track of IDs sent to LLM
    if not candidate_ids: candidate_list_str = "No candidate resources available."
    else:
        candidate_lines = []
        max_candidates_in_prompt = 50
        selected_candidate_ids = random.sample(list(candidate_ids), k=min(len(candidate_ids), max_candidates_in_prompt))
        for res_id in selected_candidate_ids:
            res_info = all_resource_info_map.get(res_id, {})
            title = res_info.get('title', res_id); topic_id = res_info.get('topicId')
            topic_name = topic_id_map.get(topic_id, "N/A"); modality = res_info.get('modality', 'N/A')
            candidate_lines.append(f"- ID: {res_id}, Title: '{title}', Topic: {topic_name}, Modality: {modality}")
        candidate_list_str = "\n".join(candidate_lines)
        if len(candidate_ids) > max_candidates_in_prompt: candidate_list_str += f"\n- ... (and {len(candidate_ids) - max_candidates_in_prompt} more)"

    history_summary = format_history_for_prompt(history, all_resource_info_map, topic_id_map)

    # Define the desired JSON output schema
    output_schema = {"type": "array", "items": {"type": "object", "properties": {"resource_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["resource_id", "reason"]}}

    # --- Adaptation Instruction ---
    adaptation_instruction = ""
    if adapt_prompt:
        adaptation_instruction = ("\n4.  **Promote Diversity:** The student's recent interactions seem narrow. "
                                  "Prioritize suggesting resources from topics the student *hasn't* interacted with recently, "
                                  "even if they are neutral or slightly disliked, especially if the modality matches their style. "
                                  "Explain this exploration benefit in the reason.")
    else:
        adaptation_instruction = ("\n4.  **Output Format:** Provide the recommendations as a JSON array of objects, matching the schema provided. "
                                  "Each object must have a 'resource_id' key (with a value from the candidate list) and a 'reason' key (1 sentence justification referencing the student's profile/history/candidate details).")


    # --- Construct the REVISED prompt ---
    prompt = f"""You are an expert AI Tutor designing a personalized learning path for a 9th-grade student (ID: {student_id}).

Student Profile:
- Preferred Learning Style: {learning_style}
- Enjoys Topics Like: {loved_str}
- Dislikes Topics Like: {disliked_str}

Recent Interaction History (last {len(history[-5:])} items):
{history_summary}

--- Candidate Resources ---
Below is a list of available resources the student has NOT yet consumed. You MUST choose recommendations ONLY from this list.

{candidate_list_str}
--- End of Candidate Resources ---

Task:
Based *only* on the provided profile, interaction history, and the candidate resource list, recommend exactly {num_recommendations} specific resources for this student to engage with next.

Instructions for Recommendation:
1.  **Select ONLY from Candidates:** Choose Resource IDs strictly from the 'Candidate Resources' list provided above. Do NOT invent resource IDs.
2.  **Prioritize Alignment:** Suggest resources that align with the student's learning style ({learning_style}) and topics they enjoy ({loved_str}), if suitable candidates exist.
3.  **Avoid Exploration:** Do NOT suggest resources outside the student's narrow band of recent/liked topics unless explicitly instructed by an adaptation flag.

{adaptation_instruction }  # Keep adaptation logic separate

output schema is this:
{output_schema}

Begin Recommendation (JSON output only):
"""

    # System prompt can also be adjusted based on adaptation
    system_prompt = """
    You are an expert AI Tutor generating personalized resource recommendations.
    You MUST strictly adhere to the output format specified and only use Resource IDs provided in the prompt's candidate list.
    """
    if adapt_prompt:
         system_prompt += "\nYour primary goal now is to encourage topic diversity and exploration for this student."

    # --- Call LLM ---
    llm_response = chat_with_llm(prompt, system_prompt=system_prompt)

    # --- Return parsed JSON or None (with validation) ---
    if llm_response and isinstance(llm_response, list):
        validated_response = []
        candidate_id_set = set(selected_candidate_ids) # Use the IDs actually sent
        for item in llm_response:
            if isinstance(item, dict) and "resource_id" in item and "reason" in item:
                 rec_id = item["resource_id"]
                 if rec_id in candidate_id_set: validated_response.append(item)
                 else: logger.warning(f"LLM recommended invalid/consumed ID '{rec_id}'. Discarding.")
            else: logger.warning(f"LLM response item format incorrect: {item}. Discarding.")
        # Return only validated recommendations, up to the requested number
        # If fewer valid recs than requested, return what's valid
        return validated_response[:num_recommendations]
    else:
        logger.warning(f"LLM response was not a list or failed: {llm_response}")
        return None

# This block is now just for testing this specific file,
# the simulation loop will be in simulation.py
if __name__ == "__main__":
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
            recommendations = recommend_to_student(student_data, resource_map, topic_map, num_recommendations=3, adapt_prompt=False)
            if recommendations: print(json.dumps(recommendations, indent=2))
            else: print("Failed to get standard recommendations.")

            print("\n--- Testing Adaptive Recommendation ---")
            adapt_recommendations = recommend_to_student(student_data, resource_map, topic_map, num_recommendations=3, adapt_prompt=True)
            if adapt_recommendations: print(json.dumps(adapt_recommendations, indent=2))
            else: print("Failed to get adaptive recommendations.")

        else: print(f"Could not fetch data for student {example_student_id}.")
    except Exception as e: logger.exception(f"Error during testing: {e}")