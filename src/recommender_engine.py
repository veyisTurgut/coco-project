import json, logging, os, random, pandas as pd
from py2neo import Graph # Import Neo4j driver
from llm.curl_vertex import CurlVertex, vertex_credentials
from llm.mylib import *
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Best practice: Use environment variables or a config file
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://gcloud.madlen.io:7687") # Default if env var not set
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") 

graph_db = None

def connect_neo4j():
    """Establishes connection to Neo4j."""
    global graph_db
    if graph_db is None:
        try:
            graph_db = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            # Test connection
            graph_db.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j at {NEO4J_URI}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            graph_db = None # Ensure it's None if connection fails
            raise # Re-raise exception to halt execution if connection is critical
    return graph_db


def fetch_lookup_maps_from_neo4j(graph: Graph):
    """Fetches topic and resource information from Neo4j."""
    topic_id_map = {}
    resource_info_map = {}

    try:
        # Fetch Topics
        topic_query = "MATCH (t:Topic) RETURN t.topicId AS topicId, t.name AS name"
        topic_results = graph.run(topic_query)
        for record in topic_results:
            topic_id_map[record['topicId']] = record['name']
        logger.info(f"Fetched {len(topic_id_map)} topics.")

        # Fetch Resources
        resource_query = """
        MATCH (r:Resource)
        OPTIONAL MATCH (r)-[:ABOUT_TOPIC]->(t:Topic) // Use OPTIONAL MATCH in case some resources aren't linked
        RETURN r.resourceId AS resourceId, r.title AS title, r.type AS type,
               r.modality AS modality, t.topicId AS topicId
        """
        resource_results = graph.run(resource_query)
        for record in resource_results:
            resource_info_map[record['resourceId']] = {
                'resourceId': record['resourceId'],
                'title': record['title'],
                'type': record['type'],
                'modality': record['modality'],
                'topicId': record['topicId'] # Will be None if not linked
            }
        logger.info(f"Fetched {len(resource_info_map)} resources.")

    except Exception as e:
        logger.error(f"Error fetching lookup maps from Neo4j: {e}")
        # Depending on requirements, you might want to raise the error or return empty maps
        raise

    return topic_id_map, resource_info_map


def fetch_student_data_from_neo4j(graph: Graph, student_id: str, history_limit=15):
    """Fetches profile and recent interaction history for a specific student."""
    student_data = {"studentId": student_id, "profile": {}, "history": []}

    try:
        # Fetch Profile
        profile_query = """
        MATCH (s:Student {studentId: $studentId})
        RETURN s.learningStyle AS learningStyle,
               s.lovedTopicIds AS lovedTopicIds,    // Assuming these are stored as JSON strings
               s.dislikedTopicIds AS dislikedTopicIds // Assuming these are stored as JSON strings
        LIMIT 1
        """
        profile_result = graph.run(profile_query, studentId=student_id).data()

        if not profile_result:
            logger.warning(f"Student profile not found for ID: {student_id}")
            return None # Or raise an error

        profile_record = profile_result[0]
        student_data["profile"]["learningStyle"] = profile_record.get("learningStyle", "Unknown")
        # Parse JSON strings back into lists
        try:
            student_data["profile"]["lovedTopics"] = json.loads(profile_record.get("lovedTopicIds") or '[]')
        except (json.JSONDecodeError, TypeError):
             logger.warning(f"Could not parse lovedTopicIds for student {student_id}. Defaulting to empty list.")
             student_data["profile"]["lovedTopics"] = []
        try:
            student_data["profile"]["dislikedTopics"] = json.loads(profile_record.get("dislikedTopicIds") or '[]')
        except (json.JSONDecodeError, TypeError):
             logger.warning(f"Could not parse dislikedTopicIds for student {student_id}. Defaulting to empty list.")
             student_data["profile"]["dislikedTopics"] = []


        # Fetch History (Consumed relationships with properties)
        history_query = """
        MATCH (s:Student {studentId: $studentId})-[c:CONSUMED]->(r:Resource)
        RETURN r.resourceId AS resourceId,
               c.rating AS rating,
               c.comment AS comment,
               c.timestamp AS timestamp // Make sure timestamp was loaded correctly
        ORDER BY c.timestamp DESC
        LIMIT $limit
        """
        history_result = graph.run(history_query, studentId=student_id, limit=history_limit)
        student_data["history"] = [dict(record) for record in history_result] # Convert records to list of dicts

        logger.info(f"Fetched profile and {len(student_data['history'])} history items for student {student_id}.")

    except Exception as e:
        logger.error(f"Error fetching data for student {student_id} from Neo4j: {e}")
        return None # Indicate failure

    return student_data


def chat_with_llm(prompt: str, system_prompt: str = None, schema: str = None):
    if not system_prompt:
        system_prompt = """
        You are an expert AI Tutor designing a personalized learning path for a 9th-grade student using resources from a curriculum covering Physics, Biology, Chemistry, and Mathematics.
        """
    vertex_credentials.initialize()
    # Step 1: Configure LLM
    llm_config = LLMConfig({
        "model_arch": ChatMainLLM.VERTEXLLM.value,
        "model_name": ChatMainLLMName.VERTEX_GEMINI_25_FLASH_PREVIEW,  # Correct: Pass enum, not string
        "temperature": 0.7,
        "top_k": 10,
        "top_p": 0.95,
        "max_output_tokens": 500,
        "llm_region": "us-central1",  # Adjust the region as necessary
        "response_mimetype": "application/json",
        "responseSchema": schema
    })

    # Initialize the CurlVertex instance
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

    # Step 3: Generate response
    response_content = ""
    for response in curl_vertex.generate(
        instruction_prompt=system_prompt,  # Include the system prompt here
        chat_history=chat_history,
        is_streaming=True,  # Enable streaming for token-by-token responses
        return_tokens=True,  # Retrieve token usage info
        timeout=60
    ):
        response_content += response.content

    # Step 4: Print and return final response
    #print("Final Response:")
    #print(response_content)
    return response_content


def format_history_for_prompt(history, resource_info_map, topic_id_map, max_items=5):
    """Formats recent history for inclusion in the LLM prompt."""
    if not history:
        return "No recent interaction history available."

    formatted_lines = []
    # Sort history by timestamp if available, otherwise take last items
    # Assuming history is a list of dicts like {'resourceId': ..., 'rating': ..., 'comment': ...}
    # If timestamps were loaded, sort by them descending. Here, we just take the latest entries.
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

        formatted_lines.append(
            f"- Resource: '{title}' (ID: {resource_id}, Topic: {topic_name}, Modality: {modality})\n"
            f"  Rating: {rating}/5\n"
            f"  Comment: \"{comment}\""
        )
    return "\n".join(formatted_lines)


def recommend_to_student(student_data, all_resource_info_map, topic_id_map, num_recommendations=3):
    """
    Generates a prompt for an LLM to recommend resources based on student profile and history.

    Args:
        student_data (dict): Containing 'studentId', 'profile' (with 'learningStyle',
                             'lovedTopics', 'dislikedTopics'), and 'history' (list of interactions).
        all_resource_info_map (dict): Lookup map for resource details (title, topicId, modality).
        topic_id_map (dict): Lookup map for topic names.
        num_recommendations (int): Number of recommendations to ask for.

    Returns:
        str: The generated prompt for the LLM.
    """

    student_id = student_data.get("studentId", "Unknown")
    profile = student_data.get("profile", {})
    history = student_data.get("history", [])

    learning_style = profile.get('learningStyle', 'Unknown')
    loved_topics = [topic_id_map.get(tid, tid) for tid in profile.get('lovedTopics', [])] # Get names
    disliked_topics = [topic_id_map.get(tid, tid) for tid in profile.get('dislikedTopics', [])]

    # Format loved/disliked topics for readability
    loved_str = ", ".join(loved_topics) if loved_topics else "None specified"
    disliked_str = ", ".join(disliked_topics) if disliked_topics else "None specified"

 # --- Filter candidate resources ---
    consumed_resource_ids = {item['resourceId'] for item in history}
    candidate_ids = set(all_resource_info_map.keys()) - consumed_resource_ids

    # Format candidate list for the prompt
    candidate_list_str = ""
    if not candidate_ids:
        candidate_list_str = "No candidate resources available (student consumed all?)."
    else:
        candidate_lines = []
        max_candidates_in_prompt = 50
        selected_candidate_ids = random.sample(list(candidate_ids), k=min(len(candidate_ids), max_candidates_in_prompt))

        for res_id in selected_candidate_ids:
            res_info = all_resource_info_map.get(res_id, {})
            title = res_info.get('title', res_id)
            topic_id = res_info.get('topicId')
            topic_name = topic_id_map.get(topic_id, "Unknown Topic")
            modality = res_info.get('modality', 'Unknown')
            candidate_lines.append(f"- ID: {res_id}, Title: '{title}', Topic: {topic_name}, Modality: {modality}")
        candidate_list_str = "\n".join(candidate_lines)
        if len(candidate_ids) > max_candidates_in_prompt:
             candidate_list_str += f"\n- ... (and {len(candidate_ids) - max_candidates_in_prompt} more available resources)"


    # Format history
    history_summary = format_history_for_prompt(history, all_resource_info_map, topic_id_map)

  # --- Define the desired JSON output schema (unchanged) ---
    output_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "Resource ID": {"type": "string"},
                "Reason": {"type": "string"}
            },
            "required": ["Resource ID", "Reason"]
        }
    }
    
    # Construct the prompt
    prompt = f"""You are an expert AI Tutor designing a personalized learning path for a 9th-grade student (ID: {student_id}) using resources from a curriculum covering Physics, Biology, Chemistry, and Mathematics.

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

Available resource modalities mentioned in the candidate list include: Visual (Video, Slideshow, Simulation), Audio (Podcast), Reading/Writing (Article), Kinaesthetic (Quiz, Simulation), Mixed.

Task:
Based *only* on the provided profile, interaction history, and the candidate resource list, recommend exactly {num_recommendations} specific resources for this student to engage with next.

Instructions for Recommendation:
1.  **Select ONLY from Candidates:** Choose Resource IDs strictly from the 'Candidate Resources' list provided above. Do NOT invent resource IDs like R0XXX.
2.  **Prioritize Alignment:** Suggest resources that align with the student's learning style ({learning_style}) and topics they enjoy ({loved_str}), if suitable candidates exist.
3.  **Consider Exploration:** If appropriate, select a candidate resource in a topic related to their liked topics or gently re-introduce a disliked topic ({disliked_str}) using a preferred modality, explaining the rationale.
4.  **Output Format:** Provide the recommendations as a JSON array of objects, matching the schema provided. Each object must have a 'Resource ID' key (with a value from the candidate list) and a 'Reason' key (1 sentence justification referencing the student's profile/history/candidate details).

Begin Recommendation (JSON output only):
"""

    res = chat_with_llm(prompt)
    return res


if __name__ == "__main__":
    logger.info("Starting recommendation engine...")

    try:
        # Connect to Neo4j
        db = connect_neo4j()
        if not db:
             exit() # Stop if connection failed

        # Fetch lookup maps once
        topic_map, resource_map = fetch_lookup_maps_from_neo4j(db)
        if not topic_map or not resource_map:
            logger.error("Failed to fetch necessary lookup maps from Neo4j. Exiting.")
            exit()

        # --- Get recommendations for a specific student ---
        example_student_id = "S0001" # Or get from input, loop, etc.
        logger.info(f"Fetching data for student: {example_student_id}")
        student_data = fetch_student_data_from_neo4j(db, example_student_id)

        if student_data:
            logger.info(f"Generating recommendations for student: {example_student_id}")
            recommendations = recommend_to_student(
                student_data,
                resource_map,
                topic_map,
                num_recommendations=3
            )

            print(f"\n--- LLM Recommendations for {example_student_id} ---")
            if recommendations:
                print(recommendations)
            else:
                print("Failed to get valid recommendations from LLM.")
            print("------------------------------------")
        else:
            print(f"Could not fetch data for student {example_student_id}.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred in the main execution block: {e}")