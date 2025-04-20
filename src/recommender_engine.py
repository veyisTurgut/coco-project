import json
import pandas as pd

students_df = pd.read_csv('synthetic_data/students.csv')
resources_df = pd.read_csv('synthetic_data/resources.csv')
consumed_df = pd.read_csv('synthetic_data/consumed.csv')
topics_df = pd.read_csv('synthetic_data/topics.csv')

from llm.curl_vertex import CurlVertex, vertex_credentials
from llm.mylib import *
from datetime import datetime, timedelta
import logging
logger = logging.getLogger(__name__)


def chat_with_llm(prompt: str, system_prompt: str = None, schema: str = None):
    if not system_prompt:
        system_prompt = """
        You are an expert AI Tutor designing a personalized learning path for a 9th-grade student using resources from a curriculum covering Physics, Biology, Chemistry, and Mathematics.
        """
    global vertex_last_initialized
    if vertex_last_initialized < datetime.now() - timedelta(minutes=30):
        vertex_credentials.initialize()
        vertex_last_initialized = datetime.now()
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
    print("Final Response:")
    print(response_content)
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
            f"- Resource: '{title}' (Topic: {topic_name}, Modality: {modality})\n"
            f"  Rating: {rating}/5\n"
            f"  Comment: \"{comment}\""
        )
    return "\n".join(formatted_lines)

def recommend_to_student(student_data, resource_info_map, topic_id_map, num_recommendations=3):
    """
    Generates a prompt for an LLM to recommend resources based on student profile and history.

    Args:
        student_data (dict): Containing 'studentId', 'profile' (with 'learningStyle',
                             'lovedTopics', 'dislikedTopics'), and 'history' (list of interactions).
        resource_info_map (dict): Lookup map for resource details (title, topicId, modality).
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

    # Format history
    history_summary = format_history_for_prompt(history, resource_info_map, topic_id_map)

    # Construct the prompt
    prompt = f"""You are an expert AI Tutor designing a personalized learning path for a 9th-grade student (ID: {student_id}) using resources from a curriculum covering Physics, Biology, Chemistry, and Mathematics.

Student Profile:
- Preferred Learning Style: {learning_style}
- Enjoys Topics Like: {loved_str}
- Dislikes Topics Like: {disliked_str}

Recent Interaction History (last {len(history[-5:])} items):
{history_summary}

Available resource modalities include: Visual (Video, Slideshow, Simulation), Audio (Podcast), Reading/Writing (Article), Kinaesthetic (Quiz, Simulation), Mixed.

Task:
Based *only* on the provided profile and interaction history, recommend {num_recommendations} specific resources for this student to engage with next.

Instructions for Recommendation:
1.  **Prioritize Alignment:** Suggest resources that align with the student's learning style ({learning_style}) and topics they enjoy ({loved_str}).
2.  **Consider Exploration:** If appropriate, suggest resources in topics related to their liked topics or gently re-introduce a disliked topic ({disliked_str}) using a preferred modality or a different angle, explaining why it might be useful.
3.  **Avoid Poorly Rated:** Do not recommend resources the student has already interacted with and rated poorly (e.g., 1 or 2 stars).
4.  **Output Format:** Provide the recommendations as a list, including the Resource ID and a *brief* justification (1 sentence) for why each resource is suitable for *this specific student*, referencing their profile/history.

Example Output Format:
Recommended Resources:
1.  Resource ID: R0XXX - Reason: This video matches your visual learning style and covers [Liked Topic Name], which you enjoy.
2.  Resource ID: R0YYY - Reason: Since you found [Resource Title] confusing, this interactive simulation on [Related Topic] might offer a different, more engaging approach.
3.  Resource ID: R0ZZZ - Reason: Exploring [Adjacent Topic] could build upon your interest in [Liked Topic Name]; this article provides a solid introduction.

Begin Recommendation:
"""

    res = chat_with_llm(prompt)
    return res


# --- Example Usage (Illustrative) ---
if __name__ == "__main__":
    # In a real scenario, load these from CSV or Neo4j
    topic_id_map = {t['topicId']: t['name'] for t in topics_df.to_dict('records')}
    resource_info_map = {r['resourceId']: r for r in resources_df.to_dict('records')}

    # Example: Fetch or construct data for one student (S0001)
    example_student_id = "S0001"
    student_row = students_df[students_df['studentId'] == example_student_id].iloc[0]
    student_history_df = consumed_df[consumed_df['studentId'] == example_student_id]

    example_student_data = {
        "studentId": student_row["studentId"],
        "profile": {
            "learningStyle": student_row["learningStyle"],
            # Load JSON strings back to lists
            "lovedTopics": json.loads(student_row["lovedTopicIds"]),
            "dislikedTopics": json.loads(student_row["dislikedTopicIds"])
        },
        "history": student_history_df.to_dict('records') # Pass interaction history
    }

    # Generate the prompt
    generated_prompt = recommend_to_student(example_student_data, resource_info_map, topic_id_map)

    print("--- Example Generated Prompt for S0001 ---")
    print(generated_prompt)
    print("------------------------------------------")