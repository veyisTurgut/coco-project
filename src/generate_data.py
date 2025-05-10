# src/generate_data.py
# Purpose:
# This script generates a synthetic dataset for a simulated learning environment. It creates
# CSV files containing information about students, topics, learning resources, and initial
# interactions (resource consumptions with LLM-generated feedback, and topic participations).
# This dataset serves as the foundational data for loading into a Neo4j graph database,
# which is then used by the simulation and analysis scripts.
#
# What it does:
# 1.  Initialization and Configuration:
#     - Loads environment variables (e.g., for LLM access if needed externally, though primarily uses local config).
#     - Sets up logging.
#     - Defines core parameters: number of students, resources, min/max initial interactions,
#       learning styles, output directory for CSVs, LLM batch size, and probability of initial participation.
#     - Defines file paths for all output CSVs.
#     - Initializes Faker for generating synthetic names.
#
# 2.  Curriculum Definition:
#     - Defines a hierarchical curriculum structure with subjects (e.g., Physics, Biology)
#       and specific topics within each subject.
#     - Generates unique topic IDs and maintains mappings between topic IDs, names, and subject codes.
#
# 3.  Base Node Data Generation (Conditional - `REGENERATE_BASE_DATA` flag or missing files):
#     - Topics: Creates a list of topics based on the curriculum, assigning unique IDs and names.
#       Saves to `topics.csv`.
#     - Resources:
#       - Generates a specified number of resources with unique IDs.
#       - Assigns titles using Faker, types (e.g., 'Article', 'Video'), and corresponding
#         modalities (e.g., 'Reading/Writing', 'Visual').
#       - Randomly associates each resource with one or more topics from the curriculum.
#       - Saves resource data to `resources.csv` and resource-topic relationships to `resource_topic.csv`.
#     - Students:
#       - Generates a specified number of student profiles.
#       - Assigns unique student IDs and names (using Faker).
#       - Randomly assigns a learning style from a predefined list.
#       - Randomly assigns a set of 'loved' and 'disliked' topic IDs from the curriculum.
#       - Assigns a random social engagement score.
#       - Saves student data to `students.csv`.
#
# 4.  Initial Interaction Data Generation:
#     - Consumed Interactions (`consumed_initially.csv`):
#       - For each student, simulates a random number of initial resource consumption events.
#       - Resources are chosen to generally align with student preferences (avoiding disliked topics if possible).
#       - Generates random timestamps for these interactions within a defined historical period.
#       - **LLM-Powered Feedback**: For batches of these consumption events, it calls a
#         Large Language Model (Vertex AI Gemini via `llm.curl_vertex` and `llm.mylib`)
#         to generate realistic initial feedback (a rating from 1-5 and a short comment).
#         - Prompts include student profile snippets (style, topic preferences) and resource details.
#         - Uses a `ThreadPoolExecutor` to make parallel LLM calls for feedback generation on batches
#           of interactions to improve efficiency.
#         - Stores the `studentId`, `resourceId`, `timestamp`, `rating`, `comment`, and
#           `feedback_generated_by` (marked as 'llm_initial').
#     - Participated Interactions (`participated.csv`):
#       - After a simulated consumption, there's a defined probability that the student also
#         'participates' in the topic of the consumed resource.
#       - These interactions are recorded with `studentId`, `topicId`, `timestamp`, and
#         `interactionType` (marked as 'initial_post').
#
# 5.  Output:
#     - All generated data (students, topics, resources, resource-topic links, initial
#       consumptions with feedback, initial participations) are saved as CSV files in the
#       directory specified by `OUTPUT_DIR` (default: "synthetic_data").
#
# Key Libraries/Modules Used:
# - pandas: For creating and managing DataFrames, and saving to CSV.
# - numpy: For numerical operations, especially random number generation and handling NaNs.
# - Faker: For generating realistic synthetic data like names and text snippets.
# - datetime, timedelta: For generating and managing timestamps.
# - concurrent.futures (ThreadPoolExecutor): For parallelizing LLM calls for feedback generation.
# - llm.curl_vertex, llm.mylib (custom): For interfacing with Google Vertex AI LLMs to generate feedback.
# - json: For handling JSON data, especially LLM responses and schema definitions.
# - os, random, logging, time: Standard Python libraries.

import pandas as pd, numpy as np, random, os, json, logging, time
from faker import Faker
from datetime import datetime, timedelta
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed # Import for parallelism

from llm.curl_vertex import CurlVertex, vertex_credentials
from llm.mylib import * # Assuming this imports LLMConfig, ChatMainLLM, Message, etc.

load_dotenv()

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REGENERATE_BASE_DATA = True # Set to True to regenerate students, topics, resources

# --- Simulation Parameters ---
NUM_STUDENTS = 250; NUM_RESOURCES = 500
MIN_INITIAL_INTERACTIONS = 3; MAX_INITIAL_INTERACTIONS = 15
LEARNING_STYLES = ['Visual', 'Audio', 'Kinaesthetic', 'Reading/Writing', 'Mixed']
AVG_LOVED_SUBJECTS = 2; AVG_DISLIKED_SUBJECTS = 4
OUTPUT_DIR = "synthetic_data" # Ensure this matches where files should be read/written
LLM_BATCH_SIZE = 50; MAX_WORKERS = 50
PROBABILITY_INITIAL_PARTICIPATION = 0.05

# --- File Paths ---
STUDENTS_FILE = os.path.join(OUTPUT_DIR, "students.csv")
TOPICS_FILE = os.path.join(OUTPUT_DIR, "topics.csv")
RESOURCES_FILE = os.path.join(OUTPUT_DIR, "resources.csv")
RESOURCE_TOPIC_FILE = os.path.join(OUTPUT_DIR, "resource_topic.csv")
CONSUMED_FILE = os.path.join(OUTPUT_DIR, "consumed_initially.csv")
PARTICIPATED_FILE = os.path.join(OUTPUT_DIR, "participated.csv")

# --- Curriculum Definition ---
curriculum_topics = {
    "PHY": [ # Physics
        "PHY01: Introduction to Physics", "PHY02: Matter and Properties",
        "PHY03: Force and Motion", "PHY04: Energy", "PHY05: Heat and Temperature",
    ],
    "BIO": [ # Biology
        "BIO01: Biology and Living Things", "BIO02: Cell Structure",
        "BIO03: Classification", "BIO04: Cell Division", "BIO05: Genetics Intro",
    ],
    "CHM": [ # Chemistry
        "CHM01: Intro to Chemistry", "CHM02: Atomic Structure",
        "CHM03: Bonding Basics", "CHM04: States of Matter", "CHM05: Reactions Intro",
    ],
    "MAT": [ # Mathematics
        "MAT01: Logic", "MAT02: Sets", "MAT03: Equations",
        "MAT04: Functions Intro", "MAT05: Triangles",
    ],
    "HIS": [ # History
        "HIS01: What is History?", "HIS02: Early Civilizations",
        "HIS03: Ancient Greece & Rome", "HIS04: Middle Ages Overview", "HIS05: Age of Exploration",
    ],
    "GEO": [ # Geography
        "GEO01: Introduction to Geography", "GEO02: Maps and Mapmaking",
        "GEO03: Physical Geography (Landforms, Climate)", "GEO04: Human Geography (Population, Culture)", "GEO05: Regional Geography Intro",
    ],
    "LIT": [ # Literature
        "LIT01: Elements of Literature (Plot, Character)", "LIT02: Introduction to Poetry",
        "LIT03: Short Stories Analysis", "LIT04: Introduction to Drama", "LIT05: The Novel Genre",
    ],
    "TRK": [ # Turkish
        "TRK01: Introduction to Turkish", "TRK02: Turkish Grammar",
        "TRK03: Turkish Vocabulary", "TRK04: Turkish Pronunciation", "TRK05: Turkish Culture",
    ]
}
# SUBJECT_CODES will now automatically include "HIS", "GEO", "LIT"
SUBJECT_CODES = list(curriculum_topics.keys())

# --- Declare variables needed in both branches ---
resource_info_map_internal = {}
student_profiles = {}
all_resource_ids = []
all_student_ids = []

# The rest of the topic processing logic remains the same:
all_curriculum_topics_list = []
topic_id_map = {} # ID -> Name
topic_subject_map = {} # ID -> Subject Code (PHY, BIO, etc.)
topic_id_counter = 1
for subject_code, topics in curriculum_topics.items():
    for topic_name in topics:
        # Use subject code and counter for ID
        topic_id = f"{subject_code}{topic_id_counter:03d}" # IDs will now mix subjects, e.g., PHY001, BIO006, HIS026...
        all_curriculum_topics_list.append({"topicId": topic_id, "name": topic_name})
        topic_id_map[topic_id] = topic_name
        topic_subject_map[topic_id] = subject_code
        topic_id_counter += 1

NUM_TOPICS = len(all_curriculum_topics_list) # Automatically updated
all_topic_ids = list(topic_id_map.keys())

# --- End Curriculum ---

# --- Initialization ---
fake = Faker('tr_TR'); random.seed(42); np.random.seed(42)
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
# ---

# --- Helper Functions ---
def generate_random_timestamp(start_date, end_date):
    time_delta = end_date - start_date
    random_seconds = random.uniform(0, time_delta.total_seconds())
    return start_date + timedelta(seconds=random_seconds)

# --- LLM Interaction Function (Revised response handling) ---
def chat_with_llm(prompt: str, curl_vertex, system_prompt: str = None):
    if not system_prompt: system_prompt = "Sen yardımcı bir yapay zeka asistanısın."

    chat_history = [Message(
            role="user",
            message=prompt,
            message_content_type=ChatInputContentType.TEXT,
            message_uri=None,
            message_type=None
        )]
    try:
        for i in range(5):
            # Call generate (non-streaming)
            response = curl_vertex.generate( instruction_prompt=system_prompt, chat_history=chat_history,
                                        is_streaming=False, return_tokens=False, timeout=80 )

            # Accumulate content if generate returns an iterable
            response_content = ""
            for item in response:
                response_content += item.content

            logger.debug(f"Raw LLM Batch Response Length: {len(response_content)}")
            try:
                # Attempt to parse the accumulated string
                return json.loads(response_content)
            except json.JSONDecodeError as json_err: logger.error(f"LLM JSON parse fail: {json_err}. Resp: {response_content}"); return None # Log truncated response
            except TypeError as type_err: logger.error(f"LLM response content not string? Type: {type(response_content)}. Err: {type_err}"); return None
    except Exception as e: logger.exception(f"LLM generation error: {e}"); return None

# --- NEW: Function to Generate *Batch* Initial Feedback via LLM ---
BATCH_LLM_FEEDBACK_SYSTEM_PROMPT = """
You are simulating feedback from 9th-grade students for multiple learning resource interactions provided in a list.
For EACH interaction in the list, generate a realistic feedback object containing a rating and comment, based ONLY on that interaction's student profile and resource details.
Output ONLY a single JSON array where each element corresponds directly to an interaction in the input list, maintaining the order and including the provided 'interaction_id'.
"""
# Define the schema for the *output* of the batch call (an array)
BATCH_LLM_FEEDBACK_SCHEMA = {
    "type": "array",
    "description": "A list of feedback objects, one for each interaction provided in the prompt.",
    "items": {
        "type": "object",
        "properties": {
            "interaction_id": {"type": "integer", "description": "The unique ID provided for the interaction in the prompt."},
            "rating": {"type": "integer", "description": "A realistic integer rating from 1 to 5."},
            "comment": {"type": "string", "description": "A brief (1-2 sentence) comment reflecting the student's likely initial reaction."}
        },
        "required": ["interaction_id", "rating", "comment"]
    }
}

def generate_batch_initial_feedback_llm(batch_scenarios: list, topic_id_map: dict, curl_vertex) -> dict:
    """
    Generates simulated feedback for a batch of interactions using a single LLM call.
    Args:
        batch_scenarios: A list of dicts, each containing:
                         {'interaction_id': int, 'profile': dict, 'resource_info': dict}
        topic_id_map: Lookup map for topic names.
    Returns:
        A dictionary mapping interaction_id to {'rating': int, 'comment': str} or None if failed.
    """
    if not batch_scenarios: return {}
    prompt_parts = ["Simulate initial feedback for each of the following student-resource interactions:", ""]
    for i, scenario in enumerate(batch_scenarios): # ... (rest of prompt building is same) ...
        interaction_id=scenario['interaction_id']; profile=scenario['profile']; resource_info=scenario['resource_info']
        learning_style=profile.get('learningStyle', 'N/A'); loved_str=", ".join([topic_id_map.get(tid, tid) for tid in profile.get('lovedTopics', [])]) or "None"
        disliked_str=", ".join([topic_id_map.get(tid, tid) for tid in profile.get('dislikedTopics', [])]) or "None"; res_title=resource_info.get('title', 'N/A')
        res_topic_name=topic_id_map.get(resource_info.get('topicId'), "N/A"); res_modality=resource_info.get('modality', 'N/A')
        prompt_parts.append(f"--- Interaction {i+1} (ID: {interaction_id}) ---")
        prompt_parts.append(f"Student Profile: Style={learning_style}, Enjoys=[{loved_str}], Dislikes=[{disliked_str}]")
        prompt_parts.append(f"Consumed Resource: Title='{res_title}', Topic={res_topic_name}, Modality={res_modality}")
        prompt_parts.append("")
    prompt_parts.append("Task: For EACH interaction above, generate a realistic integer rating (1-5) and a brief comment (1-2 sentences).")
    prompt_parts.append("Consider the student's style/modality match and topic affinity.")
    prompt_parts.append(f"Output Format (JSON array only, matching schema, {len(batch_scenarios)} items):")
    full_prompt = "\n".join(prompt_parts)
    
    batch_feedback_list = chat_with_llm(full_prompt, curl_vertex, system_prompt=BATCH_LLM_FEEDBACK_SYSTEM_PROMPT)
    results = {} 
    if batch_feedback_list and isinstance(batch_feedback_list, list):
        if len(batch_feedback_list) != len(batch_scenarios): logger.warning(f"LLM batch length mismatch! Exp {len(batch_scenarios)}, got {len(batch_feedback_list)}.")
        for feedback_item in batch_feedback_list:
            if isinstance(feedback_item, dict) and all(k in feedback_item for k in ["interaction_id", "rating", "comment"]):
                try:
                    interaction_id=int(feedback_item["interaction_id"]); rating=int(feedback_item["rating"]); comment=str(feedback_item["comment"])
                    if 1 <= rating <= 5: results[interaction_id] = {"rating": rating, "comment": comment}
                    else: logger.warning(f"LLM rating out of range {interaction_id}: {rating}.")
                except (ValueError, TypeError) as e: logger.warning(f"Error processing item {feedback_item}: {e}")
            else: logger.warning(f"Invalid feedback item format: {feedback_item}")
    else: logger.error("LLM batch feedback failed or invalid format.")
    return results

# --- Wrapper function for parallel execution ---
def process_feedback_batch(batch_data, curl_vertex):
    """Wrapper to call LLM feedback generation for use with ThreadPoolExecutor."""
    batch_scenarios, topic_map = batch_data # Unpack arguments
    # **Crucial:** Add error handling within the thread if needed
    try:
        logger.info(f"Thread {os.getpid()} processing batch of {len(batch_scenarios)}...")
        result = generate_batch_initial_feedback_llm(batch_scenarios, topic_map, curl_vertex)
        logger.info(f"Thread {os.getpid()} finished batch.")
        return result
    except Exception as e:
        logger.error(f"Error in thread {os.getpid()} processing batch: {e}")
        return {} # Return empty dict on error to avoid crashing the main loop

# --- End LLM ---



# --- 1. Generate or Load Base Nodes ---
if REGENERATE_BASE_DATA or not all(os.path.exists(f) for f in [STUDENTS_FILE, TOPICS_FILE, RESOURCES_FILE, RESOURCE_TOPIC_FILE]):
    if not REGENERATE_BASE_DATA:
        logger.warning(f"Base data files not found in {OUTPUT_DIR}. Regenerating base data even though REGENERATE_BASE_DATA is False.")

    logger.info("--- Regenerating Base Data (Students, Topics, Resources) ---")

    # --- Generate Topics ---
    logger.info("Generating Curriculum Topics...")
    all_curriculum_topics_list = []; topic_subject_map = {}; topic_id_counter = 1
    for subject_code, topics in curriculum_topics.items():
        for topic_name in topics:
            topic_id = f"{subject_code}{topic_id_counter:03d}"
            all_curriculum_topics_list.append({"topicId": topic_id, "name": topic_name})
            topic_id_map[topic_id] = topic_name
            topic_subject_map[topic_id] = subject_code
            topic_id_counter += 1
    topics_df = pd.DataFrame(all_curriculum_topics_list)
    all_topic_ids = list(topic_id_map.keys())
    logger.info(f"Generated {len(topics_df)} topics.")
    topics_df.to_csv(TOPICS_FILE, index=False) # Save immediately

    # --- Generate Resources ---
    logger.info("Generating Resources with Modality...")
    resources_data = []; resource_topic_rels = []
    resource_types = ['Article', 'Video', 'Podcast', 'Quiz', 'Simulation',
                  'Slideshow', 'Map', 'Timeline', 'Document']
    # Map types to likely modalities
    type_modality_map = {
        'Article': 'Reading/Writing',
        'Video': 'Visual',
        'Podcast': 'Audio',
        'Quiz': 'Kinaesthetic', # Or Reading/Writing depending on quiz type
        'Simulation': 'Kinaesthetic', # Also Visual
        'Slideshow': 'Visual',
        'Map': 'Visual', # Primarily visual
        'Timeline': 'Visual', # Primarily visual, some Reading/Writing
        'Document': 'Reading/Writing' # e.g., primary source document
    }
    for i in range(NUM_RESOURCES):
        resource_id = f"R{i+1:04d}"; resource_type = random.choice(resource_types); assigned_topic_id = random.choice(all_topic_ids)
        assigned_topic_name = topic_id_map[assigned_topic_id]; modality = type_modality_map.get(resource_type, 'Mixed'); resource_title = f"{resource_type} on {assigned_topic_name}"
        resource_record = { "resourceId": resource_id, "title": resource_title, "type": resource_type, "modality": modality }
        resources_data.append(resource_record)
        resource_info_map_internal[resource_id] = {**resource_record, "topicId": assigned_topic_id}
        resource_topic_rels.append({"resourceId": resource_id, "topicId": assigned_topic_id})
    resources_df = pd.DataFrame(resources_data); resource_topic_df = pd.DataFrame(resource_topic_rels)
    all_resource_ids = resources_df["resourceId"].tolist()
    logger.info(f"Generated {len(resources_df)} resources.")
    resources_df.to_csv(RESOURCES_FILE, index=False) # Save immediately
    resource_topic_df.to_csv(RESOURCE_TOPIC_FILE, index=False) # Save immediately

    # --- Generate Students ---
    logger.info("Generating Students with Profiles...")
    students_data = []
    for i in range(NUM_STUDENTS):
        student_id = f"S{i+1:04d}"; learning_style = random.choice(LEARNING_STYLES)
        loved_subjects = random.sample(SUBJECT_CODES, k=random.randint(0, AVG_LOVED_SUBJECTS + 1))
        remaining_subjects = list(set(SUBJECT_CODES) - set(loved_subjects))
        disliked_subjects = random.sample(remaining_subjects, k=random.randint(0, AVG_DISLIKED_SUBJECTS + 1)) if remaining_subjects else []
        all_loved_topic_ids = [tid for tid, subj in topic_subject_map.items() if subj in loved_subjects]
        # Get a random subset of loved topics, between 1 and all of them
        loved_topic_ids = random.sample(all_loved_topic_ids, k=random.randint(1, max(1, len(all_loved_topic_ids)))) if all_loved_topic_ids else []
        disliked_topic_ids = [tid for tid, subj in topic_subject_map.items() if subj in disliked_subjects]
        social_engagement_score = np.random.beta(a=2, b=5) # scaled to 0-1
        students_data.append({ "studentId": student_id, "name": fake.name(), "learningStyle": learning_style, "lovedTopicIds": json.dumps(loved_topic_ids), "dislikedTopicIds": json.dumps(disliked_topic_ids), "socialEngagementScore": social_engagement_score })
        student_profiles[student_id] = { "learningStyle": learning_style, "lovedTopics": loved_topic_ids, "dislikedTopics": disliked_topic_ids, "socialEngagementScore": social_engagement_score }
    students_df = pd.DataFrame(students_data)
    all_student_ids = students_df["studentId"].tolist()
    logger.info(f"Generated {len(students_df)} students.")
    students_df.to_csv(STUDENTS_FILE, index=False) # Save immediately

else:
    logger.info(f"--- Loading Existing Base Data from {OUTPUT_DIR} ---")
    try:
        # --- Load Topics ---
        logger.info(f"Loading topics from {TOPICS_FILE}...")
        topics_df = pd.read_csv(TOPICS_FILE)
        topic_subject_map = {}
        for _, row in topics_df.iterrows():
             topic_id = row['topicId']
             topic_id_map[topic_id] = row['name']
             # Infer subject code from topic ID prefix
             subject_code = topic_id[:3]
             if subject_code in SUBJECT_CODES:
                 topic_subject_map[topic_id] = subject_code
             else:
                  logger.warning(f"Could not infer subject code for topic {topic_id}")
        all_topic_ids = list(topic_id_map.keys())
        logger.info(f"Loaded {len(topics_df)} topics.")

        # --- Load Resources ---
        logger.info(f"Loading resources from {RESOURCES_FILE} and {RESOURCE_TOPIC_FILE}...")
        resources_df = pd.read_csv(RESOURCES_FILE)
        resource_topic_df = pd.read_csv(RESOURCE_TOPIC_FILE)
        # Merge to easily create the internal map
        res_merged_df = pd.merge(resources_df, resource_topic_df, on='resourceId', how='left')
        for _, row in res_merged_df.iterrows():
            resource_info_map_internal[row['resourceId']] = row.to_dict()
        all_resource_ids = resources_df["resourceId"].tolist()
        logger.info(f"Loaded {len(resources_df)} resources.")

        # --- Load Students ---
        logger.info(f"Loading students from {STUDENTS_FILE}...")
        students_df = pd.read_csv(STUDENTS_FILE)
        all_student_ids = students_df["studentId"].tolist()
        # Rebuild student_profiles map from loaded data
        for _, row in students_df.iterrows():
            student_id = row['studentId']
            try: loved_topics = json.loads(row.get("lovedTopicIds") or '[]')
            except: loved_topics = []
            try: disliked_topics = json.loads(row.get("dislikedTopicIds") or '[]')
            except: disliked_topics = []
            student_profiles[student_id] = {
                "learningStyle": row.get("learningStyle", "Unknown"),
                "lovedTopics": loved_topics,
                "dislikedTopics": disliked_topics
            }
        logger.info(f"Loaded {len(students_df)} students and reconstructed profiles.")

    except FileNotFoundError as e:
        logger.error(f"Error loading base data file: {e}. Cannot proceed with REGENERATE_BASE_DATA=False.")
        exit()
    except Exception as e:
        logger.exception(f"An unexpected error occurred loading base data:")
        exit()

# --- End Generate/Load Base Nodes ---


# --- 2. Simulate Initial Interactions (Pairs Only First) ---
# This part runs regardless of the flag, using the loaded/generated maps
logger.info("Simulating initial interaction pairs...")
interaction_pairs_to_process = []
interaction_counter = 0
initial_participated_rels = []

# ...(The loop to select pairs and potentially trigger participation is IDENTICAL to previous version)...
for i, student_id in enumerate(all_student_ids):
    profile = student_profiles[student_id]; num_interactions = random.randint(MIN_INITIAL_INTERACTIONS, MAX_INITIAL_INTERACTIONS); interacted_resources = set()
    logger.debug(f"Planning {num_interactions} interactions for {student_id}...")
    for interaction_num in range(num_interactions): # ...(Resource selection logic identical)...
        chosen_resource_id = None; attempts = 0; MAX_ATTEMPTS = 5
        while not chosen_resource_id and attempts < MAX_ATTEMPTS:
            potential_topics = list(set(all_topic_ids) - set(profile['dislikedTopics'])); potential_topics = potential_topics or all_topic_ids
            if profile['lovedTopics'] and random.random() < 0.8: chosen_topic_id = random.choice(profile['lovedTopics'])
            else: chosen_topic_id = random.choice(potential_topics)
            resources_in_topic = [rid for rid, res_info in resource_info_map_internal.items() if res_info.get("topicId") == chosen_topic_id]
            if not resources_in_topic: attempts += 1; continue
            available_resources = list(set(resources_in_topic) - interacted_resources)
            if not available_resources: attempts += 1; continue
            chosen_resource_id = random.choice(available_resources); attempts += 1
        if chosen_resource_id: # ...(Storing context & probabilistic participation identical)...
            interacted_resources.add(chosen_resource_id); chosen_resource_info = resource_info_map_internal.get(chosen_resource_id)
            if chosen_resource_info:
                interaction_counter += 1; interaction_pairs_to_process.append({ "interaction_id": interaction_counter, "student_id": student_id, "resource_id": chosen_resource_id, "profile": profile, "resource_info": chosen_resource_info })
                if random.random() < PROBABILITY_INITIAL_PARTICIPATION:
                    topic_id = chosen_resource_info.get('topicId')
                    if topic_id: timestamp = generate_random_timestamp(datetime(2024,9,1), datetime(2025,1,30)); initial_participated_rels.append({"studentId": student_id, "topicId": topic_id, "timestamp": timestamp.isoformat(), "interactionType": random.choice(["initial_post", "initial_reply"]) }); logger.debug(f"  -> Student {student_id} will also initially participate in {topic_id}")
            else: logger.error(f"Consistency error: Info missing for {chosen_resource_id}")
        else: logger.warning(f"Could not find resource for {student_id} interaction {interaction_num+1}")
logger.info(f"Planned {len(interaction_pairs_to_process)} initial CONSUMED interactions.")
logger.info(f"Generated {len(initial_participated_rels)} initial PARTICIPATED_IN interactions.")

# --- 3. Generate Feedback in Parallel Batches using LLM ---
# ...(This section is IDENTICAL to previous version)...
logger.info(f"Generating feedback for {len(interaction_pairs_to_process)} interactions concurrently...")
all_feedback_results = {}; llm_feedback_failures = 0; start_batch_processing_time = time.time()
tasks = [];

llm_config = LLMConfig({ "model_arch": ChatMainLLM.VERTEXLLM.value, "model_name": ChatMainLLMName.VERTEX_GEMINI_25_FLASH_PREVIEW,
                           "temperature": 0.7, "top_k": 10, "top_p": 0.95, "max_output_tokens": 50000, 
                           "llm_region": "us-central1", "response_mimetype": "application/json", "responseSchema": BATCH_LLM_FEEDBACK_SCHEMA })
curl_vertex = CurlVertex(llm_config=llm_config, logger=logger) 

for i in range(0, len(interaction_pairs_to_process), LLM_BATCH_SIZE): tasks.append((interaction_pairs_to_process[i : i + LLM_BATCH_SIZE], topic_id_map))
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_batch_num = {executor.submit(process_feedback_batch, task_data, curl_vertex): i for i, task_data in enumerate(tasks)}
    total_batches = len(tasks)
    for i, future in enumerate(as_completed(future_to_batch_num)):
        batch_num = future_to_batch_num[future]
        try: # ...(Result processing identical)...
            batch_result = future.result(); all_feedback_results.update(batch_result); original_batch_size = len(tasks[batch_num][0]); failures_in_batch = original_batch_size - len(batch_result)
            if failures_in_batch > 0: llm_feedback_failures += failures_in_batch; logger.warning(f"Batch {batch_num + 1}/{total_batches} completed with {failures_in_batch} failures.")
            else: logger.info(f"Batch {batch_num + 1}/{total_batches} completed successfully.")
        except Exception as exc: original_batch_size = len(tasks[batch_num][0]); llm_feedback_failures += original_batch_size; logger.error(f'Batch {batch_num + 1}/{total_batches} generated an exception: {exc}')
logger.info(f"Finished generating feedback. Time: {time.time() - start_batch_processing_time:.2f}s. Failures: {llm_feedback_failures}")


# --- 4. Create Final consumed_rels List ---
# ...(This section is IDENTICAL to previous version)...
consumed_rels = []; fallback_rating = 3; fallback_comment = "Feedback unavailable."
for interaction_data in interaction_pairs_to_process:
    interaction_id = interaction_data["interaction_id"]; feedback = all_feedback_results.get(interaction_id); rating = fallback_rating; comment = fallback_comment; feedback_source = "fallback"
    if feedback: rating = feedback["rating"]; comment = feedback["comment"]; feedback_source = "llm"
    consumed_rels.append({ "studentId": interaction_data["student_id"], "resourceId": interaction_data["resource_id"], "timestamp": generate_random_timestamp(datetime(2024,9,1), datetime(2025,1,30)).isoformat(), "rating": rating, "comment": comment, "feedback_generated_by": feedback_source })


# --- 5. Convert and Export ---
# Only export interactions if base data was reused
consumed_df = pd.DataFrame(consumed_rels)
participated_df = pd.DataFrame(initial_participated_rels)

if REGENERATE_BASE_DATA:
    logger.info(f"Exporting ALL data (including base nodes) to CSV files in {OUTPUT_DIR}...")
    students_df.to_csv(STUDENTS_FILE, index=False)
    topics_df.to_csv(TOPICS_FILE, index=False)
    resources_df.to_csv(RESOURCES_FILE, index=False)
    resource_topic_df.to_csv(RESOURCE_TOPIC_FILE, index=False)
    consumed_df.to_csv(CONSUMED_FILE, index=False)
    participated_df.to_csv(PARTICIPATED_FILE, index=False)
else:
    logger.info(f"Exporting ONLY interaction data (consumed, participated) to CSV files in {OUTPUT_DIR}...")
    consumed_df.to_csv(CONSUMED_FILE, index=False)
    participated_df.to_csv(PARTICIPATED_FILE, index=False)


logger.info(f"Successfully generated/updated data in directory: {OUTPUT_DIR}")
print("\nGenerated/Updated files:")
if REGENERATE_BASE_DATA:
    print(f"- {STUDENTS_FILE} ({len(students_df)} rows)")
    print(f"- {TOPICS_FILE} ({len(topics_df)} rows)")
    print(f"- {RESOURCES_FILE} ({len(resources_df)} rows)")
    print(f"- {RESOURCE_TOPIC_FILE} ({len(resource_topic_df)} rows)")
print(f"- {CONSUMED_FILE} ({len(consumed_df)} rows)")
print(f"- {PARTICIPATED_FILE} ({len(participated_df)} rows)")