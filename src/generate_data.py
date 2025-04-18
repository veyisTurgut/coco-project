import pandas as pd
import numpy as np
import random
from faker import Faker # Using Faker for student names and comments
from datetime import datetime, timedelta
import os
import json # To store lists in CSV cells properly

# --- Simulation Parameters ---
NUM_STUDENTS = 50 # Smaller number to focus on richer profiles
NUM_TOPICS = 20 # Determined by curriculum
NUM_RESOURCES = 150 # Fewer resources initially
NUM_INITIAL_INTERACTIONS_PER_STUDENT = random.randint(5, 15) # Fewer, richer interactions

# Student Profile Parameters
LEARNING_STYLES = ['Visual', 'Audio', 'Kinaesthetic', 'Reading/Writing', 'Mixed']
AVG_LOVED_SUBJECTS = 1
AVG_DISLIKED_SUBJECTS = 1

# Interaction Parameters
MIN_RATING = 1
MAX_RATING = 5

# Timestamps (Less critical now, but keep for structure)
SIMULATION_START_DATE = datetime(2023, 9, 1)
SIMULATION_END_DATE = datetime(2023, 9, 30) # Shorter period for initial interactions

# Output directory
OUTPUT_DIR = "synthetic_data"

# --- Curriculum Definition (same as before) ---
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
    ]
}
SUBJECT_CODES = list(curriculum_topics.keys())

# Flatten the list and create IDs/Map
all_curriculum_topics_list = []
topic_id_map = {} # ID -> Name
topic_subject_map = {} # ID -> Subject Code (PHY, BIO, etc.)
topic_id_counter = 1
for subject_code, topics in curriculum_topics.items():
    for topic_name in topics:
        topic_id = f"{subject_code}{topic_id_counter:03d}"
        all_curriculum_topics_list.append({"topicId": topic_id, "name": topic_name})
        topic_id_map[topic_id] = topic_name
        topic_subject_map[topic_id] = subject_code
        topic_id_counter += 1

NUM_TOPICS = len(all_curriculum_topics_list)
all_topic_ids = list(topic_id_map.keys())

# --- Initialization ---
fake = Faker()
random.seed(42)
np.random.seed(42)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Helper Functions ---
def generate_random_timestamp(start_date, end_date):
    time_delta = end_date - start_date
    random_seconds = random.uniform(0, time_delta.total_seconds())
    return start_date + timedelta(seconds=random_seconds)

# --- 1. Generate Nodes with Profiles ---

# Topics (Same as before)
print("Generating Curriculum Topics...")
topics_df = pd.DataFrame(all_curriculum_topics_list)
print(f"Generated {len(topics_df)} curriculum topics.")

# Resources (Add Modality)
print("Generating Resources with Modality...")
resources_data = []
resource_topic_map = {} # resourceId -> topicId
resource_modality_map = {} # resourceId -> modality
resource_topic_rels = []
resource_types = ['Article', 'Video', 'Podcast', 'Quiz', 'Simulation', 'Slideshow']
# Map types to likely modalities
type_modality_map = {
    'Article': 'Reading/Writing', 'Video': 'Visual', 'Podcast': 'Audio',
    'Quiz': 'Kinaesthetic', 'Simulation': 'Kinaesthetic', 'Slideshow': 'Visual'
}

for i in range(NUM_RESOURCES):
    resource_id = f"R{i+1:04d}"
    resource_type = random.choice(resource_types)
    assigned_topic_id = random.choice(all_topic_ids)
    assigned_topic_name = topic_id_map[assigned_topic_id]
    modality = type_modality_map.get(resource_type, 'Mixed') # Assign modality based on type

    resource_title = f"{resource_type} on {assigned_topic_name}"
    resources_data.append({
        "resourceId": resource_id,
        "title": resource_title,
        "type": resource_type,
        "modality": modality, # Added modality property
    })
    resource_topic_map[resource_id] = assigned_topic_id
    resource_modality_map[resource_id] = modality
    resource_topic_rels.append({
        "resourceId": resource_id,
        "topicId": assigned_topic_id
    })

resources_df = pd.DataFrame(resources_data)
resource_topic_df = pd.DataFrame(resource_topic_rels)
all_resource_ids = resources_df["resourceId"].tolist()
print(f"Generated {len(resources_df)} resources with modalities.")

# Students (Add Learning Style and Affinities)
print("Generating Students with Profiles (Style, Affinities)...")
students_data = []
student_profiles = {} # Store profile details for interaction generation

for i in range(NUM_STUDENTS):
    student_id = f"S{i+1:04d}"
    learning_style = random.choice(LEARNING_STYLES)

    # Determine loved/disliked subjects
    loved_subjects = random.sample(SUBJECT_CODES, k=random.randint(0, AVG_LOVED_SUBJECTS + 1))
    remaining_subjects = list(set(SUBJECT_CODES) - set(loved_subjects))
    disliked_subjects = random.sample(remaining_subjects, k=random.randint(0, AVG_DISLIKED_SUBJECTS + 1)) if remaining_subjects else []

    # Get corresponding topic IDs
    loved_topic_ids = [tid for tid, subj in topic_subject_map.items() if subj in loved_subjects]
    disliked_topic_ids = [tid for tid, subj in topic_subject_map.items() if subj in disliked_subjects]

    students_data.append({
        "studentId": student_id,
        "name": fake.name(),
        "learningStyle": learning_style,
        # Store lists as JSON strings for CSV compatibility
        "lovedTopicIds": json.dumps(loved_topic_ids),
        "dislikedTopicIds": json.dumps(disliked_topic_ids)
    })
    # Store usable lists in memory for interaction generation
    student_profiles[student_id] = {
        "learningStyle": learning_style,
        "lovedTopics": loved_topic_ids,
        "dislikedTopics": disliked_topic_ids
    }

students_df = pd.DataFrame(students_data)
all_student_ids = students_df["studentId"].tolist()
print(f"Generated {len(students_df)} students with profiles.")


# --- 2. Simulate Initial Rich Interactions ---

print("Simulating Initial Interactions with Ratings/Comments...")
consumed_rels = []
interactions_generated = 0

# Generate comments based on rating and potential mismatches
def generate_comment(rating, student_style, resource_modality, is_loved, is_disliked):
    if rating >= 4:
        templates = [
            "Excellent explanation!", "Really helpful, thanks.", "Very clear.",
            "Loved this resource.", "Well presented."
        ]
        if student_style == resource_modality or student_style == 'Mixed':
            templates.append(f"Perfect for my {student_style.lower()} style.")
        if is_loved:
            templates.append("Great content on a topic I enjoy!")
        return random.choice(templates)
    elif rating <= 2:
        templates = [
            "Found this hard to follow.", "A bit confusing.", "Not very engaging.",
            "Could be explained better.", "Didn't quite get it."
        ]
        if student_style != resource_modality and student_style != 'Mixed' and resource_modality != 'Mixed':
             templates.append(f"As a {student_style.lower()} learner, this {resource_modality.lower()} format was tough.")
        if is_disliked:
            templates.append("Struggled with this topic.")
        templates.append("More examples would be good.")
        return random.choice(templates)
    else: # Rating is 3
        templates = [
            "It was okay.", "Covered the basics.", "Decent overview.", "Neither good nor bad.",
            "Helped a little."
        ]
        return random.choice(templates)

for student_id in all_student_ids:
    profile = student_profiles[student_id]
    num_interactions = random.randint(5, 15)
    interacted_resources = set() # Track resources interacted with by this student

    for _ in range(num_interactions):
        chosen_resource_id = None
        # Bias selection towards loved/neutral topics, away from disliked
        potential_topics = list(set(all_topic_ids) - set(profile['dislikedTopics']))
        if not potential_topics: potential_topics = all_topic_ids # Fallback

        # Higher chance for loved topics if any exist
        if profile['lovedTopics'] and random.random() < 0.6: # 60% chance to pick from loved topics
             chosen_topic_id = random.choice(profile['lovedTopics'])
        else:
             chosen_topic_id = random.choice(potential_topics)

        # Find resources for this topic
        resources_in_topic = [rid for rid, tid in resource_topic_map.items() if tid == chosen_topic_id]
        if not resources_in_topic: continue # Skip if topic has no resources

        # Select a resource, avoiding ones already interacted with
        available_resources = list(set(resources_in_topic) - interacted_resources)
        if not available_resources:
            # If all resources in topic interacted with, try another topic
            continue
        chosen_resource_id = random.choice(available_resources)
        interacted_resources.add(chosen_resource_id)


        # Generate rating based on affinity
        is_loved = chosen_topic_id in profile['lovedTopics']
        is_disliked = chosen_topic_id in profile['dislikedTopics'] # Should be rare due to selection bias
        rating_bias = 0
        if is_loved:
            rating_bias = random.uniform(0.5, 1.5) # Higher ratings
        elif is_disliked:
             rating_bias = random.uniform(-1.5, -0.5) # Lower ratings

        # Base rating + bias + noise, clamped
        base_rating = random.uniform(2.5, 4.0) # Tend towards slightly positive baseline
        rating = round(np.clip(base_rating + rating_bias + random.uniform(-0.5, 0.5), MIN_RATING, MAX_RATING))

        # Generate comment
        resource_modality = resource_modality_map[chosen_resource_id]
        comment = generate_comment(rating, profile['learningStyle'], resource_modality, is_loved, is_disliked)

        consumed_rels.append({
            "studentId": student_id,
            "resourceId": chosen_resource_id,
            "timestamp": generate_random_timestamp(SIMULATION_START_DATE, SIMULATION_END_DATE).isoformat(),
            "rating": rating,
            "comment": comment
        })
        interactions_generated += 1

print(f"Simulated {interactions_generated} initial CONSUMED interactions with ratings/comments.")

# --- 3. Convert Interaction Lists to DataFrames ---
consumed_df = pd.DataFrame(consumed_rels)

# --- 4. Export to CSV ---
print(f"Exporting rich synthetic data to CSV files in {OUTPUT_DIR}...")

students_df.to_csv(os.path.join(OUTPUT_DIR, "students.csv"), index=False)
topics_df.to_csv(os.path.join(OUTPUT_DIR, "topics.csv"), index=False)
resources_df.to_csv(os.path.join(OUTPUT_DIR, "resources.csv"), index=False)
consumed_df.to_csv(os.path.join(OUTPUT_DIR, "consumed.csv"), index=False)
resource_topic_df.to_csv(os.path.join(OUTPUT_DIR, "resource_topic.csv"), index=False)
# No participation or recommendation data generated in this version

print(f"Successfully generated rich synthetic data in directory: {OUTPUT_DIR}")
print("\nGenerated files:")
print(f"- {os.path.join(OUTPUT_DIR, 'students.csv')} ({len(students_df)} rows)")
print(f"- {os.path.join(OUTPUT_DIR, 'topics.csv')} ({len(topics_df)} rows)")
print(f"- {os.path.join(OUTPUT_DIR, 'resources.csv')} ({len(resources_df)} rows)")
print(f"- {os.path.join(OUTPUT_DIR, 'consumed.csv')} ({len(consumed_df)} rows)")
print(f"- {os.path.join(OUTPUT_DIR, 'resource_topic.csv')} ({len(resource_topic_df)} rows)")