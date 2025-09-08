import json
import random

# --- The Final, Corrected Logic ---
NUM_SAMPLES = 100000
MAX_FAILS = 50

# --- Clear, Hard Thresholds ---
FAIL_THRESHOLD = 10       # Any fail count above this is "high"
# --- THIS IS THE CORRECTED THRESHOLD ---
ACTIVITY_THRESHOLD = 0.35   # A more balanced threshold for the activity score

def calculate_adaptive_class(fail_count, activity_score):
    """
    Determines the adaptive difficulty class based on a clear interaction
    between player failure and their level of activity.

    Class Mapping:
    - 0 (Decrease):   High fail count OR Low fail count with High activity.
    - 1 (Increase):   Low fail count AND Low activity.
    """
    # Rule 1: Player is failing a lot. Make it easier.
    if fail_count > FAIL_THRESHOLD:
        return 0  # Decrease

    # Rule 2: Player is NOT failing much. We check their activity.
    if activity_score > ACTIVITY_THRESHOLD:
        # Low fail, high activity. The game is likely too easy.
        return 0 # Decrease
    else:
        # Low fail, low activity. The player is succeeding but bored. Make it harder.
        return 1 # Increase


# --- Generation Script ---
synthetic_data = []
print("Generating the TRUE synthetic dataset (Fail/Activity Interaction)...")

for _ in range(NUM_SAMPLES):
    # Generate random base data
    fail_count = random.randint(0, MAX_FAILS)
    movement = random.uniform(0.0, 1.0)
    rotation = random.uniform(0.0, 1.0)
    action = random.uniform(0.0, 1.0)

    activity_score = (movement + rotation + action) / 3.0
    adaptive_difficulty_class = calculate_adaptive_class(fail_count, activity_score)

    synthetic_data.append({
        "fail": fail_count,
        "activity": round(activity_score, 6),
        "adaptive_difficulty": adaptive_difficulty_class,
    })

# Save the dataset to a JSON file
file_path = "data/classification_dataset.json"
with open(file_path, "w") as f:
    json.dump(synthetic_data, f, indent=4)

print(f"\nSuccessfully created '{file_path}'.")

# --- Verify the final class distribution ---
print("\n--- Final Dataset Class Distribution ---")
class_counts = {0: 0, 1: 0}
for entry in synthetic_data:
    if entry["adaptive_difficulty"] not in class_counts:
        class_counts[entry["adaptive_difficulty"]] = 0
    class_counts[entry["adaptive_difficulty"]] += 1

total_samples = len(synthetic_data)
print(f"Total Samples: {total_samples}")
print(f"Class 0 (Decrease): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/total_samples:.2%})")
print(f"Class 1 (Increase):  {class_counts.get(1, 0)} ({class_counts.get(1, 0)/total_samples:.2%})")

# Let's test your specific cases with the new threshold:
print("\n--- Verifying Your Test Cases with NEW Threshold (0.55) ---")
print(f"Case: fail 7, activity 0.7 -> Class: {calculate_adaptive_class(7, 0.7)} (Should be 0)")
print(f"Case: fail 5, activity 0.3 -> Class: {calculate_adaptive_class(5, 0.3)} (Should be 1)")
print(f"Case: fail 3, activity 0.6 -> Class: {calculate_adaptive_class(3, 0.6)} (Should be 0)")
