import json
import os

PROGRESS_DIR = "data/progress"
os.makedirs(PROGRESS_DIR, exist_ok=True)

def set_progress(image_name, message,percentage):
    with open(f"{PROGRESS_DIR}/{image_name}.json", "w") as f:
        json.dump({"status": message,"percentage": percentage}, f)

def get_progress(image_name):
    path = f"{PROGRESS_DIR}/{image_name}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"status": "pending"}

def clear_progress(image_name):
    path = f"{PROGRESS_DIR}/{image_name}.json"
    if os.path.exists(path):
        os.remove(path)
