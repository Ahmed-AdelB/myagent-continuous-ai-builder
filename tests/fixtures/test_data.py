import json
import os
from pathlib import Path

# Get the directory of the current file
current_dir = Path(__file__).parent

# Load the JSON data
json_path = current_dir / "test_data.json"

with open(json_path, "r") as f:
    TEST_DATA = json.load(f)
