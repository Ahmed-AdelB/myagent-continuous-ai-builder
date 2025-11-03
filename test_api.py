#!/usr/bin/env python3
import requests
import json

# Test creating a project
url = "http://localhost:8000/projects"
data = {
    "name": "Test AI Project",
    "description": "Testing the continuous AI builder",
    "requirements": ["Feature 1", "Feature 2"],
    "target_metrics": {"test_coverage": 90},
    "max_iterations": 100
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

    if response.status_code == 200:
        print("\n✅ Project created successfully!")

        # Now get all projects
        projects = requests.get(url)
        print(f"\nAll projects: {projects.json()}")
    else:
        print(f"\n❌ Failed to create project: {response.text}")

except Exception as e:
    print(f"Error: {e}")