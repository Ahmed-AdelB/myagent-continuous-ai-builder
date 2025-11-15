#!/usr/bin/env python3
"""
22_MyAgent Live Demo Project
Creates and runs a Smart Todo App to demonstrate continuous AI development
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.append('.')

class DemoProjectLauncher:
    def __init__(self):
        self.project_spec = {
            "name": "SmartTodoApp",
            "description": "Intelligent Todo List with AI-powered categorization and priority suggestions",
            "requirements": [
                "Create, edit, and delete todo items",
                "AI-powered automatic categorization (work, personal, shopping, health)",
                "Priority level suggestions based on keywords",
                "Due date tracking and reminders",
                "Simple, clean user interface",
                "Local storage persistence",
                "Search and filter functionality",
                "Export todos to text/CSV"
            ],
            "tech_stack": {
                "frontend": "React with TypeScript",
                "backend": "Python FastAPI", 
                "database": "SQLite for simplicity",
                "ai": "OpenAI GPT for categorization",
                "styling": "TailwindCSS"
            },
            "quality_targets": {
                "code_quality": 90,
                "test_coverage": 80,
                "performance": 85,
                "security": 85,
                "accessibility": 80,
                "maintainability": 85,
                "documentation": 75,
                "ui_ux": 90
            },
            "demo_features": [
                "Add todo: 'Buy groceries for dinner party'",
                "Watch AI categorize as 'Shopping' and set priority 'Medium'",
                "Add todo: 'Finish quarterly report by Friday'", 
                "Watch AI categorize as 'Work' and set priority 'High'",
                "Test search, filter, and export features"
            ]
        }
        
    def print_demo_plan(self):
        """Print the demo project plan"""
        print("🎯 22_MYAGENT LIVE DEMO PROJECT")
        print("="*50)
        print(f"📱 Project: {self.project_spec['name']}")
        print(f"📝 Description: {self.project_spec['description']}")
        
        print("\n🔧 Features to Build:")
        for i, req in enumerate(self.project_spec['requirements'], 1):
            print(f"   {i}. {req}")
            
        print("\n🛠️ Technology Stack:")
        for tech, choice in self.project_spec['tech_stack'].items():
            print(f"   • {tech.title()}: {choice}")
            
        print("\n📊 Quality Targets:")
        for metric, target in self.project_spec['quality_targets'].items():
            print(f"   • {metric.replace('_', ' ').title()}: {target}%")
            
        print("\n🎪 Demo Scenarios:")
        for i, demo in enumerate(self.project_spec['demo_features'], 1):
            print(f"   {i}. {demo}")
            
        print("\n🚀 Next Steps:")
        print("   1. Run this demo launcher")
        print("   2. Monitor continuous development process")  
        print("   3. Watch AI agents collaborate to build the app")
        print("   4. Test the completed Smart Todo App")
        print("   5. Evaluate quality improvements over iterations")
        
    def create_demo_workspace(self):
        """Create dedicated workspace for demo project"""
        demo_dir = Path('demo_projects/smart_todo_app')
        demo_dir.mkdir(parents=True, exist_ok=True)
        
        # Create project specification file
        spec_file = demo_dir / 'project_spec.json'
        with open(spec_file, 'w') as f:
            json.dump(self.project_spec, f, indent=2)
            
        print(f"✅ Created demo workspace: {demo_dir}")
        print(f"✅ Saved project spec: {spec_file}")
        
        return demo_dir
        
    def start_demo_monitoring(self):
        """Start monitoring script for the demo"""
        monitor_script = '''#!/bin/bash
# Demo Project Monitor
# Tracks progress of Smart Todo App development

echo "🎬 Starting Smart Todo App Demo Monitoring..."
echo "📊 Monitoring continuous AI development process..."

# Monitor for 10 minutes or until completion
for i in {1..120}; do
    echo "⏱️ Demo minute 0: Checking development progress..."
    
    # Check if project files are being created
    if [ -d "demo_projects/smart_todo_app/src" ]; then
        echo "📁 Source code generation detected!"
    fi
    
    # Check for specific files
    if [ -f "demo_projects/smart_todo_app/src/App.tsx" ]; then
        echo "⚛️ React frontend being built..."
    fi
    
    if [ -f "demo_projects/smart_todo_app/main.py" ]; then
        echo "🐍 Python backend being created..."
    fi
    
    # Wait 30 seconds between checks
    sleep 30
done

echo "✅ Demo monitoring complete!"
'''
        
        with open('start_demo_monitoring.sh', 'w') as f:
            f.write(monitor_script)
            
        os.chmod('start_demo_monitoring.sh', 0o755)
        print("✅ Created demo monitoring script")
        
def main():
    """Launch the demo project"""
    launcher = DemoProjectLauncher()
    
    print("🎬 Welcome to 22_MyAgent Live Demo!")
    print("This will demonstrate continuous AI development in action.\n")
    
    launcher.print_demo_plan()
    
    print("\n" + "="*50)
    response = input("🚀 Ready to start the demo? (y/n): ")
    
    if response.lower() == 'y':
        demo_dir = launcher.create_demo_workspace()
        launcher.start_demo_monitoring()
        
        print("\n🎯 DEMO READY TO LAUNCH!")
        print("\nNext steps:")
        print("1. ./start_demo_monitoring.sh  # Start monitoring (in background)")
        print("2. Start your 22_MyAgent orchestrator with the demo project")
        print("3. Watch the continuous AI development process!")
        
        # Save demo spec for easy access
        print(f"\n📋 Demo project spec saved to: {demo_dir}/project_spec.json")
    else:
        print("👋 Demo cancelled. Run again when ready!")

if __name__ == "__main__":
    main()
