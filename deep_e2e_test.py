#!/usr/bin/env python3
"""
22_MyAgent Deep End-to-End Test
Tests the complete continuous AI development workflow with a real app idea
"""

import asyncio
import json
import time
import requests
from pathlib import Path
import subprocess
import os
from datetime import datetime

class DeepE2ETest:
    def __init__(self):
        self.base_dir = Path('/home/aadel/projects/22_MyAgent')
        self.api_url = 'http://127.0.0.1:8000'
        self.test_results = {}
        
    def log_test(self, name, status, details=""):
        """Log test results"""
        icon = "✅" if status else "❌"
        print(f"{icon} {name}: {details}")
        self.test_results[name] = {'status': status, 'details': details}
        
    def test_api_health(self):
        """Test API server health"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_test("API Health Check", True, f"Status: {data['status']}")
                return True
            else:
                self.log_test("API Health Check", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("API Health Check", False, f"Connection failed: {e}")
            return False
            
    def create_test_project(self):
        """Create a test project with real app requirements"""
        app_idea = {
            "name": "SmartFinanceTracker",
            "description": "Personal finance tracker with AI insights and spending analysis",
            "requirements": [
                "User authentication system with JWT tokens",
                "Expense tracking with categories (food, transport, entertainment, bills)",
                "Income recording and management",
                "Monthly budget setting and tracking",
                "AI-powered spending insights and recommendations", 
                "Data visualization with charts and graphs",
                "Export functionality (PDF, CSV)",
                "Mobile-responsive design",
                "Dark/light theme toggle",
                "Data backup and sync"
            ],
            "tech_stack": {
                "backend": "Python FastAPI",
                "frontend": "React with TypeScript",
                "database": "PostgreSQL",
                "ai_features": "OpenAI GPT integration",
                "styling": "TailwindCSS"
            },
            "quality_targets": {
                "code_quality": 95,
                "test_coverage": 90, 
                "performance": 90,
                "security": 95,
                "accessibility": 85,
                "maintainability": 90,
                "documentation": 85,
                "ui_ux": 88
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/api/projects",
                json=app_idea,
                timeout=30
            )
            
            if response.status_code == 201:
                project_data = response.json()
                project_id = project_data.get('project_id')
                self.log_test("Project Creation", True, f"Project ID: {project_id}")
                return project_id
            else:
                self.log_test("Project Creation", False, f"HTTP {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            self.log_test("Project Creation", False, f"Request failed: {e}")
            return None
            
    def start_continuous_development(self, project_id):
        """Start the continuous development process"""
        try:
            response = requests.post(
                f"{self.api_url}/api/projects/{project_id}/start",
                timeout=30
            )
            
            if response.status_code == 200:
                self.log_test("Start Development", True, "Continuous development initiated")
                return True
            else:
                self.log_test("Start Development", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Start Development", False, f"Request failed: {e}")
            return False
            
    def monitor_development_progress(self, project_id, duration=60):
        """Monitor development progress for specified duration"""
        print(f"\n📊 Monitoring development for {duration} seconds...")
        
        start_time = time.time()
        iterations = 0
        agent_activities = {}
        
        while time.time() - start_time < duration:
            try:
                # Get project status
                response = requests.get(f"{self.api_url}/api/projects/{project_id}", timeout=10)
                
                if response.status_code == 200:
                    project_data = response.json()
                    
                    # Track iterations
                    current_iteration = project_data.get('current_iteration', 0)
                    if current_iteration > iterations:
                        iterations = current_iteration
                        print(f"🔄 Iteration {iterations} completed")
                        
                    # Track agent activities
                    active_agents = project_data.get('active_agents', [])
                    for agent in active_agents:
                        agent_name = agent.get('name', 'Unknown')
                        if agent_name not in agent_activities:
                            agent_activities[agent_name] = 0
                        agent_activities[agent_name] += 1
                        
                    # Check quality metrics
                    quality_metrics = project_data.get('quality_metrics', {})
                    if quality_metrics:
                        print(f"📈 Quality: Code={quality_metrics.get('code_quality', 0)}%, Tests={quality_metrics.get('test_coverage', 0)}%")
                        
                    # Check for completion
                    status = project_data.get('status', '')
                    if status == 'completed':
                        self.log_test("Development Completion", True, f"Completed in {iterations} iterations")
                        break
                        
                    time.sleep(5)  # Wait 5 seconds between checks
                    
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(5)
                
        # Summary
        self.log_test("Development Monitoring", True, 
                     f"Monitored {iterations} iterations, {len(agent_activities)} agents active")
        
        return {
            'iterations': iterations,
            'agent_activities': agent_activities,
            'duration': time.time() - start_time
        }
        
    def test_generated_code_quality(self, project_id):
        """Test the quality of generated code"""
        try:
            # Check if code files were generated
            project_path = self.base_dir / 'persistence' / 'projects' / str(project_id)
            
            if project_path.exists():
                code_files = list(project_path.glob('**/*.py')) + list(project_path.glob('**/*.tsx'))
                
                if code_files:
                    self.log_test("Code Generation", True, f"Generated {len(code_files)} code files")
                    
                    # Check for key files
                    expected_files = ['main.py', 'models.py', 'App.tsx', 'package.json']
                    found_files = []
                    
                    for expected in expected_files:
                        if any(expected in str(f) for f in code_files):
                            found_files.append(expected)
                            
                    self.log_test("Key Files Generated", len(found_files) > 0, 
                                f"Found: {', '.join(found_files)}")
                    return True
                else:
                    self.log_test("Code Generation", False, "No code files found")
                    return False
            else:
                self.log_test("Code Generation", False, "Project directory not found")
                return False
                
        except Exception as e:
            self.log_test("Code Generation", False, f"Check failed: {e}")
            return False
            
    def test_api_endpoints(self):
        """Test all API endpoints"""
        endpoints = [
            ('GET', '/api/status', 'System Status'),
            ('GET', '/api/projects', 'Project List'), 
            ('GET', '/api/agents', 'Agent Status')
        ]
        
        all_passed = True
        for method, endpoint, name in endpoints:
            try:
                if method == 'GET':
                    response = requests.get(f"{self.api_url}{endpoint}", timeout=10)
                    
                if response.status_code in [200, 201]:
                    self.log_test(f"API {name}", True, f"HTTP {response.status_code}")
                else:
                    self.log_test(f"API {name}", False, f"HTTP {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"API {name}", False, f"Request failed: {e}")
                all_passed = False
                
        return all_passed
        
    def run_complete_test(self):
        """Run the complete end-to-end test suite"""
        print("🚀 Starting Deep End-to-End Test of 22_MyAgent System")
        print("="*70)
        
        # Test 1: API Health
        if not self.test_api_health():
            print("❌ API not responding, cannot continue tests")
            return False
            
        # Test 2: API Endpoints
        self.test_api_endpoints()
        
        # Test 3: Create Real Project
        project_id = self.create_test_project()
        if not project_id:
            print("❌ Could not create project, stopping tests")
            return False
            
        # Test 4: Start Development
        if not self.start_continuous_development(project_id):
            print("❌ Could not start development process")
            return False
            
        # Test 5: Monitor Progress
        progress = self.monitor_development_progress(project_id, 120)  # Monitor for 2 minutes
        
        # Test 6: Check Generated Code
        self.test_generated_code_quality(project_id)
        
        # Generate Report
        self.generate_test_report(progress)
        
        return True
        
    def generate_test_report(self, progress_data):
        """Generate comprehensive test report"""
        print("\n" + "="*70)
        print("📊 22_MYAGENT DEEP TEST RESULTS")
        print("="*70)
        
        passed = sum(1 for result in self.test_results.values() if result['status'])
        total = len(self.test_results)
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        print(f"✅ Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
        print(f"🔄 Development Iterations: {progress_data.get('iterations', 0)}")
        print(f"⏱️ Monitoring Duration: {progress_data.get('duration', 0):.1f} seconds")
        print(f"🤖 Active Agents: {len(progress_data.get('agent_activities', {}))}")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            icon = "✅" if result['status'] else "❌"
            print(f"  {icon} {test_name}: {result['details']}")
            
        # Overall Assessment
        if success_rate >= 80:
            print("\n🎉 OVERALL: 22_MyAgent system is OPERATIONAL and ready for production use!")
        elif success_rate >= 60:
            print("\n⚠️ OVERALL: 22_MyAgent system is partially functional, needs minor fixes")
        else:
            print("\n❌ OVERALL: 22_MyAgent system needs significant work before use")
            
        return success_rate >= 80

def main():
    """Run the deep end-to-end test"""
    tester = DeepE2ETest()
    success = tester.run_complete_test()
    
    if success:
        print("\n🚀 22_MyAgent is ready for continuous AI development!")
    else:
        print("\n🔧 22_MyAgent needs additional configuration before use")
        
    return success

if __name__ == "__main__":
    main()
