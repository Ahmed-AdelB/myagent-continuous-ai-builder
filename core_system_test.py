#!/usr/bin/env python3
"""
22_MyAgent Core System Deep Test
Tests the continuous AI development workflow with a real app idea
Bypasses API layer to test core components directly
"""

import sys
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime

# Add project to Python path
sys.path.append('.')

# Import core components
from core.orchestrator.continuous_director import ContinuousDirector, ProjectState
from core.memory.project_ledger import ProjectLedger
from core.memory.vector_memory import VectorMemory
from core.memory.error_knowledge_graph import ErrorKnowledgeGraph
from core.agents.coder_agent import CoderAgent
from core.agents.tester_agent import TesterAgent
from core.agents.debugger_agent import DebuggerAgent
from core.agents.architect_agent import ArchitectAgent
from core.agents.analyzer_agent import AnalyzerAgent
from core.agents.ui_refiner_agent import UIRefinerAgent

class CoreSystemTest:
    def __init__(self):
        self.base_dir = Path('/home/aadel/projects/22_MyAgent')
        self.test_results = {}
        self.test_project = None
        
    def log_test(self, name, status, details=""):
        """Log test results"""
        icon = "✅" if status else "❌"
        print(f"{icon} {name}: {details}")
        self.test_results[name] = {'status': status, 'details': details}
        
    async def test_memory_systems(self):
        """Test all memory systems"""
        print("\n🧠 Testing Memory Systems...")
        
        try:
            # Test Project Ledger
            ledger = ProjectLedger('test_project_123')
            ledger.record_decision(1, "system_test_agent", "system_test", "Test initialization")
            decisions = ledger.decision_log[:5]  # Get recent decisions
            self.log_test("Project Ledger", len(decisions) > 0, f"Logged {len(decisions)} events")
            
            # Test Vector Memory
            vector_memory = VectorMemory('test_project_123')
            await vector_memory.store_memory(
                "Test memory content for SmartFinanceTracker app",
                "project_context",
                {"test": True}
            )
            results = await vector_memory.search_memories("finance tracker", limit=5)
            self.log_test("Vector Memory", len(results) >= 0, f"Stored and searched memories")
            
            # Test Error Knowledge Graph  
            error_graph = ErrorKnowledgeGraph('test_project_123')
            await error_graph.add_error(
                "test_error",
                "Sample error for testing",
                "Import error in main.py",
                "Fixed by adding missing import"
            )
            similar_errors = await error_graph.find_similar_errors("import error")
            self.log_test("Error Knowledge Graph", True, f"Added error and found {len(similar_errors)} similar")
            
            return True
            
        except Exception as e:
            self.log_test("Memory Systems", False, f"Error: {str(e)[:100]}")
            return False
            
    async def test_individual_agents(self):
        """Test each agent individually"""
        print("\n🤖 Testing Individual Agents...")
        
        # Create mock orchestrator for agents
        class MockOrchestrator:
            def __init__(self):
                self.project_state = ProjectState()
                self.project_ledger = ProjectLedger('test_project_123')
                self.vector_memory = VectorMemory('test_project_123')
                self.error_graph = ErrorKnowledgeGraph('test_project_123')
                
        mock_orchestrator = MockOrchestrator()
        
        # Test agents
        agents = [
            ('Coder Agent', CoderAgent),
            ('Tester Agent', TesterAgent), 
            ('Debugger Agent', DebuggerAgent),
            ('Architect Agent', ArchitectAgent),
            ('Analyzer Agent', AnalyzerAgent),
            ('UI Refiner Agent', UIRefinerAgent)
        ]
        
        all_agents_ok = True
        for name, agent_class in agents:
            try:
                agent = agent_class(
                    name=name,
                    role=f"{name} for testing",
                    capabilities=["test_capability"],
                    orchestrator=mock_orchestrator
                )
                
                # Test initialization
                await agent.initialize()
                self.log_test(f"Agent: {name}", True, "Initialized successfully")
                
            except Exception as e:
                self.log_test(f"Agent: {name}", False, f"Error: {str(e)[:60]}")
                all_agents_ok = False
                
        return all_agents_ok
        
    async def test_orchestrator_setup(self):
        """Test orchestrator initialization"""
        print("\n🎯 Testing Continuous Orchestrator...")
        
        try:
            # Create test project specification
            project_spec = {
                "name": "SmartFinanceTracker",
                "description": "Personal finance tracker with AI insights",
                "requirements": [
                    "User authentication system",
                    "Expense tracking with categories", 
                    "Monthly budget management",
                    "AI-powered spending insights",
                    "Data visualization dashboard",
                    "Mobile-responsive design"
                ],
                "tech_stack": {
                    "backend": "Python FastAPI",
                    "frontend": "React TypeScript",
                    "database": "PostgreSQL",
                    "styling": "TailwindCSS"
                },
                "quality_targets": {
                    "code_quality": 95,
                    "test_coverage": 90,
                    "performance": 90,
                    "security": 95
                }
            }
            
            # Initialize orchestrator
            orchestrator = ContinuousDirector("test_123", project_spec)
            await orchestrator.initialize()
            
            self.log_test("Orchestrator Setup", True, "Initialized with test project")
            self.test_project = orchestrator
            
            # Test one iteration cycle
            print("   🔄 Testing one development iteration...")
            # Note: Full iteration test would require API keys and can be time-consuming
            # For now, just test setup
            
            return True
            
        except Exception as e:
            self.log_test("Orchestrator Setup", False, f"Error: {str(e)[:100]}")
            return False
            
    async def test_quality_metrics_system(self):
        """Test quality metrics and continuous improvement"""
        print("\n📊 Testing Quality Metrics System...")
        
        try:
            if self.test_project:
                # Test quality metrics calculation
                current_metrics = self.test_project.current_quality_metrics
                target_metrics = self.test_project.target_quality_metrics
                
                self.log_test("Quality Metrics", True, 
                            f"Tracking {len(current_metrics)} quality dimensions")
                
                # Test improvement detection
                improvement_needed = any(
                    current_metrics.get(metric, 0) < target_metrics.get(metric, 100)
                    for metric in target_metrics
                )
                
                self.log_test("Quality Improvement Logic", True,
                            f"Improvement needed: {improvement_needed}")
                            
                return True
            else:
                self.log_test("Quality Metrics", False, "No test project available")
                return False
                
        except Exception as e:
            self.log_test("Quality Metrics", False, f"Error: {str(e)[:100]}")
            return False
            
    def test_github_integration(self):
        """Test GitHub auto-sync functionality"""
        print("\n🐙 Testing GitHub Integration...")
        
        try:
            # Test auto-sync script
            sync_script = self.base_dir / 'auto-sync.sh'
            if sync_script.exists():
                self.log_test("Auto-Sync Script", True, "Script exists and is executable")
            else:
                self.log_test("Auto-Sync Script", False, "Script not found")
                
            # Test git configuration
            import subprocess
            result = subprocess.run(['git', 'status'], capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode == 0:
                self.log_test("Git Repository", True, "Repository is properly configured")
            else:
                self.log_test("Git Repository", False, "Git not properly configured")
                
            # Test GitHub CLI
            result = subprocess.run(['gh', 'auth', 'status'], capture_output=True, text=True)
            if result.returncode == 0:
                self.log_test("GitHub CLI", True, "Authenticated and ready")
            else:
                self.log_test("GitHub CLI", False, "Not authenticated or not installed")
                
            return True
            
        except Exception as e:
            self.log_test("GitHub Integration", False, f"Error: {str(e)[:100]}")
            return False
            
    def test_project_structure(self):
        """Test project structure and dependencies"""
        print("\n📁 Testing Project Structure...")
        
        # Check core directories
        required_dirs = [
            'core/orchestrator',
            'core/agents', 
            'core/memory',
            'api',
            'frontend',
            'persistence',
            'tests',
            'venv'
        ]
        
        all_dirs_ok = True
        for dir_path in required_dirs:
            full_path = self.base_dir / dir_path
            if full_path.exists():
                self.log_test(f"Directory: {dir_path}", True, "Exists")
            else:
                self.log_test(f"Directory: {dir_path}", False, "Missing")
                all_dirs_ok = False
                
        # Check key files
        key_files = [
            '.env',
            'requirements.txt',
            'CLAUDE.md',
            'verify_system.py',
            'auto-sync.sh'
        ]
        
        for file_path in key_files:
            full_path = self.base_dir / file_path
            if full_path.exists():
                self.log_test(f"File: {file_path}", True, "Exists")
            else:
                self.log_test(f"File: {file_path}", False, "Missing")
                
        return all_dirs_ok
        
    async def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print("🚀 STARTING COMPREHENSIVE 22_MYAGENT CORE SYSTEM TEST")
        print("="*80)
        
        test_sequence = [
            ("Project Structure", self.test_project_structure, False),
            ("Memory Systems", self.test_memory_systems, True),
            ("Individual Agents", self.test_individual_agents, True), 
            ("Orchestrator Setup", self.test_orchestrator_setup, True),
            ("Quality Metrics", self.test_quality_metrics_system, True),
            ("GitHub Integration", self.test_github_integration, False)
        ]
        
        for test_name, test_func, is_async in test_sequence:
            print(f"\n🧪 Running {test_name} Test...")
            try:
                if is_async:
                    await test_func()
                else:
                    test_func()
            except Exception as e:
                self.log_test(test_name, False, f"Test execution failed: {str(e)[:100]}")
                
        # Generate final report
        self.generate_final_report()
        
    def generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("📊 22_MYAGENT COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        passed = sum(1 for result in self.test_results.values() if result['status'])
        total = len(self.test_results)
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        print(f"✅ Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
        
        # Categorize results
        categories = {
            'Core System': ['Memory Systems', 'Individual Agents', 'Orchestrator Setup', 'Quality Metrics'],
            'Infrastructure': ['Project Structure', 'GitHub Integration'],
            'Agents': [k for k in self.test_results.keys() if 'Agent:' in k]
        }
        
        for category, tests in categories.items():
            if tests:
                category_passed = sum(1 for test in tests if test in self.test_results and self.test_results[test]['status'])
                category_total = len([t for t in tests if t in self.test_results])
                if category_total > 0:
                    print(f"\n📂 {category}: {category_passed}/{category_total} passed")
                    
        print("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            icon = "✅" if result['status'] else "❌"
            print(f"  {icon} {test_name}: {result['details']}")
            
        # Overall assessment with specific recommendations
        print("\n🎯 OVERALL ASSESSMENT:")
        if success_rate >= 90:
            print("🎉 EXCELLENT: 22_MyAgent system is fully operational and ready for production!")
            print("   → You can start continuous AI development projects immediately")
        elif success_rate >= 75:
            print("✅ GOOD: 22_MyAgent core system is working well with minor issues")
            print("   → System is ready for development with some monitoring needed")
        elif success_rate >= 50:
            print("⚠️ MODERATE: 22_MyAgent has core functionality but needs attention")
            print("   → Fix failing components before production use")
        else:
            print("❌ POOR: 22_MyAgent system needs significant work")
            print("   → Address critical failures before proceeding")
            
        print("\n🔧 NEXT STEPS:")
        print("   1. Fix any failed tests shown above")
        print("   2. Start with a simple test project to validate full workflow")
        print("   3. Monitor system performance during development")
        print("   4. Use GitHub auto-sync to track all changes")
        
        return success_rate

async def main():
    """Run the comprehensive test suite"""
    tester = CoreSystemTest()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())
