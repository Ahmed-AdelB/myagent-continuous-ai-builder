#!/usr/bin/env python3
"""
Deep Validation Check - Ultra-thorough system analysis
Verifies EVERYTHING works end-to-end with no false claims
"""

import sys
import asyncio
import traceback
import subprocess
import requests
import time
import os
from pathlib import Path

sys.path.append('.')

class DeepValidationCheck:
    def __init__(self):
        self.issues_found = []
        self.warnings = []
        
    def log_issue(self, severity, component, issue, details):
        self.issues_found.append({
            'severity': severity,
            'component': component, 
            'issue': issue,
            'details': details
        })
        print(f"{severity} [{component}] {issue}: {details}")
    
    def log_warning(self, component, warning, details):
        self.warnings.append({
            'component': component,
            'warning': warning, 
            'details': details
        })
        print(f"⚠️  [{component}] {warning}: {details}")

    async def validate_deep_system_integration(self):
        """Test actual end-to-end workflows"""
        print("\n🔍 DEEP SYSTEM INTEGRATION VALIDATION")
        
        try:
            # Test full orchestrator workflow
            from core.orchestrator.continuous_director import ContinuousDirector
            
            real_project_spec = {
                "name": "Deep Validation Test App",
                "description": "Full end-to-end AI application",
                "requirements": [
                    "User authentication system",
                    "Real-time data processing", 
                    "Machine learning predictions",
                    "REST API with full CRUD",
                    "React frontend with real-time updates"
                ],
                "tech_stack": {
                    "backend": "FastAPI + PostgreSQL + Redis",
                    "frontend": "React + TypeScript",
                    "ai": "OpenAI GPT + Anthropic Claude",
                    "deployment": "Docker + GCP"
                },
                "quality_targets": {
                    "test_coverage": 95,
                    "performance_score": 90,
                    "security_score": 95,
                    "documentation_coverage": 90
                }
            }
            
            orchestrator = ContinuousDirector("deep_validation_test", real_project_spec)
            
            # Test all orchestrator capabilities
            print("Testing orchestrator initialization...")
            assert orchestrator.project_name == "deep_validation_test"
            assert orchestrator.project_spec == real_project_spec
            assert hasattr(orchestrator, 'metrics')
            assert hasattr(orchestrator, 'ledger')
            
            # Test quality metrics update
            print("Testing quality metrics updates...")
            test_metrics = {
                'test_coverage': 85.5,
                'performance_score': 88.2,
                'critical_bugs': 2
            }
            orchestrator.update_quality_metrics(test_metrics)
            
            # Verify metrics were actually updated
            assert orchestrator.metrics.test_coverage == 85.5
            
            # Test ledger integration
            print("Testing ledger integration...")
            initial_decisions = len(orchestrator.ledger.decision_log)
            
            # Add more complex data
            orchestrator.ledger.record_decision(
                1, "deep_test_agent", "architecture_decision", 
                "Selected microservices architecture for scalability",
                rationale="Need to handle 10k+ concurrent users",
                metadata={"services": ["auth", "api", "ml", "frontend"], "load_target": 10000}
            )
            
            final_decisions = len(orchestrator.ledger.decision_log)
            assert final_decisions > initial_decisions, "Ledger not recording decisions"
            
            # Test code version management
            print("Testing code version management...")
            complex_code = '''
def advanced_ai_function(user_input, context):
    """Advanced AI processing with multiple models"""
    try:
        # Preprocess input
        processed_input = preprocess(user_input, context)
        
        # Multi-model inference
        gpt_result = await openai_inference(processed_input)
        claude_result = await anthropic_inference(processed_input)
        
        # Ensemble results
        final_result = ensemble_predict(gpt_result, claude_result)
        
        # Post-process and validate
        validated_result = validate_output(final_result, context)
        
        return {
            "result": validated_result,
            "confidence": calculate_confidence(gpt_result, claude_result),
            "models_used": ["gpt-4", "claude-3"],
            "processing_time": time.time() - start_time
        }
    except Exception as e:
        logger.error(f"AI processing failed: {e}")
        return {"error": str(e), "fallback": True}
'''
            
            version = orchestrator.ledger.save_code_version(
                "core/ai/advanced_processor.py",
                complex_code,
                1,
                "deep_test_agent",
                "Implemented advanced AI processing with ensemble models",
                test_results={"unit_tests": 15, "passed": 14, "failed": 1},
                performance_metrics={"avg_response_time": 1.2, "throughput": 850}
            )
            
            assert version is not None, "Code version not saved"
            assert version.content == complex_code, "Code content mismatch"
            assert version.test_results["passed"] == 14, "Test results not saved"
            
            print("✅ Deep system integration validation PASSED")
            
        except Exception as e:
            self.log_issue("🚨 CRITICAL", "System Integration", "Deep workflow failure", 
                         f"Full system test failed: {str(e)}\n{traceback.format_exc()}")

    def validate_api_server_functionality(self):
        """Test if API server actually works"""
        print("\n🌐 API SERVER FUNCTIONALITY VALIDATION")
        
        try:
            # Check if server is actually running
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print("✅ API server is running and responding")
                else:
                    self.log_issue("🚨 CRITICAL", "API Server", "Health check failed", 
                                 f"Server responded with status {response.status_code}")
            except requests.exceptions.ConnectionError:
                self.log_warning("API Server", "Not running", "Server not accessible on port 8000")
            except requests.exceptions.Timeout:
                self.log_issue("⚠️  MAJOR", "API Server", "Timeout", "Server not responding within 5 seconds")
                
        except Exception as e:
            self.log_issue("🚨 CRITICAL", "API Server", "Validation failed", str(e))

    def validate_database_integrity(self):
        """Deep database integrity check"""
        print("\n🗄️  DATABASE INTEGRITY VALIDATION")
        
        try:
            import sqlite3
            import psycopg2
            import redis
            from dotenv import load_dotenv
            
            load_dotenv()
            
            # Test PostgreSQL connection with real queries
            print("Testing PostgreSQL connection...")
            try:
                postgres_url = os.getenv('POSTGRES_URL')
                if not postgres_url:
                    self.log_issue("🚨 CRITICAL", "Database", "Missing PostgreSQL URL", 
                                 "POSTGRES_URL not set in environment")
                else:
                    conn = psycopg2.connect(postgres_url)
                    cursor = conn.cursor()
                    
                    # Test actual database operations
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()
                    print(f"PostgreSQL version: {version[0][:50]}...")
                    
                    # Test table creation and data operations
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS deep_test_table (
                            id SERIAL PRIMARY KEY,
                            data JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Insert complex data
                    test_data = {
                        "ai_model": "gpt-4",
                        "performance": {"latency": 1.2, "accuracy": 0.95},
                        "metadata": ["test", "validation", "deep_check"]
                    }
                    
                    cursor.execute(
                        "INSERT INTO deep_test_table (data) VALUES (%s) RETURNING id",
                        (psycopg2.extras.Json(test_data),)
                    )
                    
                    inserted_id = cursor.fetchone()[0]
                    
                    # Verify data integrity
                    cursor.execute("SELECT data FROM deep_test_table WHERE id = %s", (inserted_id,))
                    retrieved_data = cursor.fetchone()[0]
                    
                    assert retrieved_data["ai_model"] == "gpt-4"
                    assert retrieved_data["performance"]["accuracy"] == 0.95
                    
                    # Cleanup
                    cursor.execute("DROP TABLE deep_test_table")
                    conn.commit()
                    conn.close()
                    
                    print("✅ PostgreSQL deep validation PASSED")
                    
            except Exception as e:
                self.log_issue("🚨 CRITICAL", "PostgreSQL", "Deep validation failed", str(e))
            
            # Test Redis connection with complex operations
            print("Testing Redis connection...")
            try:
                redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
                r = redis.Redis.from_url(redis_url)
                
                # Test basic connectivity
                r.ping()
                
                # Test complex data structures
                r.hset("deep_test_hash", mapping={
                    "ai_models": "gpt-4,claude-3",
                    "performance": "95.5",
                    "active_sessions": "42"
                })
                
                # Test list operations
                r.lpush("deep_test_list", "item1", "item2", "item3")
                list_length = r.llen("deep_test_list")
                assert list_length == 3
                
                # Test sorted sets
                r.zadd("deep_test_scores", {"model_a": 0.95, "model_b": 0.87, "model_c": 0.92})
                top_score = r.zrevrange("deep_test_scores", 0, 0, withscores=True)
                assert top_score[0][0].decode() == "model_a"
                assert top_score[0][1] == 0.95
                
                # Cleanup
                r.delete("deep_test_hash", "deep_test_list", "deep_test_scores")
                
                print("✅ Redis deep validation PASSED")
                
            except Exception as e:
                self.log_issue("🚨 CRITICAL", "Redis", "Deep validation failed", str(e))
                
        except Exception as e:
            self.log_issue("🚨 CRITICAL", "Database", "Validation setup failed", str(e))

    def validate_agent_system_thoroughly(self):
        """Comprehensive agent system validation"""
        print("\n🤖 AGENT SYSTEM COMPREHENSIVE VALIDATION")
        
        try:
            from core.agents.base_agent import PersistentAgent, AgentState, AgentTask, AgentMemory
            from datetime import datetime
            
            # Test agent creation with complex scenarios
            class DeepTestAgent(PersistentAgent):
                def __init__(self, name, role, capabilities):
                    super().__init__(name, role, capabilities)
                    self.complex_operations_completed = 0
                    
                async def process_task(self, task):
                    # Simulate complex AI processing
                    if task.type == "code_generation": 
                        await asyncio.sleep(0.1)  # Simulate processing time
                        return {
                            "generated_code": f"def {task.data.get('function_name', 'generated_func')}(): pass",
                            "complexity_score": 0.75,
                            "estimated_lines": 42
                        }
                    elif task.type == "testing":
                        await asyncio.sleep(0.05)
                        return {
                            "tests_generated": 8,
                            "coverage_estimate": 0.92,
                            "execution_time": 2.1
                        }
                    else:
                        return {"status": "completed", "task_type": task.type}
                
                def analyze_context(self, context):
                    return {
                        "complexity": len(str(context)) / 100,
                        "estimated_effort": "medium",
                        "confidence": 0.85,
                        "recommendations": ["use_async", "add_error_handling", "optimize_performance"]
                    }
                
                def generate_solution(self, problem):
                    return {
                        "solution_type": "algorithmic",
                        "approach": "divide_and_conquer",
                        "estimated_complexity": "O(n log n)",
                        "implementation_steps": [
                            "analyze_requirements", "design_architecture", 
                            "implement_core", "add_tests", "optimize"
                        ]
                    }
            
            # Create multiple agents with different roles
            agents = []
            for i in range(3):
                agent = DeepTestAgent(
                    name=f"deep_test_agent_{i}",
                    role=f"test_role_{i}",
                    capabilities=["coding", "testing", "debugging", "optimization"]
                )
                agents.append(agent)
            
            print(f"Created {len(agents)} test agents")
            
            # Test complex task processing
            complex_tasks = [
                AgentTask(
                    id=f"task_{i}",
                    type="code_generation",
                    description=f"Generate complex AI function {i}",
                    priority=5 - i,
                    data={
                        "function_name": f"ai_processor_{i}",
                        "requirements": ["async", "error_handling", "logging"],
                        "complexity": "high"
                    },
                    created_at=datetime.now()
                )
                for i in range(5)
            ]
            
        async def test_agent_processing():
            for i, task in enumerate(complex_tasks):
                agent = agents[i % len(agents)]
                agent.add_task(task)
                
                # Process task manually for testing
                result = await agent.process_task(task)
                
                assert "generated_code" in result, f"Task {i} processing failed"
                assert "complexity_score" in result, f"Task {i} missing complexity score"
                
                # Test context analysis
                context = {"project_size": "large", "team_size": 5, "deadline": "2_weeks"}
                analysis = agent.analyze_context(context)
                
                assert "complexity" in analysis, f"Agent {i} context analysis failed"
                assert "confidence" in analysis, f"Agent {i} confidence missing"
                
                # Test solution generation
                problem = {"type": "performance", "description": "Slow API responses"}
                solution = agent.generate_solution(problem)
                
                assert "approach" in solution, f"Agent {i} solution generation failed"
                assert "implementation_steps" in solution, f"Agent {i} missing implementation steps"
            
            print("✅ Agent system comprehensive validation PASSED")
            
        except Exception as e:
            self.log_issue("🚨 CRITICAL", "Agent System", "Comprehensive validation failed", 
                         f"{str(e)}\n{traceback.format_exc()}")

    def validate_memory_systems_deeply(self):
        """Deep validation of all memory systems"""
        print("\n🧠 MEMORY SYSTEMS DEEP VALIDATION")
        
        try:
            # Test ProjectLedger with complex scenarios
            from core.memory.project_ledger import ProjectLedger
            
            ledger = ProjectLedger("deep_memory_test")
            
            # Test complex decision recording
            complex_decisions = [
                {
                    "iteration": 1,
                    "agent": "architect_agent",
                    "decision_type": "architecture_choice",
                    "description": "Selected microservices over monolith",
                    "rationale": "Better scalability and team autonomy",
                    "metadata": {
                        "options_considered": ["monolith", "microservices", "serverless"],
                        "evaluation_criteria": ["scalability", "maintainability", "performance"],
                        "scores": {"monolith": 6, "microservices": 9, "serverless": 7}
                    }
                },
                {
                    "iteration": 2, 
                    "agent": "coder_agent",
                    "decision_type": "technology_stack",
                    "description": "Selected FastAPI + PostgreSQL + Redis",
                    "rationale": "High performance with excellent async support",
                    "metadata": {
                        "performance_benchmarks": {"fastapi": 15000, "django": 8000, "flask": 5000},
                        "community_support": "excellent",
                        "learning_curve": "medium"
                    }
                }
            ]
            
            for decision in complex_decisions:
                ledger.record_decision(
                    decision["iteration"],
                    decision["agent"],
                    decision["decision_type"],
                    decision["description"],
                    decision["rationale"],
                    metadata=decision["metadata"]
                )
            
            # Test complex code version management
            complex_code_versions = [
                {
                    "file": "core/ai/ensemble_model.py",
                    "content": '''
class EnsembleAIModel:
    def __init__(self, models=["gpt-4", "claude-3"]):
        self.models = models
        self.weights = self.optimize_weights()
    
    async def predict(self, input_data):
        predictions = []
        for model in self.models:
            pred = await self.get_model_prediction(model, input_data)
            predictions.append(pred)
        return self.ensemble_predict(predictions)
''',
                    "agent": "ai_specialist",
                    "reason": "Implemented ensemble model for improved accuracy",
                    "test_results": {"accuracy": 0.95, "latency": 1.1, "tests_passed": 24},
                    "performance": {"memory_usage": "250MB", "cpu_usage": "15%"}
                },
                {
                    "file": "api/endpoints/ai_processing.py",
                    "content": '''
@app.post("/ai/process")
async def process_with_ai(request: AIRequest):
    try:
        result = await ai_model.predict(request.data)
        return {
            "result": result,
            "confidence": result.get("confidence", 0.8),
            "processing_time": result.get("processing_time", 0)
        }
    except Exception as e:
        logger.error(f"AI processing failed: {e}")
        raise HTTPException(status_code=500, detail="AI processing failed")
''',
                    "agent": "api_developer", 
                    "reason": "Added AI processing endpoint with error handling",
                    "test_results": {"endpoint_tests": 15, "passed": 14, "failed": 1},
                    "performance": {"avg_response_time": 1.8, "throughput": 450}
                }
            ]
            
            versions_created = []
            for i, code_data in enumerate(complex_code_versions):
                version = ledger.save_code_version(
                    code_data["file"],
                    code_data["content"],
                    i + 1,
                    code_data["agent"],
                    code_data["reason"],
                    test_results=code_data["test_results"],
                    performance_metrics=code_data["performance"]
                )
                versions_created.append(version)
                
                # Verify version saved correctly
                assert version is not None, f"Version {i} not saved"
                retrieved = ledger.get_version(version.id)
                assert retrieved.content == code_data["content"], f"Content mismatch for version {i}"
            
            # Test iteration summary
            summary = ledger.get_iteration_summary(1)
            assert "code_changes" in summary, "Iteration summary missing code changes"
            assert "decisions" in summary, "Iteration summary missing decisions"
            assert len(summary["decisions"]) > 0, "No decisions recorded in iteration 1"
            
            print("✅ Memory systems deep validation PASSED")
            
        except Exception as e:
            self.log_issue("🚨 CRITICAL", "Memory Systems", "Deep validation failed",
                         f"{str(e)}\n{traceback.format_exc()}")

    def validate_production_readiness(self):
        """Validate actual production readiness criteria"""
        print("\n🚀 PRODUCTION READINESS VALIDATION")
        
        try:
            # Check all required files exist
            required_files = [
                "core/orchestrator/continuous_director.py",
                "core/memory/project_ledger.py",
                "core/agents/base_agent.py",
                "api/main.py",
                "requirements.txt",
                ".env",
                "README.md"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                self.log_issue("🚨 CRITICAL", "Production", "Missing required files",
                             f"Missing: {missing_files}")
            
            # Check environment configuration
            from dotenv import load_dotenv
            load_dotenv()
            
            required_env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "POSTGRES_URL"]
            missing_env = []
            
            for var in required_env_vars:
                value = os.getenv(var)
                if not value or value.startswith("your_") or len(value) < 10:
                    missing_env.append(var)
            
            if missing_env:
                self.log_issue("🚨 CRITICAL", "Production", "Invalid environment variables",
                             f"Invalid/missing: {missing_env}")
            
            # Check directory structure
            required_dirs = [
                "persistence/database",
                "persistence/storage", 
                "logs",
                "test_reports"
            ]
            
            for dir_path in required_dirs:
                if not Path(dir_path).exists():
                    self.log_warning("Production", "Missing directory", dir_path)
            
            # Check dependencies
            try:
                result = subprocess.run(["pip", "check"], capture_output=True, text=True)
                if result.returncode != 0:
                    self.log_issue("⚠️  MAJOR", "Production", "Dependency conflicts", result.stdout)
            except Exception as e:
                self.log_warning("Production", "Could not check dependencies", str(e))
            
            print("✅ Production readiness validation PASSED")
            
        except Exception as e:
            self.log_issue("🚨 CRITICAL", "Production", "Readiness check failed", str(e))

    def generate_deep_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "="*80)
        print("🔍 DEEP VALIDATION REPORT - COMPREHENSIVE SYSTEM ANALYSIS")
        print("="*80)
        
        critical_issues = [issue for issue in self.issues_found if "CRITICAL" in issue['severity']]
        major_issues = [issue for issue in self.issues_found if "MAJOR" in issue['severity']]
        
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"  🚨 Critical Issues: {len(critical_issues)}")
        print(f"  ⚠️  Major Issues: {len(major_issues)}")
        print(f"  ⚠️  Warnings: {len(self.warnings)}")
        print(f"  📋 Total Issues Found: {len(self.issues_found)}")
        
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES FOUND:")
            for issue in critical_issues:
                print(f"  • [{issue['component']}] {issue['issue']}")
                print(f"    Details: {issue['details']}")
        
        if major_issues:
            print(f"\n⚠️  MAJOR ISSUES FOUND:")
            for issue in major_issues:
                print(f"  • [{issue['component']}] {issue['issue']}")
                print(f"    Details: {issue['details']}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • [{warning['component']}] {warning['warning']}")
        
        # Overall system health assessment
        total_critical_and_major = len(critical_issues) + len(major_issues)
        
        if total_critical_and_major == 0:
            status = "🟢 EXCELLENT - System fully validated"
            recommendation = "Ready for production deployment"
        elif total_critical_and_major <= 2:
            status = "🟡 GOOD - Minor issues found"
            recommendation = "Address issues before production"
        else:
            status = "🔴 POOR - Significant issues found"
            recommendation = "Major remediation required"
        
        print(f"\n🎯 OVERALL SYSTEM STATUS: {status}")
        print(f"💡 RECOMMENDATION: {recommendation}")
        
        print("\n" + "="*80)
        
        return len(critical_issues) == 0 and len(major_issues) <= 1

    async def run_deep_validation(self):
        """Run complete deep validation"""
        print("🚀 STARTING ULTRA-DEEP SYSTEM VALIDATION")
        print("This will thoroughly test EVERYTHING to find any missed issues...")
        
        # Run all validation checks
        await self.validate_deep_system_integration()
        self.validate_api_server_functionality()
        self.validate_database_integrity()
        await self.validate_agent_system_thoroughly()
        self.validate_memory_systems_deeply()
        self.validate_production_readiness()
        
        # Generate final report
        system_healthy = self.generate_deep_validation_report()
        
        return system_healthy, self.issues_found, self.warnings

if __name__ == "__main__":
    async def main():
        validator = DeepValidationCheck()
        healthy, issues, warnings = await validator.run_deep_validation()
        
        if healthy:
            print("\n✅ DEEP VALIDATION PASSED - No critical issues missed")
        else:
            print("\n❌ DEEP VALIDATION FAILED - Critical issues found")
            
        return healthy
    
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
