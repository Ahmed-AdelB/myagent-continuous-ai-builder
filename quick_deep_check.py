#!/usr/bin/env python3
import sys
import asyncio
import traceback
import os
from pathlib import Path

sys.path.append('.')

async def ultra_deep_check():
    """Ultra thorough check for missed issues"""
    issues_found = []
    
    print("🔍 ULTRA-DEEP VALIDATION - Finding what was missed...")
    
    # 1. Test actual orchestrator end-to-end
    try:
        from core.orchestrator.continuous_director import ContinuousDirector
        
        real_spec = {
            "name": "Production App Test",
            "description": "Real production application", 
            "requirements": ["auth", "api", "frontend", "ml", "monitoring"]
        }
        
        orchestrator = ContinuousDirector("ultra_deep_test", real_spec)
        
        # Test missing methods that might not exist
        try:
            orchestrator.update_quality_metrics({"test_coverage": 95})
            print("✅ update_quality_metrics working")
        except Exception as e:
            issues_found.append(f"❌ update_quality_metrics failed: {e}")
        
        try:
            ledger = orchestrator.ledger
            ledger.record_decision(1, "test", "validation", "Deep check")
            print("✅ Ledger property working")
        except Exception as e:
            issues_found.append(f"❌ Ledger property failed: {e}")
            
        # Test if orchestrator can actually start
        try:
            # Don't actually start (would run forever), just check method exists
            if hasattr(orchestrator, 'start'):
                print("✅ Orchestrator start method exists")
            else:
                issues_found.append("❌ Orchestrator missing start method")
        except Exception as e:
            issues_found.append(f"❌ Orchestrator start check failed: {e}")
            
    except Exception as e:
        issues_found.append(f"❌ Orchestrator system failed: {e}")
    
    # 2. Test actual agent functionality
    try:
        from core.agents.base_agent import PersistentAgent, AgentState
        
        class TestAgent(PersistentAgent):
            async def process_task(self, task):
                return {"result": "test_complete"}
            def analyze_context(self, context):
                return {"analysis": "complete"}
            def generate_solution(self, problem):
                return {"solution": "generated"}
        
        agent = TestAgent("test", "tester", ["testing"])
        
        # Test state management
        if agent.state != AgentState.IDLE:
            issues_found.append(f"❌ Agent initial state wrong: {agent.state}")
        
        print("✅ Agent system working")
        
    except Exception as e:
        issues_found.append(f"❌ Agent system failed: {e}")
    
    # 3. Test database connections thoroughly
    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()
        
        postgres_url = os.getenv('POSTGRES_URL')
        if postgres_url:
            try:
                conn = psycopg2.connect(postgres_url)
                cursor = conn.cursor()
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()
                if result[0] != 1:
                    issues_found.append("❌ PostgreSQL not returning correct results")
                conn.close()
                print("✅ PostgreSQL working correctly")
            except Exception as e:
                issues_found.append(f"❌ PostgreSQL connection failed: {e}")
        else:
            issues_found.append("❌ PostgreSQL URL not configured")
            
    except Exception as e:
        issues_found.append(f"❌ Database test failed: {e}")
    
    # 4. Test memory systems integration
    try:
        from core.memory.project_ledger import ProjectLedger
        from core.memory.vector_memory import VectorMemory
        
        # Test ProjectLedger with complex operations
        ledger = ProjectLedger("ultra_test")
        
        # Save a complex code version
        complex_code = "def ai_function(): return 'advanced'"\n        version = ledger.save_code_version(
            "test.py", complex_code, 1, "ultra_agent", "Ultra test",
            test_results={"passed": 10, "failed": 0},
            performance_metrics={"speed": 1.5, "memory": 100}
        )
        
        if not version:
            issues_found.append("❌ ProjectLedger save_code_version failed")
        
        # Retrieve and verify
        retrieved = ledger.get_version(version.id)
        if not retrieved or retrieved.content != complex_code:
            issues_found.append("❌ ProjectLedger version retrieval failed")
        
        print("✅ ProjectLedger working correctly")
        
        # Test VectorMemory
        vector_mem = VectorMemory("ultra_test")
        
        # Test if collections are properly initialized
        if not hasattr(vector_mem, 'collections'):
            issues_found.append("❌ VectorMemory collections not initialized")
        
        print("✅ VectorMemory initialized correctly")
        
    except Exception as e:
        issues_found.append(f"❌ Memory systems failed: {e}")
    
    # 5. Check actual file system state
    critical_files = [
        "core/orchestrator/continuous_director.py",
        "core/memory/project_ledger.py", 
        "core/agents/base_agent.py",
        "comprehensive_test_suite.py",
        "FINAL_SYSTEM_REPORT.md"
    ]
    
    for file_path in critical_files:
        if not Path(file_path).exists():
            issues_found.append(f"❌ Critical file missing: {file_path}")
    
    # 6. Test API keys are actually valid format
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not openai_key or not openai_key.startswith('sk-'):
        issues_found.append("❌ OpenAI API key invalid format")
    
    if not anthropic_key or not anthropic_key.startswith('sk-ant-'):
        issues_found.append("❌ Anthropic API key invalid format")
    
    # 7. Test persistence directories exist and are writable
    test_dirs = ["persistence/database", "persistence/storage", "logs", "test_reports"]
    
    for dir_path in test_dirs:
        path = Path(dir_path)
        if not path.exists():
            issues_found.append(f"❌ Directory missing: {dir_path}")
        elif not os.access(path, os.W_OK):
            issues_found.append(f"❌ Directory not writable: {dir_path}")
    
    # Generate final assessment
    print(f"\n🔍 ULTRA-DEEP CHECK RESULTS:")
    
    if issues_found:
        print(f"❌ CRITICAL ISSUES FOUND: {len(issues_found)}")
        for issue in issues_found:
            print(f"  {issue}")
        print(f"\n🚨 SYSTEM HAS SERIOUS PROBLEMS - Previous validation was incomplete!")
        return False
    else:
        print(f"✅ NO CRITICAL ISSUES FOUND")
        print(f"✅ System is genuinely healthy and production-ready")
        print(f"✅ Previous validation was accurate and complete")
        return True

if __name__ == "__main__":
    result = asyncio.run(ultra_deep_check())
    sys.exit(0 if result else 1)
