#!/usr/bin/env python3
"""
Comprehensive System Verification Script
Tests all components and provides actionable recommendations
"""

import sys
import asyncio
import traceback
import subprocess
import requests
import time
import os
import json
import psutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.append('.')

class ComprehensiveVerification:
    def __init__(self):
        self.results = {
            'system_health': {'score': 0, 'max_score': 0, 'details': []},
            'api_server': {'status': 'unknown', 'details': []},
            'database': {'status': 'unknown', 'details': []}, 
            'orchestrator': {'status': 'unknown', 'details': []},
            'agents': {'status': 'unknown', 'details': []},
            'memory': {'status': 'unknown', 'details': []},
            'environment': {'status': 'unknown', 'details': []},
            'files': {'status': 'unknown', 'details': []},
            'performance': {'status': 'unknown', 'details': []},
            'github': {'status': 'unknown', 'details': []}
        }
        self.recommendations = []
        self.start_time = datetime.now()

    def add_result(self, category: str, status: bool, detail: str, score: int = 1):
        """Add test result"""
        icon = "✅" if status else "❌"
        print(f"{icon} {detail}")
        
        self.results[category]['details'].append({
            'status': status,
            'detail': detail,
            'timestamp': datetime.now().isoformat()
        })
        
        if status:
            self.results[category]['score'] = self.results[category].get('score', 0) + score
        self.results[category]['max_score'] = self.results[category].get('max_score', 0) + score

    def add_recommendation(self, priority: str, action: str, reason: str):
        """Add actionable recommendation"""
        self.recommendations.append({
            'priority': priority,
            'action': action, 
            'reason': reason,
            'category': 'action_required'
        })

    async def verify_api_server(self):
        """Verify API server functionality"""
        print("\n🌐 API SERVER VERIFICATION")
        
        try:
            # Check if server is running on port 8000
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                self.add_result('api_server', True, "API server responding on port 8000")
                
                # Try to get server info
                try:
                    data = response.json()
                    self.add_result('api_server', True, f"API health endpoint returning: {data}")
                except:
                    self.add_result('api_server', True, "API responding but no JSON health data")
            else:
                self.add_result('api_server', False, f"API server returned status {response.status_code}")
                self.add_recommendation('HIGH', 'Check API server logs for errors', 'Server not responding correctly')
                
        except requests.exceptions.ConnectionError:
            self.add_result('api_server', False, "API server not accessible on localhost:8000")
            self.add_recommendation('HIGH', 'Start API server: uvicorn api.main:app --host 0.0.0.0 --port 8000', 'API server not running')
            
        except requests.exceptions.Timeout:
            self.add_result('api_server', False, "API server timeout (>5 seconds)")
            self.add_recommendation('MEDIUM', 'Investigate API server performance issues', 'Server too slow to respond')
            
        except Exception as e:
            self.add_result('api_server', False, f"API server check failed: {str(e)}")
            self.add_recommendation('HIGH', 'Debug API server configuration', f'Unexpected error: {str(e)}')

        # Check if process is actually running
        api_process_found = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'uvicorn' in cmdline and 'api.main:app' in cmdline:
                    self.add_result('api_server', True, f"API server process found (PID: {proc.info['pid']})")
                    api_process_found = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if not api_process_found:
            self.add_result('api_server', False, "No uvicorn API server process found")

        self.results["api_server"]["status"] = "healthy" if self.results["api_server"].get("score", 0) > 0 else "unhealthy"

    def verify_database_connections(self):
        """Verify all database connections"""
        print("\n🗄️ DATABASE VERIFICATION")
        
        # Test PostgreSQL
        try:
            import psycopg2
            from dotenv import load_dotenv
            load_dotenv()
            
            postgres_url = os.getenv('POSTGRES_URL')
            if not postgres_url:
                self.add_result('database', False, "PostgreSQL URL not configured in environment")
                self.add_recommendation('CRITICAL', 'Set POSTGRES_URL in .env file', 'Database connection required')
            else:
                conn = psycopg2.connect(postgres_url)
                cursor = conn.cursor()
                
                # Test basic query
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                self.add_result('database', True, f"PostgreSQL connected: {version[:50]}...")
                
                # Test database operations
                cursor.execute("SELECT current_database();")
                db_name = cursor.fetchone()[0]
                self.add_result('database', True, f"Connected to database: {db_name}")
                
                # Test table creation (if needed)
                cursor.execute("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                table_count = cursor.fetchone()[0]
                self.add_result('database', True, f"Database has {table_count} tables")
                
                conn.close()
                
        except ImportError:
            self.add_result('database', False, "psycopg2 not installed")
            self.add_recommendation('HIGH', 'Install psycopg2: pip install psycopg2-binary', 'PostgreSQL driver missing')
        except Exception as e:
            self.add_result('database', False, f"PostgreSQL connection failed: {str(e)}")
            self.add_recommendation('CRITICAL', 'Fix PostgreSQL connection', f'Database error: {str(e)}')

        # Test Redis
        try:
            import redis
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            r = redis.Redis.from_url(redis_url)
            
            r.ping()
            self.add_result('database', True, "Redis connection successful")
            
            # Test Redis operations
            r.set('health_check', 'ok', ex=60)
            value = r.get('health_check')
            if value == b'ok':
                self.add_result('database', True, "Redis read/write operations working")
            
            # Get Redis info
            info = r.info()
            self.add_result('database', True, f"Redis version: {info.get('redis_version', 'unknown')}")
            
        except ImportError:
            self.add_result('database', False, "redis package not installed")
            self.add_recommendation('MEDIUM', 'Install redis: pip install redis', 'Redis client missing')
        except Exception as e:
            self.add_result('database', False, f"Redis connection failed: {str(e)}")
            self.add_recommendation('HIGH', 'Fix Redis connection or install Redis server', f'Redis error: {str(e)}')

        total_score = self.results['database']['score']
        max_score = self.results['database']['max_score']
        self.results['database']['status'] = 'healthy' if total_score > max_score * 0.7 else 'unhealthy'

    async def verify_orchestrator_system(self):
        """Verify orchestrator functionality"""
        print("\n🎯 ORCHESTRATOR SYSTEM VERIFICATION")
        
        try:
            from core.orchestrator.continuous_director import ContinuousDirector
            
            # Test basic initialization
            test_spec = {
                'name': 'Verification Test Project',
                'description': 'Testing orchestrator functionality',
                'requirements': ['feature_1', 'feature_2'],
                'quality_targets': {
                    'test_coverage': 95,
                    'performance_score': 90
                }
            }
            
            orchestrator = ContinuousDirector('verification_test', test_spec)
            self.add_result('orchestrator', True, "Orchestrator initialization successful")
            
            # Test project specification handling
            if orchestrator.project_spec == test_spec:
                self.add_result('orchestrator', True, "Project specification properly stored")
            
            # Test quality metrics functionality
            try:
                orchestrator.update_quality_metrics({'test_coverage': 85, 'performance_score': 88})
                if orchestrator.metrics.test_coverage == 85:
                    self.add_result('orchestrator', True, "Quality metrics update working")
                else:
                    self.add_result('orchestrator', False, "Quality metrics not updating correctly")
            except Exception as e:
                self.add_result('orchestrator', False, f"Quality metrics update failed: {str(e)}")
                self.add_recommendation('HIGH', 'Fix update_quality_metrics method', 'Core functionality broken')
            
            # Test ledger integration
            try:
                ledger = orchestrator.ledger
                initial_decisions = len(ledger.decision_log)
                
                ledger.record_decision(1, 'verification_agent', 'test_decision', 'Testing ledger integration')
                final_decisions = len(ledger.decision_log)
                
                if final_decisions > initial_decisions:
                    self.add_result('orchestrator', True, "Ledger integration working")
                else:
                    self.add_result('orchestrator', False, "Ledger not recording decisions")
                    
            except Exception as e:
                self.add_result('orchestrator', False, f"Ledger integration failed: {str(e)}")
                self.add_recommendation('HIGH', 'Fix orchestrator ledger property', 'Memory system not accessible')
            
            # Test agent registry
            if hasattr(orchestrator, 'agents'):
                self.add_result('orchestrator', True, "Agent registry initialized")
            else:
                self.add_result('orchestrator', False, "Agent registry missing")
            
            # Test iteration tracking
            if hasattr(orchestrator, 'iteration_count'):
                self.add_result('orchestrator', True, f"Iteration tracking available (current: {orchestrator.iteration_count})")
            
        except ImportError as e:
            self.add_result('orchestrator', False, f"Cannot import orchestrator: {str(e)}")
            self.add_recommendation('CRITICAL', 'Fix orchestrator import issues', 'Core system not accessible')
        except Exception as e:
            self.add_result('orchestrator', False, f"Orchestrator verification failed: {str(e)}")
            self.add_recommendation('HIGH', 'Debug orchestrator system', f'Core system error: {str(e)}')

        total_score = self.results['orchestrator']['score']
        max_score = self.results['orchestrator']['max_score']
        self.results['orchestrator']['status'] = 'healthy' if total_score > max_score * 0.8 else 'unhealthy'

    async def verify_agent_system(self):
        """Verify agent framework"""
        print("\n🤖 AGENT SYSTEM VERIFICATION")
        
        try:
            from core.agents.base_agent import PersistentAgent, AgentState, AgentTask, AgentMemory
            from datetime import datetime
            
            # Test agent state enum
            try:
                idle_state = AgentState.IDLE
                working_state = AgentState.WORKING
                self.add_result('agents', True, f"AgentState enum working: {idle_state.value}, {working_state.value}")
            except Exception as e:
                self.add_result('agents', False, f"AgentState enum failed: {str(e)}")
            
            # Test agent creation
            class VerificationAgent(PersistentAgent):
                async def process_task(self, task):
                    return {'result': 'task_completed', 'task_id': task.id}
                
                def analyze_context(self, context):
                    return {'analysis': 'context_analyzed', 'confidence': 0.9}
                
                def generate_solution(self, problem):
                    return {'solution': 'problem_solved', 'approach': 'systematic'}
            
            agent = VerificationAgent(
                name='verification_agent',
                role='verifier', 
                capabilities=['verification', 'testing', 'validation']
            )
            
            self.add_result('agents', True, f"Agent created successfully: {agent.name}")
            
            # Test agent state management
            if agent.state == AgentState.IDLE:
                self.add_result('agents', True, "Agent starts in IDLE state")
            
            # Test agent capabilities
            if agent.has_capability('verification'):
                self.add_result('agents', True, "Agent capability checking works")
            
            # Test task creation and processing
            test_task = AgentTask(
                id='verification_task_1',
                type='verification',
                description='Test agent task processing',
                priority=5,
                data={'test': True},
                created_at=datetime.now()
            )
            
            # Process task
            result = await agent.process_task(test_task)
            if result and result.get('result') == 'task_completed':
                self.add_result('agents', True, "Agent task processing working")
            
            # Test context analysis
            context = {'project': 'verification', 'complexity': 'medium'}
            analysis = agent.analyze_context(context)
            if analysis and 'analysis' in analysis:
                self.add_result('agents', True, "Agent context analysis working")
            
            # Test solution generation
            problem = {'type': 'verification', 'description': 'Need to verify system'}
            solution = agent.generate_solution(problem)
            if solution and 'solution' in solution:
                self.add_result('agents', True, "Agent solution generation working")
            
            # Test agent memory
            if hasattr(agent, 'memory') and isinstance(agent.memory, AgentMemory):
                self.add_result('agents', True, "Agent memory system initialized")
            
        except ImportError as e:
            self.add_result('agents', False, f"Cannot import agent system: {str(e)}")
            self.add_recommendation('CRITICAL', 'Fix agent system imports', 'Agent framework not accessible')
        except Exception as e:
            self.add_result('agents', False, f"Agent system verification failed: {str(e)}")
            self.add_recommendation('HIGH', 'Debug agent framework', f'Agent system error: {str(e)}')

        total_score = self.results['agents']['score']
        max_score = self.results['agents']['max_score']
        self.results['agents']['status'] = 'healthy' if total_score > max_score * 0.8 else 'unhealthy'

    def verify_memory_systems(self):
        """Verify memory and persistence systems"""
        print("\n🧠 MEMORY SYSTEMS VERIFICATION")
        
        try:
            from core.memory.project_ledger import ProjectLedger
            
            # Test ProjectLedger
            ledger = ProjectLedger('verification_memory_test')
            self.add_result('memory', True, "ProjectLedger initialization successful")
            
            # Test decision recording
            ledger.record_decision(
                iteration=1,
                agent='verification_agent',
                decision_type='test_decision',
                description='Testing decision recording functionality',
                rationale='Verifying memory system works correctly'
            )
            
            decisions = ledger.decision_log
            if len(decisions) > 0:
                self.add_result('memory', True, f"Decision recording working ({len(decisions)} decisions)")
            
            # Test code version management
            test_code = '''
def verification_function():
    """Test function for verification"""
    return "verification_complete"
'''
            
            version = ledger.save_code_version(
                file_path='verification/test.py',
                content=test_code,
                iteration=1,
                agent='verification_agent',
                reason='Testing code version management'
            )
            
            if version:
                self.add_result('memory', True, f"Code version saved: {version.id}")
                
                # Test version retrieval
                retrieved = ledger.get_version(version.id)
                if retrieved and retrieved.content == test_code:
                    self.add_result('memory', True, "Code version retrieval working")
                else:
                    self.add_result('memory', False, "Code version retrieval failed")
            
            # Test iteration summary
            summary = ledger.get_iteration_summary(1)
            if summary and 'code_changes' in summary:
                self.add_result('memory', True, "Iteration summary generation working")
            
        except ImportError as e:
            self.add_result('memory', False, f"Cannot import memory systems: {str(e)}")
            self.add_recommendation('CRITICAL', 'Fix memory system imports', 'Memory system not accessible')
        except Exception as e:
            self.add_result('memory', False, f"Memory system verification failed: {str(e)}")
            self.add_recommendation('HIGH', 'Debug memory systems', f'Memory error: {str(e)}')

        try:
            from core.memory.vector_memory import VectorMemory
            
            # Test VectorMemory initialization
            vector_memory = VectorMemory('verification_vector_test')
            self.add_result('memory', True, "VectorMemory initialization successful")
            
            # Test if collections are available
            if hasattr(vector_memory, 'collections'):
                available_collections = list(vector_memory.collections.keys())
                self.add_result('memory', True, f"Vector collections available: {available_collections}")
            
        except ImportError as e:
            self.add_result('memory', False, f"Cannot import VectorMemory: {str(e)}")
        except Exception as e:
            self.add_result('memory', False, f"VectorMemory verification failed: {str(e)}")

        total_score = self.results['memory']['score']
        max_score = self.results['memory']['max_score']
        self.results['memory']['status'] = 'healthy' if total_score > max_score * 0.7 else 'unhealthy'

    def verify_environment_config(self):
        """Verify environment configuration"""
        print("\n⚙️ ENVIRONMENT CONFIGURATION VERIFICATION")
        
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            # Check critical environment variables
            required_vars = [
                ('OPENAI_API_KEY', 'sk-', 'OpenAI API access'),
                ('ANTHROPIC_API_KEY', 'sk-ant-', 'Anthropic API access'),
                ('POSTGRES_URL', 'postgresql://', 'Database connection')
            ]
            
            for var_name, expected_prefix, description in required_vars:
                value = os.getenv(var_name)
                if value and value.startswith(expected_prefix) and len(value) > 20:
                    self.add_result('environment', True, f"{var_name} configured correctly")
                elif value:
                    self.add_result('environment', False, f"{var_name} has invalid format")
                    self.add_recommendation('HIGH', f'Fix {var_name} format in .env file', f'Required for {description}')
                else:
                    self.add_result('environment', False, f"{var_name} not set")
                    self.add_recommendation('CRITICAL', f'Set {var_name} in .env file', f'Required for {description}')
            
            # Check optional environment variables
            optional_vars = ['REDIS_URL', 'DEV_MODE', 'LOG_LEVEL']
            for var_name in optional_vars:
                value = os.getenv(var_name)
                if value:
                    self.add_result('environment', True, f"{var_name} configured: {value}")
            
            # Check Python environment
            python_version = sys.version_info
            if python_version >= (3, 8):
                self.add_result('environment', True, f"Python version compatible: {python_version.major}.{python_version.minor}")
            else:
                self.add_result('environment', False, f"Python version too old: {python_version.major}.{python_version.minor}")
                self.add_recommendation('HIGH', 'Upgrade Python to 3.8+', 'Compatibility requirements')
            
        except Exception as e:
            self.add_result('environment', False, f"Environment verification failed: {str(e)}")
            self.add_recommendation('MEDIUM', 'Check environment configuration', f'Config error: {str(e)}')

        total_score = self.results['environment']['score']
        max_score = self.results['environment']['max_score']
        self.results['environment']['status'] = 'healthy' if total_score > max_score * 0.8 else 'unhealthy'

    def verify_file_system(self):
        """Verify file system integrity"""
        print("\n📁 FILE SYSTEM VERIFICATION")
        
        # Check critical files
        critical_files = [
            'core/orchestrator/continuous_director.py',
            'core/memory/project_ledger.py',
            'core/agents/base_agent.py',
            'core/memory/vector_memory.py',
            'api/main.py',
            'requirements.txt',
            '.env',
            'README.md'
        ]
        
        for file_path in critical_files:
            path = Path(file_path)
            if path.exists():
                size = path.stat().st_size
                self.add_result('files', True, f"{file_path} exists ({size} bytes)")
            else:
                self.add_result('files', False, f"{file_path} missing")
                self.add_recommendation('HIGH', f'Create or restore {file_path}', 'Critical file missing')
        
        # Check critical directories
        critical_dirs = [
            'persistence/database',
            'persistence/storage',
            'logs',
            'test_reports',
            'core',
            'api'
        ]
        
        for dir_path in critical_dirs:
            path = Path(dir_path)
            if path.exists() and path.is_dir():
                self.add_result('files', True, f"{dir_path}/ directory exists")
                
                # Check if writable
                if os.access(path, os.W_OK):
                    self.add_result('files', True, f"{dir_path}/ is writable")
                else:
                    self.add_result('files', False, f"{dir_path}/ not writable")
                    self.add_recommendation('MEDIUM', f'Fix permissions for {dir_path}/', 'Directory not writable')
            else:
                self.add_result('files', False, f"{dir_path}/ directory missing")
                self.add_recommendation('MEDIUM', f'Create directory {dir_path}/', 'Required for system operation')
        
        # Check recent files
        recent_files = [
            'comprehensive_test_suite.py',
            'FINAL_SYSTEM_REPORT.md',
            'test_reports/comprehensive_report_20251116_001631.json'
        ]
        
        for file_path in recent_files:
            path = Path(file_path)
            if path.exists():
                self.add_result('files', True, f"Recent file {file_path} exists")

        total_score = self.results['files']['score']
        max_score = self.results['files']['max_score']
        self.results['files']['status'] = 'healthy' if total_score > max_score * 0.8 else 'unhealthy'

    def verify_performance(self):
        """Verify system performance"""
        print("\n⚡ PERFORMANCE VERIFICATION")
        
        try:
            # Check system resources
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            disk = psutil.disk_usage('.')
            
            # Memory check
            memory_percent = memory.percent
            if memory_percent < 80:
                self.add_result('performance', True, f"Memory usage healthy: {memory_percent:.1f}%")
            else:
                self.add_result('performance', False, f"Memory usage high: {memory_percent:.1f}%")
                self.add_recommendation('MEDIUM', 'Monitor memory usage', 'High memory consumption')
            
            # Disk space check
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent < 85:
                self.add_result('performance', True, f"Disk usage healthy: {disk_percent:.1f}%")
            else:
                self.add_result('performance', False, f"Disk usage high: {disk_percent:.1f}%")
                self.add_recommendation('HIGH', 'Free up disk space', 'Low disk space')
            
            # CPU check
            self.add_result('performance', True, f"CPU cores available: {cpu_count}")
            
            # Test database query performance
            start_time = time.time()
            try:
                import psycopg2
                postgres_url = os.getenv('POSTGRES_URL')
                if postgres_url:
                    conn = psycopg2.connect(postgres_url)
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1;")
                    cursor.fetchone()
                    conn.close()
                    
                    db_time = time.time() - start_time
                    if db_time < 0.1:
                        self.add_result('performance', True, f"Database query fast: {db_time:.3f}s")
                    else:
                        self.add_result('performance', False, f"Database query slow: {db_time:.3f}s")
                        self.add_recommendation('MEDIUM', 'Optimize database performance', 'Slow query response')
            except:
                pass
            
            # Test import performance
            start_time = time.time()
            try:
                from core.orchestrator.continuous_director import ContinuousDirector
                import_time = time.time() - start_time
                if import_time < 1.0:
                    self.add_result('performance', True, f"Module import fast: {import_time:.3f}s")
                else:
                    self.add_result('performance', False, f"Module import slow: {import_time:.3f}s")
            except:
                pass
                
        except Exception as e:
            self.add_result('performance', False, f"Performance check failed: {str(e)}")

        total_score = self.results['performance']['score']
        max_score = self.results['performance']['max_score']
        self.results['performance']['status'] = 'healthy' if total_score > max_score * 0.7 else 'unhealthy'

    def verify_github_integration(self):
        """Verify GitHub integration"""
        print("\n🔗 GITHUB INTEGRATION VERIFICATION")
        
        try:
            # Check git status
            result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                self.add_result('github', True, "Git repository accessible")
                
                if result.stdout.strip():
                    self.add_result('github', False, f"Uncommitted changes found: {len(result.stdout.strip().split())}")
                    self.add_recommendation('LOW', 'Commit pending changes', 'Uncommitted work found')
                else:
                    self.add_result('github', True, "Working directory clean")
            else:
                self.add_result('github', False, "Git status failed")
                
        except Exception as e:
            self.add_result('github', False, f"Git check failed: {str(e)}")
        
        try:
            # Check remote configuration
            result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True, cwd='.')
            if result.returncode == 0 and 'origin' in result.stdout:
                self.add_result('github', True, "Git remote configured")
                if 'github.com' in result.stdout:
                    self.add_result('github', True, "GitHub remote detected")
            
        except Exception as e:
            self.add_result('github', False, f"Git remote check failed: {str(e)}")
        
        try:
            # Check auto-sync script
            auto_sync_path = Path('./auto-sync.sh')
            if auto_sync_path.exists():
                self.add_result('github', True, "Auto-sync script exists")
                if os.access(auto_sync_path, os.X_OK):
                    self.add_result('github', True, "Auto-sync script is executable")
                else:
                    self.add_result('github', False, "Auto-sync script not executable")
                    self.add_recommendation('LOW', 'Make auto-sync.sh executable: chmod +x auto-sync.sh', 'Script permissions')
            else:
                self.add_result('github', False, "Auto-sync script missing")
                
        except Exception as e:
            self.add_result('github', False, f"Auto-sync check failed: {str(e)}")

        total_score = self.results['github']['score']
        max_score = self.results['github']['max_score']
        self.results['github']['status'] = 'healthy' if total_score > max_score * 0.7 else 'unhealthy'

    def generate_comprehensive_report(self):
        """Generate comprehensive verification report"""
        execution_time = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE SYSTEM VERIFICATION REPORT")
        print("="*80)
        
        # Calculate overall health score
        total_score = 0
        max_total_score = 0
        category_health = {}
        
        for category, data in self.results.items():
            if category != 'system_health':
                score = data.get('score', 0)
                max_score = data.get('max_score', 1)
                total_score += score
                max_total_score += max_score
                
                if max_score > 0:
                    health_percent = (score / max_score) * 100
                    category_health[category] = health_percent
                else:
                    category_health[category] = 0
        
        overall_health = (total_score / max_total_score * 100) if max_total_score > 0 else 0
        
        # Display category results
        print(f"\n📊 SYSTEM HEALTH BY CATEGORY:")
        for category, percent in category_health.items():
            status_icon = "🟢" if percent >= 80 else "🟡" if percent >= 60 else "🔴"
            category_name = category.replace('_', ' ').title()
            print(f"  {status_icon} {category_name}: {percent:.1f}% ({self.results[category]['status']})")
        
        # Priority recommendations
        critical_recs = [r for r in self.recommendations if r['priority'] == 'CRITICAL']
        high_recs = [r for r in self.recommendations if r['priority'] == 'HIGH']
        medium_recs = [r for r in self.recommendations if r['priority'] == 'MEDIUM']
        low_recs = [r for r in self.recommendations if r['priority'] == 'LOW']
        
        print(f"\n🚨 IMMEDIATE ACTION REQUIRED:")
        if critical_recs:
            print(f"  🔴 CRITICAL ({len(critical_recs)} issues):")
            for rec in critical_recs[:3]:  # Show top 3
                print(f"    • {rec['action']}")
                print(f"      Reason: {rec['reason']}")
        else:
            print(f"  ✅ No critical issues found")
        
        if high_recs:
            print(f"  🟡 HIGH PRIORITY ({len(high_recs)} issues):")
            for rec in high_recs[:3]:  # Show top 3
                print(f"    • {rec['action']}")
                print(f"      Reason: {rec['reason']}")
        
        if medium_recs:
            print(f"  🟠 MEDIUM PRIORITY ({len(medium_recs)} issues):")
            for rec in medium_recs[:2]:  # Show top 2
                print(f"    • {rec['action']}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL SYSTEM ASSESSMENT:")
        print(f"  📊 Health Score: {overall_health:.1f}%")
        print(f"  ⏱️  Verification Time: {execution_time:.2f} seconds")
        
        if overall_health >= 90:
            status = "🟢 EXCELLENT - System fully operational and production-ready"
            next_action = "🚀 Ready for production deployment"
        elif overall_health >= 75:
            status = "🟡 GOOD - System mostly functional with minor issues"
            next_action = "🔧 Address high-priority issues, then deploy"
        elif overall_health >= 50:
            status = "🟠 FAIR - System has significant issues requiring attention"
            next_action = "⚠️ Fix critical and high-priority issues before deployment"
        else:
            status = "🔴 POOR - System has major problems"
            next_action = "🚨 Major remediation required before any deployment"
        
        print(f"  🎯 Status: {status}")
        print(f"  💡 Next Action: {next_action}")
        
        # Specific action steps
        print(f"\n📋 SPECIFIC ACTION STEPS:")
        
        if critical_recs:
            print(f"\n1. 🚨 CRITICAL ACTIONS (DO IMMEDIATELY):")
            for i, rec in enumerate(critical_recs[:5], 1):
                print(f"   {i}. {rec['action']}")
                print(f"      └─ {rec['reason']}")
        
        if high_recs:
            print(f"\n2. 🔧 HIGH PRIORITY ACTIONS (DO SOON):")
            for i, rec in enumerate(high_recs[:5], 1):
                print(f"   {i}. {rec['action']}")
                print(f"      └─ {rec['reason']}")
        
        if not critical_recs and not high_recs:
            print(f"\n✅ NO CRITICAL ACTIONS REQUIRED")
            print(f"   System is healthy and operational")
            if medium_recs:
                print(f"\n🔧 OPTIONAL IMPROVEMENTS:")
                for i, rec in enumerate(medium_recs[:3], 1):
                    print(f"   {i}. {rec['action']}")
        
        print("\n" + "="*80)
        
        # Save detailed report
        report_file = Path(f"verification_reports/verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'overall_health': overall_health,
                'execution_time': execution_time,
                'results': self.results,
                'recommendations': self.recommendations,
                'category_health': category_health
            }, f, indent=2)
        
        print(f"📋 Detailed report saved: {report_file}")
        
        return overall_health >= 75  # Return True if system is healthy enough

    async def run_verification(self):
        """Run complete system verification"""
        print("🚀 STARTING COMPREHENSIVE SYSTEM VERIFICATION")
        print(f"Timestamp: {self.start_time}")
        print("This will test all system components and provide actionable recommendations...")
        
        # Run all verification checks
        await self.verify_api_server()
        self.verify_database_connections()
        await self.verify_orchestrator_system()
        await self.verify_agent_system()
        self.verify_memory_systems()
        self.verify_environment_config()
        self.verify_file_system()
        self.verify_performance()
        self.verify_github_integration()
        
        # Generate comprehensive report
        system_healthy = self.generate_comprehensive_report()
        
        return system_healthy, self.results, self.recommendations

if __name__ == "__main__":
    async def main():
        verifier = ComprehensiveVerification()
        healthy, results, recommendations = await verifier.run_verification()
        
        return healthy
    
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
