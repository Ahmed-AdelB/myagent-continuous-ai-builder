#!/usr/bin/env python3
"""
22_MyAgent System Verification Script
Comprehensive check of all components before starting the continuous AI system
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
import json

def print_status(status, message):
    """Print status with emoji"""
    icon = "✅" if status else "❌"
    print(f"{icon} {message}")
    return status

def check_environment():
    """Check basic environment setup"""
    print("\n🔍 Checking Environment Setup...")
    
    # Check Python version
    py_version = sys.version_info
    python_ok = py_version.major == 3 and py_version.minor >= 11
    print_status(python_ok, f"Python {py_version.major}.{py_version.minor}.{py_version.micro} (need 3.11+)")
    
    # Check project structure
    required_dirs = ['core', 'api', 'frontend', 'persistence', 'tests', 'venv']
    dirs_ok = all(Path(d).exists() for d in required_dirs)
    print_status(dirs_ok, f"Project directories: {', '.join(required_dirs)}")
    
    # Check .env file
    env_ok = Path('.env').exists()
    print_status(env_ok, ".env file exists")
    
    if env_ok:
        with open('.env') as f:
            env_content = f.read()
            openai_ok = 'OPENAI_API_KEY=sk-' in env_content
            anthropic_ok = 'ANTHROPIC_API_KEY=sk-ant-' in env_content
            print_status(openai_ok, "OpenAI API key configured")
            print_status(anthropic_ok, "Anthropic API key configured")
    
    return python_ok and dirs_ok and env_ok

def check_dependencies():
    """Check Python package dependencies"""
    print("\n📦 Checking Dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'sqlalchemy', 'psycopg2', 'redis',
        'langchain', 'langchain_openai', 'langchain_anthropic', 'chromadb',
        'numpy', 'pandas', 'loguru', 'websockets'
    ]
    
    all_good = True
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print_status(True, f"{package} installed")
        except ImportError:
            print_status(False, f"{package} MISSING")
            all_good = False
    
    return all_good

def check_core_imports():
    """Check if core modules can be imported"""
    print("\n🧠 Checking Core Module Imports...")
    
    core_modules = [
        'core.orchestrator.continuous_director',
        'core.agents.coder_agent',
        'core.agents.tester_agent', 
        'core.agents.debugger_agent',
        'core.agents.architect_agent',
        'core.agents.analyzer_agent',
        'core.agents.ui_refiner_agent',
        'core.memory.project_ledger',
        'core.memory.vector_memory',
        'core.memory.error_knowledge_graph'
    ]
    
    all_good = True
    sys.path.append('.')
    
    for module in core_modules:
        try:
            importlib.import_module(module)
            print_status(True, f"{module} imports successfully")
        except Exception as e:
            print_status(False, f"{module} FAILED: {str(e)[:60]}...")
            all_good = False
    
    return all_good

def check_databases():
    """Check database connections"""
    print("\n🗄️ Checking Database Connections...")
    
    # Test Redis
    redis_ok = False
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        redis_ok = True
        print_status(True, "Redis connection successful")
    except Exception as e:
        print_status(False, f"Redis connection failed: {e}")
    
    # Test PostgreSQL
    postgres_ok = False
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            database="myagent_db", 
            user="myagent_user",
            password="myagent_pass"
        )
        conn.close()
        postgres_ok = True
        print_status(True, "PostgreSQL connection successful")
    except Exception as e:
        print_status(False, f"PostgreSQL connection failed: {e}")
    
    return redis_ok and postgres_ok

def check_api_structure():
    """Check API structure"""
    print("\n🌐 Checking API Structure...")
    
    api_files = [
        'api/main.py',
        'api/routes.py', 
        'api/websocket.py',
        'api/middleware.py'
    ]
    
    all_good = True
    for file in api_files:
        exists = Path(file).exists()
        print_status(exists, f"{file} exists")
        if not exists:
            all_good = False
    
    return all_good

def check_frontend_structure():
    """Check frontend structure"""
    print("\n⚛️ Checking Frontend Structure...")
    
    frontend_files = [
        'frontend/package.json',
        'frontend/src/App.tsx',
        'frontend/src/main.tsx'
    ]
    
    all_good = True
    for file in frontend_files:
        exists = Path(file).exists()
        print_status(exists, f"{file} exists")
        if not exists:
            all_good = False
    
    # Check if node_modules exists
    node_modules_ok = Path('frontend/node_modules').exists()
    print_status(node_modules_ok, "Frontend dependencies installed")
    
    return all_good and node_modules_ok

def generate_startup_commands():
    """Generate startup commands"""
    print("\n🚀 Startup Commands Ready:")
    
    commands = [
        "# Terminal 1: Start API Server",
        "cd /home/aadel/projects/22_MyAgent && source venv/bin/activate",
        "uvicorn api.main:app --reload --host 0.0.0.0 --port 8000",
        "",
        "# Terminal 2: Start Frontend", 
        "cd /home/aadel/projects/22_MyAgent/frontend",
        "npm run dev -- --host 0.0.0.0 --port 5173",
        "",
        "# Terminal 3: Start Orchestrator",
        "cd /home/aadel/projects/22_MyAgent && source venv/bin/activate", 
        "python -m core.orchestrator.continuous_director",
        "",
        "# Access URLs:",
        "# API: http://136.112.55.140:8000",
        "# Frontend: http://136.112.55.140:5173",
        "# API Docs: http://136.112.55.140:8000/docs"
    ]
    
    for cmd in commands:
        print(f"  {cmd}")

def main():
    """Run complete system verification"""
    print("🎯 22_MyAgent System Verification Starting...")
    
    checks = [
        ('Environment', check_environment),
        ('Dependencies', check_dependencies), 
        ('Core Imports', check_core_imports),
        ('Databases', check_databases),
        ('API Structure', check_api_structure),
        ('Frontend Structure', check_frontend_structure)
    ]
    
    results = {}
    overall_status = True
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
            overall_status = overall_status and results[name]
        except Exception as e:
            print_status(False, f"{name} check failed: {e}")
            results[name] = False
            overall_status = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    for name, status in results.items():
        print_status(status, f"{name} Check")
    
    print("\n" + ("✅ SYSTEM READY" if overall_status else "❌ ISSUES FOUND"))
    
    if overall_status:
        print("\n🎉 All systems operational! Ready to start 22_MyAgent.")
        generate_startup_commands()
    else:
        print("\n⚠️ Please fix the issues above before starting the system.")
        
    return overall_status

if __name__ == "__main__":
    main()
