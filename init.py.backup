#!/usr/bin/env python3
"""
MyAgent Continuous AI Development System - Initialization Script
This script initializes the entire system from scratch.
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path
from loguru import logger
import json
import sqlite3
import psycopg2
import redis
from datetime import datetime

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

class SystemInitializer:
    """Initializes the MyAgent system"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.errors = []
        self.successes = []

    def log_success(self, message):
        """Log successful operations"""
        self.successes.append(message)
        logger.success(f"✅ {message}")
        print(f"✅ {message}")

    def log_error(self, message):
        """Log error operations"""
        self.errors.append(message)
        logger.error(f"❌ {message}")
        print(f"❌ {message}")

    def run_command(self, command, description):
        """Run shell command"""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                self.log_success(f"{description}")
                return True
            else:
                self.log_error(f"{description}: {result.stderr}")
                return False
        except Exception as e:
            self.log_error(f"{description}: {str(e)}")
            return False

    def check_dependencies(self):
        """Check system dependencies"""
        print("\n🔍 Checking System Dependencies...")

        dependencies = [
            ("python3", "Python 3.11+"),
            ("node", "Node.js 18+"),
            ("npm", "NPM package manager"),
            ("psql", "PostgreSQL client"),
            ("redis-cli", "Redis client"),
            ("docker", "Docker"),
            ("git", "Git version control")
        ]

        for cmd, desc in dependencies:
            if subprocess.run(f"which {cmd}", shell=True, capture_output=True).returncode == 0:
                self.log_success(f"{desc} found")
            else:
                self.log_error(f"{desc} not found - please install {cmd}")

    def setup_python_environment(self):
        """Setup Python virtual environment"""
        print("\n🐍 Setting up Python Environment...")

        # Create virtual environment
        if not (self.base_dir / "venv").exists():
            self.run_command("python3 -m venv venv", "Created Python virtual environment")
        else:
            self.log_success("Python virtual environment already exists")

        # Install Python dependencies
        pip_cmd = str(self.base_dir / "venv" / "bin" / "pip")
        requirements = [
            "fastapi",
            "uvicorn",
            "sqlalchemy",
            "psycopg2-binary",
            "redis",
            "langchain",
            "langchain-openai",
            "chromadb",
            "numpy",
            "pandas",
            "loguru",
            "python-jose[cryptography]",
            "passlib[bcrypt]",
            "python-multipart",
            "websockets",
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "black",
            "flake8",
            "mypy"
        ]

        for package in requirements:
            self.run_command(f"{pip_cmd} install {package}", f"Installed {package}")

    def setup_node_environment(self):
        """Setup Node.js environment"""
        print("\n📦 Setting up Node.js Environment...")

        frontend_dir = self.base_dir / "frontend"
        if frontend_dir.exists():
            os.chdir(frontend_dir)

            # Install dependencies
            if not (frontend_dir / "node_modules").exists():
                self.run_command("npm install", "Installed Node.js dependencies")
            else:
                self.log_success("Node.js dependencies already installed")

            # Install additional packages
            additional_packages = [
                "react-chartjs-2",
                "chart.js",
                "socket.io-client",
                "@types/react",
                "@types/react-dom",
                "typescript"
            ]

            for package in additional_packages:
                self.run_command(f"npm install {package}", f"Installed {package}")

        os.chdir(self.base_dir)

    def setup_database(self):
        """Setup PostgreSQL database"""
        print("\n🗄️ Setting up Database...")

        try:
            # Check if PostgreSQL is running
            subprocess.run("pg_isready", check=True, capture_output=True)
            self.log_success("PostgreSQL is running")

            # Create database and user
            commands = [
                "createdb myagent_db",
                "createuser myagent_user",
                "psql -d myagent_db -f scripts/init_database.sql"
            ]

            for cmd in commands:
                try:
                    subprocess.run(cmd, shell=True, check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    pass  # Database/user might already exist

            self.log_success("Database initialized")

        except subprocess.CalledProcessError:
            self.log_error("PostgreSQL not running - please start PostgreSQL service")

    def setup_redis(self):
        """Setup Redis"""
        print("\n🔴 Setting up Redis...")

        try:
            # Check if Redis is running
            subprocess.run("redis-cli ping", shell=True, check=True, capture_output=True)
            self.log_success("Redis is running")

            # Test Redis connection
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.set('myagent_init_test', 'success')
            if r.get('myagent_init_test') == 'success':
                self.log_success("Redis connection test passed")
                r.delete('myagent_init_test')
            else:
                self.log_error("Redis connection test failed")

        except Exception as e:
            self.log_error(f"Redis setup failed: {str(e)}")

    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating Directory Structure...")

        directories = [
            "persistence",
            "persistence/database",
            "persistence/checkpoints",
            "persistence/learning",
            "logs",
            "tmp",
            "models",
            "exports"
        ]

        for directory in directories:
            dir_path = self.base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.log_success(f"Created directory: {directory}")

    def setup_environment_file(self):
        """Create environment configuration file"""
        print("\n⚙️ Setting up Environment Configuration...")

        env_content = """# MyAgent Environment Configuration
# Database Configuration
POSTGRES_URL=postgresql://myagent_user:password@localhost/myagent_db
REDIS_URL=redis://localhost:6379

# API Keys (Replace with your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_here

# Development Settings
DEV_MODE=true
DEBUG=true
LOG_LEVEL=INFO

# System Configuration
MAX_ITERATIONS=1000
CHECKPOINT_INTERVAL=10
QUALITY_THRESHOLD=95
TEST_COVERAGE_TARGET=95
PERFORMANCE_TARGET=90

# Agent Configuration
AGENT_TIMEOUT=300
MAX_CONCURRENT_AGENTS=6
LEARNING_RATE=0.1

# API Configuration
API_HOST=localhost
API_PORT=8000
FRONTEND_PORT=3000

# Vector Database
CHROMA_PERSIST_DIRECTORY=./chroma_db
"""

        env_file = self.base_dir / ".env"
        if not env_file.exists():
            with open(env_file, 'w') as f:
                f.write(env_content)
            self.log_success("Created .env configuration file")
        else:
            self.log_success(".env file already exists")

    def initialize_agents(self):
        """Initialize AI agents"""
        print("\n🤖 Initializing AI Agents...")

        try:
            # Import and initialize agents
            from core.agents.coder_agent import CoderAgent
            from core.agents.tester_agent import TesterAgent
            from core.agents.debugger_agent import DebuggerAgent
            from core.agents.architect_agent import ArchitectAgent
            from core.agents.analyzer_agent import AnalyzerAgent
            from core.agents.ui_refiner_agent import UIRefinerAgent

            agents = [
                CoderAgent(),
                TesterAgent(),
                DebuggerAgent(),
                ArchitectAgent(),
                AnalyzerAgent(),
                UIRefinerAgent()
            ]

            for agent in agents:
                agent.save_state()
                self.log_success(f"Initialized {agent.name}")

        except Exception as e:
            self.log_error(f"Agent initialization failed: {str(e)}")

    def setup_logging(self):
        """Setup logging configuration"""
        print("\n📝 Setting up Logging...")

        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "[{time:YYYY-MM-DD HH:mm:ss}] {level} | {name}:{function}:{line} - {message}",
                    "style": "{"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": "INFO"
                },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": "logs/myagent.log",
                    "formatter": "default",
                    "level": "DEBUG"
                }
            },
            "root": {
                "level": "DEBUG",
                "handlers": ["console", "file"]
            }
        }

        config_file = self.base_dir / "logging_config.json"
        with open(config_file, 'w') as f:
            json.dump(log_config, f, indent=2)

        self.log_success("Logging configuration created")

    def run_initial_tests(self):
        """Run initial system tests"""
        print("\n🧪 Running Initial Tests...")

        # Test Python environment
        python_cmd = str(self.base_dir / "venv" / "bin" / "python")
        test_commands = [
            f"{python_cmd} -c 'import fastapi; print(\"FastAPI OK\")'",
            f"{python_cmd} -c 'import langchain; print(\"LangChain OK\")'",
            f"{python_cmd} -c 'import redis; print(\"Redis module OK\")'",
        ]

        for cmd in test_commands:
            self.run_command(cmd, f"Python module test")

    def generate_startup_scripts(self):
        """Generate startup scripts"""
        print("\n🚀 Generating Startup Scripts...")

        # Development startup script
        dev_script = """#!/bin/bash
echo "🚀 Starting MyAgent Development Environment..."

# Activate Python environment
source venv/bin/activate

# Start Redis (if not running)
if ! pgrep redis-server > /dev/null; then
    echo "Starting Redis..."
    redis-server --daemonize yes
fi

# Start PostgreSQL (if not running)
if ! pgrep postgres > /dev/null; then
    echo "Starting PostgreSQL..."
    brew services start postgresql@14 2>/dev/null || sudo service postgresql start 2>/dev/null || echo "Please start PostgreSQL manually"
fi

# Start API server
echo "Starting API server..."
uvicorn api.main:app --reload --port 8000 &
API_PID=$!

# Start frontend
echo "Starting frontend..."
cd frontend && npm run dev &
FRONTEND_PID=$!

echo "✅ MyAgent is now running!"
echo "🌐 API: http://localhost:8000"
echo "🖥️  Frontend: http://localhost:3000"
echo "📚 Docs: http://localhost:8000/docs"

# Wait for user input to stop
read -p "Press Enter to stop all services..."

# Kill background processes
kill $API_PID $FRONTEND_PID 2>/dev/null
echo "🛑 Services stopped"
"""

        with open(self.base_dir / "start_dev.sh", 'w') as f:
            f.write(dev_script)

        self.run_command("chmod +x start_dev.sh", "Made dev startup script executable")

        # Production startup script
        prod_script = """#!/bin/bash
echo "🏭 Starting MyAgent Production Environment..."

# Build frontend
cd frontend && npm run build
cd ..

# Start with production settings
export NODE_ENV=production
export DEV_MODE=false

# Start services
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

        with open(self.base_dir / "start_prod.sh", 'w') as f:
            f.write(prod_script)

        self.run_command("chmod +x start_prod.sh", "Made prod startup script executable")

    def print_summary(self):
        """Print initialization summary"""
        print("\n" + "="*60)
        print("🎉 MYAGENT INITIALIZATION COMPLETE!")
        print("="*60)

        print(f"\n✅ Successes: {len(self.successes)}")
        for success in self.successes[-5:]:  # Show last 5
            print(f"   • {success}")

        if self.errors:
            print(f"\n❌ Errors: {len(self.errors)}")
            for error in self.errors:
                print(f"   • {error}")

        print(f"\n🚀 Quick Start:")
        print(f"   1. ./start_dev.sh")
        print(f"   2. Open http://localhost:3000")
        print(f"   3. Check API docs at http://localhost:8000/docs")

        print(f"\n📚 Next Steps:")
        print(f"   • Configure API keys in .env file")
        print(f"   • Run tests: pytest tests/")
        print(f"   • Read documentation in docs/")
        print(f"   • Start developing with the continuous AI system!")

        print(f"\n🔗 Repository: https://github.com/Ahmed-AdelB/myagent-continuous-ai-builder")
        print("="*60)

async def main():
    """Main initialization function"""
    print("🤖 MyAgent Continuous AI Development System")
    print("🔧 Initialization Starting...")
    print("⚡ Using Maximum Parallel Processing")

    initializer = SystemInitializer()

    # Run initialization steps
    steps = [
        initializer.check_dependencies,
        initializer.create_directories,
        initializer.setup_environment_file,
        initializer.setup_logging,
        initializer.setup_python_environment,
        initializer.setup_node_environment,
        initializer.setup_database,
        initializer.setup_redis,
        initializer.initialize_agents,
        initializer.run_initial_tests,
        initializer.generate_startup_scripts
    ]

    for step in steps:
        try:
            step()
        except Exception as e:
            initializer.log_error(f"Step {step.__name__} failed: {str(e)}")

    initializer.print_summary()

if __name__ == "__main__":
    asyncio.run(main())