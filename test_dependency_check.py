#!/usr/bin/env python3
"""
Dependency and Configuration Check
Tests which dependencies are available and configuration state
"""

import sys
import subprocess
from pathlib import Path


def check_dependency(package_name: str) -> bool:
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def main():
    print("=" * 80)
    print("📦 DEPENDENCY CHECK")
    print("=" * 80)

    # Critical dependencies
    critical_deps = {
        "loguru": "Logging",
        "pydantic": "Data validation",
        "pydantic_settings": "Settings management",
        "fastapi": "API framework",
        "uvicorn": "ASGI server",
        "chromadb": "Vector database",
        "sentence_transformers": "Embeddings",
        "networkx": "Graph database",
        "langchain": "LLM framework",
        "langchain_core": "LangChain core",
        "langchain_openai": "OpenAI integration",
        "asyncpg": "PostgreSQL async driver",
        "redis": "Redis client",
        "numpy": "Numerical computing",
        "pandas": "Data analysis"
    }

    print("\n🔍 Checking critical dependencies...")
    missing = []
    installed = []

    for package, description in critical_deps.items():
        if check_dependency(package):
            installed.append(package)
            print(f"  ✅ {package:25} - {description}")
        else:
            missing.append(package)
            print(f"  ❌ {package:25} - {description} [MISSING]")

    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Installed:  {len(installed)}/{len(critical_deps)}")
    print(f"Missing:    {len(missing)}/{len(critical_deps)}")

    if missing:
        print("\n" + "=" * 80)
        print("💡 TO INSTALL MISSING DEPENDENCIES:")
        print("=" * 80)
        print(f"pip install {' '.join(missing)}")

    # Check configuration files
    print("\n" + "=" * 80)
    print("📝 CONFIGURATION FILES")
    print("=" * 80)

    config_files = {
        ".env": "Environment variables",
        ".env.example": "Environment template",
        "requirements.txt": "Python dependencies",
        "frontend/package.json": "Frontend dependencies",
        "docker-compose.yml": "Docker configuration",
        "CLAUDE.md": "Project documentation"
    }

    for file, description in config_files.items():
        path = Path(file)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✅ {file:30} - {description} ({size:,} bytes)")
        else:
            print(f"  ❌ {file:30} - {description} [MISSING]")

    # Check directories
    print("\n" + "=" * 80)
    print("📁 REQUIRED DIRECTORIES")
    print("=" * 80)

    required_dirs = [
        "core", "api", "config", "persistence",
        "frontend", "tests", "scripts", "docs"
    ]

    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists():
            py_files = list(path.rglob("*.py")) if path.is_dir() else []
            print(f"  ✅ {dir_name:20} - {len(py_files)} Python files")
        else:
            print(f"  ❌ {dir_name:20} [MISSING]")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
