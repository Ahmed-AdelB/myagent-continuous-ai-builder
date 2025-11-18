# GEMINI.md - AI Assistant Guide for the Continuous AI App Builder

## 🎯 Project Overview

This project is the **Continuous AI App Builder**, a sophisticated, multi-agent system designed for autonomous and iterative software development. Its core philosophy is not speed, but **guaranteed eventual perfection**. It employs a team of specialized AI agents managed by a central orchestrator to continuously write, test, debug, and refine a software application until a predefined set of high-quality metrics (e.g., test coverage, bug counts, performance) are met.

The system is built on a modern technology stack:
*   **Backend:** Python with FastAPI, providing a comprehensive API to manage and monitor the development process.
*   **Frontend:** A React-based dashboard for real-time visualization of agents, metrics, and progress.
*   **Core Logic:** A multi-agent system written in Python, using `langchain` for LLM interaction.
*   **Persistence:** A robust memory system utilizing PostgreSQL for structured data, Redis for messaging and caching, and ChromaDB for vector-based semantic memory.
*   **Orchestration:** The entire stack is containerized and managed via Docker and `docker-compose`.

The architecture is centered around the `ContinuousDirector` (`core/orchestrator/continuous_director.py`), which coordinates the agents (Coder, Tester, Debugger, Architect, etc.) and leverages a persistent memory system to ensure context and learning are never lost between sessions.

## 🚀 Building and Running

This project can be run either via Docker (recommended for a complete environment) or manually for more granular control.

### Using Docker (Recommended)

The `docker-compose.yml` file orchestrates all necessary services.

1.  **Prerequisites:** Docker and Docker Compose installed.
2.  **Configuration:** Create a `.env` file from the `.env.example` template and fill in the required API keys (e.g., `OPENAI_API_KEY`) and passwords.
3.  **Build and Run:**
    ```bash
    # Build and start all services (Postgres, Redis, ChromaDB, Backend)
    docker-compose up --build
    ```
4.  **Development Mode:** To include the live-reloading frontend development server:
    ```bash
    # Use the 'dev' profile to add the frontend service
    docker-compose --profile dev up --build
    ```

### Running Manually

For more direct control over each component.

1.  **Prerequisites:** Python 3.11+, Node.js 18+, and running instances of PostgreSQL and Redis.
2.  **Setup Backend:**
    ```bash
    # Install Python dependencies
    pip install -r requirements.txt
    ```
3.  **Setup Frontend:**
    ```bash
    # Install Node.js dependencies
    cd frontend
    npm install
    cd ..
    ```
4.  **Launch Services:** Run each command in a separate terminal.
    *   **Start the API Server:**
        ```bash
        uvicorn api.main:app --reload --port 8000
        ```
    *   **Start the Frontend:**
        ```bash
        cd frontend
        npm run dev
        ```
    *   **Start the Main Orchestrator:**
        ```bash
        python -m core.orchestrator.continuous_director
        ```
    *   **All-in-One Launcher:** The `start_22_myagent.py` script is also available to launch and manage all components.
        ```bash
        python start_22_myagent.py
        ```

### Running Tests

The project uses `pytest`. The primary way to run tests is through the provided test runner script.

*   **Run Quick Test Suite (Unit, Integration, System):**
    ```bash
    python tests/test_runner.py --quick
    ```
*   **Run All Test Categories:**
    ```bash
    python tests/test_runner.py
    ```
*   **Run a Specific Category:**
    ```bash
    python tests/test_runner.py --category unit
    ```

## 🛠️ Development Conventions

*   **Configuration:** All configuration is managed centrally in `config/settings.py` and loaded from a `.env` file. This provides a single source of truth for environment-specific variables.
*   **Code Style:** The codebase uses Python type hints extensively. The presence of `black` and `pylint` in `requirements.txt` indicates a convention for well-formatted and linted code.
*   **Persistence:** A core principle. All agent states, decisions, and learnings are designed to be persisted, as detailed in `CLAUDE.md`. This ensures the system can be stopped and restarted without losing context.
*   **Testing:** Tests are organized by category (`unit`, `integration`, `system`, etc.) within the `tests/` directory. The `tests/test_runner.py` script is the standard entry point for executing test suites.
*   **API-Driven:** The system is designed to be controlled and monitored via the FastAPI backend, with a clear separation between the core logic and the API layer.
*   **Documentation:** There is a strong emphasis on documentation, both in code (docstrings) and in the project root (numerous `.md` files), which should be maintained.
