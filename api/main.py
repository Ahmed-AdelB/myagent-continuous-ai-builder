"""
FastAPI Backend - Main API server for the Continuous AI Builder
"""

from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import json
from loguru import logger
from pydantic import BaseModel, Field
import uuid
import asyncpg
import os

# Import core components
from core.orchestrator.continuous_director import ContinuousDirector, ProjectState
from core.memory.project_ledger import ProjectLedger
from core.memory.error_knowledge_graph import ErrorKnowledgeGraph
from core.memory.vector_memory import VectorMemory
from core.agents.coder_agent import CoderAgent
from core.agents.tester_agent import TesterAgent
from core.agents.debugger_agent import DebuggerAgent
from config.settings import settings
from config.database import db_manager, init_database


# Pydantic models for API
class ProjectCreate(BaseModel):
    """Model for creating a new project"""
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    requirements: List[str] = Field(default_factory=list)
    target_metrics: Dict[str, float] = Field(default_factory=dict)
    max_iterations: int = Field(default=1000)


class TaskCreate(BaseModel):
    """Model for creating a new task"""
    type: str = Field(..., description="Task type")
    description: str = Field(..., description="Task description")
    priority: int = Field(default=5, ge=1, le=10)
    agent: Optional[str] = Field(None, description="Target agent")
    data: Dict = Field(default_factory=dict)


class ProjectStatus(BaseModel):
    """Model for project status response"""
    id: str
    name: str
    state: str
    iteration: int
    metrics: Dict[str, float]
    milestones: Dict
    agents: List[Dict]
    estimated_completion: Optional[datetime]


class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, project_id: str):
        await websocket.accept()
        if project_id not in self.active_connections:
            self.active_connections[project_id] = []
        self.active_connections[project_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, project_id: str):
        if project_id in self.active_connections:
            self.active_connections[project_id].remove(websocket)
            if not self.active_connections[project_id]:
                del self.active_connections[project_id]
    
    async def broadcast(self, message: dict, project_id: str):
        if project_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[project_id]:
                try:
                    await connection.send_json(message)
                except (RuntimeError, ConnectionError, Exception) as e:
                    logger.warning(f"WebSocket send failed, marking for disconnect: {e}")
                    disconnected.append(connection)

            # Clean up disconnected clients
            for conn in disconnected:
                self.disconnect(conn, project_id)


# Global instances
orchestrators: Dict[str, ContinuousDirector] = {}
ws_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Continuous AI Builder API")

    # Initialize database
    try:
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Continuous AI Builder API")

    # Stop all orchestrators
    for orchestrator in orchestrators.values():
        await orchestrator.stop()

    # Close database pool
    await db_manager.disconnect()
    logger.info("Database connection closed")


# Create FastAPI app
app = FastAPI(
    title="Continuous AI Builder API",
    description="API for the never-stopping AI development system",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_projects": len(orchestrators)
    }


# Project management endpoints
@app.post("/projects", response_model=ProjectStatus)
async def create_project(
    project: ProjectCreate,
    background_tasks: BackgroundTasks
):
    """Create a new project and start continuous development"""
    project_id = str(uuid.uuid4())

    # Prepare project spec
    project_spec = {
        "description": project.description,
        "requirements": project.requirements,
        "target_metrics": project.target_metrics,
        "max_iterations": project.max_iterations
    }

    # Insert into database
    await db_manager.execute(
        """INSERT INTO projects (name, spec, state, metrics)
        VALUES ($1, $2, $3, $4)""",
        project.name,
        json.dumps(project_spec),
        "initializing",
        json.dumps({})
    )

    # Create orchestrator
    orchestrator = ContinuousDirector(
        project_name=project.name,
        project_spec=project_spec
    )

    orchestrators[project_id] = orchestrator
    
    # Start orchestrator in background
    background_tasks.add_task(orchestrator.start)
    
    logger.info(f"Created project {project.name} with ID {project_id}")
    
    return ProjectStatus(
        id=project_id,
        name=project.name,
        state=orchestrator.state.value,
        iteration=orchestrator.iteration_count,
        metrics={},
        milestones={},
        agents=[],
        estimated_completion=None
    )


@app.get("/projects", response_model=List[ProjectStatus])
async def list_projects():
    """List all active projects"""
    projects = []
    
    for project_id, orchestrator in orchestrators.items():
        projects.append(ProjectStatus(
            id=project_id,
            name=orchestrator.project_name,
            state=orchestrator.state.value,
            iteration=orchestrator.iteration_count,
            metrics=orchestrator.metrics.to_dict() if hasattr(orchestrator, 'metrics') else {},
            milestones=orchestrator.milestone_tracker.get_progress_summary() if hasattr(orchestrator, 'milestone_tracker') else {},
            agents=[],
            estimated_completion=None
        ))
    
    return projects


@app.get("/projects/{project_id}", response_model=ProjectStatus)
async def get_project(project_id: str):
    """Get project details"""
    if project_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Project not found")
    
    orchestrator = orchestrators[project_id]
    
    # Get agent statuses safely
    agent_statuses = []
    if hasattr(orchestrator, 'agents'):
        for agent_name, agent in orchestrator.agents.items():
            agent_statuses.append({
                "name": agent_name,
                "role": getattr(agent, 'role', agent_name),
                "status": getattr(agent, 'status', 'active')
            })

    # Get milestones safely
    milestones = {}
    if hasattr(orchestrator, 'milestone_tracker'):
        try:
            milestones = orchestrator.milestone_tracker.get_progress_summary()
        except (AttributeError, KeyError, Exception) as e:
            logger.warning(f"Failed to get milestones: {e}")
            milestones = {}

    # Get estimated completion safely
    estimated_completion = None
    if hasattr(orchestrator, 'progress_analyzer'):
        try:
            status = orchestrator.progress_analyzer.get_current_status()
            estimated_completion = status.get('estimated_completion')
        except (AttributeError, KeyError, Exception) as e:
            logger.warning(f"Failed to get estimated completion: {e}")
            pass

    return ProjectStatus(
        id=project_id,
        name=orchestrator.project_name,
        state=orchestrator.state.value,
        iteration=orchestrator.iteration_count,
        metrics=orchestrator.metrics.to_dict() if hasattr(orchestrator, 'metrics') else {},
        milestones=milestones,
        agents=agent_statuses,
        estimated_completion=estimated_completion
    )


@app.post("/projects/{project_id}/pause")
async def pause_project(project_id: str):
    """Pause project execution"""
    if project_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Project not found")
    
    orchestrator = orchestrators[project_id]
    await orchestrator.pause()
    
    return {"status": "paused", "project_id": project_id}


@app.post("/projects/{project_id}/resume")
async def resume_project(project_id: str, background_tasks: BackgroundTasks):
    """Resume project execution"""
    if project_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Project not found")
    
    orchestrator = orchestrators[project_id]
    background_tasks.add_task(orchestrator.resume)
    
    return {"status": "resumed", "project_id": project_id}


@app.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Stop and delete a project"""
    if project_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Project not found")
    
    orchestrator = orchestrators[project_id]
    await orchestrator.stop()
    del orchestrators[project_id]
    
    return {"status": "deleted", "project_id": project_id}


# Task management endpoints
@app.post("/projects/{project_id}/tasks")
async def create_task(project_id: str, task: TaskCreate):
    """Create a new task for a project"""
    if project_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Project not found")

    orchestrator = orchestrators[project_id]

    # Add task to orchestrator task queue
    task_id = str(uuid.uuid4())

    from core.orchestrator.continuous_director import DevelopmentTask, TaskPriority

    dev_task = DevelopmentTask(
        id=task_id,
        type=task.type,
        description=task.description,
        priority=TaskPriority(task.priority) if task.priority <= 5 else TaskPriority.LOW,
        assigned_agent=task.agent,
        data=task.data
    )

    orchestrator.task_queue.append(dev_task)

    return {"task_id": task_id, "status": "queued"}


# Agent endpoints
@app.get("/projects/{project_id}/agents")
async def get_agents(project_id: str):
    """Get all agents for a project"""
    if project_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Project not found")

    orchestrator = orchestrators[project_id]

    agent_statuses = []
    if hasattr(orchestrator, 'agents'):
        for agent_name, agent in orchestrator.agents.items():
            agent_statuses.append({
                "name": agent_name,
                "role": getattr(agent, 'role', agent_name),
                "status": getattr(agent, 'status', 'active'),
                "capabilities": getattr(agent, 'capabilities', [])
            })

    return agent_statuses


@app.get("/projects/{project_id}/agents/{agent_id}")
async def get_agent(project_id: str, agent_id: str):
    """Get specific agent details"""
    if project_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Project not found")

    orchestrator = orchestrators[project_id]

    if agent_id not in orchestrator.agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = orchestrator.agents[agent_id]
    return {
        "name": agent_id,
        "role": getattr(agent, 'role', agent_id),
        "status": getattr(agent, 'status', 'active'),
        "capabilities": getattr(agent, 'capabilities', [])
    }


# Metrics endpoints
@app.get("/projects/{project_id}/metrics")
async def get_metrics(project_id: str):
    """Get project metrics"""
    if project_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Project not found")
    
    orchestrator = orchestrators[project_id]
    
    return {
        "quality_metrics": orchestrator.metrics.to_dict(),
        "progress": orchestrator.progress_analyzer.get_current_status(),
        "trends": orchestrator.progress_analyzer.get_trend_analysis(),
        "bottlenecks": orchestrator.progress_analyzer.get_bottlenecks()
    }


@app.get("/projects/{project_id}/iterations")
async def get_iterations(project_id: str, limit: int = 10):
    """Get iteration history"""
    if project_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Project not found")
    
    orchestrator = orchestrators[project_id]
    
    # Get recent iterations from project ledger
    iterations = []
    for i in range(
        max(1, orchestrator.iteration_count - limit + 1),
        orchestrator.iteration_count + 1
    ):
        summary = orchestrator.project_ledger.get_iteration_summary(i)
        iterations.append(summary)
    
    return iterations


# Memory endpoints
@app.get("/projects/{project_id}/memory/errors")
async def get_error_knowledge(project_id: str):
    """Get error knowledge graph data"""
    if project_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Project not found")

    orchestrator = orchestrators[project_id]

    if not hasattr(orchestrator, 'error_graph'):
        return {"nodes": [], "edges": []}

    graph_data = orchestrator.error_graph.export_graph()

    return graph_data


@app.post("/projects/{project_id}/memory/search")
async def search_memory(project_id: str, query: str):
    """Search project memory"""
    if project_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Project not found")
    
    orchestrator = orchestrators[project_id]
    
    # Search vector memory
    results = orchestrator.vector_memory.search_memories(
        query=query,
        top_k=10
    )
    
    return [
        {
            "id": r.id,
            "content": r.content,
            "relevance": r.relevance_score,
            "metadata": r.metadata
        }
        for r in results
    ]


# WebSocket endpoint for real-time updates (with project ID)
@app.websocket("/ws/{project_id}")
async def websocket_endpoint_with_project(websocket: WebSocket, project_id: str):
    """WebSocket connection for real-time project updates"""
    if project_id not in orchestrators:
        await websocket.close(code=4004, reason="Project not found")
        return
    
    await ws_manager.connect(websocket, project_id)
    
    try:
        orchestrator = orchestrators[project_id]
        
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "data": {
                "state": orchestrator.state.value,
                "iteration": orchestrator.iteration_count,
                "metrics": orchestrator.metrics.to_dict()
            }
        })
        
        # Keep connection alive and send updates
        while True:
            # Wait for messages or send periodic updates
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                # Handle incoming messages
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    
            except asyncio.TimeoutError:
                # Send periodic update
                await websocket.send_json({
                    "type": "update",
                    "data": {
                        "iteration": orchestrator.iteration_count,
                        "metrics": orchestrator.metrics.to_dict()
                    }
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        ws_manager.disconnect(websocket, project_id)


# WebSocket endpoint for general connection (without project ID)
@app.websocket("/ws")
async def websocket_endpoint_general(websocket: WebSocket):
    """WebSocket connection for general updates (defaults to first project)"""
    await websocket.accept()

    try:
        # Use first available project or wait for one
        project_id = list(orchestrators.keys())[0] if orchestrators else "default"

        if project_id == "default":
            # No projects yet, just keep connection alive
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=30.0
                    )
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "heartbeat"})
        else:
            # Connect to first project
            await ws_manager.connect(websocket, project_id)
            await websocket_endpoint_with_project(websocket, project_id)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if project_id != "default":
            ws_manager.disconnect(websocket, project_id)


# Code endpoints
@app.get("/projects/{project_id}/code/{file_path:path}")
async def get_code_version(project_id: str, file_path: str, version: Optional[str] = None):
    """Get code version from project ledger"""
    if project_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Project not found")
    
    orchestrator = orchestrators[project_id]
    
    if version:
        code_version = orchestrator.project_ledger.get_version(version)
    else:
        code_version = orchestrator.project_ledger.get_current_version(file_path)
    
    if not code_version:
        raise HTTPException(status_code=404, detail="Code version not found")
    
    return {
        "version_id": code_version.id,
        "file_path": code_version.file_path,
        "content": code_version.content,
        "timestamp": code_version.timestamp,
        "iteration": code_version.iteration,
        "agent": code_version.agent,
        "reason": code_version.reason
    }


# Checkpoint endpoints
@app.post("/projects/{project_id}/checkpoint")
async def create_checkpoint(project_id: str, description: str = ""):
    """Create a manual checkpoint"""
    if project_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Project not found")

    orchestrator = orchestrators[project_id]
    await orchestrator._create_checkpoint()

    checkpoint_id = f"checkpoint_{orchestrator.iteration_count}"

    return {"checkpoint_id": checkpoint_id, "status": "created"}


@app.post("/projects/{project_id}/restore")
async def restore_from_checkpoint(project_id: str):
    """Restore from latest checkpoint"""
    if project_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Project not found")

    orchestrator = orchestrators[project_id]
    await orchestrator._load_project_state()

    return {"status": "restored", "iteration": orchestrator.iteration_count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)