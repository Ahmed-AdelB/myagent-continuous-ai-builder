"""
Advanced System Observability and Monitoring
Implements enterprise-grade monitoring with metrics, traces, and alerts
Based on 2024 best practices for multi-agent AI systems
"""

import asyncio
import time
import logging
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
import aioredis
from sqlalchemy import create_engine, text
import httpx
from contextlib import asynccontextmanager

# Metrics collection
REGISTRY = CollectorRegistry()

# System metrics
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage percentage', registry=REGISTRY)
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_percent', 'Memory usage percentage', registry=REGISTRY)
SYSTEM_DISK_USAGE = Gauge('system_disk_usage_percent', 'Disk usage percentage', registry=REGISTRY)

# Agent metrics
AGENT_TASK_DURATION = Histogram('agent_task_duration_seconds', 'Agent task execution time', 
                                ['agent_name', 'task_type'], registry=REGISTRY)
AGENT_TASK_COUNTER = Counter('agent_task_total', 'Total agent tasks executed',
                            ['agent_name', 'task_type', 'status'], registry=REGISTRY)
AGENT_ERROR_COUNTER = Counter('agent_error_total', 'Total agent errors',
                             ['agent_name', 'error_type'], registry=REGISTRY)

# API metrics  
API_REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration',
                                 ['method', 'endpoint'], registry=REGISTRY)
API_REQUEST_COUNTER = Counter('api_request_total', 'Total API requests',
                             ['method', 'endpoint', 'status_code'], registry=REGISTRY)

# Database metrics
DB_CONNECTION_POOL = Gauge('db_connection_pool_size', 'Database connection pool size',
                          ['database'], registry=REGISTRY)
DB_QUERY_DURATION = Histogram('db_query_duration_seconds', 'Database query duration',
                              ['database', 'query_type'], registry=REGISTRY)

# LLM metrics
LLM_TOKEN_USAGE = Counter('llm_token_usage_total', 'Total LLM tokens used',
                         ['provider', 'model', 'type'], registry=REGISTRY)
LLM_COST_USAGE = Counter('llm_cost_usd_total', 'Total LLM cost in USD',
                        ['provider', 'model'], registry=REGISTRY)
LLM_REQUEST_DURATION = Histogram('llm_request_duration_seconds', 'LLM request duration',
                                ['provider', 'model'], registry=REGISTRY)

@dataclass
class SystemHealth:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_agents: int
    db_connections: Dict[str, int]
    api_response_time: float
    error_count_last_hour: int
    status: str  # healthy, degraded, critical

@dataclass
class AgentMetrics:
    agent_name: str
    tasks_completed: int
    tasks_failed: int
    avg_task_duration: float
    last_activity: datetime
    error_rate: float
    status: str

@dataclass
class Alert:
    id: str
    severity: str  # info, warning, critical
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class SystemObservability:
    """Advanced system monitoring and observability"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 postgres_url: str = "postgresql://myagent_user:myagent_pass@localhost:5432/myagent_db"):
        self.redis_url = redis_url
        self.postgres_url = postgres_url
        self.alerts: List[Alert] = []
        self.metrics_history: List[SystemHealth] = []
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.logger = logging.getLogger(__name__)
        self.thresholds = {
            'cpu_critical': 90.0,
            'cpu_warning': 75.0,
            'memory_critical': 95.0,
            'memory_warning': 80.0,
            'disk_critical': 95.0,
            'disk_warning': 85.0,
            'error_rate_critical': 5.0,
            'error_rate_warning': 2.0,
            'response_time_critical': 5.0,
            'response_time_warning': 2.0
        }
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get current system health summary"""
        try:
            # Get current metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            status = self._calculate_system_status(cpu_percent, memory.percent, disk_percent)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'status': status,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk_percent,
                'active_agents': len(self.agent_metrics),
                'alerts_count': len([a for a in self.alerts if not a.resolved])
            }
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {'status': 'unknown', 'error': str(e)}
    
    def _calculate_system_status(self, cpu: float, memory: float, disk: float) -> str:
        """Calculate overall system status"""
        if (cpu > self.thresholds['cpu_critical'] or 
            memory > self.thresholds['memory_critical'] or 
            disk > self.thresholds['disk_critical']):
            return "critical"
        elif (cpu > self.thresholds['cpu_warning'] or 
              memory > self.thresholds['memory_warning'] or 
              disk > self.thresholds['disk_warning']):
            return "degraded"
        else:
            return "healthy"

# Global instance for system monitoring
_system_monitor = None

def get_system_monitor() -> SystemObservability:
    """Get or create global system monitor instance"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemObservability()
    return _system_monitor
