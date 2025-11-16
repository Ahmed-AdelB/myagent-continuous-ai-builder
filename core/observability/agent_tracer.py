"""
Agent Tracer - Individual agent monitoring and tracing
Implements detailed agent performance and behavior tracking as recommended by GPT-5.
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import inspect

from .telemetry_engine import TelemetryEngine, MetricType, LogLevel, get_telemetry, Trace

@dataclass
class AgentMetrics:
    """Individual agent performance metrics"""
    agent_name: str
    agent_type: str
    timestamp: datetime
    task_count: int
    success_count: int
    failure_count: int
    average_execution_time: float
    total_execution_time: float
    memory_usage: int
    cpu_time: float
    errors_count: int
    last_activity: datetime

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        total = self.success_count + self.failure_count
        return (self.success_count / total * 100) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_name': self.agent_name,
            'agent_type': self.agent_type,
            'timestamp': self.timestamp.isoformat(),
            'task_count': self.task_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.success_rate,
            'average_execution_time': self.average_execution_time,
            'total_execution_time': self.total_execution_time,
            'memory_usage': self.memory_usage,
            'cpu_time': self.cpu_time,
            'errors_count': self.errors_count,
            'last_activity': self.last_activity.isoformat()
        }

class AgentTaskContext:
    """Context for tracking individual agent task execution"""

    def __init__(self, agent_name: str, task_name: str, tracer: 'AgentTracer'):
        self.agent_name = agent_name
        self.task_name = task_name
        self.tracer = tracer
        self.start_time = None
        self.end_time = None
        self.trace = None
        self.success = False
        self.error = None

    async def __aenter__(self):
        """Enter async context manager"""
        self.start_time = datetime.now(timezone.utc)

        # Start distributed trace
        self.trace = self.tracer.telemetry.start_trace(
            f"{self.agent_name}.{self.task_name}",
            component=self.agent_name
        )
        self.trace.add_tag('agent_name', self.agent_name)
        self.trace.add_tag('task_name', self.task_name)

        # Log task start
        self.tracer.telemetry.log_info(
            f"Task started: {self.task_name}",
            self.agent_name,
            {
                'task_name': self.task_name,
                'trace_id': self.trace.trace_id,
                'span_id': self.trace.span_id
            }
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager"""
        self.end_time = datetime.now(timezone.utc)
        duration = (self.end_time - self.start_time).total_seconds()

        if exc_type is None:
            self.success = True
            status = "success"
            log_level = LogLevel.INFO
            message = f"Task completed: {self.task_name}"
        else:
            self.success = False
            self.error = str(exc_val) if exc_val else str(exc_type)
            status = "error"
            log_level = LogLevel.ERROR
            message = f"Task failed: {self.task_name} - {self.error}"

        # Finish trace
        if self.trace:
            self.trace.add_tag('success', self.success)
            self.trace.add_tag('duration_ms', duration * 1000)
            if self.error:
                self.trace.add_tag('error', self.error)
            self.tracer.telemetry.finish_trace(self.trace, status)

        # Log task completion
        self.tracer.telemetry.log_entry(
            log_level,
            message,
            self.agent_name,
            {
                'task_name': self.task_name,
                'success': self.success,
                'duration_ms': duration * 1000,
                'error': self.error,
                'trace_id': self.trace.trace_id if self.trace else None
            }
        )

        # Update agent metrics
        await self.tracer._record_task_completion(
            self.agent_name, self.task_name, duration, self.success, self.error
        )

        return False  # Don't suppress exceptions

class AgentTracer:
    """
    Agent-specific tracing and monitoring system.
    Tracks individual agent performance, behavior, and errors.
    """

    def __init__(self, telemetry: TelemetryEngine = None):
        self.telemetry = telemetry or get_telemetry()
        self.telemetry.register_component('agent_tracer')

        # Agent tracking
        self.tracked_agents = {}
        self.agent_metrics = {}
        self.task_history = defaultdict(deque)  # agent_name -> task history

        # Performance tracking
        self.execution_times = defaultdict(deque)  # agent_name -> execution times
        self.error_patterns = defaultdict(list)    # agent_name -> error patterns

        # Thread safety
        self._lock = threading.Lock()

        # Configuration
        self.max_history_size = 1000
        self.metric_update_interval = 60  # seconds

    def register_agent(self, agent_name: str, agent_type: str, metadata: Dict[str, Any] = None):
        """Register an agent for monitoring"""
        with self._lock:
            self.tracked_agents[agent_name] = {
                'agent_type': agent_type,
                'metadata': metadata or {},
                'registered_at': datetime.now(timezone.utc),
                'last_seen': datetime.now(timezone.utc)
            }

            # Initialize metrics
            self.agent_metrics[agent_name] = AgentMetrics(
                agent_name=agent_name,
                agent_type=agent_type,
                timestamp=datetime.now(timezone.utc),
                task_count=0,
                success_count=0,
                failure_count=0,
                average_execution_time=0.0,
                total_execution_time=0.0,
                memory_usage=0,
                cpu_time=0.0,
                errors_count=0,
                last_activity=datetime.now(timezone.utc)
            )

        self.telemetry.log_info(
            f"Agent registered for monitoring: {agent_name}",
            'agent_tracer',
            {
                'agent_name': agent_name,
                'agent_type': agent_type,
                'metadata': metadata
            }
        )

        # Emit registration metric
        self.telemetry.increment_counter(
            'agent_registrations_total',
            1,
            {'agent_name': agent_name, 'agent_type': agent_type},
            'agent_tracer'
        )

    def unregister_agent(self, agent_name: str):
        """Unregister an agent from monitoring"""
        with self._lock:
            if agent_name in self.tracked_agents:
                del self.tracked_agents[agent_name]
                del self.agent_metrics[agent_name]
                if agent_name in self.task_history:
                    del self.task_history[agent_name]
                if agent_name in self.execution_times:
                    del self.execution_times[agent_name]
                if agent_name in self.error_patterns:
                    del self.error_patterns[agent_name]

        self.telemetry.log_info(
            f"Agent unregistered from monitoring: {agent_name}",
            'agent_tracer'
        )

    def trace_task(self, agent_name: str, task_name: str) -> AgentTaskContext:
        """Create a context manager for tracing agent task execution"""
        return AgentTaskContext(agent_name, task_name, self)

    def trace_method(self, agent_name: str = None):
        """Decorator for automatic method tracing"""
        def decorator(func):
            method_name = func.__name__
            actual_agent_name = agent_name or getattr(func, '__self__', {}).get('name', 'unknown')

            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    async with self.trace_task(actual_agent_name, method_name):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    # For sync functions, use a simplified tracking approach
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time
                        asyncio.create_task(
                            self._record_task_completion(actual_agent_name, method_name, duration, True, None)
                        )
                        return result
                    except Exception as e:
                        duration = time.time() - start_time
                        asyncio.create_task(
                            self._record_task_completion(actual_agent_name, method_name, duration, False, str(e))
                        )
                        raise
                return sync_wrapper
        return decorator

    async def _record_task_completion(self, agent_name: str, task_name: str,
                                     duration: float, success: bool, error: Optional[str]):
        """Record completion of an agent task"""
        with self._lock:
            # Update task history
            task_record = {
                'task_name': task_name,
                'timestamp': datetime.now(timezone.utc),
                'duration': duration,
                'success': success,
                'error': error
            }

            if len(self.task_history[agent_name]) >= self.max_history_size:
                self.task_history[agent_name].popleft()
            self.task_history[agent_name].append(task_record)

            # Update execution times
            if len(self.execution_times[agent_name]) >= self.max_history_size:
                self.execution_times[agent_name].popleft()
            self.execution_times[agent_name].append(duration)

            # Track error patterns
            if not success and error:
                self.error_patterns[agent_name].append({
                    'error': error,
                    'task_name': task_name,
                    'timestamp': datetime.now(timezone.utc)
                })

            # Update agent metrics
            if agent_name in self.agent_metrics:
                metrics = self.agent_metrics[agent_name]
                metrics.task_count += 1
                metrics.total_execution_time += duration
                metrics.average_execution_time = metrics.total_execution_time / metrics.task_count
                metrics.last_activity = datetime.now(timezone.utc)

                if success:
                    metrics.success_count += 1
                else:
                    metrics.failure_count += 1
                    metrics.errors_count += 1

        # Emit telemetry
        self.telemetry.increment_counter(
            'agent_tasks_total',
            1,
            {'agent_name': agent_name, 'task_name': task_name, 'success': str(success)},
            'agent_tracer'
        )

        self.telemetry.record_histogram(
            'agent_task_duration_ms',
            duration * 1000,
            {'agent_name': agent_name, 'task_name': task_name},
            'ms',
            'agent_tracer'
        )

        if not success:
            self.telemetry.increment_counter(
                'agent_errors_total',
                1,
                {'agent_name': agent_name, 'task_name': task_name, 'error_type': type(error).__name__ if error else 'unknown'},
                'agent_tracer'
            )

    def heartbeat(self, agent_name: str, metadata: Dict[str, Any] = None):
        """Record agent heartbeat/activity"""
        with self._lock:
            if agent_name in self.tracked_agents:
                self.tracked_agents[agent_name]['last_seen'] = datetime.now(timezone.utc)
                if metadata:
                    self.tracked_agents[agent_name]['metadata'].update(metadata)

                if agent_name in self.agent_metrics:
                    self.agent_metrics[agent_name].last_activity = datetime.now(timezone.utc)

        self.telemetry.increment_counter(
            'agent_heartbeats_total',
            1,
            {'agent_name': agent_name},
            'agent_tracer'
        )

    def get_agent_metrics(self, agent_name: str) -> Optional[AgentMetrics]:
        """Get current metrics for specific agent"""
        with self._lock:
            return self.agent_metrics.get(agent_name)

    def get_all_agent_metrics(self) -> Dict[str, AgentMetrics]:
        """Get metrics for all tracked agents"""
        with self._lock:
            return dict(self.agent_metrics)

    def get_agent_task_history(self, agent_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent task history for agent"""
        with self._lock:
            tasks = list(self.task_history.get(agent_name, []))
            return [
                {
                    **task,
                    'timestamp': task['timestamp'].isoformat()
                }
                for task in tasks[-limit:]
            ]

    def get_agent_performance_summary(self, agent_name: str) -> Dict[str, Any]:
        """Get comprehensive performance summary for agent"""
        metrics = self.get_agent_metrics(agent_name)
        if not metrics:
            return {}

        with self._lock:
            # Execution time statistics
            exec_times = list(self.execution_times.get(agent_name, []))

            if exec_times:
                min_time = min(exec_times)
                max_time = max(exec_times)
                avg_time = sum(exec_times) / len(exec_times)
                # Calculate percentiles
                sorted_times = sorted(exec_times)
                p50 = sorted_times[int(0.5 * len(sorted_times))] if sorted_times else 0
                p95 = sorted_times[int(0.95 * len(sorted_times))] if sorted_times else 0
                p99 = sorted_times[int(0.99 * len(sorted_times))] if sorted_times else 0
            else:
                min_time = max_time = avg_time = p50 = p95 = p99 = 0

            # Recent error patterns
            recent_errors = [
                {
                    'error': err['error'],
                    'task_name': err['task_name'],
                    'timestamp': err['timestamp'].isoformat()
                }
                for err in self.error_patterns.get(agent_name, [])[-10:]  # Last 10 errors
            ]

            # Activity status
            last_seen = self.tracked_agents.get(agent_name, {}).get('last_seen')
            minutes_since_activity = 0
            if last_seen:
                minutes_since_activity = (datetime.now(timezone.utc) - last_seen).total_seconds() / 60

            return {
                'metrics': metrics.to_dict(),
                'execution_times': {
                    'min_ms': min_time * 1000,
                    'max_ms': max_time * 1000,
                    'avg_ms': avg_time * 1000,
                    'p50_ms': p50 * 1000,
                    'p95_ms': p95 * 1000,
                    'p99_ms': p99 * 1000,
                    'sample_count': len(exec_times)
                },
                'recent_errors': recent_errors,
                'activity_status': {
                    'minutes_since_activity': minutes_since_activity,
                    'is_active': minutes_since_activity < 5,  # Consider active if seen in last 5 minutes
                    'status': 'active' if minutes_since_activity < 5 else
                             'idle' if minutes_since_activity < 60 else 'inactive'
                }
            }

    def get_system_agent_summary(self) -> Dict[str, Any]:
        """Get system-wide agent monitoring summary"""
        with self._lock:
            total_agents = len(self.tracked_agents)
            active_agents = 0
            total_tasks = 0
            total_successes = 0
            total_failures = 0

            agent_summaries = {}

            for agent_name, metrics in self.agent_metrics.items():
                total_tasks += metrics.task_count
                total_successes += metrics.success_count
                total_failures += metrics.failure_count

                # Check if agent is active (activity in last 5 minutes)
                last_seen = self.tracked_agents.get(agent_name, {}).get('last_seen')
                if last_seen:
                    minutes_inactive = (datetime.now(timezone.utc) - last_seen).total_seconds() / 60
                    if minutes_inactive < 5:
                        active_agents += 1

                agent_summaries[agent_name] = {
                    'agent_type': metrics.agent_type,
                    'task_count': metrics.task_count,
                    'success_rate': metrics.success_rate,
                    'avg_execution_time_ms': metrics.average_execution_time * 1000,
                    'last_activity': metrics.last_activity.isoformat(),
                    'status': 'active' if minutes_inactive < 5 else 'idle' if minutes_inactive < 60 else 'inactive'
                }

            overall_success_rate = (total_successes / total_tasks * 100) if total_tasks > 0 else 0

            return {
                'summary': {
                    'total_agents': total_agents,
                    'active_agents': active_agents,
                    'total_tasks_executed': total_tasks,
                    'overall_success_rate': overall_success_rate,
                    'total_successes': total_successes,
                    'total_failures': total_failures
                },
                'agents': agent_summaries
            }

    def detect_performance_anomalies(self, agent_name: str) -> List[Dict[str, Any]]:
        """Detect performance anomalies for specific agent"""
        anomalies = []

        metrics = self.get_agent_metrics(agent_name)
        if not metrics:
            return anomalies

        # Low success rate anomaly
        if metrics.success_rate < 80 and metrics.task_count > 10:
            anomalies.append({
                'type': 'low_success_rate',
                'severity': 'critical' if metrics.success_rate < 50 else 'warning',
                'value': metrics.success_rate,
                'message': f"Agent {agent_name} has low success rate: {metrics.success_rate:.1f}%"
            })

        # High error rate anomaly
        if metrics.errors_count > 0 and metrics.task_count > 0:
            error_rate = (metrics.errors_count / metrics.task_count) * 100
            if error_rate > 20:
                anomalies.append({
                    'type': 'high_error_rate',
                    'severity': 'critical' if error_rate > 50 else 'warning',
                    'value': error_rate,
                    'message': f"Agent {agent_name} has high error rate: {error_rate:.1f}%"
                })

        # Slow execution time anomaly
        if metrics.average_execution_time > 30:  # 30 seconds
            anomalies.append({
                'type': 'slow_execution',
                'severity': 'warning',
                'value': metrics.average_execution_time,
                'message': f"Agent {agent_name} has slow average execution: {metrics.average_execution_time:.2f}s"
            })

        # Inactive agent anomaly
        minutes_since_activity = (datetime.now(timezone.utc) - metrics.last_activity).total_seconds() / 60
        if minutes_since_activity > 60:  # No activity for 1 hour
            anomalies.append({
                'type': 'agent_inactive',
                'severity': 'warning',
                'value': minutes_since_activity,
                'message': f"Agent {agent_name} has been inactive for {minutes_since_activity:.1f} minutes"
            })

        return anomalies