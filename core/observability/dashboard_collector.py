"""
Dashboard Collector - Unified observability dashboard
Aggregates and processes metrics from all system components for visualization.
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict, deque
import statistics

from .telemetry_engine import TelemetryEngine, LogLevel, get_telemetry
from .system_monitor import SystemMonitor, ResourceMetrics
from .agent_tracer import AgentTracer, AgentMetrics

@dataclass
class DashboardMetrics:
    """Consolidated metrics for dashboard display"""
    timestamp: datetime

    # System health
    system_cpu_percent: float
    system_memory_percent: float
    system_disk_percent: float
    system_load_average: List[float]
    system_healthy: bool

    # Agent performance
    active_agents: int
    total_tasks_completed: int
    avg_task_duration_ms: float
    tasks_per_minute: float
    agent_success_rate: float

    # Quality metrics
    total_alerts: int
    critical_alerts: int
    warning_alerts: int
    error_rate_per_minute: float

    # Performance trends
    cpu_trend: List[float]
    memory_trend: List[float]
    task_rate_trend: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'system_cpu_percent': self.system_cpu_percent,
            'system_memory_percent': self.system_memory_percent,
            'system_disk_percent': self.system_disk_percent,
            'system_load_average': self.system_load_average,
            'system_healthy': self.system_healthy,
            'active_agents': self.active_agents,
            'total_tasks_completed': self.total_tasks_completed,
            'avg_task_duration_ms': self.avg_task_duration_ms,
            'tasks_per_minute': self.tasks_per_minute,
            'agent_success_rate': self.agent_success_rate,
            'total_alerts': self.total_alerts,
            'critical_alerts': self.critical_alerts,
            'warning_alerts': self.warning_alerts,
            'error_rate_per_minute': self.error_rate_per_minute,
            'cpu_trend': self.cpu_trend,
            'memory_trend': self.memory_trend,
            'task_rate_trend': self.task_rate_trend
        }

@dataclass
class SystemDashboard:
    """Complete system dashboard view"""
    last_updated: datetime
    overview: DashboardMetrics

    # Detailed breakdowns
    agent_details: List[Dict[str, Any]]
    recent_alerts: List[Dict[str, Any]]
    performance_insights: Dict[str, Any]
    resource_usage: Dict[str, Any]

    # Historical data
    metrics_history: List[DashboardMetrics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'last_updated': self.last_updated.isoformat(),
            'overview': self.overview.to_dict(),
            'agent_details': self.agent_details,
            'recent_alerts': self.recent_alerts,
            'performance_insights': self.performance_insights,
            'resource_usage': self.resource_usage,
            'metrics_history': [m.to_dict() for m in self.metrics_history]
        }

class DashboardCollector:
    """
    Centralized dashboard data collector and processor.
    Aggregates data from all observability components for unified visualization.
    """

    def __init__(self, telemetry: TelemetryEngine = None):
        self.telemetry = telemetry or get_telemetry()
        self.telemetry.register_component('dashboard_collector')

        # Component integrations
        self.system_monitor = None
        self.agent_tracers = {}  # agent_name -> AgentTracer

        # Configuration
        self.collection_interval = 10.0  # seconds
        self.history_retention = 288  # 24 hours at 10s intervals

        # State
        self._running = False
        self._collection_thread = None
        self._lock = threading.Lock()

        # Data storage
        self.metrics_history = deque(maxlen=self.history_retention)
        self.performance_cache = {}

        # Performance aggregators
        self.task_rate_calculator = TaskRateCalculator()
        self.alert_aggregator = AlertAggregator()
        self.trend_analyzer = TrendAnalyzer()

    async def start(self):
        """Start dashboard data collection"""
        self.telemetry.log_info("Starting dashboard collector", 'dashboard_collector')

        self._running = True
        self._collection_thread = threading.Thread(target=self._collection_loop)
        self._collection_thread.daemon = True
        self._collection_thread.start()

        self.telemetry.log_info("Dashboard collector started", 'dashboard_collector')

    async def stop(self):
        """Stop dashboard data collection"""
        self.telemetry.log_info("Stopping dashboard collector", 'dashboard_collector')

        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=10)

        self.telemetry.log_info("Dashboard collector stopped", 'dashboard_collector')

    def register_system_monitor(self, system_monitor: SystemMonitor):
        """Register system monitor for data collection"""
        self.system_monitor = system_monitor
        self.telemetry.log_info("System monitor registered", 'dashboard_collector')

    def register_agent_tracer(self, agent_name: str, tracer: AgentTracer):
        """Register agent tracer for data collection"""
        with self._lock:
            self.agent_tracers[agent_name] = tracer

        self.telemetry.log_info(
            f"Agent tracer registered: {agent_name}",
            'dashboard_collector',
            {'agent_name': agent_name}
        )

    def _collection_loop(self):
        """Main dashboard data collection loop"""
        while self._running:
            try:
                metrics = self._collect_dashboard_metrics()
                if metrics:
                    self._process_metrics(metrics)

                time.sleep(self.collection_interval)

            except Exception as e:
                self.telemetry.log_error(
                    f"Error in dashboard collection loop: {e}",
                    'dashboard_collector',
                    {'error_type': type(e).__name__}
                )
                time.sleep(self.collection_interval)

    def _collect_dashboard_metrics(self) -> Optional[DashboardMetrics]:
        """Collect current dashboard metrics from all sources"""
        try:
            # System metrics
            system_data = self._get_system_metrics()
            if not system_data:
                return None

            # Agent metrics
            agent_data = self._get_agent_metrics()

            # Alert metrics
            alert_data = self._get_alert_metrics()

            # Trend data
            trend_data = self._get_trend_data()

            # Aggregate into dashboard metrics
            metrics = DashboardMetrics(
                timestamp=datetime.now(timezone.utc),

                # System health
                system_cpu_percent=system_data['cpu_percent'],
                system_memory_percent=system_data['memory_percent'],
                system_disk_percent=system_data['disk_percent'],
                system_load_average=system_data['load_average'],
                system_healthy=system_data['healthy'],

                # Agent performance
                active_agents=agent_data['active_count'],
                total_tasks_completed=agent_data['total_completed'],
                avg_task_duration_ms=agent_data['avg_duration'],
                tasks_per_minute=agent_data['tasks_per_minute'],
                agent_success_rate=agent_data['success_rate'],

                # Quality metrics
                total_alerts=alert_data['total'],
                critical_alerts=alert_data['critical'],
                warning_alerts=alert_data['warning'],
                error_rate_per_minute=alert_data['error_rate'],

                # Trends
                cpu_trend=trend_data['cpu_trend'],
                memory_trend=trend_data['memory_trend'],
                task_rate_trend=trend_data['task_rate_trend']
            )

            return metrics

        except Exception as e:
            self.telemetry.log_error(
                f"Error collecting dashboard metrics: {e}",
                'dashboard_collector'
            )
            return None

    def _get_system_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current system metrics"""
        if not self.system_monitor:
            return None

        current = self.system_monitor.get_current_metrics()
        if not current:
            return None

        return {
            'cpu_percent': current.cpu_percent,
            'memory_percent': current.memory_percent,
            'disk_percent': current.disk_percent,
            'load_average': current.load_average,
            'healthy': self.system_monitor.is_system_healthy()
        }

    def _get_agent_metrics(self) -> Dict[str, Any]:
        """Get aggregated agent metrics"""
        with self._lock:
            tracers = list(self.agent_tracers.values())

        if not tracers:
            return {
                'active_count': 0,
                'total_completed': 0,
                'avg_duration': 0.0,
                'tasks_per_minute': 0.0,
                'success_rate': 0.0
            }

        # Aggregate metrics from all agents
        active_count = 0
        total_completed = 0
        total_durations = []
        total_success = 0
        total_attempts = 0

        for tracer in tracers:
            agent_summary = tracer.get_performance_summary()
            if agent_summary:
                if agent_summary['current_status'] == 'active':
                    active_count += 1

                total_completed += agent_summary['total_tasks_completed']

                if agent_summary['avg_task_duration_ms'] > 0:
                    total_durations.append(agent_summary['avg_task_duration_ms'])

                total_success += agent_summary['successful_tasks']
                total_attempts += agent_summary['total_tasks_attempted']

        # Calculate aggregated metrics
        avg_duration = statistics.mean(total_durations) if total_durations else 0.0
        success_rate = (total_success / total_attempts * 100) if total_attempts > 0 else 0.0

        # Calculate tasks per minute using rate calculator
        tasks_per_minute = self.task_rate_calculator.calculate_rate(total_completed)

        return {
            'active_count': active_count,
            'total_completed': total_completed,
            'avg_duration': avg_duration,
            'tasks_per_minute': tasks_per_minute,
            'success_rate': success_rate
        }

    def _get_alert_metrics(self) -> Dict[str, Any]:
        """Get aggregated alert metrics"""
        if not self.system_monitor:
            return {'total': 0, 'critical': 0, 'warning': 0, 'error_rate': 0.0}

        alerts = self.system_monitor.get_alerts(minutes=60)

        total = len(alerts)
        critical = sum(1 for alert in alerts if alert.get('severity') == 'critical')
        warning = sum(1 for alert in alerts if alert.get('severity') == 'warning')

        # Calculate error rate
        error_rate = self.alert_aggregator.calculate_error_rate(alerts)

        return {
            'total': total,
            'critical': critical,
            'warning': warning,
            'error_rate': error_rate
        }

    def _get_trend_data(self) -> Dict[str, Any]:
        """Get performance trend data"""
        with self._lock:
            history = list(self.metrics_history)

        if len(history) < 2:
            return {
                'cpu_trend': [],
                'memory_trend': [],
                'task_rate_trend': []
            }

        # Extract trends for last 12 data points (2 minutes at 10s intervals)
        recent_history = history[-12:]

        cpu_trend = [m.system_cpu_percent for m in recent_history]
        memory_trend = [m.system_memory_percent for m in recent_history]
        task_rate_trend = [m.tasks_per_minute for m in recent_history]

        return {
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'task_rate_trend': task_rate_trend
        }

    def _process_metrics(self, metrics: DashboardMetrics):
        """Process and store collected metrics"""
        with self._lock:
            self.metrics_history.append(metrics)

        # Update performance cache
        self.performance_cache['last_updated'] = metrics.timestamp
        self.performance_cache['current_metrics'] = metrics

        # Emit dashboard metrics to telemetry
        self.telemetry.set_gauge(
            'dashboard_system_health_score',
            self._calculate_health_score(metrics),
            component='dashboard_collector'
        )

        self.telemetry.set_gauge(
            'dashboard_agent_performance_score',
            self._calculate_performance_score(metrics),
            component='dashboard_collector'
        )

    def _calculate_health_score(self, metrics: DashboardMetrics) -> float:
        """Calculate overall system health score (0-100)"""
        factors = []

        # System resource health (0-100)
        cpu_health = max(0, 100 - metrics.system_cpu_percent)
        memory_health = max(0, 100 - metrics.system_memory_percent)
        disk_health = max(0, 100 - metrics.system_disk_percent)

        factors.extend([cpu_health, memory_health, disk_health])

        # Alert penalty
        alert_penalty = min(50, metrics.critical_alerts * 20 + metrics.warning_alerts * 5)
        alert_health = max(0, 100 - alert_penalty)
        factors.append(alert_health)

        # Agent success rate
        factors.append(metrics.agent_success_rate)

        return statistics.mean(factors)

    def _calculate_performance_score(self, metrics: DashboardMetrics) -> float:
        """Calculate overall system performance score (0-100)"""
        # Base on task throughput and success rate
        throughput_score = min(100, (metrics.tasks_per_minute / 10) * 100)  # 10 tasks/min = 100%
        success_score = metrics.agent_success_rate

        # Average task duration penalty (higher duration = lower score)
        duration_score = max(0, 100 - (metrics.avg_task_duration_ms / 1000))  # 1s = 0 points

        return statistics.mean([throughput_score, success_score, duration_score])

    def get_dashboard(self) -> SystemDashboard:
        """Get complete dashboard view"""
        with self._lock:
            current_metrics = self.performance_cache.get('current_metrics')
            history = list(self.metrics_history)

        if not current_metrics:
            # Create default dashboard
            current_metrics = DashboardMetrics(
                timestamp=datetime.now(timezone.utc),
                system_cpu_percent=0.0,
                system_memory_percent=0.0,
                system_disk_percent=0.0,
                system_load_average=[0.0, 0.0, 0.0],
                system_healthy=True,
                active_agents=0,
                total_tasks_completed=0,
                avg_task_duration_ms=0.0,
                tasks_per_minute=0.0,
                agent_success_rate=0.0,
                total_alerts=0,
                critical_alerts=0,
                warning_alerts=0,
                error_rate_per_minute=0.0,
                cpu_trend=[],
                memory_trend=[],
                task_rate_trend=[]
            )

        # Get detailed breakdowns
        agent_details = self._get_agent_details()
        recent_alerts = self._get_recent_alerts()
        performance_insights = self._get_performance_insights(current_metrics)
        resource_usage = self._get_resource_usage(current_metrics)

        return SystemDashboard(
            last_updated=datetime.now(timezone.utc),
            overview=current_metrics,
            agent_details=agent_details,
            recent_alerts=recent_alerts,
            performance_insights=performance_insights,
            resource_usage=resource_usage,
            metrics_history=history[-50:]  # Last 50 data points for charts
        )

    def _get_agent_details(self) -> List[Dict[str, Any]]:
        """Get detailed agent performance information"""
        with self._lock:
            tracers = dict(self.agent_tracers)

        details = []
        for agent_name, tracer in tracers.items():
            summary = tracer.get_performance_summary()
            if summary:
                details.append({
                    'name': agent_name,
                    'status': summary['current_status'],
                    'tasks_completed': summary['total_tasks_completed'],
                    'success_rate': summary['success_rate'],
                    'avg_duration_ms': summary['avg_task_duration_ms'],
                    'last_active': summary['last_activity'].isoformat() if summary['last_activity'] else None,
                    'health_score': self._calculate_agent_health(summary)
                })

        return sorted(details, key=lambda x: x['health_score'], reverse=True)

    def _calculate_agent_health(self, summary: Dict[str, Any]) -> float:
        """Calculate individual agent health score"""
        success_weight = 0.4
        activity_weight = 0.3
        performance_weight = 0.3

        success_score = summary['success_rate']

        # Activity score based on recent activity
        if summary['last_activity']:
            minutes_since = (datetime.now(timezone.utc) - summary['last_activity']).total_seconds() / 60
            activity_score = max(0, 100 - minutes_since)  # 100% if active, decreases over time
        else:
            activity_score = 0

        # Performance score based on task duration
        avg_duration = summary['avg_task_duration_ms']
        performance_score = max(0, 100 - (avg_duration / 1000))  # 1s = 0 points

        return (success_weight * success_score +
                activity_weight * activity_score +
                performance_weight * performance_score)

    def _get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Get recent system alerts"""
        if not self.system_monitor:
            return []

        alerts = self.system_monitor.get_alerts(minutes=30)
        return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)[:10]

    def _get_performance_insights(self, metrics: DashboardMetrics) -> Dict[str, Any]:
        """Generate performance insights and recommendations"""
        insights = {
            'health_score': self._calculate_health_score(metrics),
            'performance_score': self._calculate_performance_score(metrics),
            'recommendations': [],
            'trends': {}
        }

        # Generate recommendations
        if metrics.system_cpu_percent > 80:
            insights['recommendations'].append({
                'type': 'warning',
                'message': 'High CPU usage detected. Consider optimizing agent workloads.'
            })

        if metrics.system_memory_percent > 85:
            insights['recommendations'].append({
                'type': 'warning',
                'message': 'High memory usage detected. Monitor for memory leaks.'
            })

        if metrics.agent_success_rate < 90:
            insights['recommendations'].append({
                'type': 'error',
                'message': 'Low agent success rate. Review recent failures.'
            })

        if metrics.tasks_per_minute < 1:
            insights['recommendations'].append({
                'type': 'info',
                'message': 'Low task throughput. System may be idle or blocked.'
            })

        # Trend analysis
        if len(metrics.cpu_trend) >= 3:
            cpu_increasing = all(
                metrics.cpu_trend[i] <= metrics.cpu_trend[i+1]
                for i in range(len(metrics.cpu_trend)-1)
            )
            if cpu_increasing:
                insights['trends']['cpu'] = 'increasing'
            else:
                insights['trends']['cpu'] = 'stable'

        return insights

    def _get_resource_usage(self, metrics: DashboardMetrics) -> Dict[str, Any]:
        """Get detailed resource usage information"""
        return {
            'cpu': {
                'current': metrics.system_cpu_percent,
                'threshold_warning': 80,
                'threshold_critical': 95,
                'status': 'healthy' if metrics.system_cpu_percent < 80 else 'warning'
            },
            'memory': {
                'current': metrics.system_memory_percent,
                'threshold_warning': 85,
                'threshold_critical': 95,
                'status': 'healthy' if metrics.system_memory_percent < 85 else 'warning'
            },
            'disk': {
                'current': metrics.system_disk_percent,
                'threshold_warning': 90,
                'threshold_critical': 98,
                'status': 'healthy' if metrics.system_disk_percent < 90 else 'warning'
            },
            'load_average': {
                '1min': metrics.system_load_average[0] if len(metrics.system_load_average) > 0 else 0,
                '5min': metrics.system_load_average[1] if len(metrics.system_load_average) > 1 else 0,
                '15min': metrics.system_load_average[2] if len(metrics.system_load_average) > 2 else 0
            }
        }

# Utility classes for specialized calculations

class TaskRateCalculator:
    """Calculates task completion rates"""

    def __init__(self, window_minutes: int = 5):
        self.window_minutes = window_minutes
        self.task_history = deque(maxlen=window_minutes * 6)  # 10s intervals

    def calculate_rate(self, total_completed: int) -> float:
        """Calculate tasks per minute based on recent completion rate"""
        now = time.time()

        # Add current data point
        self.task_history.append({
            'timestamp': now,
            'total_completed': total_completed
        })

        if len(self.task_history) < 2:
            return 0.0

        # Calculate rate over window
        oldest = self.task_history[0]
        newest = self.task_history[-1]

        time_diff_minutes = (newest['timestamp'] - oldest['timestamp']) / 60
        if time_diff_minutes <= 0:
            return 0.0

        tasks_diff = newest['total_completed'] - oldest['total_completed']
        return tasks_diff / time_diff_minutes

class AlertAggregator:
    """Aggregates and analyzes alert patterns"""

    def calculate_error_rate(self, alerts: List[Dict[str, Any]]) -> float:
        """Calculate error alerts per minute"""
        if not alerts:
            return 0.0

        # Count error-level alerts in last hour
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)

        error_alerts = [
            alert for alert in alerts
            if alert.get('severity') in ['error', 'critical'] and
            datetime.fromisoformat(alert.get('timestamp', '')).replace(tzinfo=timezone.utc) > one_hour_ago
        ]

        return len(error_alerts) / 60  # errors per minute

class TrendAnalyzer:
    """Analyzes performance trends"""

    def analyze_trend(self, values: List[float]) -> str:
        """Analyze trend direction: increasing, decreasing, or stable"""
        if len(values) < 3:
            return 'unknown'

        # Simple linear trend analysis
        increases = sum(1 for i in range(len(values)-1) if values[i+1] > values[i])
        decreases = sum(1 for i in range(len(values)-1) if values[i+1] < values[i])

        if increases > decreases + 1:
            return 'increasing'
        elif decreases > increases + 1:
            return 'decreasing'
        else:
            return 'stable'