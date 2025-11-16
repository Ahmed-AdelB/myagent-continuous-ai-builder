"""
System Monitor - Real-time system health monitoring
Implements system-wide resource and performance monitoring as recommended by GPT-5.
"""

import asyncio
import psutil
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading
from collections import deque
import json

from .telemetry_engine import TelemetryEngine, MetricType, LogLevel, get_telemetry

@dataclass
class ResourceMetrics:
    """System resource metrics snapshot"""
    timestamp: datetime
    cpu_percent: float
    cpu_cores: int
    memory_total: int
    memory_available: int
    memory_percent: float
    disk_total: int
    disk_used: int
    disk_free: int
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'cpu_cores': self.cpu_cores,
            'memory_total': self.memory_total,
            'memory_available': self.memory_available,
            'memory_percent': self.memory_percent,
            'disk_total': self.disk_total,
            'disk_used': self.disk_used,
            'disk_free': self.disk_free,
            'disk_percent': self.disk_percent,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'process_count': self.process_count,
            'load_average': self.load_average
        }

class SystemMonitor:
    """
    Real-time system health and performance monitoring.
    Provides continuous monitoring of system resources and alerts on thresholds.
    """

    def __init__(self, telemetry: TelemetryEngine = None):
        self.telemetry = telemetry or get_telemetry()
        self.telemetry.register_component('system_monitor')

        # Configuration
        self.collection_interval = 5.0  # seconds
        self.alert_thresholds = {
            'cpu_percent': 85.0,
            'memory_percent': 90.0,
            'disk_percent': 95.0
        }

        # State
        self._running = False
        self._monitor_thread = None
        self._lock = threading.Lock()

        # Historical data
        self.metrics_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.alerts_history = deque(maxlen=1000)

        # Performance baselines
        self.baseline_metrics = None
        self.performance_trends = {
            'cpu': deque(maxlen=60),    # 5 minutes
            'memory': deque(maxlen=60),
            'disk': deque(maxlen=60)
        }

        # Network interface baseline
        self.network_baseline = None

    async def start(self):
        """Start system monitoring"""
        self.telemetry.log_info("Starting system monitoring", 'system_monitor')

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        # Collect initial baseline
        await self._collect_baseline()

        self.telemetry.log_info("System monitoring started", 'system_monitor')

    async def stop(self):
        """Stop system monitoring"""
        self.telemetry.log_info("Stopping system monitoring", 'system_monitor')

        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)

        self.telemetry.log_info("System monitoring stopped", 'system_monitor')

    def _monitoring_loop(self):
        """Main monitoring loop running in background thread"""
        while self._running:
            try:
                metrics = self._collect_system_metrics()
                self._process_metrics(metrics)
                self._check_alerts(metrics)
                self._update_trends(metrics)

                time.sleep(self.collection_interval)

            except Exception as e:
                self.telemetry.log_error(
                    f"Error in monitoring loop: {e}",
                    'system_monitor',
                    {'error_type': type(e).__name__}
                )
                time.sleep(self.collection_interval)

    def _collect_system_metrics(self) -> ResourceMetrics:
        """Collect current system metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_cores = psutil.cpu_count()

        # Memory metrics
        memory = psutil.virtual_memory()

        # Disk metrics (root filesystem)
        disk = psutil.disk_usage('/')

        # Network metrics
        network = psutil.net_io_counters()

        # Process count
        process_count = len(psutil.pids())

        # Load average (Unix-like systems)
        try:
            load_avg = list(psutil.getloadavg())
        except AttributeError:
            # Windows doesn't have load average
            load_avg = [0.0, 0.0, 0.0]

        return ResourceMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=cpu_percent,
            cpu_cores=cpu_cores,
            memory_total=memory.total,
            memory_available=memory.available,
            memory_percent=memory.percent,
            disk_total=disk.total,
            disk_used=disk.used,
            disk_free=disk.free,
            disk_percent=disk.percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            process_count=process_count,
            load_average=load_avg
        )

    def _process_metrics(self, metrics: ResourceMetrics):
        """Process and emit metrics to telemetry system"""
        # Emit individual metrics
        self.telemetry.set_gauge('system_cpu_percent', metrics.cpu_percent,
                                component='system_monitor')
        self.telemetry.set_gauge('system_memory_percent', metrics.memory_percent,
                                component='system_monitor')
        self.telemetry.set_gauge('system_disk_percent', metrics.disk_percent,
                                component='system_monitor')
        self.telemetry.set_gauge('system_memory_available_bytes', metrics.memory_available,
                                component='system_monitor')
        self.telemetry.set_gauge('system_disk_free_bytes', metrics.disk_free,
                                component='system_monitor')
        self.telemetry.set_gauge('system_process_count', metrics.process_count,
                                component='system_monitor')

        # Network throughput (calculate delta from baseline)
        if self.network_baseline:
            bytes_sent_delta = metrics.network_bytes_sent - self.network_baseline['bytes_sent']
            bytes_recv_delta = metrics.network_bytes_recv - self.network_baseline['bytes_recv']
            time_delta = (metrics.timestamp - self.network_baseline['timestamp']).total_seconds()

            if time_delta > 0:
                send_rate = bytes_sent_delta / time_delta
                recv_rate = bytes_recv_delta / time_delta

                self.telemetry.set_gauge('system_network_send_rate_bps', send_rate,
                                        component='system_monitor')
                self.telemetry.set_gauge('system_network_recv_rate_bps', recv_rate,
                                        component='system_monitor')

        # Update network baseline
        self.network_baseline = {
            'bytes_sent': metrics.network_bytes_sent,
            'bytes_recv': metrics.network_bytes_recv,
            'timestamp': metrics.timestamp
        }

        # Store in history
        with self._lock:
            self.metrics_history.append(metrics)

    def _check_alerts(self, metrics: ResourceMetrics):
        """Check for alert conditions"""
        alerts = []

        # CPU alert
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'cpu_high',
                'severity': 'warning',
                'value': metrics.cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent'],
                'message': f"High CPU usage: {metrics.cpu_percent:.1f}%"
            })

        # Memory alert
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'memory_high',
                'severity': 'critical' if metrics.memory_percent > 95 else 'warning',
                'value': metrics.memory_percent,
                'threshold': self.alert_thresholds['memory_percent'],
                'message': f"High memory usage: {metrics.memory_percent:.1f}%"
            })

        # Disk alert
        if metrics.disk_percent > self.alert_thresholds['disk_percent']:
            alerts.append({
                'type': 'disk_high',
                'severity': 'critical',
                'value': metrics.disk_percent,
                'threshold': self.alert_thresholds['disk_percent'],
                'message': f"High disk usage: {metrics.disk_percent:.1f}%"
            })

        # Process alerts if needed
        for alert in alerts:
            self._emit_alert(alert, metrics.timestamp)

    def _emit_alert(self, alert: Dict[str, Any], timestamp: datetime):
        """Emit system alert"""
        alert_data = {
            **alert,
            'timestamp': timestamp.isoformat(),
            'component': 'system_monitor'
        }

        # Log alert
        log_level = LogLevel.CRITICAL if alert['severity'] == 'critical' else LogLevel.WARNING
        self.telemetry.log_entry(
            log_level,
            alert['message'],
            'system_monitor',
            alert_data
        )

        # Emit alert metric
        self.telemetry.increment_counter(
            'system_alerts_total',
            1,
            {'type': alert['type'], 'severity': alert['severity']},
            'system_monitor'
        )

        # Store in alert history
        with self._lock:
            self.alerts_history.append(alert_data)

    def _update_trends(self, metrics: ResourceMetrics):
        """Update performance trends"""
        self.performance_trends['cpu'].append({
            'timestamp': metrics.timestamp,
            'value': metrics.cpu_percent
        })
        self.performance_trends['memory'].append({
            'timestamp': metrics.timestamp,
            'value': metrics.memory_percent
        })
        self.performance_trends['disk'].append({
            'timestamp': metrics.timestamp,
            'value': metrics.disk_percent
        })

    async def _collect_baseline(self):
        """Collect performance baseline"""
        metrics = self._collect_system_metrics()
        self.baseline_metrics = metrics

        self.telemetry.log_info(
            "System baseline established",
            'system_monitor',
            {
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'disk_percent': metrics.disk_percent
            }
        )

    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get most recent system metrics"""
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]
            return None

    def get_metrics_history(self, minutes: int = 60) -> List[ResourceMetrics]:
        """Get historical metrics for specified time period"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (minutes * 60)

        with self._lock:
            recent_metrics = [
                m for m in self.metrics_history
                if m.timestamp.timestamp() > cutoff_time
            ]
            return recent_metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and trends"""
        current = self.get_current_metrics()
        if not current:
            return {}

        # Calculate averages from trends
        cpu_trend = list(self.performance_trends['cpu'])
        memory_trend = list(self.performance_trends['memory'])
        disk_trend = list(self.performance_trends['disk'])

        avg_cpu = sum(p['value'] for p in cpu_trend) / len(cpu_trend) if cpu_trend else 0
        avg_memory = sum(p['value'] for p in memory_trend) / len(memory_trend) if memory_trend else 0
        avg_disk = sum(p['value'] for p in disk_trend) / len(disk_trend) if disk_trend else 0

        return {
            'current': current.to_dict(),
            'averages_5min': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory,
                'disk_percent': avg_disk
            },
            'baseline': self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            'alerts_count': len(self.alerts_history),
            'thresholds': self.alert_thresholds,
            'trends': {
                'cpu': [p['value'] for p in cpu_trend],
                'memory': [p['value'] for p in memory_trend],
                'disk': [p['value'] for p in disk_trend]
            }
        }

    def get_alerts(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (minutes * 60)

        with self._lock:
            recent_alerts = [
                alert for alert in self.alerts_history
                if datetime.fromisoformat(alert['timestamp']).timestamp() > cutoff_time
            ]
            return recent_alerts

    def set_alert_threshold(self, metric: str, threshold: float):
        """Update alert threshold"""
        if metric in self.alert_thresholds:
            old_threshold = self.alert_thresholds[metric]
            self.alert_thresholds[metric] = threshold

            self.telemetry.log_info(
                f"Alert threshold updated for {metric}",
                'system_monitor',
                {
                    'metric': metric,
                    'old_threshold': old_threshold,
                    'new_threshold': threshold
                }
            )
        else:
            self.telemetry.log_warning(
                f"Unknown metric for threshold: {metric}",
                'system_monitor'
            )

    def is_system_healthy(self) -> bool:
        """Check if system is within healthy parameters"""
        current = self.get_current_metrics()
        if not current:
            return False

        return (
            current.cpu_percent < self.alert_thresholds['cpu_percent'] and
            current.memory_percent < self.alert_thresholds['memory_percent'] and
            current.disk_percent < self.alert_thresholds['disk_percent']
        )