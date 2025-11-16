"""
Telemetry Engine - Core observability system
Implements centralized logging, metrics, and tracing as recommended by GPT-5.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
from collections import defaultdict, deque
import psutil
import uuid

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Individual metric measurement"""
    name: str
    value: Union[int, float]
    type: MetricType
    labels: Dict[str, str]
    timestamp: datetime
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'type': self.type.value,
            'labels': self.labels,
            'timestamp': self.timestamp.isoformat(),
            'unit': self.unit
        }

@dataclass
class Trace:
    """Distributed tracing span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    status: str = "unknown"
    tags: Dict[str, Any] = None
    logs: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.logs is None:
            self.logs = []

    def finish(self, status: str = "success"):
        """Mark trace as completed"""
        self.end_time = datetime.now(timezone.utc)
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status

    def add_tag(self, key: str, value: Any):
        """Add tag to trace"""
        self.tags[key] = value

    def log_event(self, event: str, fields: Dict[str, Any] = None):
        """Add log event to trace"""
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event': event,
            'fields': fields or {}
        }
        self.logs.append(log_entry)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'operation_name': self.operation_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_ms': self.duration_ms,
            'status': self.status,
            'tags': self.tags,
            'logs': self.logs
        }

@dataclass
class LogEntry:
    """Structured log entry"""
    level: LogLevel
    message: str
    timestamp: datetime
    component: str
    context: Dict[str, Any]
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'context': self.context,
            'trace_id': self.trace_id,
            'span_id': self.span_id
        }

class TelemetryEngine:
    """
    Centralized telemetry engine for observability across all system components.
    Implements GPT-5's recommendation for comprehensive system monitoring.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_buffer = deque(maxlen=10000)
        self.traces_buffer = deque(maxlen=5000)
        self.logs_buffer = deque(maxlen=20000)
        self.active_traces = {}

        # Threading for async collection
        self._lock = threading.Lock()
        self._flush_interval = self.config.get('flush_interval', 30)  # seconds
        self._running = False
        self._flush_thread = None

        # Component registrations
        self.registered_components = set()
        self.component_metrics = defaultdict(list)

        # Setup logging
        self._setup_logging()

        # Performance counters
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)

    def _setup_logging(self):
        """Configure structured logging"""
        self.logger = logging.getLogger('telemetry_engine')
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    async def start(self):
        """Start telemetry collection"""
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop)
        self._flush_thread.daemon = True
        self._flush_thread.start()
        self.logger.info("Telemetry engine started")

    async def stop(self):
        """Stop telemetry collection"""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=5)
        await self._flush_buffers()
        self.logger.info("Telemetry engine stopped")

    def register_component(self, component_name: str):
        """Register a system component for monitoring"""
        with self._lock:
            self.registered_components.add(component_name)
        self.log_info(f"Component registered: {component_name}", 'telemetry_engine')

    # Metrics Collection
    def emit_metric(self, name: str, value: Union[int, float],
                   metric_type: MetricType, labels: Dict[str, str] = None,
                   unit: str = "", component: str = "system"):
        """Emit a metric measurement"""
        metric = Metric(
            name=name,
            value=value,
            type=metric_type,
            labels=labels or {},
            timestamp=datetime.now(timezone.utc),
            unit=unit
        )

        with self._lock:
            self.metrics_buffer.append(metric)

            # Update real-time counters
            if metric_type == MetricType.COUNTER:
                self.counters[name] += value
            elif metric_type == MetricType.GAUGE:
                self.gauges[name] = value
            elif metric_type == MetricType.HISTOGRAM:
                self.histograms[name].append(value)

    def increment_counter(self, name: str, value: int = 1,
                         labels: Dict[str, str] = None, component: str = "system"):
        """Increment a counter metric"""
        self.emit_metric(name, value, MetricType.COUNTER, labels, "count", component)

    def set_gauge(self, name: str, value: float,
                  labels: Dict[str, str] = None, component: str = "system"):
        """Set a gauge metric value"""
        self.emit_metric(name, value, MetricType.GAUGE, labels, "", component)

    def record_histogram(self, name: str, value: float,
                        labels: Dict[str, str] = None, unit: str = "ms", component: str = "system"):
        """Record a histogram measurement"""
        self.emit_metric(name, value, MetricType.HISTOGRAM, labels, unit, component)

    # Distributed Tracing
    def start_trace(self, operation_name: str, parent_trace_id: str = None,
                   component: str = "system") -> Trace:
        """Start a new distributed trace"""
        trace_id = parent_trace_id or str(uuid.uuid4())
        span_id = str(uuid.uuid4())

        trace = Trace(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_trace_id,
            operation_name=operation_name,
            start_time=datetime.now(timezone.utc)
        )
        trace.add_tag('component', component)

        with self._lock:
            self.active_traces[span_id] = trace

        return trace

    def finish_trace(self, trace: Trace, status: str = "success"):
        """Finish a distributed trace"""
        trace.finish(status)

        with self._lock:
            if trace.span_id in self.active_traces:
                del self.active_traces[trace.span_id]
            self.traces_buffer.append(trace)

    def get_trace_context(self) -> Dict[str, str]:
        """Get current trace context for propagation"""
        # In a real implementation, this would get context from thread-local storage
        return {}

    # Structured Logging
    def log_entry(self, level: LogLevel, message: str, component: str,
                 context: Dict[str, Any] = None, trace_id: str = None, span_id: str = None):
        """Emit a structured log entry"""
        log_entry = LogEntry(
            level=level,
            message=message,
            timestamp=datetime.now(timezone.utc),
            component=component,
            context=context or {},
            trace_id=trace_id,
            span_id=span_id
        )

        with self._lock:
            self.logs_buffer.append(log_entry)

        # Also emit to standard logger
        getattr(self.logger, level.value)(f"[{component}] {message}")

    def log_debug(self, message: str, component: str, context: Dict[str, Any] = None):
        self.log_entry(LogLevel.DEBUG, message, component, context)

    def log_info(self, message: str, component: str, context: Dict[str, Any] = None):
        self.log_entry(LogLevel.INFO, message, component, context)

    def log_warning(self, message: str, component: str, context: Dict[str, Any] = None):
        self.log_entry(LogLevel.WARNING, message, component, context)

    def log_error(self, message: str, component: str, context: Dict[str, Any] = None):
        self.log_entry(LogLevel.ERROR, message, component, context)

    def log_critical(self, message: str, component: str, context: Dict[str, Any] = None):
        self.log_entry(LogLevel.CRITICAL, message, component, context)

    # System Health Monitoring
    def collect_system_metrics(self):
        """Collect system-wide health metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.set_gauge('system_cpu_percent', cpu_percent, component='system')

        # Memory metrics
        memory = psutil.virtual_memory()
        self.set_gauge('system_memory_percent', memory.percent, component='system')
        self.set_gauge('system_memory_available_bytes', memory.available, component='system')

        # Disk metrics
        disk = psutil.disk_usage('/')
        self.set_gauge('system_disk_percent', disk.percent, component='system')
        self.set_gauge('system_disk_free_bytes', disk.free, component='system')

        # Process metrics
        process = psutil.Process()
        self.set_gauge('process_memory_rss_bytes', process.memory_info().rss, component='process')
        self.set_gauge('process_cpu_percent', process.cpu_percent(), component='process')

    # Data Retrieval
    def get_metrics(self, component: str = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve recent metrics"""
        with self._lock:
            metrics = list(self.metrics_buffer)

        if component:
            # Filter by component would need to be implemented based on metric labels
            pass

        return [metric.to_dict() for metric in metrics[-limit:]]

    def get_traces(self, trace_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent traces"""
        with self._lock:
            traces = list(self.traces_buffer)

        if trace_id:
            traces = [t for t in traces if t.trace_id == trace_id]

        return [trace.to_dict() for trace in traces[-limit:]]

    def get_logs(self, component: str = None, level: LogLevel = None,
                limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve recent logs"""
        with self._lock:
            logs = list(self.logs_buffer)

        if component:
            logs = [log for log in logs if log.component == component]

        if level:
            logs = [log for log in logs if log.level == level]

        return [log.to_dict() for log in logs[-limit:]]

    def get_summary(self) -> Dict[str, Any]:
        """Get telemetry summary statistics"""
        with self._lock:
            return {
                'metrics_count': len(self.metrics_buffer),
                'traces_count': len(self.traces_buffer),
                'logs_count': len(self.logs_buffer),
                'active_traces_count': len(self.active_traces),
                'registered_components': list(self.registered_components),
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'last_flush': datetime.now(timezone.utc).isoformat()
            }

    # Background Processing
    def _flush_loop(self):
        """Background thread for flushing telemetry data"""
        while self._running:
            try:
                asyncio.run(self._flush_buffers())
                self.collect_system_metrics()
                time.sleep(self._flush_interval)
            except Exception as e:
                self.logger.error(f"Error in flush loop: {e}")

    async def _flush_buffers(self):
        """Flush telemetry data to storage/external systems"""
        # In a real implementation, this would write to:
        # - Time series database (InfluxDB, Prometheus)
        # - Log aggregation system (ELK, Loki)
        # - Distributed tracing system (Jaeger, Zipkin)

        with self._lock:
            metrics_count = len(self.metrics_buffer)
            traces_count = len(self.traces_buffer)
            logs_count = len(self.logs_buffer)

        if metrics_count > 0 or traces_count > 0 or logs_count > 0:
            self.logger.info(
                f"Flushed telemetry: {metrics_count} metrics, "
                f"{traces_count} traces, {logs_count} logs"
            )

# Global telemetry instance
_telemetry_engine = None

def get_telemetry() -> TelemetryEngine:
    """Get global telemetry engine instance"""
    global _telemetry_engine
    if _telemetry_engine is None:
        _telemetry_engine = TelemetryEngine()
    return _telemetry_engine

def init_telemetry(config: Dict[str, Any] = None) -> TelemetryEngine:
    """Initialize global telemetry engine"""
    global _telemetry_engine
    _telemetry_engine = TelemetryEngine(config)
    return _telemetry_engine