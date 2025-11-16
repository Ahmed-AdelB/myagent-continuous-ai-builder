"""
Performance Tests for MyAgent API Endpoints
Tests response times, throughput, resource usage, and load handling
"""

import asyncio
import time
import statistics
import psutil
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import aiohttp
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
from dataclasses import dataclass

# Performance test configuration
PERFORMANCE_CONFIG = {
    "api_base_url": "http://localhost:8000",
    "load_test_users": [1, 5, 10, 25, 50],
    "duration_seconds": 30,
    "acceptable_response_time": 2.0,  # seconds
    "acceptable_error_rate": 0.05,  # 5%
    "memory_threshold_mb": 512,
    "cpu_threshold_percent": 80
}

@dataclass
class PerformanceMetric:
    """Performance measurement data"""
    endpoint: str
    response_time: float
    status_code: int
    memory_usage_mb: float
    cpu_percent: float
    timestamp: datetime
    payload_size: int = 0

@dataclass
class LoadTestResult:
    """Load test results"""
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput_rps: float
    error_rate: float
    max_memory_mb: float
    avg_cpu_percent: float

class PerformanceMonitor:
    """Monitors system performance during tests"""

    def __init__(self):
        self.monitoring = False
        self.metrics: List[Dict[str, Any]] = []
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        if not self.metrics:
            return {"error": "No metrics collected"}

        memory_values = [m["memory_mb"] for m in self.metrics]
        cpu_values = [m["cpu_percent"] for m in self.metrics]

        return {
            "duration_seconds": len(self.metrics),
            "avg_memory_mb": statistics.mean(memory_values),
            "max_memory_mb": max(memory_values),
            "avg_cpu_percent": statistics.mean(cpu_values),
            "max_cpu_percent": max(cpu_values),
            "sample_count": len(self.metrics)
        }

    def _monitor_system(self):
        """Internal monitoring loop"""
        while self.monitoring:
            try:
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)

                self.metrics.append({
                    "timestamp": datetime.now().isoformat(),
                    "memory_mb": memory_info.used / (1024 * 1024),
                    "memory_percent": memory_info.percent,
                    "cpu_percent": cpu_percent,
                    "available_memory_mb": memory_info.available / (1024 * 1024)
                })

                time.sleep(1)
            except Exception as e:
                logging.error(f"Error monitoring system: {e}")
                break

class ApiPerformanceTester:
    """Performance testing for API endpoints"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.monitor = PerformanceMonitor()

    async def setup(self):
        """Setup test session"""
        self.session = aiohttp.ClientSession()

    async def cleanup(self):
        """Cleanup test session"""
        if self.session:
            await self.session.close()

    async def measure_endpoint_performance(self, endpoint: str, method: str = "GET",
                                         payload: Optional[Dict] = None,
                                         headers: Optional[Dict] = None) -> PerformanceMetric:
        """Measure performance of a single endpoint call"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024 * 1024)
        start_cpu = psutil.cpu_percent()

        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == "GET":
                async with self.session.get(url, headers=headers) as response:
                    await response.text()
                    status_code = response.status
                    payload_size = len(await response.read()) if hasattr(response, 'read') else 0
            elif method.upper() == "POST":
                async with self.session.post(url, json=payload, headers=headers) as response:
                    await response.text()
                    status_code = response.status
                    payload_size = len(json.dumps(payload)) if payload else 0
            else:
                raise ValueError(f"Unsupported method: {method}")

        except Exception as e:
            logging.error(f"Error calling {endpoint}: {e}")
            status_code = 500
            payload_size = 0

        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024 * 1024)
        end_cpu = psutil.cpu_percent()

        return PerformanceMetric(
            endpoint=endpoint,
            response_time=end_time - start_time,
            status_code=status_code,
            memory_usage_mb=end_memory - start_memory,
            cpu_percent=(start_cpu + end_cpu) / 2,
            timestamp=datetime.now(),
            payload_size=payload_size
        )

    async def load_test_endpoint(self, endpoint: str, concurrent_users: int,
                               duration_seconds: int, method: str = "GET",
                               payload: Optional[Dict] = None) -> LoadTestResult:
        """Perform load testing on an endpoint"""

        self.monitor.start_monitoring()

        start_time = time.time()
        end_time = start_time + duration_seconds

        results: List[PerformanceMetric] = []
        tasks = []

        # Create concurrent tasks
        async def make_requests():
            while time.time() < end_time:
                try:
                    metric = await self.measure_endpoint_performance(endpoint, method, payload)
                    results.append(metric)
                    await asyncio.sleep(0.1)  # Small delay between requests
                except Exception as e:
                    logging.error(f"Error in load test: {e}")

        # Start concurrent users
        for _ in range(concurrent_users):
            tasks.append(asyncio.create_task(make_requests()))

        # Wait for all tasks to complete
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logging.error(f"Error in load test execution: {e}")

        system_metrics = self.monitor.stop_monitoring()

        # Calculate statistics
        if not results:
            return LoadTestResult(
                concurrent_users=concurrent_users,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                throughput_rps=0,
                error_rate=1.0,
                max_memory_mb=0,
                avg_cpu_percent=0
            )

        successful_requests = sum(1 for r in results if 200 <= r.status_code < 400)
        failed_requests = len(results) - successful_requests
        response_times = [r.response_time for r in results]

        return LoadTestResult(
            concurrent_users=concurrent_users,
            total_requests=len(results),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=statistics.mean(response_times),
            p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
            p99_response_time=statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times),
            throughput_rps=len(results) / duration_seconds,
            error_rate=failed_requests / len(results),
            max_memory_mb=system_metrics.get("max_memory_mb", 0),
            avg_cpu_percent=system_metrics.get("avg_cpu_percent", 0)
        )

# Test Fixtures and Mock Data
@pytest.fixture
def performance_tester():
    """Fixture providing performance tester instance"""
    tester = ApiPerformanceTester(PERFORMANCE_CONFIG["api_base_url"])
    return tester

@pytest.fixture
def test_payloads():
    """Test payloads for different scenarios"""
    return {
        "small_project": {
            "name": "test_project",
            "description": "Performance test project",
            "requirements": ["basic functionality"]
        },
        "medium_project": {
            "name": "medium_test_project",
            "description": "Medium complexity performance test project with more requirements",
            "requirements": [
                "advanced functionality",
                "complex business logic",
                "data processing capabilities",
                "integration with multiple services"
            ],
            "metadata": {
                "priority": "high",
                "estimated_duration": "2 weeks",
                "team_size": 5,
                "technologies": ["python", "fastapi", "postgresql", "redis"]
            }
        },
        "large_project": {
            "name": "large_test_project",
            "description": "Large scale performance test project with comprehensive requirements and complex structure",
            "requirements": [
                "microservices architecture",
                "real-time data processing",
                "machine learning capabilities",
                "distributed systems integration",
                "high availability and scalability",
                "comprehensive monitoring and logging",
                "advanced security features",
                "multi-tenant support"
            ],
            "metadata": {
                "priority": "critical",
                "estimated_duration": "6 months",
                "team_size": 20,
                "technologies": [
                    "python", "fastapi", "postgresql", "redis", "elasticsearch",
                    "kafka", "kubernetes", "docker", "prometheus", "grafana"
                ],
                "compliance_requirements": ["GDPR", "SOC2", "HIPAA"],
                "performance_requirements": {
                    "max_response_time": "100ms",
                    "throughput": "10000 rps",
                    "availability": "99.99%"
                }
            }
        }
    }

# Performance Test Cases

@pytest.mark.asyncio
@pytest.mark.performance
class TestApiPerformance:
    """Test suite for API performance"""

    async def test_health_endpoint_performance(self, performance_tester):
        """Test health endpoint response time"""
        await performance_tester.setup()

        try:
            metric = await performance_tester.measure_endpoint_performance("/health")

            assert metric.response_time < PERFORMANCE_CONFIG["acceptable_response_time"]
            assert 200 <= metric.status_code < 300
            assert metric.memory_usage_mb < PERFORMANCE_CONFIG["memory_threshold_mb"]

            logging.info(f"Health endpoint performance: {metric.response_time:.3f}s")

        finally:
            await performance_tester.cleanup()

    async def test_projects_list_performance(self, performance_tester):
        """Test projects listing endpoint performance"""
        await performance_tester.setup()

        try:
            metric = await performance_tester.measure_endpoint_performance("/api/v1/projects")

            assert metric.response_time < PERFORMANCE_CONFIG["acceptable_response_time"] * 2
            assert 200 <= metric.status_code < 300

            logging.info(f"Projects list performance: {metric.response_time:.3f}s")

        finally:
            await performance_tester.cleanup()

    async def test_project_creation_performance(self, performance_tester, test_payloads):
        """Test project creation performance with different payload sizes"""
        await performance_tester.setup()

        try:
            for payload_name, payload in test_payloads.items():
                metric = await performance_tester.measure_endpoint_performance(
                    "/api/v1/projects",
                    method="POST",
                    payload=payload
                )

                # Adjust timeout based on payload complexity
                timeout_multiplier = 1 if "small" in payload_name else 2 if "medium" in payload_name else 3
                max_response_time = PERFORMANCE_CONFIG["acceptable_response_time"] * timeout_multiplier

                assert metric.response_time < max_response_time
                assert 200 <= metric.status_code < 300

                logging.info(f"Project creation ({payload_name}) performance: {metric.response_time:.3f}s")

        finally:
            await performance_tester.cleanup()

    @pytest.mark.parametrize("concurrent_users", PERFORMANCE_CONFIG["load_test_users"])
    async def test_load_testing_health_endpoint(self, performance_tester, concurrent_users):
        """Load testing for health endpoint"""
        await performance_tester.setup()

        try:
            result = await performance_tester.load_test_endpoint(
                "/health",
                concurrent_users=concurrent_users,
                duration_seconds=10  # Shorter duration for health endpoint
            )

            assert result.error_rate <= PERFORMANCE_CONFIG["acceptable_error_rate"]
            assert result.avg_response_time < PERFORMANCE_CONFIG["acceptable_response_time"]
            assert result.max_memory_mb < PERFORMANCE_CONFIG["memory_threshold_mb"]

            logging.info(f"Load test ({concurrent_users} users): {result.throughput_rps:.1f} RPS, "
                        f"{result.avg_response_time:.3f}s avg response time")

        finally:
            await performance_tester.cleanup()

    @pytest.mark.parametrize("concurrent_users", [1, 5, 10])
    async def test_load_testing_project_operations(self, performance_tester, test_payloads, concurrent_users):
        """Load testing for project operations"""
        await performance_tester.setup()

        try:
            # Test project creation load
            result = await performance_tester.load_test_endpoint(
                "/api/v1/projects",
                concurrent_users=concurrent_users,
                duration_seconds=15,
                method="POST",
                payload=test_payloads["small_project"]
            )

            # More relaxed thresholds for project operations
            assert result.error_rate <= 0.1  # 10% error rate acceptable for load testing
            assert result.avg_response_time < PERFORMANCE_CONFIG["acceptable_response_time"] * 3

            logging.info(f"Project operations load test ({concurrent_users} users): "
                        f"{result.throughput_rps:.1f} RPS, {result.error_rate:.2%} error rate")

        finally:
            await performance_tester.cleanup()

    async def test_memory_usage_stability(self, performance_tester):
        """Test memory usage stability during extended operation"""
        await performance_tester.setup()

        try:
            initial_memory = psutil.virtual_memory().used / (1024 * 1024)

            # Perform multiple operations
            for i in range(20):
                await performance_tester.measure_endpoint_performance("/health")
                await asyncio.sleep(0.1)

            final_memory = psutil.virtual_memory().used / (1024 * 1024)
            memory_increase = final_memory - initial_memory

            # Memory increase should be minimal
            assert memory_increase < 50  # Less than 50MB increase

            logging.info(f"Memory stability test: {memory_increase:.1f}MB increase")

        finally:
            await performance_tester.cleanup()

    async def test_response_time_consistency(self, performance_tester):
        """Test response time consistency across multiple calls"""
        await performance_tester.setup()

        try:
            response_times = []

            for _ in range(10):
                metric = await performance_tester.measure_endpoint_performance("/health")
                response_times.append(metric.response_time)
                await asyncio.sleep(0.1)

            avg_time = statistics.mean(response_times)
            std_dev = statistics.stdev(response_times)

            # Standard deviation should be low for consistent performance
            assert std_dev < avg_time * 0.5  # Standard deviation less than 50% of mean

            logging.info(f"Response time consistency: {avg_time:.3f}s ± {std_dev:.3f}s")

        finally:
            await performance_tester.cleanup()

@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarks"""

    async def test_throughput_benchmark(self, performance_tester):
        """Benchmark maximum throughput"""
        await performance_tester.setup()

        try:
            # Test with increasing load
            results = []
            for users in [1, 5, 10, 20, 30]:
                result = await performance_tester.load_test_endpoint(
                    "/health",
                    concurrent_users=users,
                    duration_seconds=10
                )
                results.append({
                    "users": users,
                    "throughput": result.throughput_rps,
                    "avg_response_time": result.avg_response_time,
                    "error_rate": result.error_rate
                })

                # Stop if error rate becomes too high
                if result.error_rate > 0.1:
                    break

            # Find optimal throughput point
            max_throughput = max(r["throughput"] for r in results if r["error_rate"] < 0.05)

            logging.info(f"Maximum sustainable throughput: {max_throughput:.1f} RPS")

            # Throughput should be reasonable
            assert max_throughput > 10  # At least 10 RPS

        finally:
            await performance_tester.cleanup()

    async def test_stress_testing(self, performance_tester):
        """Stress test with high load"""
        await performance_tester.setup()

        try:
            # Gradually increase load to find breaking point
            breaking_point_found = False
            max_stable_users = 0

            for users in range(10, 101, 10):  # 10 to 100 users
                result = await performance_tester.load_test_endpoint(
                    "/health",
                    concurrent_users=users,
                    duration_seconds=5  # Shorter duration for stress test
                )

                if result.error_rate > 0.2 or result.avg_response_time > 5.0:
                    breaking_point_found = True
                    break
                else:
                    max_stable_users = users

            logging.info(f"System stable up to {max_stable_users} concurrent users")

            # System should handle at least some concurrent load
            assert max_stable_users >= 20

        finally:
            await performance_tester.cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v", "--tb=short"])