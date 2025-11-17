"""
Performance Test Runner for MyAgent
Orchestrates and executes all performance testing suites
"""

import asyncio
import time
import json
import logging
import statistics
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pytest

# Import our performance test modules
from test_api_performance import ApiPerformanceTester, PERFORMANCE_CONFIG
from test_database_performance import DatabasePerformanceTester, DB_CONFIG

# Performance test runner configuration
RUNNER_CONFIG = {
    "output_directory": "test_results/performance",
    "report_formats": ["json", "html", "csv"],
    "test_suites": [
        "api_performance",
        "database_performance",
        "memory_performance",
        "concurrent_load"
    ],
    "performance_thresholds": {
        "api_response_time": 2.0,
        "database_query_time": 0.1,
        "memory_usage_mb": 512,
        "cpu_usage_percent": 80,
        "error_rate_threshold": 0.05
    },
    "load_test_scenarios": [
        {"name": "light_load", "users": 5, "duration": 30},
        {"name": "normal_load", "users": 15, "duration": 60},
        {"name": "heavy_load", "users": 30, "duration": 90},
        {"name": "stress_test", "users": 50, "duration": 120}
    ]
}

class PerformanceTestResult:
    """Performance test result container"""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.duration: Optional[float] = None
        self.success = False
        self.metrics: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def mark_complete(self, success: bool = True):
        """Mark test as complete"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.success = success

    def add_metric(self, key: str, value: Any):
        """Add performance metric"""
        self.metrics[key] = value

    def add_error(self, error: str):
        """Add error message"""
        self.errors.append(error)

    def add_warning(self, warning: str):
        """Add warning message"""
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "test_name": self.test_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "success": self.success,
            "metrics": self.metrics,
            "errors": self.errors,
            "warnings": self.warnings
        }

class PerformanceTestRunner:
    """Main performance test runner"""

    def __init__(self):
        self.results: List[PerformanceTestResult] = []
        self.api_tester = ApiPerformanceTester(PERFORMANCE_CONFIG["api_base_url"])
        self.db_tester = DatabasePerformanceTester()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    async def run_all_tests(self, test_suites: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run all performance test suites"""
        self.start_time = datetime.now()
        logging.info("Starting comprehensive performance test run")

        if test_suites is None:
            test_suites = RUNNER_CONFIG["test_suites"]

        # Run test suites
        for suite_name in test_suites:
            try:
                await self._run_test_suite(suite_name)
            except Exception as e:
                logging.error(f"Error running test suite {suite_name}: {e}")

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        # Generate comprehensive report
        report = await self._generate_comprehensive_report()
        report["test_run_duration"] = duration
        report["test_run_timestamp"] = self.start_time.isoformat()

        logging.info(f"Performance test run completed in {duration:.1f} seconds")

        return report

    async def _run_test_suite(self, suite_name: str):
        """Run a specific test suite"""
        logging.info(f"Running test suite: {suite_name}")

        if suite_name == "api_performance":
            await self._run_api_performance_tests()
        elif suite_name == "database_performance":
            await self._run_database_performance_tests()
        elif suite_name == "memory_performance":
            await self._run_memory_performance_tests()
        elif suite_name == "concurrent_load":
            await self._run_concurrent_load_tests()
        else:
            logging.warning(f"Unknown test suite: {suite_name}")

    async def _run_api_performance_tests(self):
        """Run API performance tests"""
        result = PerformanceTestResult("api_performance")

        try:
            await self.api_tester.setup()

            # Test health endpoint
            health_metric = await self.api_tester.measure_endpoint_performance("/health")
            result.add_metric("health_endpoint_response_time", health_metric.response_time)

            if health_metric.response_time > RUNNER_CONFIG["performance_thresholds"]["api_response_time"]:
                result.add_warning(f"Health endpoint slow: {health_metric.response_time:.3f}s")

            # Test projects endpoint
            projects_metric = await self.api_tester.measure_endpoint_performance("/api/v1/projects")
            result.add_metric("projects_endpoint_response_time", projects_metric.response_time)

            # Load testing
            load_results = []
            for scenario in RUNNER_CONFIG["load_test_scenarios"][:2]:  # Run light and normal load
                load_result = await self.api_tester.load_test_endpoint(
                    "/health",
                    concurrent_users=scenario["users"],
                    duration_seconds=min(scenario["duration"], 30)  # Shorter duration for testing
                )
                load_results.append({
                    "scenario": scenario["name"],
                    "throughput_rps": load_result.throughput_rps,
                    "avg_response_time": load_result.avg_response_time,
                    "error_rate": load_result.error_rate
                })

                if load_result.error_rate > RUNNER_CONFIG["performance_thresholds"]["error_rate_threshold"]:
                    result.add_warning(f"High error rate in {scenario['name']}: {load_result.error_rate:.2%}")

            result.add_metric("load_test_results", load_results)

            result.mark_complete(True)

        except Exception as e:
            result.add_error(f"API performance test failed: {str(e)}")
            result.mark_complete(False)

        finally:
            await self.api_tester.cleanup()

        self.results.append(result)

    async def _run_database_performance_tests(self):
        """Run database performance tests"""
        result = PerformanceTestResult("database_performance")

        try:
            # Test connection performance
            connection_metrics = await self.db_tester.test_connection_performance(5)
            avg_connection_time = statistics.mean(m.connection_time for m in connection_metrics)
            result.add_metric("avg_connection_time", avg_connection_time)

            if avg_connection_time > 0.1:
                result.add_warning(f"Slow database connections: {avg_connection_time:.3f}s")

            # Test query performance by type
            query_results = await self.db_tester.test_query_performance_by_type()

            query_performance = {}
            for query_type, metrics in query_results.items():
                if metrics:
                    avg_time = statistics.mean(m.execution_time for m in metrics)
                    query_performance[query_type.lower()] = avg_time

                    if avg_time > DB_CONFIG["acceptable_query_time"] * 2:
                        result.add_warning(f"Slow {query_type} queries: {avg_time:.3f}s")

            result.add_metric("query_performance_by_type", query_performance)

            # Test bulk operations
            bulk_results = await self.db_tester.test_bulk_operations_performance([100, 500])
            bulk_performance = {}
            for size, metric in bulk_results.items():
                time_per_record = metric.execution_time / size
                bulk_performance[f"bulk_{size}"] = {
                    "total_time": metric.execution_time,
                    "time_per_record": time_per_record
                }

            result.add_metric("bulk_operation_performance", bulk_performance)

            # Test concurrent connections
            concurrent_result = await self.db_tester.stress_test_concurrent_connections(
                concurrent_connections=10, duration_seconds=15
            )

            result.add_metric("concurrent_connections", {
                "queries_per_second": concurrent_result["queries_per_second"],
                "error_rate": concurrent_result["error_rate"],
                "avg_connection_time": concurrent_result["avg_connection_time"]
            })

            if concurrent_result["error_rate"] > RUNNER_CONFIG["performance_thresholds"]["error_rate_threshold"]:
                result.add_warning(f"High error rate in concurrent test: {concurrent_result['error_rate']:.2%}")

            result.mark_complete(True)

        except Exception as e:
            result.add_error(f"Database performance test failed: {str(e)}")
            result.mark_complete(False)

        self.results.append(result)

    async def _run_memory_performance_tests(self):
        """Run memory performance tests"""
        result = PerformanceTestResult("memory_performance")

        try:
            import psutil

            # Initial memory measurement
            initial_memory = psutil.virtual_memory().used / (1024 * 1024)
            result.add_metric("initial_memory_mb", initial_memory)

            # Simulate memory-intensive operations
            memory_measurements = []

            for i in range(10):
                # Simulate some operations
                await self.api_tester.measure_endpoint_performance("/health")

                current_memory = psutil.virtual_memory().used / (1024 * 1024)
                memory_measurements.append(current_memory)

                await asyncio.sleep(0.1)

            # Final memory measurement
            final_memory = psutil.virtual_memory().used / (1024 * 1024)
            memory_increase = final_memory - initial_memory

            result.add_metric("final_memory_mb", final_memory)
            result.add_metric("memory_increase_mb", memory_increase)
            result.add_metric("peak_memory_mb", max(memory_measurements))
            result.add_metric("avg_memory_mb", statistics.mean(memory_measurements))

            # Check memory thresholds
            if memory_increase > 100:  # More than 100MB increase
                result.add_warning(f"High memory increase: {memory_increase:.1f}MB")

            if max(memory_measurements) > RUNNER_CONFIG["performance_thresholds"]["memory_usage_mb"]:
                result.add_warning(f"Peak memory usage exceeded threshold: {max(memory_measurements):.1f}MB")

            result.mark_complete(True)

        except Exception as e:
            result.add_error(f"Memory performance test failed: {str(e)}")
            result.mark_complete(False)

        self.results.append(result)

    async def _run_concurrent_load_tests(self):
        """Run concurrent load tests"""
        result = PerformanceTestResult("concurrent_load")

        try:
            await self.api_tester.setup()

            # Test concurrent API and database operations
            concurrent_results = []

            async def concurrent_api_worker(worker_id: int):
                """Worker for concurrent API testing"""
                start_time = time.time()
                requests_made = 0
                errors = 0

                while time.time() - start_time < 20:  # 20 seconds
                    try:
                        await self.api_tester.measure_endpoint_performance("/health")
                        requests_made += 1
                    except Exception:
                        errors += 1

                    await asyncio.sleep(0.1)

                return {
                    "worker_id": worker_id,
                    "requests_made": requests_made,
                    "errors": errors,
                    "duration": time.time() - start_time
                }

            async def concurrent_db_worker(worker_id: int):
                """Worker for concurrent database testing"""
                await self.db_tester.db.connect()
                start_time = time.time()
                queries_made = 0
                errors = 0

                try:
                    while time.time() - start_time < 20:  # 20 seconds
                        try:
                            await self.db_tester.db.execute_query(f"SELECT * FROM test WHERE id = {worker_id}")
                            queries_made += 1
                        except Exception:
                            errors += 1

                        await asyncio.sleep(0.1)
                finally:
                    await self.db_tester.db.disconnect()

                return {
                    "worker_id": worker_id,
                    "queries_made": queries_made,
                    "errors": errors,
                    "duration": time.time() - start_time
                }

            # Start concurrent workers
            api_workers = [concurrent_api_worker(i) for i in range(5)]
            db_workers = [concurrent_db_worker(i) for i in range(3)]

            all_workers = api_workers + db_workers
            worker_results = await asyncio.gather(*all_workers, return_exceptions=True)

            # Process results
            api_results = [r for r in worker_results[:5] if isinstance(r, dict)]
            db_results = [r for r in worker_results[5:] if isinstance(r, dict)]

            total_api_requests = sum(r["requests_made"] for r in api_results)
            total_api_errors = sum(r["errors"] for r in api_results)
            total_db_queries = sum(r["queries_made"] for r in db_results)
            total_db_errors = sum(r["errors"] for r in db_results)

            result.add_metric("concurrent_api_performance", {
                "total_requests": total_api_requests,
                "total_errors": total_api_errors,
                "error_rate": total_api_errors / max(total_api_requests, 1),
                "requests_per_second": total_api_requests / 20
            })

            result.add_metric("concurrent_db_performance", {
                "total_queries": total_db_queries,
                "total_errors": total_db_errors,
                "error_rate": total_db_errors / max(total_db_queries, 1),
                "queries_per_second": total_db_queries / 20
            })

            # Check error rates
            api_error_rate = total_api_errors / max(total_api_requests, 1)
            db_error_rate = total_db_errors / max(total_db_queries, 1)

            if api_error_rate > RUNNER_CONFIG["performance_thresholds"]["error_rate_threshold"]:
                result.add_warning(f"High API error rate in concurrent test: {api_error_rate:.2%}")

            if db_error_rate > RUNNER_CONFIG["performance_thresholds"]["error_rate_threshold"]:
                result.add_warning(f"High database error rate in concurrent test: {db_error_rate:.2%}")

            result.mark_complete(True)

        except Exception as e:
            result.add_error(f"Concurrent load test failed: {str(e)}")
            result.mark_complete(False)

        finally:
            await self.api_tester.cleanup()

        self.results.append(result)

    async def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": sum(1 for r in self.results if r.success),
                "failed_tests": sum(1 for r in self.results if not r.success),
                "total_warnings": sum(len(r.warnings) for r in self.results),
                "total_errors": sum(len(r.errors) for r in self.results)
            },
            "test_results": [result.to_dict() for result in self.results],
            "performance_analysis": {},
            "recommendations": [],
            "thresholds": RUNNER_CONFIG["performance_thresholds"]
        }

        # Analyze overall performance
        api_result = next((r for r in self.results if r.test_name == "api_performance"), None)
        db_result = next((r for r in self.results if r.test_name == "database_performance"), None)

        if api_result and api_result.success:
            report["performance_analysis"]["api"] = {
                "health_endpoint_time": api_result.metrics.get("health_endpoint_response_time"),
                "projects_endpoint_time": api_result.metrics.get("projects_endpoint_response_time"),
                "load_test_summary": api_result.metrics.get("load_test_results")
            }

        if db_result and db_result.success:
            report["performance_analysis"]["database"] = {
                "connection_time": db_result.metrics.get("avg_connection_time"),
                "query_performance": db_result.metrics.get("query_performance_by_type"),
                "concurrent_performance": db_result.metrics.get("concurrent_connections")
            }

        # Generate recommendations
        recommendations = []

        # Check for performance issues
        total_warnings = sum(len(r.warnings) for r in self.results)
        if total_warnings > 5:
            recommendations.append("Multiple performance warnings detected - review and optimize")

        if any("slow" in warning.lower() for r in self.results for warning in r.warnings):
            recommendations.append("Address slow response times across the application")

        if any("error rate" in warning.lower() for r in self.results for warning in r.warnings):
            recommendations.append("Investigate and fix high error rates under load")

        failed_tests = [r.test_name for r in self.results if not r.success]
        if failed_tests:
            recommendations.append(f"Fix failed test suites: {', '.join(failed_tests)}")

        report["recommendations"] = recommendations

        # Save report to files
        await self._save_report(report)

        return report

    async def _save_report(self, report: Dict[str, Any]):
        """Save performance report to files"""
        output_dir = Path(RUNNER_CONFIG["output_directory"])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        json_path = output_dir / f"performance_report_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        # Save simple text summary
        summary_path = output_dir / f"performance_summary_{timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write("MyAgent Performance Test Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Test Run: {report.get('test_run_timestamp', 'Unknown')}\n")
            f.write(f"Duration: {report.get('test_run_duration', 0):.1f} seconds\n\n")

            summary = report["summary"]
            f.write(f"Tests: {summary['successful_tests']}/{summary['total_tests']} passed\n")
            f.write(f"Warnings: {summary['total_warnings']}\n")
            f.write(f"Errors: {summary['total_errors']}\n\n")

            if report["recommendations"]:
                f.write("Recommendations:\n")
                for rec in report["recommendations"]:
                    f.write(f"- {rec}\n")

        logging.info(f"Performance report saved to {json_path}")

# Main execution function
async def run_performance_tests():
    """Main function to run performance tests"""
    runner = PerformanceTestRunner()
    report = await runner.run_all_tests()

    print("\nPerformance Test Summary:")
    print(f"Tests: {report['summary']['successful_tests']}/{report['summary']['total_tests']} passed")
    print(f"Warnings: {report['summary']['total_warnings']}")
    print(f"Errors: {report['summary']['total_errors']}")

    if report["recommendations"]:
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"- {rec}")

    return report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_performance_tests())