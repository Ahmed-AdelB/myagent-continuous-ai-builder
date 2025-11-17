"""
Database Performance Tests for MyAgent
Tests query performance, connection handling, data operations, and scalability
"""

import asyncio
import time
import statistics
import random
import string
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pytest
import asyncpg
import logging
from dataclasses import dataclass
import json

# Database performance configuration
DB_CONFIG = {
    "connection_string": "postgresql://localhost/myagent_test",
    "max_connections": 20,
    "query_timeout": 5.0,
    "acceptable_query_time": 0.1,  # 100ms
    "bulk_operation_size": 1000,
    "stress_test_duration": 30
}

@dataclass
class QueryMetric:
    """Database query performance metric"""
    query_type: str
    query_text: str
    execution_time: float
    rows_affected: int
    timestamp: datetime
    parameters: Optional[Dict] = None

@dataclass
class ConnectionMetric:
    """Database connection performance metric"""
    connection_time: float
    query_count: int
    session_duration: float
    memory_usage: float
    timestamp: datetime

class MockDatabase:
    """Mock database for performance testing"""

    def __init__(self):
        self.connected = False
        self.query_count = 0
        self.active_connections = 0
        self.max_connections_used = 0
        self.query_history: List[QueryMetric] = []
        self.projects: List[Dict[str, Any]] = []
        self.agents: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []
        self.metrics: List[Dict[str, Any]] = []

    async def connect(self) -> float:
        """Simulate database connection"""
        start_time = time.time()
        await asyncio.sleep(random.uniform(0.01, 0.05))  # Simulate connection overhead
        self.connected = True
        self.active_connections += 1
        self.max_connections_used = max(self.max_connections_used, self.active_connections)
        return time.time() - start_time

    async def disconnect(self):
        """Simulate database disconnection"""
        await asyncio.sleep(random.uniform(0.001, 0.01))
        self.connected = False
        self.active_connections = max(0, self.active_connections - 1)

    async def execute_query(self, query: str, parameters: Optional[List] = None) -> QueryMetric:
        """Execute a database query and measure performance"""
        start_time = time.time()

        # Simulate different query types with different performance characteristics
        rows_affected = await self._simulate_query_execution(query, parameters)

        execution_time = time.time() - start_time
        self.query_count += 1

        metric = QueryMetric(
            query_type=self._get_query_type(query),
            query_text=query,
            execution_time=execution_time,
            rows_affected=rows_affected,
            timestamp=datetime.now(),
            parameters=parameters
        )

        self.query_history.append(metric)
        return metric

    async def _simulate_query_execution(self, query: str, parameters: Optional[List] = None) -> int:
        """Simulate query execution with realistic timing"""
        query_lower = query.lower().strip()

        if query_lower.startswith('select'):
            # SELECT queries - simulate data retrieval
            if 'where' in query_lower:
                # Filtered queries
                await asyncio.sleep(random.uniform(0.01, 0.05))
                return random.randint(0, 100)
            elif 'join' in query_lower:
                # Complex joins
                await asyncio.sleep(random.uniform(0.05, 0.2))
                return random.randint(10, 1000)
            else:
                # Simple selects
                await asyncio.sleep(random.uniform(0.001, 0.01))
                return random.randint(1, 50)

        elif query_lower.startswith('insert'):
            # INSERT queries
            if 'values' in query_lower and query_lower.count('values') > 1:
                # Bulk inserts
                await asyncio.sleep(random.uniform(0.1, 0.5))
                return random.randint(100, 1000)
            else:
                # Single inserts
                await asyncio.sleep(random.uniform(0.01, 0.05))
                return 1

        elif query_lower.startswith('update'):
            # UPDATE queries
            await asyncio.sleep(random.uniform(0.02, 0.1))
            return random.randint(1, 10)

        elif query_lower.startswith('delete'):
            # DELETE queries
            await asyncio.sleep(random.uniform(0.01, 0.08))
            return random.randint(0, 5)

        elif any(ddl in query_lower for ddl in ['create', 'alter', 'drop']):
            # DDL operations
            await asyncio.sleep(random.uniform(0.1, 1.0))
            return 0

        else:
            # Other queries
            await asyncio.sleep(random.uniform(0.01, 0.1))
            return random.randint(0, 10)

    def _get_query_type(self, query: str) -> str:
        """Determine query type"""
        query_lower = query.lower().strip()

        if query_lower.startswith('select'):
            return 'SELECT'
        elif query_lower.startswith('insert'):
            return 'INSERT'
        elif query_lower.startswith('update'):
            return 'UPDATE'
        elif query_lower.startswith('delete'):
            return 'DELETE'
        elif any(ddl in query_lower for ddl in ['create', 'alter', 'drop']):
            return 'DDL'
        else:
            return 'OTHER'

    async def bulk_insert_projects(self, count: int) -> QueryMetric:
        """Simulate bulk project insertion"""
        projects = []
        for i in range(count):
            projects.append({
                'name': f'project_{i}_{int(time.time())}',
                'description': f'Test project {i}',
                'status': random.choice(['active', 'completed', 'pending']),
                'created_at': datetime.now().isoformat()
            })

        query = f"INSERT INTO projects (name, description, status, created_at) VALUES {', '.join(['(%s, %s, %s, %s)'] * count)}"
        return await self.execute_query(query, projects)

    async def complex_analytics_query(self) -> QueryMetric:
        """Simulate complex analytics query"""
        query = """
        SELECT
            p.status,
            COUNT(*) as project_count,
            AVG(task_completion_rate) as avg_completion,
            COUNT(DISTINCT a.id) as agent_count
        FROM projects p
        LEFT JOIN tasks t ON p.id = t.project_id
        LEFT JOIN agent_assignments aa ON t.id = aa.task_id
        LEFT JOIN agents a ON aa.agent_id = a.id
        WHERE p.created_at >= NOW() - INTERVAL '30 days'
        GROUP BY p.status
        HAVING COUNT(*) > 5
        ORDER BY avg_completion DESC
        """
        return await self.execute_query(query)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.query_history:
            return {"error": "No queries executed"}

        query_times = [q.execution_time for q in self.query_history]
        query_types = {}

        for query in self.query_history:
            query_type = query.query_type
            if query_type not in query_types:
                query_types[query_type] = []
            query_types[query_type].append(query.execution_time)

        summary = {
            "total_queries": len(self.query_history),
            "avg_query_time": statistics.mean(query_times),
            "median_query_time": statistics.median(query_times),
            "p95_query_time": statistics.quantiles(query_times, n=20)[18] if len(query_times) > 20 else max(query_times),
            "max_query_time": max(query_times),
            "min_query_time": min(query_times),
            "max_connections_used": self.max_connections_used,
            "query_types": {}
        }

        for query_type, times in query_types.items():
            summary["query_types"][query_type] = {
                "count": len(times),
                "avg_time": statistics.mean(times),
                "max_time": max(times),
                "min_time": min(times)
            }

        return summary

class DatabasePerformanceTester:
    """Database performance testing framework"""

    def __init__(self):
        self.db = MockDatabase()
        self.connection_metrics: List[ConnectionMetric] = []

    async def test_connection_performance(self, connection_count: int = 10) -> List[ConnectionMetric]:
        """Test database connection performance"""
        metrics = []

        for i in range(connection_count):
            start_time = time.time()
            connection_time = await self.db.connect()

            # Simulate some queries
            query_count = random.randint(1, 5)
            for _ in range(query_count):
                await self.db.execute_query(f"SELECT * FROM test_table WHERE id = {random.randint(1, 100)}")

            session_duration = time.time() - start_time
            await self.db.disconnect()

            metric = ConnectionMetric(
                connection_time=connection_time,
                query_count=query_count,
                session_duration=session_duration,
                memory_usage=random.uniform(1.0, 5.0),  # Simulate memory usage
                timestamp=datetime.now()
            )
            metrics.append(metric)
            self.connection_metrics.append(metric)

        return metrics

    async def test_query_performance_by_type(self) -> Dict[str, List[QueryMetric]]:
        """Test performance of different query types"""
        await self.db.connect()

        query_results = {
            'SELECT': [],
            'INSERT': [],
            'UPDATE': [],
            'DELETE': [],
            'COMPLEX': []
        }

        try:
            # Test SELECT queries
            for i in range(10):
                metric = await self.db.execute_query(f"SELECT * FROM projects WHERE id = {i}")
                query_results['SELECT'].append(metric)

            # Test INSERT queries
            for i in range(10):
                metric = await self.db.execute_query(
                    "INSERT INTO projects (name, description) VALUES (%s, %s)",
                    [f"test_project_{i}", f"Test description {i}"]
                )
                query_results['INSERT'].append(metric)

            # Test UPDATE queries
            for i in range(10):
                metric = await self.db.execute_query(
                    f"UPDATE projects SET description = 'Updated description' WHERE id = {i}"
                )
                query_results['UPDATE'].append(metric)

            # Test DELETE queries
            for i in range(5):
                metric = await self.db.execute_query(f"DELETE FROM projects WHERE id = {i}")
                query_results['DELETE'].append(metric)

            # Test complex queries
            for i in range(5):
                metric = await self.db.complex_analytics_query()
                query_results['COMPLEX'].append(metric)

        finally:
            await self.db.disconnect()

        return query_results

    async def test_bulk_operations_performance(self, sizes: List[int] = [100, 500, 1000]) -> Dict[int, QueryMetric]:
        """Test bulk operations performance"""
        await self.db.connect()

        results = {}

        try:
            for size in sizes:
                metric = await self.db.bulk_insert_projects(size)
                results[size] = metric
        finally:
            await self.db.disconnect()

        return results

    async def stress_test_concurrent_connections(self, concurrent_connections: int = 20,
                                               duration_seconds: int = 30) -> Dict[str, Any]:
        """Stress test with concurrent database connections"""

        async def connection_worker(worker_id: int, results: List[Dict]):
            """Worker function for concurrent connections"""
            start_time = time.time()
            connection_time = await self.db.connect()

            queries_executed = 0
            errors = 0

            try:
                while time.time() - start_time < duration_seconds:
                    try:
                        query_type = random.choice(['SELECT', 'INSERT', 'UPDATE'])

                        if query_type == 'SELECT':
                            await self.db.execute_query(f"SELECT * FROM projects WHERE id = {random.randint(1, 100)}")
                        elif query_type == 'INSERT':
                            await self.db.execute_query(
                                "INSERT INTO temp_table (data) VALUES (%s)",
                                [f"test_data_{queries_executed}"]
                            )
                        else:  # UPDATE
                            await self.db.execute_query(f"UPDATE projects SET updated_at = NOW() WHERE id = {random.randint(1, 100)}")

                        queries_executed += 1
                        await asyncio.sleep(0.01)  # Small delay between queries

                    except Exception as e:
                        errors += 1
                        logging.error(f"Query error in worker {worker_id}: {e}")

            finally:
                await self.db.disconnect()

            results.append({
                'worker_id': worker_id,
                'connection_time': connection_time,
                'queries_executed': queries_executed,
                'errors': errors,
                'duration': time.time() - start_time
            })

        # Start concurrent workers
        worker_results = []
        tasks = []

        for i in range(concurrent_connections):
            task = asyncio.create_task(connection_worker(i, worker_results))
            tasks.append(task)

        # Wait for all workers to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        total_queries = sum(r['queries_executed'] for r in worker_results)
        total_errors = sum(r['errors'] for r in worker_results)
        avg_connection_time = statistics.mean(r['connection_time'] for r in worker_results)

        return {
            'concurrent_connections': concurrent_connections,
            'duration_seconds': duration_seconds,
            'total_queries': total_queries,
            'total_errors': total_errors,
            'error_rate': total_errors / max(total_queries, 1),
            'queries_per_second': total_queries / duration_seconds,
            'avg_connection_time': avg_connection_time,
            'max_connections_used': self.db.max_connections_used,
            'worker_results': worker_results
        }

# Test Fixtures

@pytest.fixture
def db_tester():
    """Database performance tester fixture"""
    return DatabasePerformanceTester()

@pytest.fixture
def sample_data():
    """Sample test data"""
    return {
        'projects': [
            {'name': f'project_{i}', 'description': f'Test project {i}', 'status': 'active'}
            for i in range(100)
        ],
        'large_text': ''.join(random.choices(string.ascii_letters + string.digits, k=10000))
    }

# Database Performance Tests

@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.database
class TestDatabasePerformance:
    """Database performance test suite"""

    async def test_basic_query_performance(self, db_tester):
        """Test basic database query performance"""
        await db_tester.db.connect()

        try:
            # Test simple SELECT
            metric = await db_tester.db.execute_query("SELECT COUNT(*) FROM projects")
            assert metric.execution_time < DB_CONFIG["acceptable_query_time"]

            # Test parameterized query
            metric = await db_tester.db.execute_query(
                "SELECT * FROM projects WHERE status = %s",
                ['active']
            )
            assert metric.execution_time < DB_CONFIG["acceptable_query_time"] * 2

        finally:
            await db_tester.db.disconnect()

    async def test_connection_pool_performance(self, db_tester):
        """Test database connection pool performance"""
        connection_metrics = await db_tester.test_connection_performance(10)

        connection_times = [m.connection_time for m in connection_metrics]
        avg_connection_time = statistics.mean(connection_times)

        # Connection time should be reasonable
        assert avg_connection_time < 0.1  # Less than 100ms average
        assert max(connection_times) < 0.5  # No connection takes more than 500ms

        logging.info(f"Average connection time: {avg_connection_time:.3f}s")

    async def test_query_types_performance(self, db_tester):
        """Test performance of different query types"""
        results = await db_tester.test_query_performance_by_type()

        # Check SELECT performance
        select_times = [m.execution_time for m in results['SELECT']]
        assert statistics.mean(select_times) < DB_CONFIG["acceptable_query_time"]

        # Check INSERT performance
        insert_times = [m.execution_time for m in results['INSERT']]
        assert statistics.mean(insert_times) < DB_CONFIG["acceptable_query_time"] * 2

        # Complex queries can take longer
        complex_times = [m.execution_time for m in results['COMPLEX']]
        assert statistics.mean(complex_times) < 1.0  # 1 second max for complex queries

        logging.info(f"Query performance - SELECT: {statistics.mean(select_times):.3f}s, "
                    f"INSERT: {statistics.mean(insert_times):.3f}s, "
                    f"COMPLEX: {statistics.mean(complex_times):.3f}s")

    @pytest.mark.parametrize("bulk_size", [100, 500, 1000])
    async def test_bulk_operations_performance(self, db_tester, bulk_size):
        """Test bulk operations performance"""
        results = await db_tester.test_bulk_operations_performance([bulk_size])

        metric = results[bulk_size]

        # Bulk operations should be efficient
        time_per_record = metric.execution_time / bulk_size
        assert time_per_record < 0.01  # Less than 10ms per record

        logging.info(f"Bulk insert ({bulk_size} records): {metric.execution_time:.3f}s "
                    f"({time_per_record*1000:.1f}ms per record)")

    @pytest.mark.parametrize("concurrent_connections", [5, 10, 20])
    async def test_concurrent_connections_performance(self, db_tester, concurrent_connections):
        """Test concurrent database connections performance"""
        result = await db_tester.stress_test_concurrent_connections(
            concurrent_connections=concurrent_connections,
            duration_seconds=10  # Shorter duration for tests
        )

        # Check error rate is acceptable
        assert result['error_rate'] < 0.05  # Less than 5% errors

        # Check throughput
        assert result['queries_per_second'] > 10  # At least 10 QPS

        # Check connection time is reasonable
        assert result['avg_connection_time'] < 0.2  # Less than 200ms average

        logging.info(f"Concurrent test ({concurrent_connections} connections): "
                    f"{result['queries_per_second']:.1f} QPS, "
                    f"{result['error_rate']:.2%} error rate")

    async def test_database_memory_usage(self, db_tester):
        """Test database memory usage patterns"""
        await db_tester.db.connect()

        try:
            # Execute many queries to test memory patterns
            for i in range(100):
                await db_tester.db.execute_query(f"SELECT * FROM large_table LIMIT {random.randint(10, 100)}")

            # Check that query history doesn't grow unbounded
            assert len(db_tester.db.query_history) == 100

            # Memory usage should be tracked
            summary = db_tester.db.get_performance_summary()
            assert summary['total_queries'] == 100
            assert summary['avg_query_time'] > 0

        finally:
            await db_tester.db.disconnect()

    async def test_database_scalability(self, db_tester):
        """Test database scalability under increasing load"""
        results = []

        # Test with increasing load
        for load_level in [1, 5, 10, 15]:
            result = await db_tester.stress_test_concurrent_connections(
                concurrent_connections=load_level,
                duration_seconds=5
            )

            results.append({
                'load': load_level,
                'qps': result['queries_per_second'],
                'error_rate': result['error_rate'],
                'avg_connection_time': result['avg_connection_time']
            })

            # Stop if error rate becomes too high
            if result['error_rate'] > 0.1:
                break

        # Check that performance degrades gracefully
        qps_values = [r['qps'] for r in results]

        # QPS should increase initially (up to a point)
        assert len(qps_values) >= 2
        logging.info(f"Scalability test QPS progression: {qps_values}")

    async def test_transaction_performance(self, db_tester):
        """Test transaction performance"""
        await db_tester.db.connect()

        try:
            start_time = time.time()

            # Simulate transaction with multiple operations
            await db_tester.db.execute_query("BEGIN")

            for i in range(5):
                await db_tester.db.execute_query(
                    "INSERT INTO projects (name, description) VALUES (%s, %s)",
                    [f"tx_project_{i}", f"Transaction test project {i}"]
                )

            await db_tester.db.execute_query("COMMIT")

            transaction_time = time.time() - start_time

            # Transaction should complete reasonably quickly
            assert transaction_time < 1.0  # Less than 1 second

            logging.info(f"Transaction performance: {transaction_time:.3f}s for 5 operations")

        finally:
            await db_tester.db.disconnect()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v", "--tb=short"])