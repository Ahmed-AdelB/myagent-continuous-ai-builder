"""
Tests for Project Ledger Persistence

Validates that project history is correctly:
- Recorded to database
- Retrieved with full fidelity
- Versioned properly
- Queryable for analysis
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from core.memory.project_ledger import ProjectLedger, ProjectVersion


class TestProjectLedger:
    """Test suite for ProjectLedger persistence"""

    @pytest.fixture
    async def ledger(self, tmp_path):
        """Create ProjectLedger with temporary database"""
        db_path = tmp_path / "test_ledger.db"
        ledger = ProjectLedger(db_path=str(db_path))
        await ledger.initialize()
        return ledger

    @pytest.fixture
    def sample_version(self):
        """Sample project version for testing"""
        return ProjectVersion(
            version_number=1,
            timestamp=datetime.now(),
            files_changed=['file1.py', 'file2.py'],
            changes_summary='Added feature X',
            metrics={'test_coverage': 85, 'bugs': 3},
            agent='coder'
        )

    @pytest.mark.asyncio
    async def test_initialization(self, ledger):
        """Test ledger initializes correctly"""
        assert ledger is not None
        assert await ledger.is_initialized()

    @pytest.mark.asyncio
    async def test_record_version(self, ledger, sample_version):
        """Test recording a new version"""
        version_id = await ledger.record_version(sample_version)

        assert version_id is not None
        assert version_id > 0

    @pytest.mark.asyncio
    async def test_retrieve_version(self, ledger, sample_version):
        """Test retrieving a specific version"""
        version_id = await ledger.record_version(sample_version)

        retrieved = await ledger.get_version(version_id)

        assert retrieved is not None
        assert retrieved.version_number == sample_version.version_number
        assert retrieved.changes_summary == sample_version.changes_summary

    @pytest.mark.asyncio
    async def test_version_history(self, ledger):
        """Test retrieving complete version history"""
        # Record multiple versions
        for i in range(5):
            version = ProjectVersion(
                version_number=i + 1,
                timestamp=datetime.now(),
                files_changed=[f'file{i}.py'],
                changes_summary=f'Change {i}',
                metrics={'iteration': i},
                agent='coder'
            )
            await ledger.record_version(version)

        # Get history
        history = await ledger.get_history(limit=10)

        assert len(history) == 5
        assert history[0].version_number == 5  # Most recent first
        assert history[-1].version_number == 1

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, ledger):
        """Test that metrics are tracked across versions"""
        versions = [
            ProjectVersion(
                version_number=i + 1,
                timestamp=datetime.now(),
                files_changed=[],
                changes_summary=f'Version {i}',
                metrics={'test_coverage': 70 + i * 5},
                agent='coder'
            )
            for i in range(4)
        ]

        for v in versions:
            await ledger.record_version(v)

        # Get metrics trend
        coverage_trend = await ledger.get_metric_trend('test_coverage')

        assert len(coverage_trend) == 4
        assert coverage_trend[0]['value'] == 70
        assert coverage_trend[-1]['value'] == 85  # 70 + 3*5

    @pytest.mark.asyncio
    async def test_file_change_tracking(self, ledger):
        """Test tracking which files changed in each version"""
        version = ProjectVersion(
            version_number=1,
            timestamp=datetime.now(),
            files_changed=['core/agent.py', 'tests/test_agent.py'],
            changes_summary='Modified agent',
            metrics={},
            agent='coder'
        )

        version_id = await ledger.record_version(version)
        retrieved = await ledger.get_version(version_id)

        assert 'core/agent.py' in retrieved.files_changed
        assert 'tests/test_agent.py' in retrieved.files_changed

    @pytest.mark.asyncio
    async def test_version_diff(self, ledger):
        """Test calculating diff between versions"""
        v1 = ProjectVersion(
            version_number=1,
            timestamp=datetime.now(),
            files_changed=['file1.py'],
            changes_summary='Initial',
            metrics={'bugs': 10, 'test_coverage': 50},
            agent='coder'
        )

        v2 = ProjectVersion(
            version_number=2,
            timestamp=datetime.now(),
            files_changed=['file1.py', 'file2.py'],
            changes_summary='Fixed bugs',
            metrics={'bugs': 5, 'test_coverage': 75},
            agent='debugger'
        )

        v1_id = await ledger.record_version(v1)
        v2_id = await ledger.record_version(v2)

        diff = await ledger.get_version_diff(v1_id, v2_id)

        assert diff['metrics_changed']['bugs'] == -5  # Decreased
        assert diff['metrics_changed']['test_coverage'] == 25  # Increased
        assert len(diff['files_added']) == 1
        assert 'file2.py' in diff['files_added']

    @pytest.mark.asyncio
    async def test_search_by_agent(self, ledger):
        """Test searching versions by agent"""
        agents = ['coder', 'tester', 'debugger', 'coder']

        for i, agent in enumerate(agents):
            version = ProjectVersion(
                version_number=i + 1,
                timestamp=datetime.now(),
                files_changed=[],
                changes_summary=f'By {agent}',
                metrics={},
                agent=agent
            )
            await ledger.record_version(version)

        # Search for coder versions
        coder_versions = await ledger.search_by_agent('coder')

        assert len(coder_versions) == 2
        assert all(v.agent == 'coder' for v in coder_versions)

    @pytest.mark.asyncio
    async def test_search_by_file(self, ledger):
        """Test finding versions that modified specific file"""
        versions_data = [
            (['file1.py'], 'v1'),
            (['file2.py'], 'v2'),
            (['file1.py', 'file3.py'], 'v3'),
        ]

        for i, (files, summary) in enumerate(versions_data):
            version = ProjectVersion(
                version_number=i + 1,
                timestamp=datetime.now(),
                files_changed=files,
                changes_summary=summary,
                metrics={},
                agent='coder'
            )
            await ledger.record_version(version)

        # Find versions that modified file1.py
        file1_versions = await ledger.search_by_file('file1.py')

        assert len(file1_versions) == 2
        assert file1_versions[0].changes_summary in ['v1', 'v3']

    @pytest.mark.asyncio
    async def test_cleanup_old_versions(self, ledger):
        """Test cleaning up old versions while keeping recent ones"""
        # Create 20 versions
        for i in range(20):
            version = ProjectVersion(
                version_number=i + 1,
                timestamp=datetime.now(),
                files_changed=[],
                changes_summary=f'Version {i}',
                metrics={},
                agent='coder'
            )
            await ledger.record_version(version)

        # Cleanup, keep only last 10
        await ledger.cleanup_old_versions(keep_last=10)

        history = await ledger.get_history(limit=100)

        assert len(history) <= 10
        assert history[0].version_number == 20  # Most recent kept

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, ledger, sample_version):
        """Test that failed transactions don't corrupt ledger"""
        try:
            async with ledger.begin_transaction():
                await ledger.record_version(sample_version)
                # Simulate error
                raise Exception("Simulated error")
        except:
            pass

        # Version should not be recorded due to rollback
        history = await ledger.get_history()
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, ledger):
        """Test ledger handles concurrent writes correctly"""
        async def write_version(n):
            version = ProjectVersion(
                version_number=n,
                timestamp=datetime.now(),
                files_changed=[],
                changes_summary=f'Concurrent {n}',
                metrics={},
                agent='coder'
            )
            return await ledger.record_version(version)

        # Write 10 versions concurrently
        tasks = [write_version(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 10
        assert all(r is not None for r in results)

        # Check all recorded
        history = await ledger.get_history(limit=20)
        assert len(history) == 10
