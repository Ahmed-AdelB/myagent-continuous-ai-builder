"""
Unit tests for TaskLedger.

Tests:
- Task enqueue and state transitions
- Retry count increment
- Dead-letter queue functionality
- Task history audit trail
- Task queries by state/agent

Issue #5: Task Ledger with retries, timeouts, circuit breakers
"""

import pytest
import asyncio
import tempfile
from pathlib import Path

from core.resilience.task_ledger import TaskLedger, Task, TaskState


@pytest.fixture
async def task_ledger():
    """Create TaskLedger with temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_task_ledger.db"
        ledger = TaskLedger(db_path)
        await ledger.initialize()
        yield ledger


@pytest.mark.asyncio
async def test_enqueue_task(task_ledger):
    """Test task enqueue."""
    task_id = await task_ledger.enqueue(
        agent_name="test_agent",
        task_type="test_task",
        payload={"key": "value"},
        priority=1,
        max_retries=3
    )

    assert task_id is not None
    assert len(task_id) == 36  # UUID format

    # Verify task in database
    task = await task_ledger.get_task(task_id)
    assert task is not None
    assert task.agent_name == "test_agent"
    assert task.task_type == "test_task"
    assert task.state == TaskState.QUEUED
    assert task.priority == 1
    assert task.max_retries == 3
    assert task.retry_count == 0


@pytest.mark.asyncio
async def test_state_transition_queued_to_running(task_ledger):
    """Test QUEUED → RUNNING transition."""
    task_id = await task_ledger.enqueue(
        agent_name="test_agent",
        task_type="test_task",
        payload={}
    )

    # Transition to RUNNING
    await task_ledger.mark_running(task_id)

    task = await task_ledger.get_task(task_id)
    assert task.state == TaskState.RUNNING
    assert task.started_at is not None

    # Verify history
    history = await task_ledger.get_task_history(task_id)
    assert len(history) == 1
    assert history[0]["from_state"] == TaskState.QUEUED.value
    assert history[0]["to_state"] == TaskState.RUNNING.value


@pytest.mark.asyncio
async def test_state_transition_running_to_done(task_ledger):
    """Test RUNNING → DONE transition."""
    task_id = await task_ledger.enqueue(
        agent_name="test_agent",
        task_type="test_task",
        payload={}
    )

    await task_ledger.mark_running(task_id)
    await task_ledger.mark_done(task_id, {"result": "success"})

    task = await task_ledger.get_task(task_id)
    assert task.state == TaskState.DONE
    assert task.result == {"result": "success"}
    assert task.completed_at is not None


@pytest.mark.asyncio
async def test_state_transition_running_to_retry(task_ledger):
    """Test RUNNING → RETRY transition with retry count increment."""
    task_id = await task_ledger.enqueue(
        agent_name="test_agent",
        task_type="test_task",
        payload={},
        max_retries=3
    )

    await task_ledger.mark_running(task_id)

    # First retry
    await task_ledger.mark_retry(task_id, Exception("Transient error"))

    task = await task_ledger.get_task(task_id)
    assert task.state == TaskState.RETRY
    assert task.retry_count == 1
    assert "Transient error" in task.error


@pytest.mark.asyncio
async def test_retry_count_increment_and_exhaustion(task_ledger):
    """Test retry count increment and max retries exhaustion."""
    task_id = await task_ledger.enqueue(
        agent_name="test_agent",
        task_type="test_task",
        payload={},
        max_retries=2  # Allow 2 retries
    )

    await task_ledger.mark_running(task_id)

    # First retry
    await task_ledger.mark_retry(task_id, Exception("Error 1"))
    task = await task_ledger.get_task(task_id)
    assert task.retry_count == 1
    assert task.state == TaskState.RETRY

    # Second retry - should mark as FAILED (retry_count=2 >= max_retries=2)
    await task_ledger.mark_retry(task_id, Exception("Error 2"))
    task = await task_ledger.get_task(task_id)
    assert task.retry_count == 2
    assert task.state == TaskState.FAILED


@pytest.mark.asyncio
async def test_state_transition_running_to_blocked(task_ledger):
    """Test RUNNING → BLOCKED transition."""
    task_id = await task_ledger.enqueue(
        agent_name="test_agent",
        task_type="test_task",
        payload={}
    )

    await task_ledger.mark_running(task_id)
    await task_ledger.mark_blocked(task_id, "Circuit breaker open")

    task = await task_ledger.get_task(task_id)
    assert task.state == TaskState.BLOCKED
    assert "Circuit breaker open" in task.error


@pytest.mark.asyncio
async def test_dead_letter_queue(task_ledger):
    """Test dead-letter queue functionality."""
    task_id = await task_ledger.enqueue(
        agent_name="test_agent",
        task_type="test_task",
        payload={"important": "data"},
        max_retries=1
    )

    await task_ledger.mark_running(task_id)
    await task_ledger.mark_failed(task_id, "Permanent failure")

    # Verify task in dead-letter queue
    dlq = await task_ledger.get_dead_letter_queue()
    assert len(dlq) == 1
    assert dlq[0]["task_id"] == task_id
    assert dlq[0]["agent_name"] == "test_agent"
    assert dlq[0]["error"] == "Permanent failure"


@pytest.mark.asyncio
async def test_get_tasks_by_state(task_ledger):
    """Test querying tasks by state."""
    # Create tasks in different states
    task1_id = await task_ledger.enqueue("agent1", "task_type", {})
    task2_id = await task_ledger.enqueue("agent2", "task_type", {})
    task3_id = await task_ledger.enqueue("agent3", "task_type", {})

    await task_ledger.mark_running(task1_id)
    await task_ledger.mark_running(task2_id)
    await task_ledger.mark_done(task2_id, {})

    # Query by state
    queued_tasks = await task_ledger.get_tasks_by_state(TaskState.QUEUED)
    running_tasks = await task_ledger.get_tasks_by_state(TaskState.RUNNING)
    done_tasks = await task_ledger.get_tasks_by_state(TaskState.DONE)

    assert len(queued_tasks) == 1
    assert queued_tasks[0].task_id == task3_id

    assert len(running_tasks) == 1
    assert running_tasks[0].task_id == task1_id

    assert len(done_tasks) == 1
    assert done_tasks[0].task_id == task2_id


@pytest.mark.asyncio
async def test_get_tasks_by_agent(task_ledger):
    """Test querying tasks by agent."""
    task1_id = await task_ledger.enqueue("agent_A", "task1", {})
    task2_id = await task_ledger.enqueue("agent_A", "task2", {})
    task3_id = await task_ledger.enqueue("agent_B", "task3", {})

    tasks_A = await task_ledger.get_tasks_by_agent("agent_A")
    tasks_B = await task_ledger.get_tasks_by_agent("agent_B")

    assert len(tasks_A) == 2
    assert len(tasks_B) == 1


@pytest.mark.asyncio
async def test_task_history(task_ledger):
    """Test task history audit trail."""
    task_id = await task_ledger.enqueue("test_agent", "test_task", {})

    await task_ledger.mark_running(task_id)
    await task_ledger.mark_retry(task_id, Exception("Error"))
    await task_ledger.mark_running(task_id)
    await task_ledger.mark_done(task_id, {})

    history = await task_ledger.get_task_history(task_id)

    # Should have 4 transitions
    assert len(history) >= 4

    # Verify transition sequence
    assert history[0]["from_state"] == TaskState.QUEUED.value
    assert history[0]["to_state"] == TaskState.RUNNING.value


@pytest.mark.asyncio
async def test_get_stats(task_ledger):
    """Test statistics collection."""
    task1_id = await task_ledger.enqueue("agent1", "task1", {})
    task2_id = await task_ledger.enqueue("agent2", "task2", {})

    await task_ledger.mark_running(task1_id)
    await task_ledger.mark_done(task1_id, {})

    await task_ledger.mark_running(task2_id)
    await task_ledger.mark_failed(task2_id, "Error")

    stats = await task_ledger.get_stats()

    assert stats[TaskState.DONE.value] == 1
    assert stats[TaskState.FAILED.value] == 1
    assert stats["dead_letter_queue"] == 1


@pytest.mark.asyncio
async def test_task_priority_ordering(task_ledger):
    """Test that tasks are ordered by priority."""
    low_priority_id = await task_ledger.enqueue("agent", "task", {}, priority=10)
    high_priority_id = await task_ledger.enqueue("agent", "task", {}, priority=1)
    medium_priority_id = await task_ledger.enqueue("agent", "task", {}, priority=5)

    tasks = await task_ledger.get_tasks_by_state(TaskState.QUEUED)

    # Should be ordered by priority DESC (1, 5, 10)
    assert tasks[0].task_id == high_priority_id
    assert tasks[1].task_id == medium_priority_id
    assert tasks[2].task_id == low_priority_id
