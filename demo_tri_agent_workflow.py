#!/usr/bin/env python3
"""
Demo script to test Tri-Agent SDLC workflow

This script demonstrates the tri-agent collaboration system:
- Claude Code (Sonnet 4.5): Requirements analysis and integration
- Codex (o1): Code implementation
- Gemini (1.5 Pro): Code review and approval

Tests the complete 5-phase SDLC with consensus voting.
"""

import asyncio
from pathlib import Path
from loguru import logger
from core.orchestrator.tri_agent_sdlc import TriAgentSDLCOrchestrator


async def main():
    logger.info("=" * 80)
    logger.info("TRI-AGENT SDLC WORKFLOW DEMONSTRATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Agents:")
    logger.info("  🤖 Claude Code (Sonnet 4.5) - Requirements & Integration")
    logger.info("  🤖 Codex (o1 via codex CLI) - Code Implementation")
    logger.info("  🤖 Gemini (1.5 Pro via SDK) - Code Review & Approval")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    # Initialize tri-agent SDLC orchestrator
    orchestrator = TriAgentSDLCOrchestrator(
        project_name="MyAgent-TriAgent-Demo",
        working_dir=Path.cwd()
    )

    logger.info("✅ Tri-Agent SDLC Orchestrator initialized")
    logger.info("")

    # Create a work item for remaining frontend component tests
    work_item_id = orchestrator.add_work_item(
        title="Complete remaining frontend component tests",
        description="""
Create comprehensive tests for the remaining 5 React components:
1. Dashboard.jsx - Main dashboard with live indicators
2. ErrorAnalytics.jsx - Error analysis and patterns
3. IterationHistory.jsx - Iteration timeline
4. MetricsView.jsx - Metrics container/tabs
5. AgentStatus.jsx - Agent status display

Requirements:
- Use Vitest + @testing-library/react
- Follow existing test patterns from ProjectManager.test.jsx
- Aim for 80%+ coverage per component
- Test user interactions with fireEvent
- Mock fetch calls and WebSocket connections
- Use waitFor for async operations

Acceptance Criteria:
- All 5 components have test files
- Each component has 6-8 test cases
- Tests cover happy path, error cases, and user interactions
- All tests pass
- Code coverage meets 80% threshold
        """.strip(),
        priority=2,  # HIGH
        file_paths=[
            "frontend/src/components/__tests__/Dashboard.test.jsx",
            "frontend/src/components/__tests__/ErrorAnalytics.test.jsx",
            "frontend/src/components/__tests__/IterationHistory.test.jsx",
            "frontend/src/components/__tests__/MetricsView.test.jsx",
            "frontend/src/components/__tests__/AgentStatus.test.jsx"
        ],
        acceptance_criteria=[
            "All 5 test files created with proper structure",
            "Each component has 6-8 comprehensive test cases",
            "Tests cover rendering, user interactions, and error handling",
            "All tests pass when run with `npm run test`",
            "Code coverage >= 80% for each component",
            "Tests follow existing patterns from ProjectManager.test.jsx"
        ]
    )

    logger.info(f"✅ Created work item: {work_item_id}")
    logger.info("")

    # Process the work item through full 5-phase SDLC
    logger.info("🚀 Starting Tri-Agent SDLC workflow...")
    logger.info("")
    logger.info("Phases:")
    logger.info("  1. REQUIREMENTS - Claude analyzes, all agents vote (3/3 required)")
    logger.info("  2. DESIGN - Create implementation plan")
    logger.info("  3. DEVELOPMENT - Codex generates code")
    logger.info("  4. TESTING - Run tests and verify")
    logger.info("  5. DEPLOYMENT - Git commit with tri-agent approval")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    try:
        result = await orchestrator.process_work_item(work_item_id)

        logger.info("")
        logger.info("=" * 80)
        logger.info("TRI-AGENT WORKFLOW RESULT")
        logger.info("=" * 80)
        logger.info(f"Success: {result.get('success')}")
        logger.info(f"Phase Reached: {result.get('phase')}")
        logger.info(f"Message: {result.get('message', 'N/A')}")

        if result.get("success"):
            logger.success("🎉 Tri-Agent workflow completed successfully!")

            logger.info("")
            logger.info("Metrics:")
            metrics = orchestrator.get_metrics()
            logger.info(f"  Total Items: {metrics.get('total_items')}")
            logger.info(f"  Completed: {metrics.get('completed_items')}")
            logger.info(f"  Failed: {metrics.get('failed_items')}")
            logger.info(f"  Unanimous Approvals: {metrics.get('unanimous_approvals')}")
            logger.info(f"  Revisions Required: {metrics.get('revisions_required')}")
        else:
            logger.error(f"❌ Tri-Agent workflow failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("")
    logger.info("=" * 80)
    logger.info("WORKFLOW DEMONSTRATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
