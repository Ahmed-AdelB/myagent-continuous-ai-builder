"""
Claude Code Self Agent - Represents Claude's native capabilities in tri-agent system
"""

import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger
from datetime import datetime


class ClaudeCodeSelfAgent:
    """
    Self-referential agent representing Claude Code's native capabilities.

    This agent handles:
    - Orchestration and coordination
    - File operations (read, write, edit)
    - Git operations
    - Test execution
    - Code analysis
    - Integration between Aider and Gemini
    """

    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = working_dir or Path.cwd()
        self.name = "Claude Code (Sonnet 4.5)"

        # Track metrics
        self.metrics = {
            "orchestration_decisions": 0,
            "files_analyzed": 0,
            "files_modified": 0,
            "integrations_performed": 0,
            "tests_executed": 0
        }

        logger.info("Initialized ClaudeCodeSelfAgent (Sonnet 4.5)")

    async def analyze_requirements(
        self,
        issue_description: str,
        codebase_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze requirements for a given issue.

        Args:
            issue_description: Description of the issue/feature
            codebase_context: Optional context about related code

        Returns:
            Dict with requirements analysis
        """
        self.metrics["orchestration_decisions"] += 1

        logger.info(f"Analyzing requirements: {issue_description[:100]}...")

        # Use Claude's native analysis capabilities
        requirements = {
            "success": True,
            "issue": issue_description,
            "dependencies": self._identify_dependencies(issue_description),
            "affected_files": self._identify_affected_files(issue_description, codebase_context),
            "complexity": self._assess_complexity(issue_description),
            "estimated_effort": self._estimate_effort(issue_description),
            "prerequisites": self._identify_prerequisites(issue_description),
            "acceptance_criteria": self._generate_acceptance_criteria(issue_description)
        }

        logger.info(f"Requirements analysis complete: {requirements['complexity']} complexity")

        return requirements

    def _identify_dependencies(self, description: str) -> List[str]:
        """Identify dependencies for the task"""
        dependencies = []

        # Check for common dependency keywords
        if "test" in description.lower():
            dependencies.append("pytest")
        if "api" in description.lower() or "endpoint" in description.lower():
            dependencies.append("fastapi")
        if "database" in description.lower() or "sql" in description.lower():
            dependencies.append("database_connection")
        if "agent" in description.lower():
            dependencies.append("base_agent")
        if "memory" in description.lower():
            dependencies.append("memory_systems")

        return dependencies

    def _identify_affected_files(
        self,
        description: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Identify files likely to be affected"""
        affected = []

        # Parse description for file/component mentions
        description_lower = description.lower()

        if "tester" in description_lower or "testeragent" in description_lower:
            affected.append("core/agents/tester_agent.py")
        if "coder" in description_lower or "coderagent" in description_lower:
            affected.append("core/agents/coder_agent.py")
        if "debugger" in description_lower:
            affected.append("core/agents/debugger_agent.py")
        if "orchestrator" in description_lower or "director" in description_lower:
            affected.append("core/orchestrator/continuous_director.py")
        if "memory" in description_lower:
            affected.extend([
                "core/memory/project_ledger.py",
                "core/memory/error_knowledge_graph.py",
                "core/memory/vector_memory.py"
            ])
        if "frontend" in description_lower or "ui" in description_lower:
            affected.append("frontend/src/")
        if "api" in description_lower:
            affected.append("api/main.py")

        return affected

    def _assess_complexity(self, description: str) -> str:
        """Assess task complexity"""
        description_lower = description.lower()

        # High complexity indicators
        high_indicators = ["refactor", "redesign", "architecture", "migration", "breaking change"]
        medium_indicators = ["implement", "add", "create", "build", "integrate"]
        low_indicators = ["fix", "update", "modify", "change", "adjust"]

        if any(indicator in description_lower for indicator in high_indicators):
            return "high"
        elif any(indicator in description_lower for indicator in medium_indicators):
            return "medium"
        else:
            return "low"

    def _estimate_effort(self, description: str) -> str:
        """Estimate effort required"""
        complexity = self._assess_complexity(description)

        effort_map = {
            "low": "15-30 minutes",
            "medium": "30-90 minutes",
            "high": "2-4 hours"
        }

        return effort_map.get(complexity, "1-2 hours")

    def _identify_prerequisites(self, description: str) -> List[str]:
        """Identify prerequisites that must be completed first"""
        prerequisites = []

        if "test" in description.lower() and "implement" not in description.lower():
            prerequisites.append("Implementation must be complete")

        if "integrate" in description.lower() or "connect" in description.lower():
            prerequisites.append("Individual components must exist")

        return prerequisites

    def _generate_acceptance_criteria(self, description: str) -> List[str]:
        """Generate acceptance criteria for the task"""
        criteria = [
            "Implementation matches requirements",
            "Code follows project style guidelines",
            "No new linting errors introduced"
        ]

        description_lower = description.lower()

        if "test" in description_lower:
            criteria.append("Test coverage >= 85%")
            criteria.append("All tests passing")

        if "fix" in description_lower or "bug" in description_lower:
            criteria.append("Original issue is resolved")
            criteria.append("No regressions introduced")

        if "performance" in description_lower:
            criteria.append("Performance metrics improved")

        if "security" in description_lower:
            criteria.append("No security vulnerabilities introduced")

        criteria.append("Peer review approved (Aider + Gemini)")

        return criteria

    async def integrate_changes(
        self,
        aider_output: Dict[str, Any],
        gemini_review: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate changes from Aider based on Gemini's review.

        Args:
            aider_output: Output from AiderCodexAgent
            gemini_review: Review from GeminiCLIAgent

        Returns:
            Dict with integration result
        """
        self.metrics["integrations_performed"] += 1

        logger.info("Integrating Aider changes with Gemini feedback")

        # Check Gemini's approval
        approval = gemini_review.get("approval", "REQUEST_CHANGES")

        if approval == "APPROVE":
            logger.info("Gemini approved - changes accepted")
            return {
                "success": True,
                "action": "accepted",
                "message": "Changes approved by Gemini",
                "modified_files": aider_output.get("modified_files", [])
            }

        elif approval == "REQUEST_CHANGES":
            logger.info("Gemini requested changes - revision needed")

            issues = gemini_review.get("issues", [])
            suggestions = gemini_review.get("suggestions", [])

            return {
                "success": False,
                "action": "revision_required",
                "message": "Gemini requested changes",
                "issues": issues,
                "suggestions": suggestions,
                "next_steps": self._generate_revision_plan(issues, suggestions)
            }

        else:  # REJECT
            logger.warning("Gemini rejected changes")
            return {
                "success": False,
                "action": "rejected",
                "message": "Changes rejected by Gemini",
                "issues": gemini_review.get("issues", []),
                "recommendation": "Restart with different approach"
            }

    def _generate_revision_plan(
        self,
        issues: List[Dict[str, Any]],
        suggestions: List[str]
    ) -> List[str]:
        """Generate actionable revision plan from Gemini feedback"""
        plan = []

        # Prioritize critical and high severity issues
        critical_issues = [
            issue for issue in issues
            if issue.get("severity") in ["critical", "high"]
        ]

        for issue in critical_issues:
            action = f"Fix {issue.get('severity')} issue: {issue.get('description')}"
            if issue.get("suggestion"):
                action += f" - Suggested fix: {issue.get('suggestion')}"
            plan.append(action)

        # Add general suggestions
        for suggestion in suggestions[:3]:  # Top 3 suggestions
            plan.append(f"Consider: {suggestion}")

        return plan

    async def execute_tests(
        self,
        test_paths: Optional[List[str]] = None,
        coverage_threshold: float = 85.0
    ) -> Dict[str, Any]:
        """
        Execute tests using pytest.

        Args:
            test_paths: Optional specific test files/directories
            coverage_threshold: Minimum coverage percentage required

        Returns:
            Dict with test results
        """
        self.metrics["tests_executed"] += 1

        logger.info("Executing tests with pytest")

        # This is a placeholder - in the actual implementation,
        # we would use the Bash tool to run pytest
        # For now, return a success indicator

        return {
            "success": True,
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "coverage": 0.0,
            "coverage_met": False,
            "message": "Test execution placeholder - integrate with Bash tool"
        }

    def read_file(self, file_path: str) -> str:
        """
        Read a file from the filesystem.

        Args:
            file_path: Path to file

        Returns:
            File contents
        """
        self.metrics["files_analyzed"] += 1

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return path.read_text()

    def write_file(self, file_path: str, content: str) -> bool:
        """
        Write content to a file.

        Args:
            file_path: Path to file
            content: Content to write

        Returns:
            Success status
        """
        self.metrics["files_modified"] += 1

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

        logger.info(f"Wrote file: {file_path}")
        return True

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return self.metrics.copy()
