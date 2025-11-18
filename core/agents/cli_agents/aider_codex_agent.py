"""
Aider Codex Agent - Wrapper for aider CLI using GPT-5.1 with max reasoning
"""

import asyncio
import subprocess
import json
import tempfile
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger
from datetime import datetime


class AiderCodexAgent:
    """
    Wraps the `aider` CLI tool for code generation using GPT-5.1 with maximum reasoning.

    Aider is an AI pair programming tool that can modify multiple files based on instructions.
    """

    def __init__(
        self,
        model: str = "gpt-5.1",
        max_reasoning: bool = True,
        auto_accept: bool = True,
        working_dir: Optional[Path] = None
    ):
        self.model = model
        self.max_reasoning = max_reasoning
        self.auto_accept = auto_accept
        self.working_dir = working_dir or Path.cwd()

        # Track metrics
        self.metrics = {
            "requests_made": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_files_modified": 0,
            "average_response_time": 0.0
        }

        # Verify aider is installed
        self._verify_aider_installed()

        logger.info(f"Initialized AiderCodexAgent with model={model}, max_reasoning={max_reasoning}")

    def _verify_aider_installed(self):
        """Verify that aider CLI is available"""
        try:
            result = subprocess.run(
                ["aider", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            logger.info(f"Aider version: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("Aider CLI not found. Install with: pip install aider-chat")
            raise RuntimeError("Aider CLI not installed. Run: pip install aider-chat")
        except subprocess.TimeoutExpired:
            logger.warning("Aider version check timed out")

    async def generate_code(
        self,
        instruction: str,
        files: List[str],
        context: Optional[str] = None,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Generate code using aider CLI.

        Args:
            instruction: Natural language instruction for code generation
            files: List of file paths to modify
            context: Optional additional context
            timeout: Maximum execution time in seconds

        Returns:
            Dict with success status, modified files, and output
        """
        start_time = datetime.now()
        self.metrics["requests_made"] += 1

        try:
            # Build aider command
            cmd = self._build_aider_command(instruction, files, context)

            logger.info(f"Executing aider command: {' '.join(cmd[:3])}...")
            logger.debug(f"Full instruction: {instruction}")

            # Execute aider
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.working_dir)
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                raise TimeoutError(f"Aider execution exceeded {timeout} seconds")

            stdout_text = stdout.decode('utf-8') if stdout else ""
            stderr_text = stderr.decode('utf-8') if stderr else ""

            # Parse aider output
            result = self._parse_aider_output(stdout_text, stderr_text, files)

            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            if result["success"]:
                self.metrics["successful_generations"] += 1
                self.metrics["total_files_modified"] += len(result.get("modified_files", []))
            else:
                self.metrics["failed_generations"] += 1

            # Update average response time
            total_requests = self.metrics["requests_made"]
            current_avg = self.metrics["average_response_time"]
            self.metrics["average_response_time"] = (
                (current_avg * (total_requests - 1) + execution_time) / total_requests
            )

            result["execution_time"] = execution_time

            logger.info(f"Aider execution {'succeeded' if result['success'] else 'failed'} in {execution_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Aider execution failed: {e}")
            self.metrics["failed_generations"] += 1
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "modified_files": []
            }

    def _build_aider_command(
        self,
        instruction: str,
        files: List[str],
        context: Optional[str] = None
    ) -> List[str]:
        """Build the aider CLI command"""
        cmd = ["aider"]

        # Model configuration
        if self.model:
            cmd.extend(["--model", self.model])

        # Max reasoning for GPT-5.1
        if self.max_reasoning:
            cmd.append("--max-reasoning")

        # Auto-accept changes
        if self.auto_accept:
            cmd.append("--yes")

        # Disable auto-commits (we handle git separately)
        cmd.append("--no-auto-commits")

        # Add files to modify
        for file_path in files:
            cmd.append(str(file_path))

        # Build full instruction
        full_instruction = instruction
        if context:
            full_instruction = f"{context}\n\n{instruction}"

        # Add instruction as message
        cmd.extend(["--message", full_instruction])

        return cmd

    def _parse_aider_output(
        self,
        stdout: str,
        stderr: str,
        files: List[str]
    ) -> Dict[str, Any]:
        """Parse aider output to extract results"""

        # Check for errors in stderr
        if stderr and ("error" in stderr.lower() or "failed" in stderr.lower()):
            return {
                "success": False,
                "error": stderr,
                "output": stdout,
                "modified_files": []
            }

        # Parse modified files from output
        modified_files = []
        for line in stdout.split('\n'):
            if "modified:" in line.lower() or "created:" in line.lower():
                # Extract file path from line
                for file_path in files:
                    if file_path in line:
                        modified_files.append(file_path)

        # If no explicit modifications found, assume all files were considered
        if not modified_files and stdout:
            # Check if aider successfully executed
            if "added" in stdout.lower() or "modified" in stdout.lower():
                modified_files = files.copy()

        success = len(modified_files) > 0 or "successfully" in stdout.lower()

        return {
            "success": success,
            "output": stdout,
            "modified_files": modified_files,
            "stderr": stderr if stderr else None
        }

    async def review_code(
        self,
        file_path: str,
        review_criteria: List[str]
    ) -> Dict[str, Any]:
        """
        Ask aider to review code against specific criteria.

        Args:
            file_path: Path to file to review
            review_criteria: List of criteria to check

        Returns:
            Dict with review results
        """
        criteria_text = "\n".join(f"- {criterion}" for criterion in review_criteria)

        instruction = f"""Review this code against the following criteria:
{criteria_text}

Provide a detailed review with:
1. Issues found (if any)
2. Suggestions for improvement
3. Overall assessment (APPROVE or REQUEST_CHANGES)

Format your response as:
ASSESSMENT: [APPROVE or REQUEST_CHANGES]
ISSUES:
- [issue 1]
- [issue 2]
SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]
"""

        result = await self.generate_code(
            instruction=instruction,
            files=[file_path],
            timeout=120
        )

        # Parse review from output
        output = result.get("output", "")

        assessment = "REQUEST_CHANGES"  # Default to conservative
        if "ASSESSMENT: APPROVE" in output:
            assessment = "APPROVE"

        issues = []
        suggestions = []

        # Extract issues and suggestions
        lines = output.split('\n')
        in_issues = False
        in_suggestions = False

        for line in lines:
            line = line.strip()
            if line.startswith("ISSUES:"):
                in_issues = True
                in_suggestions = False
            elif line.startswith("SUGGESTIONS:"):
                in_suggestions = True
                in_issues = False
            elif line.startswith("- "):
                if in_issues:
                    issues.append(line[2:])
                elif in_suggestions:
                    suggestions.append(line[2:])

        return {
            "success": True,
            "assessment": assessment,
            "issues": issues,
            "suggestions": suggestions,
            "raw_output": output
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return self.metrics.copy()

    async def refactor_code(
        self,
        file_path: str,
        refactoring_goal: str
    ) -> Dict[str, Any]:
        """
        Refactor code with a specific goal.

        Args:
            file_path: Path to file to refactor
            refactoring_goal: Description of refactoring objective

        Returns:
            Dict with refactoring results
        """
        instruction = f"""Refactor this code with the following goal:
{refactoring_goal}

Ensure:
1. Functionality is preserved
2. Code is more maintainable
3. Best practices are followed
4. Tests still pass (if applicable)
"""

        return await self.generate_code(
            instruction=instruction,
            files=[file_path],
            timeout=300
        )
