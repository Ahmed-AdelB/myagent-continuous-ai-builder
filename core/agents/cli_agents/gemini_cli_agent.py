"""
Gemini CLI Agent - Wrapper for google-gemini CLI using Gemini 2.5 Pro or 3.0 Pro
"""

import asyncio
import subprocess
import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger
from datetime import datetime


class GeminiCLIAgent:
    """
    Wraps the `google-gemini` CLI tool for code review and analysis using Gemini 2.5/3.0 Pro.

    This agent specializes in:
    - Code review and quality assurance
    - Security analysis
    - Best practices validation
    - Performance optimization suggestions
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",  # or "gemini-3.0-pro"
        working_dir: Optional[Path] = None
    ):
        self.model = model
        self.working_dir = working_dir or Path.cwd()

        # Track metrics
        self.metrics = {
            "requests_made": 0,
            "approvals_given": 0,
            "rejections_given": 0,
            "average_response_time": 0.0
        }

        # Verify google-gemini is installed
        self._verify_gemini_installed()

        logger.info(f"Initialized GeminiCLIAgent with model={model}")

    def _verify_gemini_installed(self):
        """Verify that google-gemini CLI is available"""
        try:
            result = subprocess.run(
                ["google-gemini", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            logger.info(f"Gemini CLI version: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("google-gemini CLI not found")
            logger.info("Install from: https://github.com/google/generative-ai-python")
            raise RuntimeError("google-gemini CLI not installed")
        except subprocess.TimeoutExpired:
            logger.warning("Gemini version check timed out")

    async def review_code(
        self,
        file_paths: List[str],
        review_type: str = "comprehensive",
        criteria: Optional[List[str]] = None,
        timeout: int = 120
    ) -> Dict[str, Any]:
        """
        Review code files using Gemini.

        Args:
            file_paths: List of files to review
            review_type: Type of review (comprehensive, security, performance, style)
            criteria: Optional specific criteria to check
            timeout: Maximum execution time in seconds

        Returns:
            Dict with review results and approval status
        """
        start_time = datetime.now()
        self.metrics["requests_made"] += 1

        try:
            # Read file contents
            file_contents = {}
            for file_path in file_paths:
                path = Path(file_path)
                if path.exists():
                    file_contents[file_path] = path.read_text()
                else:
                    logger.warning(f"File not found: {file_path}")

            if not file_contents:
                return {
                    "success": False,
                    "error": "No valid files to review"
                }

            # Build review prompt
            prompt = self._build_review_prompt(
                file_contents,
                review_type,
                criteria
            )

            # Execute gemini CLI
            result = await self._execute_gemini(prompt, timeout)

            # Parse review result
            parsed_result = self._parse_review_result(result)

            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            if parsed_result.get("approval") == "APPROVE":
                self.metrics["approvals_given"] += 1
            else:
                self.metrics["rejections_given"] += 1

            # Update average response time
            total_requests = self.metrics["requests_made"]
            current_avg = self.metrics["average_response_time"]
            self.metrics["average_response_time"] = (
                (current_avg * (total_requests - 1) + execution_time) / total_requests
            )

            parsed_result["execution_time"] = execution_time
            parsed_result["success"] = True

            logger.info(f"Gemini review completed: {parsed_result.get('approval')} in {execution_time:.2f}s")

            return parsed_result

        except Exception as e:
            logger.error(f"Gemini review failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "approval": "ERROR"
            }

    def _build_review_prompt(
        self,
        file_contents: Dict[str, str],
        review_type: str,
        criteria: Optional[List[str]]
    ) -> str:
        """Build the review prompt for Gemini"""

        # Base prompt based on review type
        prompts = {
            "comprehensive": """You are a senior software engineer conducting a comprehensive code review.
Review the following code for:
- Correctness and logic errors
- Code quality and maintainability
- Best practices adherence
- Potential bugs or edge cases
- Performance considerations
- Security vulnerabilities
- Documentation quality""",

            "security": """You are a security expert reviewing code for vulnerabilities.
Focus on:
- SQL injection risks
- XSS vulnerabilities
- Authentication/authorization issues
- Sensitive data exposure
- Insecure dependencies
- Input validation
- Cryptographic weaknesses""",

            "performance": """You are a performance optimization expert.
Review for:
- Algorithm efficiency
- Database query optimization
- Memory usage
- Unnecessary computations
- Caching opportunities
- Async/await usage
- Resource management""",

            "style": """You are a code style and maintainability expert.
Review for:
- PEP 8 compliance (Python)
- Naming conventions
- Code organization
- DRY principles
- Single Responsibility Principle
- Documentation and comments
- Type hints usage"""
        }

        base_prompt = prompts.get(review_type, prompts["comprehensive"])

        # Add custom criteria if provided
        if criteria:
            criteria_text = "\n".join(f"- {c}" for c in criteria)
            base_prompt += f"\n\nAdditional criteria:\n{criteria_text}"

        # Add file contents
        files_text = ""
        for file_path, content in file_contents.items():
            files_text += f"\n\n=== File: {file_path} ===\n```\n{content}\n```\n"

        # Format the complete prompt with structured output request
        full_prompt = f"""{base_prompt}

{files_text}

Provide your review in the following JSON format:
{{
    "approval": "APPROVE" or "REQUEST_CHANGES" or "REJECT",
    "summary": "Brief summary of review",
    "issues": [
        {{
            "severity": "critical|high|medium|low",
            "file": "file path",
            "line": line number or null,
            "description": "Issue description",
            "suggestion": "How to fix"
        }}
    ],
    "suggestions": [
        "General improvement suggestion 1",
        "General improvement suggestion 2"
    ],
    "strengths": [
        "What's good about this code"
    ],
    "overall_quality_score": 0-100
}}

Important:
- Use "APPROVE" only if code is production-ready with no critical/high issues
- Use "REQUEST_CHANGES" if there are issues that should be addressed
- Use "REJECT" only if code has critical flaws that make it unusable
"""

        return full_prompt

    async def _execute_gemini(self, prompt: str, timeout: int) -> str:
        """Execute google-gemini CLI with the given prompt"""

        cmd = [
            "google-gemini",
            "generate",
            "--model", self.model,
            "--prompt", prompt
        ]

        logger.debug(f"Executing gemini command with model={self.model}")

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
            raise TimeoutError(f"Gemini execution exceeded {timeout} seconds")

        stdout_text = stdout.decode('utf-8') if stdout else ""
        stderr_text = stderr.decode('utf-8') if stderr else ""

        if stderr_text and ("error" in stderr_text.lower() or "failed" in stderr_text.lower()):
            raise RuntimeError(f"Gemini CLI error: {stderr_text}")

        return stdout_text

    def _parse_review_result(self, output: str) -> Dict[str, Any]:
        """Parse Gemini's review output"""

        # Try to extract JSON from output
        json_match = re.search(r'\{[\s\S]*\}', output)

        if json_match:
            try:
                result = json.loads(json_match.group())
                # Validate required fields
                if "approval" in result:
                    return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from Gemini output: {e}")

        # Fallback: parse text-based output
        logger.info("Using fallback text parsing for Gemini output")

        approval = "REQUEST_CHANGES"  # Default to conservative
        if "APPROVE" in output or "approved" in output.lower():
            approval = "APPROVE"
        elif "REJECT" in output or "rejected" in output.lower():
            approval = "REJECT"

        # Extract issues (lines starting with -, *, or numbered)
        issues = []
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith(('- ', '* ', '1. ', '2. ', '3. ')):
                # Remove prefix
                cleaned = re.sub(r'^[-*\d.]\s+', '', line)
                if cleaned and len(cleaned) > 10:  # Ignore very short lines
                    issues.append({
                        "severity": "medium",
                        "description": cleaned,
                        "file": None,
                        "line": None,
                        "suggestion": ""
                    })

        return {
            "approval": approval,
            "summary": output[:200] if len(output) > 200 else output,
            "issues": issues,
            "suggestions": [],
            "strengths": [],
            "overall_quality_score": 70 if approval == "APPROVE" else 50,
            "raw_output": output
        }

    async def analyze_design(
        self,
        design_document: str,
        code_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a design document and optionally compare with implementation.

        Args:
            design_document: Path to design document or design text
            code_files: Optional list of implementation files to compare

        Returns:
            Dict with design analysis
        """
        # Read design document
        design_path = Path(design_document)
        if design_path.exists():
            design_text = design_path.read_text()
        else:
            design_text = design_document

        prompt = f"""You are a software architect reviewing a design document.

Design Document:
{design_text}
"""

        # Add implementation files if provided
        if code_files:
            prompt += "\n\nImplementation Files:\n"
            for file_path in code_files:
                path = Path(file_path)
                if path.exists():
                    content = path.read_text()
                    prompt += f"\n=== {file_path} ===\n```\n{content}\n```\n"

            prompt += "\n\nCompare the design with the implementation and identify any gaps or deviations."

        prompt += """

Provide analysis in JSON format:
{
    "alignment_score": 0-100,
    "design_quality": "excellent|good|fair|poor",
    "issues": ["issue 1", "issue 2"],
    "gaps": ["gap 1", "gap 2"],
    "recommendations": ["recommendation 1", "recommendation 2"]
}
"""

        result = await self._execute_gemini(prompt, timeout=180)
        return self._parse_design_analysis(result)

    def _parse_design_analysis(self, output: str) -> Dict[str, Any]:
        """Parse design analysis output"""
        json_match = re.search(r'\{[\s\S]*\}', output)

        if json_match:
            try:
                result = json.loads(json_match.group())
                return {
                    "success": True,
                    **result,
                    "raw_output": output
                }
            except json.JSONDecodeError:
                pass

        # Fallback
        return {
            "success": True,
            "alignment_score": 75,
            "design_quality": "good",
            "issues": [],
            "gaps": [],
            "recommendations": [],
            "raw_output": output
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        total_reviews = self.metrics["approvals_given"] + self.metrics["rejections_given"]
        approval_rate = (
            self.metrics["approvals_given"] / total_reviews * 100
            if total_reviews > 0 else 0
        )

        return {
            **self.metrics,
            "approval_rate": round(approval_rate, 2)
        }
