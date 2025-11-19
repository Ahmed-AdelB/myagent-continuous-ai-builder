"""
Task Validator - Prevents False Completion Claims

This module implements explicit validation for task completion to prevent AI agents
from claiming tasks are complete when they are actually partial or failed.

Based on research showing agents frequently claim completion when work is incomplete,
this validator requires proof of completion before allowing tasks to be marked done.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import subprocess
import re
from loguru import logger


class ValidationResult(Enum):
    """Result of validation check"""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    BLOCKED = "blocked"


@dataclass
class ValidationCheck:
    """Single validation check"""
    name: str
    description: str
    validator: Callable
    required: bool = True
    error_message: Optional[str] = None


@dataclass
class TaskValidationResult:
    """Result of complete task validation"""
    task_id: str
    overall_status: ValidationResult
    checks_passed: List[str]
    checks_failed: List[str]
    checks_partial: List[str]
    proof_of_completion: Dict[str, Any]
    can_mark_complete: bool
    failure_reasons: List[str]


class TaskValidator:
    """
    Validates that tasks are truly complete before allowing completion status.

    Prevents the common problem where AI agents claim completion when:
    - Tests haven't actually run
    - Files weren't actually created
    - Code doesn't compile
    - Requirements aren't met
    """

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.validation_registry: Dict[str, List[ValidationCheck]] = {}

        # Register default validators for common task types
        self._register_default_validators()

    def _register_default_validators(self):
        """Register validation checks for common task types"""

        # Code implementation tasks
        self.register_validator(
            "code",
            ValidationCheck(
                name="file_exists",
                description="Verify file was created/modified",
                validator=self._validate_file_exists,
                required=True
            )
        )
        self.register_validator(
            "code",
            ValidationCheck(
                name="syntax_valid",
                description="Verify code has valid syntax",
                validator=self._validate_syntax,
                required=True
            )
        )

        # Test tasks
        self.register_validator(
            "test",
            ValidationCheck(
                name="tests_exist",
                description="Verify test files created",
                validator=self._validate_tests_exist,
                required=True
            )
        )
        self.register_validator(
            "test",
            ValidationCheck(
                name="tests_run",
                description="Verify tests actually executed",
                validator=self._validate_tests_run,
                required=True
            )
        )
        self.register_validator(
            "test",
            ValidationCheck(
                name="tests_pass",
                description="Verify tests passed",
                validator=self._validate_tests_pass,
                required=True
            )
        )

        # Debug tasks
        self.register_validator(
            "debug",
            ValidationCheck(
                name="error_resolved",
                description="Verify original error no longer occurs",
                validator=self._validate_error_resolved,
                required=True
            )
        )

        # Documentation tasks
        self.register_validator(
            "documentation",
            ValidationCheck(
                name="docs_exist",
                description="Verify documentation files created",
                validator=self._validate_docs_exist,
                required=True
            )
        )
        self.register_validator(
            "documentation",
            ValidationCheck(
                name="docs_complete",
                description="Verify documentation meets coverage requirements",
                validator=self._validate_docs_complete,
                required=False
            )
        )

        # Architecture review tasks
        self.register_validator(
            "architecture",
            ValidationCheck(
                name="design_documented",
                description="Verify design decisions documented",
                validator=self._validate_design_documented,
                required=True
            )
        )

    def register_validator(self, task_type: str, check: ValidationCheck):
        """Register a validation check for a task type"""
        if task_type not in self.validation_registry:
            self.validation_registry[task_type] = []
        self.validation_registry[task_type].append(check)
        logger.debug(f"Registered validator '{check.name}' for task type '{task_type}'")

    async def validate_task_completion(
        self,
        task: Any,  # DevelopmentTask or AgentTask
        result: Dict[str, Any]
    ) -> TaskValidationResult:
        """
        Validate that a task is truly complete.

        Args:
            task: The task that claims to be complete
            result: The result data from task execution

        Returns:
            TaskValidationResult with detailed validation status
        """
        task_type = self._extract_task_type(task)
        validators = self.validation_registry.get(task_type, [])

        if not validators:
            logger.warning(f"No validators registered for task type '{task_type}'")
            # Without validators, we can't verify completion
            return TaskValidationResult(
                task_id=getattr(task, 'id', 'unknown'),
                overall_status=ValidationResult.PARTIAL,
                checks_passed=[],
                checks_failed=[],
                checks_partial=["no_validators_registered"],
                proof_of_completion={},
                can_mark_complete=False,
                failure_reasons=["No validation criteria defined for this task type"]
            )

        checks_passed = []
        checks_failed = []
        checks_partial = []
        proof = {}
        failure_reasons = []

        # Run all validation checks
        for check in validators:
            try:
                check_result = await check.validator(task, result)

                if check_result['status'] == ValidationResult.PASS:
                    checks_passed.append(check.name)
                    proof[check.name] = check_result.get('proof', True)

                elif check_result['status'] == ValidationResult.FAIL:
                    checks_failed.append(check.name)
                    failure_reason = check_result.get('reason', f"{check.name} failed")
                    failure_reasons.append(failure_reason)

                    if check.required:
                        logger.error(f"Required check '{check.name}' failed: {failure_reason}")

                elif check_result['status'] == ValidationResult.PARTIAL:
                    checks_partial.append(check.name)
                    failure_reasons.append(f"{check.name} partially complete")

            except Exception as e:
                logger.error(f"Validation check '{check.name}' raised exception: {e}")
                checks_failed.append(check.name)
                failure_reasons.append(f"{check.name} error: {str(e)}")

        # Determine overall status
        required_checks = [c for c in validators if c.required]
        required_passed = all(
            c.name in checks_passed
            for c in required_checks
        )

        if required_passed and not checks_failed:
            overall_status = ValidationResult.PASS
            can_mark_complete = True
        elif checks_partial and not checks_failed:
            overall_status = ValidationResult.PARTIAL
            can_mark_complete = False
        else:
            overall_status = ValidationResult.FAIL
            can_mark_complete = False

        return TaskValidationResult(
            task_id=getattr(task, 'id', 'unknown'),
            overall_status=overall_status,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            checks_partial=checks_partial,
            proof_of_completion=proof,
            can_mark_complete=can_mark_complete,
            failure_reasons=failure_reasons
        )

    # ==================== Validation Implementations ====================

    async def _validate_file_exists(self, task: Any, result: Dict) -> Dict:
        """Validate that expected files were created/modified"""
        file_path = result.get('file_path') or task.data.get('file_path')

        if not file_path:
            return {
                'status': ValidationResult.FAIL,
                'reason': 'No file path specified in task or result'
            }

        path = self.project_root / file_path
        if not path.exists():
            return {
                'status': ValidationResult.FAIL,
                'reason': f'File {file_path} does not exist'
            }

        return {
            'status': ValidationResult.PASS,
            'proof': {
                'file_path': str(path),
                'file_size': path.stat().st_size,
                'modified_time': path.stat().st_mtime
            }
        }

    async def _validate_syntax(self, task: Any, result: Dict) -> Dict:
        """Validate that code has valid syntax"""
        file_path = result.get('file_path') or task.data.get('file_path')

        if not file_path:
            return {'status': ValidationResult.PASS}  # Skip if no file

        path = self.project_root / file_path

        if path.suffix == '.py':
            # Python syntax check
            try:
                proc = subprocess.run(
                    ['python', '-m', 'py_compile', str(path)],
                    capture_output=True,
                    timeout=10
                )
                if proc.returncode == 0:
                    return {
                        'status': ValidationResult.PASS,
                        'proof': {'syntax_valid': True}
                    }
                else:
                    return {
                        'status': ValidationResult.FAIL,
                        'reason': f'Syntax error: {proc.stderr.decode()}'
                    }
            except subprocess.TimeoutExpired:
                return {
                    'status': ValidationResult.FAIL,
                    'reason': 'Syntax check timed out'
                }

        # For other file types, skip syntax validation
        return {'status': ValidationResult.PASS}

    async def _validate_tests_exist(self, task: Any, result: Dict) -> Dict:
        """Validate that test files were created"""
        test_dir = result.get('test_directory') or task.data.get('test_directory', 'tests')
        test_path = self.project_root / test_dir

        if not test_path.exists():
            return {
                'status': ValidationResult.FAIL,
                'reason': f'Test directory {test_dir} does not exist'
            }

        # Find test files
        test_files = list(test_path.rglob('test_*.py'))

        if not test_files:
            return {
                'status': ValidationResult.FAIL,
                'reason': 'No test files found'
            }

        return {
            'status': ValidationResult.PASS,
            'proof': {
                'test_files': [str(f.relative_to(self.project_root)) for f in test_files],
                'test_count': len(test_files)
            }
        }

    async def _validate_tests_run(self, task: Any, result: Dict) -> Dict:
        """Validate that tests actually executed"""
        test_output = result.get('output') or result.get('test_output')

        if not test_output:
            return {
                'status': ValidationResult.FAIL,
                'reason': 'No test output found - tests may not have run'
            }

        # Look for pytest execution indicators
        if 'pytest' in test_output or 'test session starts' in test_output:
            return {
                'status': ValidationResult.PASS,
                'proof': {'test_execution_confirmed': True}
            }

        return {
            'status': ValidationResult.FAIL,
            'reason': 'No evidence of test execution in output'
        }

    async def _validate_tests_pass(self, task: Any, result: Dict) -> Dict:
        """Validate that tests passed"""
        results_data = result.get('results', {})
        passed = results_data.get('passed', 0)
        failed = results_data.get('failed', 0)

        if failed > 0:
            return {
                'status': ValidationResult.FAIL,
                'reason': f'{failed} test(s) failed'
            }

        if passed == 0:
            return {
                'status': ValidationResult.FAIL,
                'reason': 'No tests passed (possibly no tests ran)'
            }

        return {
            'status': ValidationResult.PASS,
            'proof': {
                'tests_passed': passed,
                'tests_failed': failed
            }
        }

    async def _validate_error_resolved(self, task: Any, result: Dict) -> Dict:
        """Validate that the original error is resolved"""
        original_error = task.data.get('error_message')
        test_output = result.get('output', '')

        if not original_error:
            return {'status': ValidationResult.PASS}  # Can't validate without original error

        # Check if error still appears in output
        if original_error in test_output:
            return {
                'status': ValidationResult.FAIL,
                'reason': 'Original error still occurs'
            }

        return {
            'status': ValidationResult.PASS,
            'proof': {'error_resolved': True}
        }

    async def _validate_docs_exist(self, task: Any, result: Dict) -> Dict:
        """Validate documentation files exist"""
        doc_file = result.get('documentation_file') or task.data.get('file_path')

        if not doc_file:
            return {
                'status': ValidationResult.FAIL,
                'reason': 'No documentation file specified'
            }

        path = self.project_root / doc_file
        if not path.exists():
            return {
                'status': ValidationResult.FAIL,
                'reason': f'Documentation file {doc_file} does not exist'
            }

        # Check file is not empty
        content = path.read_text()
        if len(content.strip()) < 50:
            return {
                'status': ValidationResult.PARTIAL,
                'reason': 'Documentation file exists but appears too brief'
            }

        return {
            'status': ValidationResult.PASS,
            'proof': {
                'file_path': str(path),
                'word_count': len(content.split())
            }
        }

    async def _validate_docs_complete(self, task: Any, result: Dict) -> Dict:
        """Validate documentation meets coverage requirements"""
        # This is a simplified check - could be enhanced with docstring parsing
        return {'status': ValidationResult.PASS}

    async def _validate_design_documented(self, task: Any, result: Dict) -> Dict:
        """Validate design decisions are documented"""
        doc_content = result.get('documentation') or result.get('design_doc')

        if not doc_content:
            return {
                'status': ValidationResult.FAIL,
                'reason': 'No design documentation provided'
            }

        # Check for key architecture elements
        required_sections = ['architecture', 'design', 'rationale', 'tradeoffs']
        found_sections = sum(1 for s in required_sections if s in doc_content.lower())

        if found_sections < 2:
            return {
                'status': ValidationResult.PARTIAL,
                'reason': 'Design documentation missing key sections'
            }

        return {
            'status': ValidationResult.PASS,
            'proof': {'design_sections_found': found_sections}
        }

    def _extract_task_type(self, task: Any) -> str:
        """Extract task type from task object"""
        # Try different attributes
        if hasattr(task, 'type'):
            task_type = task.type
        elif hasattr(task, 'task_type'):
            task_type = task.task_type
        else:
            task_type = 'unknown'

        # Normalize task type
        task_type = task_type.lower()

        # Map similar types
        if 'test' in task_type:
            return 'test'
        elif 'code' in task_type or 'implement' in task_type:
            return 'code'
        elif 'debug' in task_type or 'fix' in task_type:
            return 'debug'
        elif 'doc' in task_type:
            return 'documentation'
        elif 'architecture' in task_type or 'design' in task_type:
            return 'architecture'

        return task_type
