"""
Guardrail System for Safe Autonomous Operation

Provides multi-layer defense that blocks dangerous inputs, filters risky outputs,
and restricts high-impact actions. Enables 24/7 autonomous operation while preventing
catastrophic failures.

Based on enterprise AI safety patterns and production guardrail frameworks.
"""

from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import re
from loguru import logger


class RiskLevel(Enum):
    """Risk assessment levels for operations"""
    SAFE = 1          # No risk - auto-approve
    LOW = 2           # Minimal risk - auto-approve with logging
    MEDIUM = 3        # Some risk - approve with validation
    HIGH = 4          # Significant risk - extra validation
    CRITICAL = 5      # Could cause data loss - block in autonomous mode


class GuardrailViolation(Exception):
    """Raised when guardrail blocks an operation"""
    pass


@dataclass
class GuardrailCheck:
    """Single guardrail validation check"""
    name: str
    description: str
    risk_level: RiskLevel
    validator: Callable
    auto_approve: bool = True


class GuardrailSystem:
    """
    Multi-layer guardrail system for safe autonomous AI operation.

    Prevents catastrophic failures by:
    - Blocking dangerous operations (eval, rm -rf, force push)
    - Assessing risk levels for all actions
    - Validating inputs/outputs/actions
    - Providing audit trail
    - Enabling safe rollback
    """

    def __init__(self, autonomous_mode: bool = False, project_root: Path = None):
        """
        Initialize guardrail system.

        Args:
            autonomous_mode: If True, blocks CRITICAL operations entirely
            project_root: Root directory for file operation validation
        """
        self.autonomous_mode = autonomous_mode
        self.project_root = project_root or Path.cwd()

        # Dangerous patterns that should NEVER execute
        self.blocked_patterns = self._init_blocked_patterns()

        # Risk indicators for assessment
        self.risk_indicators = self._init_risk_indicators()

        # Audit trail
        self.blocked_operations: List[Dict] = []
        self.high_risk_operations: List[Dict] = []
        self.operations_executed: List[Dict] = []

        logger.info(
            f"Guardrail system initialized. "
            f"Autonomous mode: {autonomous_mode}"
        )

    def _init_blocked_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns that should be blocked"""
        return {
            # Code execution vulnerabilities
            "code_injection": [
                r'\beval\s*\(',
                r'\bexec\s*\(',
                r'__import__\s*\(',
                r'compile\s*\(',
            ],

            # Destructive file operations
            "destructive_file_ops": [
                r'\brm\s+-rf\s+/',
                r'\brm\s+-rf\s+\*',
                r'shutil\.rmtree\s*\(\s*[\'"]\/[\'"]',
                r'os\.remove.*\/.*\*',
            ],

            # Dangerous git operations
            "dangerous_git": [
                r'git\s+push\s+.*--force',
                r'git\s+push\s+-f',
                r'git\s+reset\s+--hard\s+HEAD~',
                r'git\s+clean\s+-fd',
            ],

            # Database destruction
            "database_destruction": [
                r'DROP\s+DATABASE',
                r'TRUNCATE\s+TABLE',
                r'DELETE\s+FROM.*WHERE\s+1\s*=\s*1',
            ],

            # System commands
            "dangerous_system": [
                r'\bsudo\s+rm',
                r'chmod\s+777',
                r'shutdown',
                r'reboot',
            ],

            # Network/external access (in some contexts)
            "unvalidated_external": [
                r'os\.system\s*\(',
                r'subprocess\.call.*shell\s*=\s*True',
            ]
        }

    def _init_risk_indicators(self) -> Dict[RiskLevel, List[str]]:
        """Initialize risk level indicators"""
        return {
            RiskLevel.CRITICAL: [
                'DROP TABLE', 'DROP DATABASE', 'TRUNCATE',
                'rm -rf /', 'force push', '--force',
                'production', 'prod', 'master branch',
                'DELETE FROM', 'sudo', 'chmod 777'
            ],

            RiskLevel.HIGH: [
                'ALTER TABLE', 'migration', 'schema change',
                'merge to main', 'merge to master',
                'git push', 'deploy', 'configuration change',
                'env file', '.env', 'credentials'
            ],

            RiskLevel.MEDIUM: [
                'INSERT INTO', 'UPDATE', 'file deletion',
                'dependency update', 'requirements.txt',
                'package.json', 'git commit', 'database'
            ],

            RiskLevel.LOW: [
                'git status', 'git diff', 'git log',
                'pytest', 'npm test', 'read file',
                'list directory', 'logging'
            ],

            RiskLevel.SAFE: [
                'SELECT', 'read', 'GET', 'fetch',
                'log', 'print', 'display', 'show'
            ]
        }

    def validate_operation(
        self,
        operation: str,
        operation_type: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Validate an operation before execution.

        Args:
            operation: Description or code of the operation
            operation_type: Type of operation (file, git, database, etc.)
            context: Additional context for validation

        Returns:
            Dict with: {
                'allowed': bool,
                'risk_level': RiskLevel,
                'reason': str,
                'requires_approval': bool
            }

        Raises:
            GuardrailViolation: If operation is blocked
        """
        context = context or {}

        # Step 1: Check for absolutely blocked patterns
        for category, patterns in self.blocked_patterns.items():
            for pattern in patterns:
                if re.search(pattern, operation, re.IGNORECASE):
                    violation = {
                        'operation': operation,
                        'category': category,
                        'pattern': pattern,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.blocked_operations.append(violation)

                    logger.error(
                        f"GUARDRAIL VIOLATION: Blocked {category} pattern in operation"
                    )

                    if self.autonomous_mode:
                        raise GuardrailViolation(
                            f"Operation blocked by guardrail: {category} - {pattern}"
                        )
                    else:
                        # In non-autonomous mode, return for human decision
                        return {
                            'allowed': False,
                            'risk_level': RiskLevel.CRITICAL,
                            'reason': f'Matches blocked pattern: {category}',
                            'requires_approval': True
                        }

        # Step 2: Assess risk level
        risk_level = self._assess_risk(operation, operation_type, context)

        # Step 3: Determine if allowed in autonomous mode
        allowed = True
        requires_approval = False

        if self.autonomous_mode:
            if risk_level == RiskLevel.CRITICAL:
                # Block in autonomous mode
                allowed = False
                logger.warning(
                    f"CRITICAL risk operation blocked in autonomous mode: {operation[:100]}"
                )
            elif risk_level == RiskLevel.HIGH:
                # High risk - extra validation required
                if not self._validate_high_risk_operation(operation, context):
                    allowed = False
                    requires_approval = True
                    self.high_risk_operations.append({
                        'operation': operation,
                        'risk_level': risk_level,
                        'blocked': True
                    })

        # Step 4: Log and return
        result = {
            'allowed': allowed,
            'risk_level': risk_level,
            'reason': f'Risk level: {risk_level.name}',
            'requires_approval': requires_approval
        }

        if allowed:
            self.operations_executed.append({
                'operation': operation,
                'risk_level': risk_level,
                'context': context
            })

        return result

    def _assess_risk(
        self,
        operation: str,
        operation_type: str,
        context: Dict
    ) -> RiskLevel:
        """
        Assess risk level of an operation.

        Args:
            operation: The operation to assess
            operation_type: Type of operation
            context: Additional context

        Returns:
            RiskLevel enum value
        """
        operation_lower = operation.lower()

        # Check each risk level from highest to lowest
        for risk_level in [
            RiskLevel.CRITICAL,
            RiskLevel.HIGH,
            RiskLevel.MEDIUM,
            RiskLevel.LOW,
            RiskLevel.SAFE
        ]:
            indicators = self.risk_indicators.get(risk_level, [])
            if any(indicator.lower() in operation_lower for indicator in indicators):
                logger.debug(f"Operation assessed as {risk_level.name} risk")
                return risk_level

        # Default to MEDIUM if no indicators match
        return RiskLevel.MEDIUM

    def _validate_high_risk_operation(
        self,
        operation: str,
        context: Dict
    ) -> bool:
        """
        Additional validation for high-risk operations.

        Args:
            operation: The operation to validate
            context: Operation context

        Returns:
            True if operation is allowed, False otherwise
        """
        # Check if operation affects production
        if self._is_production_operation(operation, context):
            logger.warning("HIGH RISK: Operation affects production environment")
            return False

        # Check if operation modifies critical files
        critical_files = ['.env', 'credentials', 'secrets', 'private_key']
        if any(cf in operation.lower() for cf in critical_files):
            logger.warning("HIGH RISK: Operation affects critical files")
            return False

        # Check file operation scope
        if 'file_path' in context:
            file_path = Path(context['file_path'])

            # Don't allow operations outside project root
            try:
                file_path.resolve().relative_to(self.project_root.resolve())
            except ValueError:
                logger.warning(
                    f"HIGH RISK: File operation outside project root: {file_path}"
                )
                return False

            # Don't allow operations on system directories
            dangerous_paths = ['/etc', '/usr', '/bin', '/sys', '/proc', '/var']
            if any(str(file_path).startswith(dp) for dp in dangerous_paths):
                logger.warning(f"HIGH RISK: Operation on system directory: {file_path}")
                return False

        return True

    def _is_production_operation(self, operation: str, context: Dict) -> bool:
        """Check if operation affects production environment"""
        prod_indicators = [
            'production', 'prod', 'live', 'master branch',
            context.get('environment') == 'production'
        ]
        return any(
            indicator for indicator in prod_indicators
            if isinstance(indicator, str) and indicator in operation.lower()
        ) or context.get('environment') == 'production'

    def validate_file_operation(
        self,
        operation: str,
        file_path: str,
        operation_type: str = "modify"
    ) -> bool:
        """
        Validate a file operation.

        Args:
            operation: Description of operation
            file_path: Path to file
            operation_type: Type of operation (read, write, delete, execute)

        Returns:
            True if allowed, False if blocked

        Raises:
            GuardrailViolation: If operation is blocked in autonomous mode
        """
        result = self.validate_operation(
            operation=f"{operation_type} file: {file_path}\n{operation}",
            operation_type="file",
            context={'file_path': file_path, 'operation_type': operation_type}
        )

        return result['allowed']

    def validate_git_operation(
        self,
        git_command: str,
        branch: str = None
    ) -> bool:
        """
        Validate a git operation.

        Args:
            git_command: Git command to execute
            branch: Branch being operated on

        Returns:
            True if allowed, False if blocked
        """
        context = {}
        if branch:
            context['branch'] = branch
            # Extra caution for main/master branches
            if branch.lower() in ['main', 'master']:
                context['critical_branch'] = True

        result = self.validate_operation(
            operation=git_command,
            operation_type="git",
            context=context
        )

        return result['allowed']

    def validate_database_operation(
        self,
        sql_query: str,
        database: str = None
    ) -> bool:
        """
        Validate a database operation.

        Args:
            sql_query: SQL query to execute
            database: Database name

        Returns:
            True if allowed, False if blocked
        """
        result = self.validate_operation(
            operation=sql_query,
            operation_type="database",
            context={'database': database}
        )

        return result['allowed']

    def get_audit_trail(self) -> Dict[str, List]:
        """Get complete audit trail of operations"""
        return {
            'blocked_operations': self.blocked_operations,
            'high_risk_operations': self.high_risk_operations,
            'operations_executed': self.operations_executed,
            'total_blocked': len(self.blocked_operations),
            'total_high_risk': len(self.high_risk_operations),
            'total_executed': len(self.operations_executed)
        }

    def reset_audit_trail(self):
        """Reset audit trail (for testing or new sessions)"""
        self.blocked_operations.clear()
        self.high_risk_operations.clear()
        self.operations_executed.clear()
        logger.info("Audit trail reset")


# Singleton instance for global access
_guardrail_instance: Optional[GuardrailSystem] = None


def get_guardrails(autonomous_mode: bool = False) -> GuardrailSystem:
    """
    Get or create the global guardrail instance.

    Args:
        autonomous_mode: Whether to enable autonomous mode restrictions

    Returns:
        GuardrailSystem instance
    """
    global _guardrail_instance

    if _guardrail_instance is None:
        _guardrail_instance = GuardrailSystem(autonomous_mode=autonomous_mode)

    return _guardrail_instance


def with_guardrails(func: Callable) -> Callable:
    """
    Decorator to wrap function with guardrail validation.

    Usage:
        @with_guardrails
        def dangerous_operation():
            # This will be validated by guardrails
            os.system("rm -rf /")  # Would be blocked
    """
    def wrapper(*args, **kwargs):
        guardrails = get_guardrails()

        # Extract operation description from function
        operation = f"{func.__name__}: {func.__doc__ or 'No description'}"

        # Validate
        result = guardrails.validate_operation(
            operation=operation,
            operation_type="function",
            context={'function': func.__name__, 'args': args, 'kwargs': kwargs}
        )

        if not result['allowed']:
            raise GuardrailViolation(
                f"Guardrails blocked {func.__name__}: {result['reason']}"
            )

        # Execute if allowed
        return func(*args, **kwargs)

    return wrapper
