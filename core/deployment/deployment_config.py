from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime

class DeploymentStage(Enum):
    """Deployment pipeline stages"""
    CODE_CHECKOUT = "code_checkout"
    DEPENDENCY_INSTALLATION = "dependency_installation"
    UNIT_TESTS = "unit_tests"
    INTEGRATION_TESTS = "integration_tests"
    SECURITY_SCAN = "security_scan"
    BUILD_ARTIFACTS = "build_artifacts"
    DEPLOY_STAGING = "deploy_staging"
    SMOKE_TESTS = "smoke_tests"
    DEPLOY_PRODUCTION = "deploy_production"
    POST_DEPLOYMENT_VALIDATION = "post_deployment_validation"


class DeploymentStatus(Enum):
    """Deployment status values"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class DeploymentStrategy(Enum):
    """Deployment strategies"""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"


@dataclass
class QualityGate:
    """Quality gate for deployment pipeline"""
    name: str
    threshold: float
    current_value: Optional[float] = None
    passed: Optional[bool] = None
    description: str = ""
    blocking: bool = True

    def evaluate(self, value: float) -> bool:
        """Evaluate if quality gate passes"""
        self.current_value = value
        self.passed = value >= self.threshold
        return self.passed


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    deployment_id: str
    application_name: str
    version: str
    environment: Environment
    strategy: DeploymentStrategy
    quality_gates: List[QualityGate]
    environment_config: Dict[str, Any]
    secrets: Dict[str, str]
    rollback_enabled: bool = True
    max_deployment_duration: int = 1800  # 30 minutes
    health_check_endpoints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "application_name": self.application_name,
            "version": self.version,
            "environment": self.environment.value,
            "strategy": self.strategy.value,
            "quality_gates": [
                {
                    "name": qg.name,
                    "threshold": qg.threshold,
                    "current_value": qg.current_value,
                    "passed": qg.passed,
                    "description": qg.description,
                    "blocking": qg.blocking
                }
                for qg in self.quality_gates
            ],
            "environment_config": self.environment_config,
            "rollback_enabled": self.rollback_enabled,
            "max_deployment_duration": self.max_deployment_duration,
            "health_check_endpoints": self.health_check_endpoints
        }


@dataclass
class StageResult:
    """Result of a deployment stage execution"""
    stage: DeploymentStage
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    output: str = ""
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def complete(self, status: DeploymentStatus, output: str = "", error: str = None):
        """Mark stage as complete"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = status
        self.output = output
        if error:
            self.error_message = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "output": self.output,
            "error_message": self.error_message,
            "metrics": self.metrics
        }


@dataclass
class DeploymentExecution:
    """Deployment execution tracking"""
    deployment_id: str
    config: DeploymentConfig
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    current_stage: Optional[DeploymentStage] = None
    stage_results: List[StageResult] = field(default_factory=list)
    overall_metrics: Dict[str, Any] = field(default_factory=dict)
    rollback_plan: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_stage": self.current_stage.value if self.current_stage else None,
            "stage_results": [sr.to_dict() for sr in self.stage_results],
            "overall_metrics": self.overall_metrics,
            "rollback_plan": self.rollback_plan
        }


@dataclass
class InfrastructureSpec:
    """Infrastructure specification for deployment"""
    spec_id: str
    compute_resources: Dict[str, Any]
    network_config: Dict[str, Any]
    storage_config: Dict[str, Any]
    security_config: Dict[str, Any]
    scaling_config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
