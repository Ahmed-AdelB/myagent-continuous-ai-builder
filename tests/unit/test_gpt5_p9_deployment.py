#!/usr/bin/env python3
"""
GPT-5 Priority P9: Deployment Orchestrator - Comprehensive Unit Tests

Tests the automated deployment and CI/CD pipeline management capabilities including:
- Multi-stage deployment pipeline orchestration and quality gate validation
- Environment-specific configuration management and secrets handling
- Automated rollback mechanisms and deployment health monitoring
- Infrastructure provisioning and scaling automation
- DevSecOps integration and compliance validation

Testing methodologies applied:
- TDD: Test-driven development for deployment workflows
- BDD: Behavior-driven scenarios for deployment processes
- Infrastructure testing for environment validation
- Security testing for deployment security
- Contract testing for service dependencies
"""

import pytest
import asyncio
import json
import uuid
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

# Import test fixtures
from tests.fixtures.test_data import TEST_DATA


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


class MockDeploymentOrchestrator:
    """Mock implementation of Deployment Orchestrator for testing"""

    def __init__(self):
        self.active_deployments: Dict[str, DeploymentExecution] = {}
        self.deployment_history: List[DeploymentExecution] = {}
        self.environment_configs: Dict[Environment, Dict[str, Any]] = {
            Environment.DEVELOPMENT: {
                "replicas": 1,
                "cpu_limit": "500m",
                "memory_limit": "512Mi",
                "auto_deploy": True
            },
            Environment.STAGING: {
                "replicas": 2,
                "cpu_limit": "1000m",
                "memory_limit": "1Gi",
                "auto_deploy": False
            },
            Environment.PRODUCTION: {
                "replicas": 3,
                "cpu_limit": "2000m",
                "memory_limit": "2Gi",
                "auto_deploy": False
            }
        }
        self.infrastructure_templates: Dict[str, InfrastructureSpec] = {}
        self.secrets_vault: Dict[str, Dict[str, str]] = {}

    async def initialize(self):
        """Initialize deployment orchestrator"""
        # Set up default infrastructure templates
        await self._setup_default_infrastructure_templates()

        # Initialize secrets vault
        await self._setup_secrets_vault()

    async def create_deployment(self, config: DeploymentConfig) -> str:
        """Create new deployment"""
        if not config.deployment_id:
            config.deployment_id = f"deploy_{uuid.uuid4().hex[:8]}"

        # Validate deployment config
        validation_result = await self._validate_deployment_config(config)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid deployment config: {validation_result['errors']}")

        # Create deployment execution
        execution = DeploymentExecution(
            deployment_id=config.deployment_id,
            config=config,
            status=DeploymentStatus.PENDING,
            start_time=datetime.now()
        )

        # Create rollback plan
        execution.rollback_plan = await self._create_rollback_plan(config)

        self.active_deployments[config.deployment_id] = execution
        return config.deployment_id

    async def execute_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Execute deployment pipeline"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        execution = self.active_deployments[deployment_id]
        execution.status = DeploymentStatus.IN_PROGRESS

        try:
            # Execute pipeline stages
            pipeline_stages = await self._get_pipeline_stages(execution.config)

            for stage in pipeline_stages:
                execution.current_stage = stage
                stage_result = await self._execute_stage(stage, execution.config)
                execution.stage_results.append(stage_result)

                # Check if stage failed
                if stage_result.status == DeploymentStatus.FAILED:
                    execution.status = DeploymentStatus.FAILED
                    if execution.config.rollback_enabled:
                        await self._trigger_rollback(execution)
                    break

                # Evaluate quality gates after certain stages
                if stage in [DeploymentStage.UNIT_TESTS, DeploymentStage.INTEGRATION_TESTS,
                           DeploymentStage.SECURITY_SCAN, DeploymentStage.SMOKE_TESTS]:
                    gate_results = await self._evaluate_quality_gates(execution.config, stage)
                    stage_result.metrics["quality_gates"] = gate_results

                    # Check if any blocking quality gate failed
                    failed_blocking_gates = [
                        gate for gate in execution.config.quality_gates
                        if not gate.passed and gate.blocking
                    ]

                    if failed_blocking_gates:
                        execution.status = DeploymentStatus.FAILED
                        if execution.config.rollback_enabled:
                            await self._trigger_rollback(execution)
                        break

            # If all stages succeeded
            if execution.status == DeploymentStatus.IN_PROGRESS:
                execution.status = DeploymentStatus.SUCCESS
                execution.end_time = datetime.now()

                # Perform post-deployment validation
                validation_result = await self._post_deployment_validation(execution)
                execution.overall_metrics["post_deployment_validation"] = validation_result

        except Exception as e:
            execution.status = DeploymentStatus.FAILED
            execution.stage_results.append(StageResult(
                stage=execution.current_stage or DeploymentStage.CODE_CHECKOUT,
                status=DeploymentStatus.FAILED,
                start_time=datetime.now(),
                error_message=str(e)
            ))

            if execution.config.rollback_enabled:
                await self._trigger_rollback(execution)

        # Move to history
        self.deployment_history[deployment_id] = execution
        if deployment_id in self.active_deployments:
            del self.active_deployments[deployment_id]

        return execution.to_dict()

    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status"""
        if deployment_id in self.active_deployments:
            execution = self.active_deployments[deployment_id]
        elif deployment_id in self.deployment_history:
            execution = self.deployment_history[deployment_id]
        else:
            raise ValueError(f"Deployment {deployment_id} not found")

        return execution.to_dict()

    async def cancel_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Cancel active deployment"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Active deployment {deployment_id} not found")

        execution = self.active_deployments[deployment_id]
        execution.status = DeploymentStatus.CANCELLED
        execution.end_time = datetime.now()

        # Trigger rollback if enabled
        if execution.config.rollback_enabled:
            await self._trigger_rollback(execution)

        return execution.to_dict()

    async def provision_infrastructure(self, spec: InfrastructureSpec, environment: Environment) -> Dict[str, Any]:
        """Provision infrastructure for deployment"""
        provision_id = f"infra_{uuid.uuid4().hex[:8]}"

        # Simulate infrastructure provisioning
        await asyncio.sleep(0.1)

        provisioning_result = {
            "provision_id": provision_id,
            "spec_id": spec.spec_id,
            "environment": environment.value,
            "status": "provisioned",
            "resources_created": {
                "compute_instances": spec.compute_resources.get("instances", 1),
                "load_balancers": spec.network_config.get("load_balancers", 1),
                "storage_volumes": spec.storage_config.get("volumes", 1)
            },
            "provisioning_time": 120.5,  # Simulated provisioning time
            "cost_estimate": spec.compute_resources.get("instances", 1) * 0.05  # $0.05 per hour per instance
        }

        return provisioning_result

    async def scale_deployment(self, deployment_id: str, target_replicas: int) -> Dict[str, Any]:
        """Scale deployment to target replica count"""
        if deployment_id not in self.deployment_history:
            raise ValueError(f"Deployment {deployment_id} not found")

        execution = self.deployment_history[deployment_id]

        # Simulate scaling operation
        await asyncio.sleep(0.1)

        scaling_result = {
            "deployment_id": deployment_id,
            "previous_replicas": execution.config.environment_config.get("replicas", 1),
            "target_replicas": target_replicas,
            "current_replicas": target_replicas,
            "scaling_duration": 30.0,  # Simulated scaling time
            "status": "completed"
        }

        # Update config
        execution.config.environment_config["replicas"] = target_replicas

        return scaling_result

    async def get_deployment_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """Get comprehensive deployment metrics"""
        if deployment_id in self.active_deployments:
            execution = self.active_deployments[deployment_id]
        elif deployment_id in self.deployment_history:
            execution = self.deployment_history[deployment_id]
        else:
            raise ValueError(f"Deployment {deployment_id} not found")

        metrics = {
            "deployment_id": deployment_id,
            "total_duration": None,
            "stage_durations": {},
            "success_rate": 0.0,
            "quality_gate_results": {},
            "resource_utilization": {},
            "performance_metrics": {}
        }

        # Calculate total duration
        if execution.end_time:
            metrics["total_duration"] = (execution.end_time - execution.start_time).total_seconds()

        # Calculate stage durations
        for stage_result in execution.stage_results:
            if stage_result.duration:
                metrics["stage_durations"][stage_result.stage.value] = stage_result.duration

        # Calculate success rate
        total_stages = len(execution.stage_results)
        successful_stages = len([sr for sr in execution.stage_results if sr.status == DeploymentStatus.SUCCESS])
        metrics["success_rate"] = (successful_stages / total_stages) if total_stages > 0 else 0.0

        # Aggregate quality gate results
        for stage_result in execution.stage_results:
            if "quality_gates" in stage_result.metrics:
                metrics["quality_gate_results"][stage_result.stage.value] = stage_result.metrics["quality_gates"]

        # Simulate resource utilization metrics
        metrics["resource_utilization"] = {
            "cpu_average": 65.5,
            "memory_average": 72.3,
            "disk_io_average": 45.2,
            "network_io_average": 38.7
        }

        # Simulate performance metrics
        metrics["performance_metrics"] = {
            "response_time_p95": 150.5,
            "requests_per_second": 1250.0,
            "error_rate": 0.02,
            "uptime_percentage": 99.95
        }

        return metrics

    async def manage_secrets(self, environment: Environment, action: str,
                           secret_name: str = None, secret_value: str = None) -> Dict[str, Any]:
        """Manage deployment secrets"""
        env_secrets = self.secrets_vault.setdefault(environment.value, {})

        if action == "create" or action == "update":
            if not secret_name or not secret_value:
                raise ValueError("Secret name and value required for create/update")

            env_secrets[secret_name] = secret_value
            return {
                "action": action,
                "environment": environment.value,
                "secret_name": secret_name,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }

        elif action == "delete":
            if secret_name in env_secrets:
                del env_secrets[secret_name]
                return {
                    "action": action,
                    "environment": environment.value,
                    "secret_name": secret_name,
                    "status": "deleted",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise ValueError(f"Secret {secret_name} not found")

        elif action == "list":
            return {
                "action": action,
                "environment": environment.value,
                "secrets": list(env_secrets.keys()),
                "count": len(env_secrets),
                "timestamp": datetime.now().isoformat()
            }

        else:
            raise ValueError(f"Unknown action: {action}")

    # Helper methods

    async def _validate_deployment_config(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Validate deployment configuration"""
        errors = []

        # Validate application name
        if not config.application_name:
            errors.append("Application name is required")

        # Validate version
        if not config.version:
            errors.append("Version is required")

        # Validate quality gates
        for gate in config.quality_gates:
            if gate.threshold < 0 or gate.threshold > 100:
                errors.append(f"Quality gate {gate.name} threshold must be between 0 and 100")

        # Validate environment config
        required_env_keys = ["replicas", "cpu_limit", "memory_limit"]
        for key in required_env_keys:
            if key not in config.environment_config:
                errors.append(f"Environment config missing required key: {key}")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    async def _get_pipeline_stages(self, config: DeploymentConfig) -> List[DeploymentStage]:
        """Get pipeline stages based on deployment config"""
        base_stages = [
            DeploymentStage.CODE_CHECKOUT,
            DeploymentStage.DEPENDENCY_INSTALLATION,
            DeploymentStage.UNIT_TESTS,
            DeploymentStage.INTEGRATION_TESTS,
            DeploymentStage.SECURITY_SCAN,
            DeploymentStage.BUILD_ARTIFACTS
        ]

        # Add environment-specific stages
        if config.environment == Environment.STAGING:
            base_stages.append(DeploymentStage.DEPLOY_STAGING)
            base_stages.append(DeploymentStage.SMOKE_TESTS)
        elif config.environment == Environment.PRODUCTION:
            base_stages.extend([
                DeploymentStage.DEPLOY_STAGING,
                DeploymentStage.SMOKE_TESTS,
                DeploymentStage.DEPLOY_PRODUCTION,
                DeploymentStage.POST_DEPLOYMENT_VALIDATION
            ])

        return base_stages

    async def _execute_stage(self, stage: DeploymentStage, config: DeploymentConfig) -> StageResult:
        """Execute individual pipeline stage"""
        stage_result = StageResult(
            stage=stage,
            status=DeploymentStatus.IN_PROGRESS,
            start_time=datetime.now()
        )

        # Simulate stage execution
        await asyncio.sleep(0.05)  # Simulate processing time

        # Mock stage-specific logic
        if stage == DeploymentStage.CODE_CHECKOUT:
            stage_result.complete(
                DeploymentStatus.SUCCESS,
                f"Successfully checked out version {config.version}"
            )

        elif stage == DeploymentStage.DEPENDENCY_INSTALLATION:
            stage_result.complete(
                DeploymentStatus.SUCCESS,
                "Dependencies installed successfully"
            )
            stage_result.metrics["dependencies_installed"] = 25

        elif stage == DeploymentStage.UNIT_TESTS:
            # Simulate test execution
            stage_result.metrics["tests_run"] = 150
            stage_result.metrics["tests_passed"] = 148
            stage_result.metrics["coverage_percentage"] = 96.5
            stage_result.complete(
                DeploymentStatus.SUCCESS,
                "Unit tests completed: 148/150 passed"
            )

        elif stage == DeploymentStage.INTEGRATION_TESTS:
            stage_result.metrics["integration_tests_run"] = 45
            stage_result.metrics["integration_tests_passed"] = 45
            stage_result.complete(
                DeploymentStatus.SUCCESS,
                "Integration tests completed: 45/45 passed"
            )

        elif stage == DeploymentStage.SECURITY_SCAN:
            stage_result.metrics["vulnerabilities_found"] = 0
            stage_result.metrics["security_score"] = 98.5
            stage_result.complete(
                DeploymentStatus.SUCCESS,
                "Security scan completed: No critical vulnerabilities found"
            )

        elif stage == DeploymentStage.BUILD_ARTIFACTS:
            stage_result.metrics["artifact_size_mb"] = 45.2
            stage_result.metrics["build_time_seconds"] = 120
            stage_result.complete(
                DeploymentStatus.SUCCESS,
                "Artifacts built successfully"
            )

        elif stage == DeploymentStage.DEPLOY_STAGING:
            stage_result.metrics["deployment_time_seconds"] = 180
            stage_result.complete(
                DeploymentStatus.SUCCESS,
                f"Deployed to {config.environment.value} successfully"
            )

        elif stage == DeploymentStage.SMOKE_TESTS:
            stage_result.metrics["endpoints_tested"] = len(config.health_check_endpoints) if config.health_check_endpoints else 5
            stage_result.metrics["endpoints_healthy"] = stage_result.metrics["endpoints_tested"]
            stage_result.complete(
                DeploymentStatus.SUCCESS,
                "Smoke tests passed"
            )

        elif stage == DeploymentStage.DEPLOY_PRODUCTION:
            # Use deployment strategy
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                stage_result.metrics["deployment_strategy"] = "blue_green"
                stage_result.metrics["traffic_switch_time"] = 30
            elif config.strategy == DeploymentStrategy.ROLLING_UPDATE:
                stage_result.metrics["deployment_strategy"] = "rolling_update"
                stage_result.metrics["rolling_duration"] = 300

            stage_result.complete(
                DeploymentStatus.SUCCESS,
                f"Production deployment completed using {config.strategy.value}"
            )

        elif stage == DeploymentStage.POST_DEPLOYMENT_VALIDATION:
            stage_result.metrics["health_check_duration"] = 60
            stage_result.metrics["performance_baseline_met"] = True
            stage_result.complete(
                DeploymentStatus.SUCCESS,
                "Post-deployment validation completed"
            )

        return stage_result

    async def _evaluate_quality_gates(self, config: DeploymentConfig,
                                     stage: DeploymentStage) -> Dict[str, Any]:
        """Evaluate quality gates for specific stage"""
        gate_results = {}

        for gate in config.quality_gates:
            # Mock evaluation based on gate name and stage
            if gate.name == "test_coverage" and stage == DeploymentStage.UNIT_TESTS:
                gate.evaluate(96.5)
            elif gate.name == "code_quality" and stage == DeploymentStage.INTEGRATION_TESTS:
                gate.evaluate(87.2)
            elif gate.name == "security_score" and stage == DeploymentStage.SECURITY_SCAN:
                gate.evaluate(98.5)
            elif gate.name == "performance_score" and stage == DeploymentStage.SMOKE_TESTS:
                gate.evaluate(92.1)

            gate_results[gate.name] = {
                "threshold": gate.threshold,
                "current_value": gate.current_value,
                "passed": gate.passed,
                "blocking": gate.blocking
            }

        return gate_results

    async def _create_rollback_plan(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create rollback plan for deployment"""
        return {
            "rollback_strategy": "previous_version",
            "rollback_version": f"v{float(config.version.replace('v', '')) - 0.1:.1f}",
            "rollback_duration_estimate": 300,  # 5 minutes
            "rollback_validation_steps": [
                "health_check_validation",
                "performance_baseline_validation",
                "smoke_test_execution"
            ]
        }

    async def _trigger_rollback(self, execution: DeploymentExecution) -> Dict[str, Any]:
        """Trigger deployment rollback"""
        rollback_start = datetime.now()

        # Simulate rollback execution
        await asyncio.sleep(0.1)

        rollback_result = {
            "rollback_triggered_at": rollback_start.isoformat(),
            "rollback_plan": execution.rollback_plan,
            "rollback_status": "completed",
            "rollback_duration": 300,
            "rollback_validation": "passed"
        }

        execution.status = DeploymentStatus.ROLLED_BACK
        execution.overall_metrics["rollback"] = rollback_result

        return rollback_result

    async def _post_deployment_validation(self, execution: DeploymentExecution) -> Dict[str, Any]:
        """Perform post-deployment validation"""
        validation_results = {
            "health_checks": {
                "status": "passed",
                "endpoints_checked": len(execution.config.health_check_endpoints) if execution.config.health_check_endpoints else 5,
                "response_time_avg": 145.2
            },
            "performance_baseline": {
                "status": "passed",
                "cpu_utilization": 45.8,
                "memory_utilization": 62.3,
                "response_time_p95": 180.5
            },
            "security_validation": {
                "status": "passed",
                "ssl_certificate": "valid",
                "security_headers": "present",
                "vulnerability_scan": "clean"
            },
            "business_metrics": {
                "user_sessions": 1250,
                "transaction_success_rate": 99.8,
                "error_rate": 0.02
            }
        }

        return validation_results

    async def _setup_default_infrastructure_templates(self):
        """Set up default infrastructure templates"""
        # Development template
        dev_template = InfrastructureSpec(
            spec_id="dev_template",
            compute_resources={
                "instances": 1,
                "cpu_cores": 2,
                "memory_gb": 4,
                "instance_type": "t3.medium"
            },
            network_config={
                "vpc": "dev-vpc",
                "subnets": ["dev-subnet-1"],
                "security_groups": ["dev-sg"],
                "load_balancers": 0
            },
            storage_config={
                "volumes": 1,
                "volume_size_gb": 20,
                "volume_type": "gp3"
            },
            security_config={
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "iam_roles": ["dev-role"]
            },
            scaling_config={
                "min_instances": 1,
                "max_instances": 2,
                "auto_scaling": False
            }
        )

        # Production template
        prod_template = InfrastructureSpec(
            spec_id="prod_template",
            compute_resources={
                "instances": 3,
                "cpu_cores": 4,
                "memory_gb": 16,
                "instance_type": "c5.xlarge"
            },
            network_config={
                "vpc": "prod-vpc",
                "subnets": ["prod-subnet-1", "prod-subnet-2", "prod-subnet-3"],
                "security_groups": ["prod-sg-web", "prod-sg-app"],
                "load_balancers": 1
            },
            storage_config={
                "volumes": 2,
                "volume_size_gb": 100,
                "volume_type": "io2"
            },
            security_config={
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "iam_roles": ["prod-role", "backup-role"],
                "security_monitoring": True
            },
            scaling_config={
                "min_instances": 3,
                "max_instances": 10,
                "auto_scaling": True,
                "target_cpu_utilization": 70
            }
        )

        self.infrastructure_templates["development"] = dev_template
        self.infrastructure_templates["production"] = prod_template

    async def _setup_secrets_vault(self):
        """Set up secrets vault with default values"""
        self.secrets_vault = {
            Environment.DEVELOPMENT.value: {
                "DATABASE_URL": "postgresql://dev_user:dev_pass@localhost/dev_db",
                "API_KEY": "dev_api_key_12345",
                "SECRET_KEY": "dev_secret_key"
            },
            Environment.STAGING.value: {
                "DATABASE_URL": "postgresql://staging_user:staging_pass@staging-db/staging_db",
                "API_KEY": "staging_api_key_67890",
                "SECRET_KEY": "staging_secret_key"
            },
            Environment.PRODUCTION.value: {
                "DATABASE_URL": "postgresql://prod_user:***@prod-db/prod_db",
                "API_KEY": "prod_api_key_***",
                "SECRET_KEY": "prod_secret_key_***"
            }
        }


@pytest.fixture
def deployment_orchestrator():
    """Fixture providing mock deployment orchestrator"""
    return MockDeploymentOrchestrator()


@pytest.fixture
def sample_quality_gates():
    """Fixture providing sample quality gates"""
    return [
        QualityGate(
            name="test_coverage",
            threshold=95.0,
            description="Unit test coverage must be at least 95%",
            blocking=True
        ),
        QualityGate(
            name="code_quality",
            threshold=85.0,
            description="Code quality score must be at least 85%",
            blocking=True
        ),
        QualityGate(
            name="security_score",
            threshold=95.0,
            description="Security scan score must be at least 95%",
            blocking=True
        ),
        QualityGate(
            name="performance_score",
            threshold=90.0,
            description="Performance score must be at least 90%",
            blocking=False
        )
    ]


@pytest.fixture
def sample_deployment_config(sample_quality_gates):
    """Fixture providing sample deployment configuration"""
    return DeploymentConfig(
        deployment_id="test_deploy_001",
        application_name="myagent-api",
        version="v1.2.3",
        environment=Environment.STAGING,
        strategy=DeploymentStrategy.ROLLING_UPDATE,
        quality_gates=sample_quality_gates,
        environment_config={
            "replicas": 2,
            "cpu_limit": "1000m",
            "memory_limit": "1Gi",
            "auto_deploy": False
        },
        secrets={
            "database_url": "staging_db_connection",
            "api_key": "staging_api_key"
        },
        health_check_endpoints=[
            "/health",
            "/api/health",
            "/metrics"
        ]
    )


@pytest.fixture
def sample_infrastructure_spec():
    """Fixture providing sample infrastructure specification"""
    return InfrastructureSpec(
        spec_id="test_infra_spec",
        compute_resources={
            "instances": 2,
            "cpu_cores": 4,
            "memory_gb": 8,
            "instance_type": "c5.large"
        },
        network_config={
            "vpc": "test-vpc",
            "subnets": ["test-subnet-1", "test-subnet-2"],
            "security_groups": ["test-sg"],
            "load_balancers": 1
        },
        storage_config={
            "volumes": 2,
            "volume_size_gb": 50,
            "volume_type": "gp3"
        },
        security_config={
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "iam_roles": ["test-role"]
        },
        scaling_config={
            "min_instances": 2,
            "max_instances": 5,
            "auto_scaling": True
        }
    )


class TestDeploymentOrchestrator:
    """Comprehensive tests for Deployment Orchestrator"""

    @pytest.mark.asyncio
    async def test_deployment_orchestrator_initialization(self, deployment_orchestrator):
        """Test deployment orchestrator initialization"""
        await deployment_orchestrator.initialize()

        assert len(deployment_orchestrator.environment_configs) == 3
        assert Environment.DEVELOPMENT in deployment_orchestrator.environment_configs
        assert Environment.STAGING in deployment_orchestrator.environment_configs
        assert Environment.PRODUCTION in deployment_orchestrator.environment_configs

        # Verify infrastructure templates were created
        assert "development" in deployment_orchestrator.infrastructure_templates
        assert "production" in deployment_orchestrator.infrastructure_templates

        # Verify secrets vault was initialized
        assert Environment.DEVELOPMENT.value in deployment_orchestrator.secrets_vault
        assert Environment.PRODUCTION.value in deployment_orchestrator.secrets_vault

    @pytest.mark.asyncio
    async def test_deployment_config_validation(self, deployment_orchestrator, sample_deployment_config):
        """Test deployment configuration validation"""
        await deployment_orchestrator.initialize()

        # Test valid configuration
        validation_result = await deployment_orchestrator._validate_deployment_config(sample_deployment_config)
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0

        # Test invalid configuration - missing application name
        invalid_config = DeploymentConfig(
            deployment_id="invalid_deploy",
            application_name="",  # Invalid: empty
            version="v1.0.0",
            environment=Environment.DEVELOPMENT,
            strategy=DeploymentStrategy.RECREATE,
            quality_gates=[],
            environment_config={}  # Invalid: missing required keys
        )

        validation_result = await deployment_orchestrator._validate_deployment_config(invalid_config)
        assert validation_result["valid"] is False
        assert len(validation_result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_quality_gate_evaluation(self, sample_quality_gates):
        """Test quality gate evaluation logic"""
        coverage_gate = sample_quality_gates[0]  # test_coverage, threshold 95.0

        # Test passing evaluation
        assert coverage_gate.evaluate(96.5) is True
        assert coverage_gate.passed is True
        assert coverage_gate.current_value == 96.5

        # Test failing evaluation
        assert coverage_gate.evaluate(92.0) is False
        assert coverage_gate.passed is False
        assert coverage_gate.current_value == 92.0

        # Test exact threshold
        assert coverage_gate.evaluate(95.0) is True
        assert coverage_gate.passed is True

    @pytest.mark.asyncio
    async def test_deployment_creation(self, deployment_orchestrator, sample_deployment_config):
        """Test deployment creation"""
        await deployment_orchestrator.initialize()

        deployment_id = await deployment_orchestrator.create_deployment(sample_deployment_config)

        assert deployment_id == sample_deployment_config.deployment_id
        assert deployment_id in deployment_orchestrator.active_deployments

        # Verify deployment execution object
        execution = deployment_orchestrator.active_deployments[deployment_id]
        assert execution.deployment_id == deployment_id
        assert execution.config == sample_deployment_config
        assert execution.status == DeploymentStatus.PENDING
        assert execution.rollback_plan is not None

    @pytest.mark.asyncio
    async def test_deployment_creation_with_invalid_config(self, deployment_orchestrator):
        """Test deployment creation with invalid configuration"""
        await deployment_orchestrator.initialize()

        invalid_config = DeploymentConfig(
            deployment_id="invalid_deploy",
            application_name="",  # Invalid
            version="v1.0.0",
            environment=Environment.DEVELOPMENT,
            strategy=DeploymentStrategy.RECREATE,
            quality_gates=[],
            environment_config={}  # Invalid
        )

        with pytest.raises(ValueError, match="Invalid deployment config"):
            await deployment_orchestrator.create_deployment(invalid_config)

    @pytest.mark.asyncio
    async def test_deployment_pipeline_execution_success(self, deployment_orchestrator, sample_deployment_config):
        """Test successful deployment pipeline execution"""
        await deployment_orchestrator.initialize()

        # Create deployment
        deployment_id = await deployment_orchestrator.create_deployment(sample_deployment_config)

        # Execute deployment
        result = await deployment_orchestrator.execute_deployment(deployment_id)

        assert result["status"] == DeploymentStatus.SUCCESS.value
        assert result["deployment_id"] == deployment_id
        assert len(result["stage_results"]) > 0

        # Verify all stages completed successfully
        stage_results = result["stage_results"]
        for stage_result in stage_results:
            assert stage_result["status"] == DeploymentStatus.SUCCESS.value

        # Verify deployment moved to history
        assert deployment_id not in deployment_orchestrator.active_deployments
        assert deployment_id in deployment_orchestrator.deployment_history

    @pytest.mark.asyncio
    async def test_deployment_pipeline_stages(self, deployment_orchestrator, sample_deployment_config):
        """Test deployment pipeline stage sequence"""
        await deployment_orchestrator.initialize()

        # Test staging environment pipeline
        stages = await deployment_orchestrator._get_pipeline_stages(sample_deployment_config)

        expected_stages = [
            DeploymentStage.CODE_CHECKOUT,
            DeploymentStage.DEPENDENCY_INSTALLATION,
            DeploymentStage.UNIT_TESTS,
            DeploymentStage.INTEGRATION_TESTS,
            DeploymentStage.SECURITY_SCAN,
            DeploymentStage.BUILD_ARTIFACTS,
            DeploymentStage.DEPLOY_STAGING,
            DeploymentStage.SMOKE_TESTS
        ]

        assert len(stages) == len(expected_stages)
        for expected_stage in expected_stages:
            assert expected_stage in stages

        # Test production environment pipeline
        sample_deployment_config.environment = Environment.PRODUCTION
        prod_stages = await deployment_orchestrator._get_pipeline_stages(sample_deployment_config)

        assert DeploymentStage.DEPLOY_PRODUCTION in prod_stages
        assert DeploymentStage.POST_DEPLOYMENT_VALIDATION in prod_stages

    @pytest.mark.asyncio
    async def test_individual_stage_execution(self, deployment_orchestrator, sample_deployment_config):
        """Test individual stage execution"""
        await deployment_orchestrator.initialize()

        # Test unit tests stage
        stage_result = await deployment_orchestrator._execute_stage(
            DeploymentStage.UNIT_TESTS,
            sample_deployment_config
        )

        assert stage_result.stage == DeploymentStage.UNIT_TESTS
        assert stage_result.status == DeploymentStatus.SUCCESS
        assert stage_result.duration is not None
        assert "tests_run" in stage_result.metrics
        assert "coverage_percentage" in stage_result.metrics

        # Test security scan stage
        security_result = await deployment_orchestrator._execute_stage(
            DeploymentStage.SECURITY_SCAN,
            sample_deployment_config
        )

        assert security_result.stage == DeploymentStage.SECURITY_SCAN
        assert security_result.status == DeploymentStatus.SUCCESS
        assert "security_score" in security_result.metrics

    @pytest.mark.asyncio
    async def test_quality_gates_integration(self, deployment_orchestrator, sample_deployment_config):
        """Test quality gates integration with pipeline"""
        await deployment_orchestrator.initialize()

        # Create deployment
        deployment_id = await deployment_orchestrator.create_deployment(sample_deployment_config)

        # Execute deployment
        result = await deployment_orchestrator.execute_deployment(deployment_id)

        # Find stage with quality gates
        unit_test_stage = next(
            (sr for sr in result["stage_results"] if sr["stage"] == DeploymentStage.UNIT_TESTS.value),
            None
        )

        assert unit_test_stage is not None
        assert "quality_gates" in unit_test_stage["metrics"]

        quality_gates = unit_test_stage["metrics"]["quality_gates"]
        assert "test_coverage" in quality_gates
        assert quality_gates["test_coverage"]["passed"] is True

    @pytest.mark.asyncio
    async def test_deployment_rollback_on_failure(self, deployment_orchestrator, sample_deployment_config):
        """Test deployment rollback on pipeline failure"""
        await deployment_orchestrator.initialize()

        # Create quality gate that will fail
        failing_gate = QualityGate(
            name="test_coverage",
            threshold=99.0,  # Very high threshold that will fail
            description="Test coverage must be 99%",
            blocking=True
        )
        sample_deployment_config.quality_gates = [failing_gate]

        # Create deployment
        deployment_id = await deployment_orchestrator.create_deployment(sample_deployment_config)

        # Execute deployment (should fail and rollback)
        result = await deployment_orchestrator.execute_deployment(deployment_id)

        # Should have failed and been rolled back
        assert result["status"] == DeploymentStatus.FAILED.value
        assert "rollback" in result["overall_metrics"]
        assert result["overall_metrics"]["rollback"]["rollback_status"] == "completed"

    @pytest.mark.asyncio
    async def test_deployment_cancellation(self, deployment_orchestrator, sample_deployment_config):
        """Test deployment cancellation"""
        await deployment_orchestrator.initialize()

        # Create deployment
        deployment_id = await deployment_orchestrator.create_deployment(sample_deployment_config)

        # Cancel deployment
        result = await deployment_orchestrator.cancel_deployment(deployment_id)

        assert result["status"] == DeploymentStatus.CANCELLED.value
        assert result["end_time"] is not None

    @pytest.mark.asyncio
    async def test_infrastructure_provisioning(self, deployment_orchestrator, sample_infrastructure_spec):
        """Test infrastructure provisioning"""
        await deployment_orchestrator.initialize()

        # Provision infrastructure
        result = await deployment_orchestrator.provision_infrastructure(
            sample_infrastructure_spec,
            Environment.STAGING
        )

        assert "provision_id" in result
        assert result["status"] == "provisioned"
        assert result["environment"] == Environment.STAGING.value
        assert "resources_created" in result
        assert "cost_estimate" in result

        # Verify resource counts match specification
        assert result["resources_created"]["compute_instances"] == sample_infrastructure_spec.compute_resources["instances"]

    @pytest.mark.asyncio
    async def test_deployment_scaling(self, deployment_orchestrator, sample_deployment_config):
        """Test deployment scaling"""
        await deployment_orchestrator.initialize()

        # Create and execute deployment first
        deployment_id = await deployment_orchestrator.create_deployment(sample_deployment_config)
        await deployment_orchestrator.execute_deployment(deployment_id)

        # Scale deployment
        scaling_result = await deployment_orchestrator.scale_deployment(deployment_id, 5)

        assert scaling_result["deployment_id"] == deployment_id
        assert scaling_result["target_replicas"] == 5
        assert scaling_result["current_replicas"] == 5
        assert scaling_result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_deployment_metrics_collection(self, deployment_orchestrator, sample_deployment_config):
        """Test deployment metrics collection"""
        await deployment_orchestrator.initialize()

        # Create and execute deployment
        deployment_id = await deployment_orchestrator.create_deployment(sample_deployment_config)
        await deployment_orchestrator.execute_deployment(deployment_id)

        # Get metrics
        metrics = await deployment_orchestrator.get_deployment_metrics(deployment_id)

        assert metrics["deployment_id"] == deployment_id
        assert "total_duration" in metrics
        assert "stage_durations" in metrics
        assert "success_rate" in metrics
        assert "resource_utilization" in metrics
        assert "performance_metrics" in metrics

        # Verify success rate calculation
        assert 0.0 <= metrics["success_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_secrets_management(self, deployment_orchestrator):
        """Test secrets management functionality"""
        await deployment_orchestrator.initialize()

        # Create secret
        create_result = await deployment_orchestrator.manage_secrets(
            Environment.DEVELOPMENT,
            "create",
            "test_secret",
            "test_value"
        )

        assert create_result["action"] == "create"
        assert create_result["status"] == "success"

        # List secrets
        list_result = await deployment_orchestrator.manage_secrets(
            Environment.DEVELOPMENT,
            "list"
        )

        assert "test_secret" in list_result["secrets"]
        assert list_result["count"] > 0

        # Update secret
        update_result = await deployment_orchestrator.manage_secrets(
            Environment.DEVELOPMENT,
            "update",
            "test_secret",
            "updated_value"
        )

        assert update_result["action"] == "update"
        assert update_result["status"] == "success"

        # Delete secret
        delete_result = await deployment_orchestrator.manage_secrets(
            Environment.DEVELOPMENT,
            "delete",
            "test_secret"
        )

        assert delete_result["action"] == "delete"
        assert delete_result["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_deployment_status_retrieval(self, deployment_orchestrator, sample_deployment_config):
        """Test deployment status retrieval"""
        await deployment_orchestrator.initialize()

        # Create deployment
        deployment_id = await deployment_orchestrator.create_deployment(sample_deployment_config)

        # Get status while active
        active_status = await deployment_orchestrator.get_deployment_status(deployment_id)
        assert active_status["status"] == DeploymentStatus.PENDING.value

        # Execute deployment
        await deployment_orchestrator.execute_deployment(deployment_id)

        # Get status from history
        completed_status = await deployment_orchestrator.get_deployment_status(deployment_id)
        assert completed_status["status"] == DeploymentStatus.SUCCESS.value

    @pytest.mark.asyncio
    async def test_deployment_with_different_strategies(self, deployment_orchestrator, sample_deployment_config):
        """Test deployment with different deployment strategies"""
        await deployment_orchestrator.initialize()

        strategies = [
            DeploymentStrategy.BLUE_GREEN,
            DeploymentStrategy.ROLLING_UPDATE,
            DeploymentStrategy.CANARY
        ]

        for strategy in strategies:
            # Configure deployment with strategy
            sample_deployment_config.deployment_id = f"deploy_{strategy.value}"
            sample_deployment_config.strategy = strategy
            sample_deployment_config.environment = Environment.PRODUCTION

            # Create and execute deployment
            deployment_id = await deployment_orchestrator.create_deployment(sample_deployment_config)
            result = await deployment_orchestrator.execute_deployment(deployment_id)

            assert result["status"] == DeploymentStatus.SUCCESS.value

            # Check that strategy-specific metrics were recorded
            prod_deploy_stage = next(
                (sr for sr in result["stage_results"]
                 if sr["stage"] == DeploymentStage.DEPLOY_PRODUCTION.value),
                None
            )

            if prod_deploy_stage:
                assert "deployment_strategy" in prod_deploy_stage["metrics"]
                assert prod_deploy_stage["metrics"]["deployment_strategy"] == strategy.value

    @pytest.mark.asyncio
    async def test_concurrent_deployments(self, deployment_orchestrator):
        """Test handling of concurrent deployments"""
        await deployment_orchestrator.initialize()

        # Create multiple deployment configs
        configs = []
        for i in range(3):
            config = DeploymentConfig(
                deployment_id=f"concurrent_deploy_{i}",
                application_name=f"app_{i}",
                version=f"v1.{i}.0",
                environment=Environment.DEVELOPMENT,
                strategy=DeploymentStrategy.RECREATE,
                quality_gates=[],
                environment_config={
                    "replicas": 1,
                    "cpu_limit": "500m",
                    "memory_limit": "512Mi"
                }
            )
            configs.append(config)

        # Create deployments concurrently
        create_tasks = [
            deployment_orchestrator.create_deployment(config)
            for config in configs
        ]
        deployment_ids = await asyncio.gather(*create_tasks)

        assert len(deployment_ids) == 3
        assert len(deployment_orchestrator.active_deployments) == 3

        # Execute deployments concurrently
        execute_tasks = [
            deployment_orchestrator.execute_deployment(deployment_id)
            for deployment_id in deployment_ids
        ]
        results = await asyncio.gather(*execute_tasks)

        assert len(results) == 3
        assert all(result["status"] == DeploymentStatus.SUCCESS.value for result in results)

    @pytest.mark.asyncio
    async def test_deployment_timeout_handling(self, deployment_orchestrator, sample_deployment_config):
        """Test deployment timeout handling"""
        await deployment_orchestrator.initialize()

        # Set very short timeout
        sample_deployment_config.max_deployment_duration = 1  # 1 second

        # Create deployment
        deployment_id = await deployment_orchestrator.create_deployment(sample_deployment_config)

        # In a real implementation, this would timeout
        # For testing, we verify the timeout configuration is stored
        execution = deployment_orchestrator.active_deployments[deployment_id]
        assert execution.config.max_deployment_duration == 1

    @pytest.mark.asyncio
    async def test_post_deployment_validation(self, deployment_orchestrator, sample_deployment_config):
        """Test post-deployment validation"""
        await deployment_orchestrator.initialize()

        # Set up production deployment with validation
        sample_deployment_config.environment = Environment.PRODUCTION
        sample_deployment_config.health_check_endpoints = ["/health", "/api/status"]

        # Create and execute deployment
        deployment_id = await deployment_orchestrator.create_deployment(sample_deployment_config)
        result = await deployment_orchestrator.execute_deployment(deployment_id)

        # Verify post-deployment validation was performed
        assert "post_deployment_validation" in result["overall_metrics"]
        validation = result["overall_metrics"]["post_deployment_validation"]

        assert "health_checks" in validation
        assert "performance_baseline" in validation
        assert "security_validation" in validation
        assert "business_metrics" in validation

        # All validations should pass
        assert validation["health_checks"]["status"] == "passed"
        assert validation["performance_baseline"]["status"] == "passed"
        assert validation["security_validation"]["status"] == "passed"


class TestDeploymentPipelineIntegration:
    """Tests for deployment pipeline integration scenarios"""

    @pytest.fixture
    def gpt5_test_data(self):
        """Load GPT-5 specific test data"""
        return TEST_DATA.get('gpt5_test_data', {}).get('deployment', {})

    @pytest.mark.asyncio
    async def test_pipeline_stages_from_test_data(self, deployment_orchestrator, gpt5_test_data):
        """Test pipeline stages configuration from test data"""
        if not gpt5_test_data or 'pipeline_stages' not in gpt5_test_data:
            pytest.skip("GPT-5 deployment test data not available")

        await deployment_orchestrator.initialize()

        expected_stages = gpt5_test_data['pipeline_stages']

        # Create deployment config
        config = DeploymentConfig(
            deployment_id="pipeline_test",
            application_name="test_app",
            version="v1.0.0",
            environment=Environment.PRODUCTION,
            strategy=DeploymentStrategy.BLUE_GREEN,
            quality_gates=[],
            environment_config={"replicas": 1, "cpu_limit": "1000m", "memory_limit": "1Gi"}
        )

        # Get pipeline stages
        stages = await deployment_orchestrator._get_pipeline_stages(config)

        # Verify expected stages are present
        stage_names = [stage.value for stage in stages]
        for expected_stage in expected_stages:
            assert expected_stage in stage_names

    @pytest.mark.asyncio
    async def test_quality_gates_from_test_data(self, deployment_orchestrator, gpt5_test_data):
        """Test quality gates configuration from test data"""
        if not gpt5_test_data or 'quality_gates' not in gpt5_test_data:
            pytest.skip("GPT-5 deployment test data not available")

        await deployment_orchestrator.initialize()

        test_gates_data = gpt5_test_data['quality_gates']

        # Create quality gates from test data
        quality_gates = []
        for gate_data in test_gates_data:
            gate = QualityGate(
                name=gate_data['name'],
                threshold=gate_data['threshold'],
                description=f"Quality gate for {gate_data['name']}",
                blocking=True
            )
            quality_gates.append(gate)

        # Create deployment with test data quality gates
        config = DeploymentConfig(
            deployment_id="quality_gate_test",
            application_name="test_app",
            version="v1.0.0",
            environment=Environment.STAGING,
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            quality_gates=quality_gates,
            environment_config={"replicas": 2, "cpu_limit": "1000m", "memory_limit": "1Gi"}
        )

        deployment_id = await deployment_orchestrator.create_deployment(config)
        result = await deployment_orchestrator.execute_deployment(deployment_id)

        # Verify quality gates were evaluated
        stage_with_gates = next(
            (sr for sr in result["stage_results"]
             if "quality_gates" in sr.get("metrics", {})),
            None
        )

        assert stage_with_gates is not None
        quality_gate_results = stage_with_gates["metrics"]["quality_gates"]

        # Verify all test data quality gates were evaluated
        for gate_data in test_gates_data:
            assert gate_data['name'] in quality_gate_results

    @pytest.mark.asyncio
    async def test_deployment_strategies_from_test_data(self, deployment_orchestrator, gpt5_test_data):
        """Test deployment strategies from test data"""
        if not gpt5_test_data or 'deployment_strategies' not in gpt5_test_data:
            pytest.skip("GPT-5 deployment test data not available")

        await deployment_orchestrator.initialize()

        test_strategies = gpt5_test_data['deployment_strategies']

        for strategy_name in test_strategies:
            # Map string to enum
            strategy_mapping = {
                "rolling_update": DeploymentStrategy.ROLLING_UPDATE,
                "blue_green": DeploymentStrategy.BLUE_GREEN,
                "canary": DeploymentStrategy.CANARY
            }

            if strategy_name not in strategy_mapping:
                continue

            strategy = strategy_mapping[strategy_name]

            # Create deployment with strategy
            config = DeploymentConfig(
                deployment_id=f"strategy_test_{strategy_name}",
                application_name="test_app",
                version="v1.0.0",
                environment=Environment.PRODUCTION,
                strategy=strategy,
                quality_gates=[],
                environment_config={"replicas": 3, "cpu_limit": "1000m", "memory_limit": "1Gi"}
            )

            deployment_id = await deployment_orchestrator.create_deployment(config)
            result = await deployment_orchestrator.execute_deployment(deployment_id)

            assert result["status"] == DeploymentStatus.SUCCESS.value

    @pytest.mark.asyncio
    async def test_multi_environment_deployment_workflow(self, deployment_orchestrator, sample_deployment_config):
        """Test complete multi-environment deployment workflow"""
        await deployment_orchestrator.initialize()

        environments = [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]
        deployment_results = {}

        for env in environments:
            # Configure for environment
            config = DeploymentConfig(
                deployment_id=f"multi_env_{env.value}",
                application_name="myagent-api",
                version="v1.2.3",
                environment=env,
                strategy=DeploymentStrategy.ROLLING_UPDATE if env == Environment.PRODUCTION else DeploymentStrategy.RECREATE,
                quality_gates=sample_deployment_config.quality_gates,
                environment_config=deployment_orchestrator.environment_configs[env]
            )

            # Create and execute deployment
            deployment_id = await deployment_orchestrator.create_deployment(config)
            result = await deployment_orchestrator.execute_deployment(deployment_id)

            deployment_results[env.value] = result

        # Verify all deployments succeeded
        for env_name, result in deployment_results.items():
            assert result["status"] == DeploymentStatus.SUCCESS.value

        # Verify environment-specific differences
        dev_result = deployment_results[Environment.DEVELOPMENT.value]
        prod_result = deployment_results[Environment.PRODUCTION.value]

        # Production should have more stages
        assert len(prod_result["stage_results"]) >= len(dev_result["stage_results"])

        # Production should have post-deployment validation
        prod_has_validation = any(
            sr["stage"] == DeploymentStage.POST_DEPLOYMENT_VALIDATION.value
            for sr in prod_result["stage_results"]
        )
        assert prod_has_validation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])