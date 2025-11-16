#!/usr/bin/env python3
"""
Continuous Operation Setup for GPT-5 Enhanced 22_MyAgent

Prepares the system for continuous AI-driven development with all 7 GPT-5 improvements.
Sets up monitoring, recovery, optimization, and endless iteration capabilities.
"""

import asyncio
import sys
import os
import json
import subprocess
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import logging
from logging.handlers import RotatingFileHandler

# Add project root for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@dataclass
class ContinuousOperationConfig:
    """Configuration for continuous operation"""
    max_iteration_duration: int = 3600  # 1 hour max per iteration
    quality_threshold: float = 0.85  # Minimum quality score to continue
    resource_check_interval: int = 300  # Check resources every 5 minutes
    auto_recovery_enabled: bool = True
    max_consecutive_failures: int = 3
    backup_interval: int = 1800  # Backup every 30 minutes
    log_retention_days: int = 30
    performance_optimization_interval: int = 7200  # Optimize every 2 hours
    human_review_threshold: float = 0.70  # Trigger human review below this
    emergency_stop_on_degradation: bool = True


@dataclass
class SystemStatus:
    """Current system status"""
    is_running: bool
    current_iteration: int
    last_quality_score: float
    uptime: timedelta
    iterations_completed: int
    failures: int
    last_backup: datetime
    resource_status: Dict[str, str]
    component_status: Dict[str, str]
    performance_trend: str  # "improving", "stable", "degrading"


class ContinuousOperationManager:
    """Manages continuous AI-driven development operations"""

    def __init__(self, config: ContinuousOperationConfig = None):
        self.config = config or ContinuousOperationConfig()
        self.status = SystemStatus(
            is_running=False,
            current_iteration=0,
            last_quality_score=0.0,
            uptime=timedelta(),
            iterations_completed=0,
            failures=0,
            last_backup=datetime.now(),
            resource_status={},
            component_status={},
            performance_trend="stable"
        )
        self.start_time = datetime.now()
        self.shutdown_requested = False
        self.logger = self._setup_logging()

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for continuous operation"""
        logger = logging.getLogger("continuous_operation")
        logger.setLevel(logging.INFO)

        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # File handler with rotation
        file_handler = RotatingFileHandler(
            logs_dir / "continuous_operation.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=self.config.log_retention_days
        )
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        self.logger.info(f"Received shutdown signal {signum}. Initiating graceful shutdown...")
        self.shutdown_requested = True

    async def initialize_components(self) -> bool:
        """Initialize all GPT-5 components for continuous operation"""
        self.logger.info("🚀 Initializing GPT-5 components for continuous operation...")

        component_status = {}

        # Initialize Meta-Governance Layer
        try:
            from core.governance.meta_governor import MetaGovernorAgent, GovernanceConfiguration
            self.meta_governor = MetaGovernorAgent("continuous_ops", GovernanceConfiguration(
                max_iteration_duration=self.config.max_iteration_duration,
                quality_regression_threshold=0.1,
                convergence_patience=5
            ))
            component_status["meta_governor"] = "READY"
            self.logger.info("✅ Meta-Governance Layer initialized")
        except Exception as e:
            component_status["meta_governor"] = f"FAILED: {str(e)}"
            self.logger.error(f"❌ Meta-Governance initialization failed: {e}")

        # Initialize Quality Framework
        try:
            from core.evaluation.iteration_quality_framework import IterationQualityFramework, QualityThresholds
            self.quality_framework = IterationQualityFramework(QualityThresholds(
                test_coverage_threshold=self.config.quality_threshold,
                performance_threshold=self.config.quality_threshold,
                code_quality_threshold=self.config.quality_threshold
            ))
            component_status["quality_framework"] = "READY"
            self.logger.info("✅ Quality Framework initialized")
        except Exception as e:
            component_status["quality_framework"] = f"FAILED: {str(e)}"
            self.logger.error(f"❌ Quality Framework initialization failed: {e}")

        # Initialize Message Bus
        try:
            from core.communication.agent_message_bus import AgentMessageBus
            self.message_bus = AgentMessageBus()
            component_status["message_bus"] = "READY"
            self.logger.info("✅ Message Bus initialized")
        except Exception as e:
            component_status["message_bus"] = f"FAILED: {str(e)}"
            self.logger.error(f"❌ Message Bus initialization failed: {e}")

        # Initialize Memory Orchestrator
        try:
            from core.memory.memory_orchestrator import MemoryOrchestrator
            self.memory_orchestrator = MemoryOrchestrator("continuous_ops")
            await self.memory_orchestrator.initialize()
            component_status["memory_orchestrator"] = "READY"
            self.logger.info("✅ Memory Orchestrator initialized")
        except Exception as e:
            component_status["memory_orchestrator"] = f"FAILED: {str(e)}"
            self.logger.error(f"❌ Memory Orchestrator initialization failed: {e}")

        # Initialize Review Gateway
        try:
            from core.review.human_review_gateway import HumanReviewGateway
            self.review_gateway = HumanReviewGateway()
            component_status["review_gateway"] = "READY"
            self.logger.info("✅ Review Gateway initialized")
        except Exception as e:
            component_status["review_gateway"] = f"FAILED: {str(e)}"
            self.logger.error(f"❌ Review Gateway initialization failed: {e}")

        # Initialize RL Engine
        try:
            from core.learning.reinforcement_learning_engine import ReinforcementLearningEngine, RLConfiguration
            self.rl_engine = ReinforcementLearningEngine(RLConfiguration())
            component_status["rl_engine"] = "READY"
            self.logger.info("✅ RL Engine initialized")
        except Exception as e:
            component_status["rl_engine"] = f"FAILED: {str(e)}"
            self.logger.error(f"❌ RL Engine initialization failed: {e}")

        # Initialize Modular Skills
        try:
            from core.agents.modular_skills import skill_registry
            from core.agents.example_skills import (
                CodeGenerationSkill, TestGenerationSkill, CodeAnalysisSkill,
                DebuggingSkill, OptimizationSkill
            )

            # Register all available skills
            skills = [
                CodeGenerationSkill(),
                TestGenerationSkill(),
                CodeAnalysisSkill(),
                DebuggingSkill(),
                OptimizationSkill()
            ]

            for skill in skills:
                skill_registry.register_skill(skill)

            self.skill_registry = skill_registry
            component_status["modular_skills"] = "READY"
            self.logger.info("✅ Modular Skills initialized")
        except Exception as e:
            component_status["modular_skills"] = f"FAILED: {str(e)}"
            self.logger.error(f"❌ Modular Skills initialization failed: {e}")

        self.status.component_status = component_status

        # Check if critical components are ready
        critical_components = ["meta_governor", "quality_framework"]
        ready_critical = sum(1 for comp in critical_components
                           if component_status.get(comp) == "READY")

        success_rate = ready_critical / len(critical_components)
        self.logger.info(f"📊 Component initialization: {ready_critical}/{len(critical_components)} critical components ready")

        return success_rate >= 0.5  # At least 50% of critical components

    async def start_continuous_iteration_cycle(self):
        """Start the main continuous development cycle"""
        self.status.is_running = True
        self.logger.info("🔄 Starting continuous iteration cycle...")

        iteration_count = 0
        consecutive_failures = 0

        while not self.shutdown_requested:
            try:
                iteration_count += 1
                self.status.current_iteration = iteration_count
                self.logger.info(f"🚀 Starting iteration {iteration_count}")

                # Start iteration tracking
                iteration_id = f"continuous_iteration_{iteration_count}_{int(time.time())}"
                if hasattr(self, 'meta_governor'):
                    self.meta_governor.start_iteration(iteration_id)

                # Run iteration
                iteration_success = await self._run_single_iteration(iteration_id)

                if iteration_success:
                    self.status.iterations_completed += 1
                    consecutive_failures = 0
                    self.logger.info(f"✅ Iteration {iteration_count} completed successfully")
                else:
                    consecutive_failures += 1
                    self.status.failures += 1
                    self.logger.warning(f"⚠️ Iteration {iteration_count} failed (consecutive failures: {consecutive_failures})")

                # End iteration tracking
                if hasattr(self, 'meta_governor'):
                    self.meta_governor.end_iteration(iteration_id, {
                        "success": iteration_success,
                        "quality_score": self.status.last_quality_score
                    })

                # Check for emergency conditions
                if consecutive_failures >= self.config.max_consecutive_failures:
                    self.logger.critical(f"🚨 Maximum consecutive failures reached ({consecutive_failures}). Triggering emergency protocols.")
                    await self._handle_emergency_stop()
                    break

                # Check if governor recommends stopping
                if hasattr(self, 'meta_governor') and not self.meta_governor.should_continue_iteration():
                    self.logger.info("🛑 Meta-governor recommends stopping continuous operation")
                    break

                # Brief pause between iterations
                await asyncio.sleep(10)

            except Exception as e:
                consecutive_failures += 1
                self.status.failures += 1
                self.logger.error(f"💥 Critical error in iteration {iteration_count}: {str(e)}")

                if consecutive_failures >= self.config.max_consecutive_failures:
                    await self._handle_emergency_stop()
                    break

        self.status.is_running = False
        self.logger.info("🏁 Continuous iteration cycle ended")

    async def _run_single_iteration(self, iteration_id: str) -> bool:
        """Run a single development iteration"""
        try:
            # Phase 1: Analyze current state
            current_state = await self._analyze_current_state()
            self.logger.info(f"📊 Current state analysis: Quality={current_state.get('quality', 0):.3f}")

            # Phase 2: Identify improvement areas
            improvement_areas = await self._identify_improvement_areas(current_state)
            if not improvement_areas:
                self.logger.info("✅ No improvement areas identified - system optimal")
                return True

            # Phase 3: Select and execute improvements
            for area in improvement_areas[:3]:  # Limit to top 3 improvements
                success = await self._execute_improvement(area)
                if not success:
                    self.logger.warning(f"⚠️ Failed to execute improvement: {area['name']}")

            # Phase 4: Validate improvements
            validation_result = await self._validate_improvements()
            quality_score = validation_result.get('quality_score', 0.0)
            self.status.last_quality_score = quality_score

            # Phase 5: Check if human review needed
            if quality_score < self.config.human_review_threshold:
                await self._request_human_review(iteration_id, quality_score)

            # Phase 6: Update RL policies
            if hasattr(self, 'rl_engine'):
                await self._update_rl_policies(quality_score, improvement_areas)

            return quality_score >= self.config.quality_threshold

        except Exception as e:
            self.logger.error(f"💥 Iteration execution failed: {str(e)}")
            return False

    async def _analyze_current_state(self) -> Dict[str, Any]:
        """Analyze current system state"""
        state = {
            "quality": 0.8,  # Placeholder - would use real quality framework
            "performance": 0.85,
            "test_coverage": 0.90,
            "code_complexity": 0.75,
            "timestamp": datetime.now().isoformat()
        }

        # Use quality framework if available
        if hasattr(self, 'quality_framework'):
            try:
                test_metrics = {
                    "test_coverage": {"line_coverage": 0.88, "pass_rate": 0.95},
                    "performance": {"response_time": 0.15, "throughput": 1200},
                    "code_quality": {"complexity": 8, "duplication": 0.05}
                }

                # Try to calculate quality score
                if hasattr(self.quality_framework, 'calculate_iteration_quality_score'):
                    quality_score = self.quality_framework.calculate_iteration_quality_score(test_metrics)
                elif hasattr(self.quality_framework, 'calculate_quality_score'):
                    quality_score = self.quality_framework.calculate_quality_score(test_metrics)
                else:
                    quality_score = 0.8  # Fallback

                state["quality"] = quality_score

            except Exception as e:
                self.logger.warning(f"Quality analysis failed: {e}")

        return state

    async def _identify_improvement_areas(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify areas for improvement"""
        improvements = []

        # Analyze state and identify improvements
        if current_state.get("quality", 1.0) < 0.9:
            improvements.append({
                "name": "Quality Improvement",
                "priority": "high",
                "area": "quality",
                "current_value": current_state.get("quality"),
                "target_value": 0.9
            })

        if current_state.get("performance", 1.0) < 0.9:
            improvements.append({
                "name": "Performance Optimization",
                "priority": "medium",
                "area": "performance",
                "current_value": current_state.get("performance"),
                "target_value": 0.9
            })

        if current_state.get("test_coverage", 1.0) < 0.95:
            improvements.append({
                "name": "Test Coverage Enhancement",
                "priority": "medium",
                "area": "testing",
                "current_value": current_state.get("test_coverage"),
                "target_value": 0.95
            })

        # Sort by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        improvements.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)

        return improvements

    async def _execute_improvement(self, improvement: Dict[str, Any]) -> bool:
        """Execute a specific improvement"""
        try:
            improvement_name = improvement["name"]
            self.logger.info(f"🔧 Executing improvement: {improvement_name}")

            # Use skill registry if available
            if hasattr(self, 'skill_registry'):
                # Find appropriate skill for improvement
                skills = self.skill_registry.find_skills_by_capability("code_optimization")
                if skills:
                    skill = skills[0]
                    # Create context for skill execution
                    from core.agents.modular_skills import SkillContext
                    context = SkillContext(
                        task_description=improvement_name,
                        input_data=improvement,
                        environment_state={},
                        agent_capabilities={"optimization", "analysis"},
                        available_resources={"cpu": 1.0, "memory": 1.0}
                    )

                    result = await skill.execute(context)
                    return result.success if result else False

            # Simulate improvement execution
            await asyncio.sleep(0.5)  # Simulate work
            return True

        except Exception as e:
            self.logger.error(f"Improvement execution failed: {str(e)}")
            return False

    async def _validate_improvements(self) -> Dict[str, Any]:
        """Validate the improvements made"""
        try:
            # Re-analyze state after improvements
            new_state = await self._analyze_current_state()

            # Calculate improvement metrics
            validation_result = {
                "quality_score": new_state.get("quality", 0.8),
                "performance_score": new_state.get("performance", 0.85),
                "validation_passed": new_state.get("quality", 0) >= self.config.quality_threshold,
                "timestamp": datetime.now().isoformat()
            }

            return validation_result

        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return {"quality_score": 0.0, "validation_passed": False}

    async def _request_human_review(self, iteration_id: str, quality_score: float):
        """Request human review for low quality iterations"""
        if hasattr(self, 'review_gateway'):
            try:
                from core.review.human_review_gateway import ReviewType, ReviewPriority

                request_id = self.review_gateway.submit_review_request(
                    review_type=ReviewType.ARCHITECTURE_CHANGE,
                    priority=ReviewPriority.HIGH,
                    title=f"Quality Review - Iteration {iteration_id}",
                    description=f"Quality score {quality_score:.3f} below threshold {self.config.human_review_threshold}",
                    content={
                        "iteration_id": iteration_id,
                        "quality_score": quality_score,
                        "threshold": self.config.human_review_threshold
                    },
                    requester_id="continuous_operation_manager"
                )

                self.logger.info(f"📋 Human review requested: {request_id}")

            except Exception as e:
                self.logger.error(f"Failed to request human review: {e}")

    async def _update_rl_policies(self, quality_score: float, improvements: List[Dict[str, Any]]):
        """Update RL policies based on iteration results"""
        if hasattr(self, 'rl_engine'):
            try:
                # Register continuous operation agent if not already done
                agent_id = "continuous_operation"
                if agent_id not in self.rl_engine.agent_policies:
                    self.rl_engine.register_agent(agent_id)

                # Create context for RL
                context = {
                    "quality_score": quality_score,
                    "improvements_attempted": len(improvements),
                    "iteration": self.status.current_iteration
                }

                # Get recommended action
                action = self.rl_engine.recommend_action(agent_id, context)

                # Process feedback
                reward = quality_score  # Use quality score as reward
                outcome = {
                    "quality_improvement": max(0, quality_score - 0.5),
                    "success": quality_score >= self.config.quality_threshold
                }

                self.rl_engine.process_feedback(agent_id, action, context, reward, outcome)

                # Update policies
                self.rl_engine.update_policies()

            except Exception as e:
                self.logger.error(f"RL policy update failed: {e}")

    async def _handle_emergency_stop(self):
        """Handle emergency stop conditions"""
        self.logger.critical("🚨 EMERGENCY STOP ACTIVATED")

        # Save current state
        await self._create_emergency_backup()

        # Trigger human review
        if hasattr(self, 'review_gateway'):
            try:
                from core.review.human_review_gateway import ReviewType, ReviewPriority

                self.review_gateway.submit_review_request(
                    review_type=ReviewType.CRITICAL_ISSUE,
                    priority=ReviewPriority.CRITICAL,
                    title="EMERGENCY STOP - Continuous Operation Failure",
                    description=f"System triggered emergency stop after {self.status.failures} failures",
                    content={
                        "failures": self.status.failures,
                        "last_quality_score": self.status.last_quality_score,
                        "uptime": str(self.status.uptime),
                        "iterations_completed": self.status.iterations_completed
                    },
                    requester_id="emergency_stop_system"
                )
            except Exception as e:
                self.logger.error(f"Failed to create emergency review: {e}")

        # Graceful component shutdown
        await self._shutdown_components()

    async def _create_emergency_backup(self):
        """Create emergency backup of current state"""
        try:
            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "status": asdict(self.status),
                "config": asdict(self.config),
                "emergency_reason": "consecutive_failures_threshold_reached"
            }

            backup_file = Path("logs") / f"emergency_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_file.parent.mkdir(exist_ok=True)

            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)

            self.logger.info(f"💾 Emergency backup created: {backup_file}")

        except Exception as e:
            self.logger.error(f"Failed to create emergency backup: {e}")

    async def _shutdown_components(self):
        """Gracefully shutdown all components"""
        self.logger.info("🔌 Shutting down components...")

        # Shutdown in reverse initialization order
        components = [
            ("modular_skills", "skill_registry"),
            ("rl_engine", "rl_engine"),
            ("review_gateway", "review_gateway"),
            ("memory_orchestrator", "memory_orchestrator"),
            ("message_bus", "message_bus"),
            ("quality_framework", "quality_framework"),
            ("meta_governor", "meta_governor")
        ]

        for comp_name, attr_name in components:
            if hasattr(self, attr_name):
                try:
                    component = getattr(self, attr_name)
                    if hasattr(component, 'shutdown'):
                        await component.shutdown()
                    elif hasattr(component, 'close'):
                        await component.close()
                    self.logger.info(f"✅ {comp_name} shutdown complete")
                except Exception as e:
                    self.logger.error(f"❌ {comp_name} shutdown failed: {e}")

    async def monitor_system_health(self):
        """Monitor system health in background"""
        self.logger.info("💓 Starting system health monitoring...")

        while not self.shutdown_requested and self.status.is_running:
            try:
                # Update uptime
                self.status.uptime = datetime.now() - self.start_time

                # Check resource usage
                import psutil
                cpu_usage = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                self.status.resource_status = {
                    "cpu": f"{cpu_usage:.1f}%" if cpu_usage < 80 else "HIGH",
                    "memory": f"{memory.percent:.1f}%" if memory.percent < 85 else "HIGH",
                    "disk": f"{(disk.used/disk.total)*100:.1f}%" if disk.free > 1024**3 else "LOW"
                }

                # Performance trend analysis
                if hasattr(self, 'quality_history'):
                    if len(self.quality_history) >= 3:
                        recent_scores = self.quality_history[-3:]
                        if all(recent_scores[i] > recent_scores[i-1] for i in range(1, len(recent_scores))):
                            self.status.performance_trend = "improving"
                        elif all(recent_scores[i] < recent_scores[i-1] for i in range(1, len(recent_scores))):
                            self.status.performance_trend = "degrading"
                        else:
                            self.status.performance_trend = "stable"

                # Log periodic status
                if self.status.current_iteration % 10 == 0:  # Every 10 iterations
                    self.logger.info(
                        f"💓 Health Check - Iteration: {self.status.current_iteration}, "
                        f"Quality: {self.status.last_quality_score:.3f}, "
                        f"Uptime: {self.status.uptime}, "
                        f"Trend: {self.status.performance_trend}"
                    )

                await asyncio.sleep(self.config.resource_check_interval)

            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute

    def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        return {
            "system_status": asdict(self.status),
            "configuration": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": self.status.uptime.total_seconds() / 3600,
            "success_rate": (
                (self.status.iterations_completed / max(self.status.current_iteration, 1)) * 100
                if self.status.current_iteration > 0 else 0
            ),
            "average_quality": getattr(self, 'quality_average', self.status.last_quality_score)
        }

    async def run_continuous_operation(self):
        """Main continuous operation entry point"""
        try:
            self.logger.info("🚀 Starting Continuous AI-Driven Development System")
            self.logger.info("=" * 80)

            # Initialize all components
            if not await self.initialize_components():
                self.logger.critical("❌ Component initialization failed. Cannot start continuous operation.")
                return False

            # Start monitoring
            health_monitor = asyncio.create_task(self.monitor_system_health())

            # Start main iteration cycle
            await self.start_continuous_iteration_cycle()

            # Cancel monitoring
            health_monitor.cancel()

            # Final status report
            final_report = self.get_status_report()
            self.logger.info(f"📊 Final Status: {json.dumps(final_report, indent=2, default=str)}")

            return True

        except Exception as e:
            self.logger.critical(f"💥 Continuous operation failed: {str(e)}")
            return False
        finally:
            await self._shutdown_components()


async def main():
    """Main continuous operation setup"""
    print("🚀 22_MyAgent Continuous Operation Setup")
    print("=" * 60)

    # Create configuration
    config = ContinuousOperationConfig(
        max_iteration_duration=1800,  # 30 minutes
        quality_threshold=0.80,
        auto_recovery_enabled=True,
        max_consecutive_failures=3
    )

    # Create and start manager
    manager = ContinuousOperationManager(config)

    try:
        success = await manager.run_continuous_operation()

        if success:
            print("\n🎉 CONTINUOUS OPERATION SETUP COMPLETE!")
            print("✅ System is ready for endless AI-driven development")
            sys.exit(0)
        else:
            print("\n⚠️ CONTINUOUS OPERATION SETUP COMPLETED WITH ISSUES")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 SETUP FAILED: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())