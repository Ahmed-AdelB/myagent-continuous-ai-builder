"""
Iteration Quality Framework - GPT-5 Recommendation #2
Formalized Iteration Evaluation with Iteration Quality Score (IQS)

Provides measurable convergence and quantifies progress through data-driven iteration control.
"""

import asyncio
import json
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of quality metrics"""
    TEST_COVERAGE = "test_coverage"
    PERFORMANCE = "performance"
    CODE_QUALITY = "code_quality"
    UX_METRICS = "ux_metrics"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    RELIABILITY = "reliability"


@dataclass
class QualityThresholds:
    """Quality thresholds configuration"""
    test_coverage_threshold: float = 0.85
    performance_threshold: float = 0.80
    code_quality_threshold: float = 0.75
    ux_threshold: float = 0.80
    security_threshold: float = 0.90
    maintainability_threshold: float = 0.75
    reliability_threshold: float = 0.95


@dataclass
class QualityMetric:
    """Individual quality metric measurement"""
    metric_type: MetricType
    name: str
    value: float  # 0.0 to 1.0
    weight: float  # Importance weight
    threshold: float  # Minimum acceptable value
    target: float  # Target value for excellence
    timestamp: datetime
    details: Dict[str, any] = None


@dataclass
class IterationQualityScore:
    """Complete quality assessment for an iteration"""
    iteration_id: int
    overall_score: float  # 0.0 to 1.0
    weighted_score: float  # Weighted average
    metrics: List[QualityMetric]
    improvement_delta: float  # Change from previous iteration
    convergence_indicator: float  # 0.0 to 1.0
    quality_trends: Dict[str, List[float]]
    recommendations: List[str]
    timestamp: datetime


class IterationQualityFramework:
    """
    Framework for evaluating and tracking iteration quality

    Implements GPT-5 recommendation for measurable convergence
    and data-driven iteration control
    """

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.iteration_scores: List[IterationQualityScore] = []
        self.baseline_metrics: Optional[Dict[str, float]] = None

        # Default metric weights (can be customized per project)
        self.metric_weights = {
            MetricType.TEST_COVERAGE: 0.25,
            MetricType.PERFORMANCE: 0.20,
            MetricType.CODE_QUALITY: 0.20,
            MetricType.UX_METRICS: 0.15,
            MetricType.MAINTAINABILITY: 0.10,
            MetricType.SECURITY: 0.05,
            MetricType.RELIABILITY: 0.05
        }

        # Quality thresholds
        self.quality_thresholds = {
            MetricType.TEST_COVERAGE: {"min": 0.80, "target": 0.95},
            MetricType.PERFORMANCE: {"min": 0.70, "target": 0.90},
            MetricType.CODE_QUALITY: {"min": 0.75, "target": 0.90},
            MetricType.UX_METRICS: {"min": 0.65, "target": 0.85},
            MetricType.MAINTAINABILITY: {"min": 0.70, "target": 0.85},
            MetricType.SECURITY: {"min": 0.90, "target": 0.98},
            MetricType.RELIABILITY: {"min": 0.85, "target": 0.95}
        }

        logger.info(f"Iteration Quality Framework initialized for: {project_name}")

    async def evaluate_iteration(
        self,
        iteration_id: int,
        test_results: Dict = None,
        performance_metrics: Dict = None,
        code_analysis: Dict = None,
        ux_feedback: Dict = None
    ) -> IterationQualityScore:
        """
        Evaluate quality for a complete iteration

        Args:
            iteration_id: Current iteration identifier
            test_results: Test coverage, pass rate, etc.
            performance_metrics: Latency, throughput, resource usage
            code_analysis: Code quality, complexity, maintainability
            ux_feedback: User experience metrics and feedback

        Returns:
            Complete iteration quality score
        """
        # Collect all metrics
        metrics = []

        # Test Coverage Metrics
        if test_results:
            test_metrics = await self._evaluate_test_coverage(test_results)
            metrics.extend(test_metrics)

        # Performance Metrics
        if performance_metrics:
            perf_metrics = await self._evaluate_performance(performance_metrics)
            metrics.extend(perf_metrics)

        # Code Quality Metrics
        if code_analysis:
            code_metrics = await self._evaluate_code_quality(code_analysis)
            metrics.extend(code_metrics)

        # UX Metrics
        if ux_feedback:
            ux_metrics = await self._evaluate_ux_quality(ux_feedback)
            metrics.extend(ux_metrics)

        # Calculate overall scores
        overall_score = self._calculate_overall_score(metrics)
        weighted_score = self._calculate_weighted_score(metrics)

        # Calculate improvement delta
        improvement_delta = self._calculate_improvement_delta(overall_score)

        # Calculate convergence indicator
        convergence_indicator = self._calculate_convergence_indicator()

        # Generate quality trends
        quality_trends = self._generate_quality_trends()

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)

        # Create quality score object
        iqs = IterationQualityScore(
            iteration_id=iteration_id,
            overall_score=overall_score,
            weighted_score=weighted_score,
            metrics=metrics,
            improvement_delta=improvement_delta,
            convergence_indicator=convergence_indicator,
            quality_trends=quality_trends,
            recommendations=recommendations,
            timestamp=datetime.now()
        )

        # Store score
        self.iteration_scores.append(iqs)

        logger.info(f"Iteration {iteration_id} IQS: {overall_score:.3f} (Δ{improvement_delta:+.3f})")

        return iqs

    async def _evaluate_test_coverage(self, test_results: Dict) -> List[QualityMetric]:
        """Evaluate test coverage metrics"""
        metrics = []
        now = datetime.now()

        # Line coverage
        if "line_coverage" in test_results:
            metrics.append(QualityMetric(
                metric_type=MetricType.TEST_COVERAGE,
                name="line_coverage",
                value=test_results["line_coverage"],
                weight=self.metric_weights[MetricType.TEST_COVERAGE] * 0.4,
                threshold=self.quality_thresholds[MetricType.TEST_COVERAGE]["min"],
                target=self.quality_thresholds[MetricType.TEST_COVERAGE]["target"],
                timestamp=now,
                details={"raw_value": test_results.get("lines_covered", 0)}
            ))

        # Branch coverage
        if "branch_coverage" in test_results:
            metrics.append(QualityMetric(
                metric_type=MetricType.TEST_COVERAGE,
                name="branch_coverage",
                value=test_results["branch_coverage"],
                weight=self.metric_weights[MetricType.TEST_COVERAGE] * 0.3,
                threshold=self.quality_thresholds[MetricType.TEST_COVERAGE]["min"],
                target=self.quality_thresholds[MetricType.TEST_COVERAGE]["target"],
                timestamp=now
            ))

        # Test pass rate
        if "pass_rate" in test_results:
            metrics.append(QualityMetric(
                metric_type=MetricType.TEST_COVERAGE,
                name="test_pass_rate",
                value=test_results["pass_rate"],
                weight=self.metric_weights[MetricType.TEST_COVERAGE] * 0.3,
                threshold=0.95,  # 95% pass rate minimum
                target=1.0,
                timestamp=now,
                details={
                    "total_tests": test_results.get("total_tests", 0),
                    "passed_tests": test_results.get("passed_tests", 0)
                }
            ))

        return metrics

    async def _evaluate_performance(self, performance_metrics: Dict) -> List[QualityMetric]:
        """Evaluate performance metrics"""
        metrics = []
        now = datetime.now()

        # Response time
        if "avg_response_time" in performance_metrics:
            # Convert response time to quality score (lower is better)
            target_response_time = 200  # 200ms target
            max_acceptable = 1000  # 1000ms max
            response_time = performance_metrics["avg_response_time"]

            # Convert to 0-1 scale (1.0 = excellent, 0.0 = poor)
            if response_time <= target_response_time:
                score = 1.0
            elif response_time >= max_acceptable:
                score = 0.0
            else:
                score = 1.0 - ((response_time - target_response_time) / (max_acceptable - target_response_time))

            metrics.append(QualityMetric(
                metric_type=MetricType.PERFORMANCE,
                name="response_time",
                value=score,
                weight=self.metric_weights[MetricType.PERFORMANCE] * 0.5,
                threshold=self.quality_thresholds[MetricType.PERFORMANCE]["min"],
                target=self.quality_thresholds[MetricType.PERFORMANCE]["target"],
                timestamp=now,
                details={"response_time_ms": response_time}
            ))

        # Throughput
        if "throughput_rps" in performance_metrics:
            target_throughput = performance_metrics.get("target_throughput", 1000)
            actual_throughput = performance_metrics["throughput_rps"]

            # Calculate throughput score
            score = min(1.0, actual_throughput / target_throughput)

            metrics.append(QualityMetric(
                metric_type=MetricType.PERFORMANCE,
                name="throughput",
                value=score,
                weight=self.metric_weights[MetricType.PERFORMANCE] * 0.3,
                threshold=self.quality_thresholds[MetricType.PERFORMANCE]["min"],
                target=self.quality_thresholds[MetricType.PERFORMANCE]["target"],
                timestamp=now,
                details={"throughput_rps": actual_throughput}
            ))

        # Resource efficiency
        if "memory_usage" in performance_metrics:
            # Convert memory usage to efficiency score
            memory_usage = performance_metrics["memory_usage"]  # 0.0 to 1.0
            efficiency_score = 1.0 - memory_usage  # Lower usage = higher score

            metrics.append(QualityMetric(
                metric_type=MetricType.PERFORMANCE,
                name="memory_efficiency",
                value=efficiency_score,
                weight=self.metric_weights[MetricType.PERFORMANCE] * 0.2,
                threshold=0.6,  # 40% memory usage max for good score
                target=0.8,  # 20% memory usage for excellent score
                timestamp=now,
                details={"memory_usage_percent": memory_usage * 100}
            ))

        return metrics

    async def _evaluate_code_quality(self, code_analysis: Dict) -> List[QualityMetric]:
        """Evaluate code quality metrics"""
        metrics = []
        now = datetime.now()

        # Cyclomatic complexity
        if "avg_complexity" in code_analysis:
            complexity = code_analysis["avg_complexity"]
            # Convert complexity to quality score (lower complexity = higher quality)
            target_complexity = 5
            max_complexity = 15

            if complexity <= target_complexity:
                score = 1.0
            elif complexity >= max_complexity:
                score = 0.0
            else:
                score = 1.0 - ((complexity - target_complexity) / (max_complexity - target_complexity))

            metrics.append(QualityMetric(
                metric_type=MetricType.CODE_QUALITY,
                name="cyclomatic_complexity",
                value=score,
                weight=self.metric_weights[MetricType.CODE_QUALITY] * 0.3,
                threshold=self.quality_thresholds[MetricType.CODE_QUALITY]["min"],
                target=self.quality_thresholds[MetricType.CODE_QUALITY]["target"],
                timestamp=now,
                details={"avg_complexity": complexity}
            ))

        # Code duplication
        if "duplication_ratio" in code_analysis:
            duplication = code_analysis["duplication_ratio"]  # 0.0 to 1.0
            score = 1.0 - duplication  # Lower duplication = higher score

            metrics.append(QualityMetric(
                metric_type=MetricType.CODE_QUALITY,
                name="code_duplication",
                value=score,
                weight=self.metric_weights[MetricType.CODE_QUALITY] * 0.2,
                threshold=0.8,  # 20% duplication max for good score
                target=0.95,  # 5% duplication for excellent score
                timestamp=now,
                details={"duplication_percent": duplication * 100}
            ))

        # Documentation coverage
        if "doc_coverage" in code_analysis:
            metrics.append(QualityMetric(
                metric_type=MetricType.MAINTAINABILITY,
                name="documentation_coverage",
                value=code_analysis["doc_coverage"],
                weight=self.metric_weights[MetricType.MAINTAINABILITY] * 0.5,
                threshold=self.quality_thresholds[MetricType.MAINTAINABILITY]["min"],
                target=self.quality_thresholds[MetricType.MAINTAINABILITY]["target"],
                timestamp=now
            ))

        return metrics

    async def _evaluate_ux_quality(self, ux_feedback: Dict) -> List[QualityMetric]:
        """Evaluate UX quality metrics"""
        metrics = []
        now = datetime.now()

        # User satisfaction score
        if "satisfaction_score" in ux_feedback:
            metrics.append(QualityMetric(
                metric_type=MetricType.UX_METRICS,
                name="user_satisfaction",
                value=ux_feedback["satisfaction_score"],
                weight=self.metric_weights[MetricType.UX_METRICS] * 0.4,
                threshold=self.quality_thresholds[MetricType.UX_METRICS]["min"],
                target=self.quality_thresholds[MetricType.UX_METRICS]["target"],
                timestamp=now,
                details={"num_responses": ux_feedback.get("num_responses", 0)}
            ))

        # Task completion rate
        if "completion_rate" in ux_feedback:
            metrics.append(QualityMetric(
                metric_type=MetricType.UX_METRICS,
                name="task_completion",
                value=ux_feedback["completion_rate"],
                weight=self.metric_weights[MetricType.UX_METRICS] * 0.3,
                threshold=0.80,
                target=0.95,
                timestamp=now
            ))

        # Error rate
        if "error_rate" in ux_feedback:
            score = 1.0 - ux_feedback["error_rate"]  # Lower error rate = higher score
            metrics.append(QualityMetric(
                metric_type=MetricType.UX_METRICS,
                name="user_error_rate",
                value=score,
                weight=self.metric_weights[MetricType.UX_METRICS] * 0.3,
                threshold=0.90,  # 10% error rate max
                target=0.98,  # 2% error rate target
                timestamp=now,
                details={"error_rate_percent": ux_feedback["error_rate"] * 100}
            ))

        return metrics

    def _calculate_overall_score(self, metrics: List[QualityMetric]) -> float:
        """Calculate simple average of all metric values"""
        if not metrics:
            return 0.0

        total_score = sum(metric.value for metric in metrics)
        return total_score / len(metrics)

    def _calculate_weighted_score(self, metrics: List[QualityMetric]) -> float:
        """Calculate weighted average based on metric importance"""
        if not metrics:
            return 0.0

        weighted_sum = sum(metric.value * metric.weight for metric in metrics)
        total_weight = sum(metric.weight for metric in metrics)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_improvement_delta(self, current_score: float) -> float:
        """Calculate improvement from previous iteration"""
        if len(self.iteration_scores) == 0:
            return 0.0

        previous_score = self.iteration_scores[-1].overall_score
        return current_score - previous_score

    def _calculate_convergence_indicator(self) -> float:
        """Calculate convergence indicator based on score stability"""
        if len(self.iteration_scores) < 3:
            return 0.5  # Neutral for insufficient data

        # Get last 5 scores
        recent_scores = [iqs.overall_score for iqs in self.iteration_scores[-5:]]

        # Calculate coefficient of variation (stability indicator)
        mean_score = statistics.mean(recent_scores)
        if mean_score == 0:
            return 0.0

        std_dev = statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0
        cv = std_dev / mean_score

        # Convert to convergence score (lower CV = higher convergence)
        convergence = max(0.0, 1.0 - (cv * 2))  # CV of 0.5 = convergence score of 0.0

        return convergence

    def _generate_quality_trends(self) -> Dict[str, List[float]]:
        """Generate quality trend data for visualization"""
        trends = {}

        if not self.iteration_scores:
            return trends

        # Organize metrics by type
        for metric_type in MetricType:
            type_scores = []
            for iqs in self.iteration_scores[-10:]:  # Last 10 iterations
                type_metrics = [m for m in iqs.metrics if m.metric_type == metric_type]
                if type_metrics:
                    avg_score = sum(m.value for m in type_metrics) / len(type_metrics)
                    type_scores.append(avg_score)

            if type_scores:
                trends[metric_type.value] = type_scores

        return trends

    def _generate_recommendations(self, metrics: List[QualityMetric]) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        recommendations = []

        # Check each metric against thresholds
        for metric in metrics:
            if metric.value < metric.threshold:
                recommendations.append(
                    f"Improve {metric.name}: current {metric.value:.2f} < threshold {metric.threshold:.2f}"
                )
            elif metric.value < metric.target:
                recommendations.append(
                    f"Optimize {metric.name}: current {metric.value:.2f} < target {metric.target:.2f}"
                )

        # Add convergence recommendations
        if len(self.iteration_scores) >= 3:
            recent_improvements = [iqs.improvement_delta for iqs in self.iteration_scores[-3:]]
            if all(delta < 0.01 for delta in recent_improvements):  # Stagnation
                recommendations.append("Consider changing iteration strategy - limited improvement detected")

        return recommendations

    def get_quality_dashboard_data(self) -> Dict[str, any]:
        """Get comprehensive quality data for dashboard visualization"""
        if not self.iteration_scores:
            return {"message": "No quality data available"}

        latest = self.iteration_scores[-1]

        return {
            "current_iqs": {
                "overall_score": latest.overall_score,
                "weighted_score": latest.weighted_score,
                "improvement_delta": latest.improvement_delta,
                "convergence_indicator": latest.convergence_indicator
            },
            "metrics_by_type": {
                metric_type.value: [
                    {"name": m.name, "value": m.value, "threshold": m.threshold, "target": m.target}
                    for m in latest.metrics if m.metric_type == metric_type
                ] for metric_type in MetricType
            },
            "quality_trends": latest.quality_trends,
            "recommendations": latest.recommendations,
            "iteration_history": [
                {
                    "iteration_id": iqs.iteration_id,
                    "overall_score": iqs.overall_score,
                    "improvement_delta": iqs.improvement_delta,
                    "timestamp": iqs.timestamp.isoformat()
                }
                for iqs in self.iteration_scores[-20:]  # Last 20 iterations
            ]
        }

    def export_quality_data(self, filepath: str):
        """Export quality data to JSON file"""
        data = {
            "project_name": self.project_name,
            "metric_weights": {k.value: v for k, v in self.metric_weights.items()},
            "quality_thresholds": {k.value: v for k, v in self.quality_thresholds.items()},
            "iteration_scores": [asdict(iqs) for iqs in self.iteration_scores],
            "export_timestamp": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Quality data exported to {filepath}")