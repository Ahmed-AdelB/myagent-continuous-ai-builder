"""
Progress Analyzer - Analyzes development progress and provides insights
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics
from loguru import logger
import numpy as np
from pathlib import Path


@dataclass
class ProgressMetrics:
    """Metrics for tracking progress"""
    iteration: int
    timestamp: datetime
    code_changes: int
    tests_passed: int
    tests_failed: int
    coverage_percentage: float
    bugs_fixed: int
    bugs_introduced: int
    performance_score: float
    quality_score: float
    velocity: float  # Changes per hour
    estimated_completion: Optional[datetime] = None


class ProgressAnalyzer:
    """Analyzes and tracks development progress"""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.metrics_history: List[ProgressMetrics] = []
        self.velocity_window = 10  # iterations for velocity calculation
        self.quality_thresholds = {
            "test_coverage": 95.0,
            "performance": 90.0,
            "quality": 85.0,
            "bugs": 5
        }
        self.metrics_file = Path(f"persistence/metrics_{project_name}.json")
        self._load_metrics()

    def _load_metrics(self):
        """Load historical metrics"""
        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                data = json.load(f)
                for m in data:
                    metrics = ProgressMetrics(
                        iteration=m["iteration"],
                        timestamp=datetime.fromisoformat(m["timestamp"]),
                        code_changes=m["code_changes"],
                        tests_passed=m["tests_passed"],
                        tests_failed=m["tests_failed"],
                        coverage_percentage=m["coverage_percentage"],
                        bugs_fixed=m["bugs_fixed"],
                        bugs_introduced=m["bugs_introduced"],
                        performance_score=m["performance_score"],
                        quality_score=m["quality_score"],
                        velocity=m["velocity"]
                    )
                    if m.get("estimated_completion"):
                        metrics.estimated_completion = datetime.fromisoformat(
                            m["estimated_completion"]
                        )
                    self.metrics_history.append(metrics)

    def record_iteration_metrics(
        self,
        iteration: int,
        code_changes: int,
        tests_passed: int,
        tests_failed: int,
        coverage_percentage: float,
        bugs_fixed: int,
        bugs_introduced: int,
        performance_score: float,
        quality_score: float
    ) -> ProgressMetrics:
        """Record metrics for an iteration"""
        
        # Calculate velocity
        velocity = self._calculate_velocity(code_changes)
        
        # Estimate completion
        estimated_completion = self._estimate_completion(
            coverage_percentage,
            quality_score,
            velocity
        )
        
        metrics = ProgressMetrics(
            iteration=iteration,
            timestamp=datetime.now(),
            code_changes=code_changes,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            coverage_percentage=coverage_percentage,
            bugs_fixed=bugs_fixed,
            bugs_introduced=bugs_introduced,
            performance_score=performance_score,
            quality_score=quality_score,
            velocity=velocity,
            estimated_completion=estimated_completion
        )
        
        self.metrics_history.append(metrics)
        self._save_metrics()
        
        logger.info(f"Recorded metrics for iteration {iteration}: "
                   f"Quality={quality_score:.1f}%, Coverage={coverage_percentage:.1f}%")
        
        return metrics

    def _calculate_velocity(self, code_changes: int) -> float:
        """Calculate development velocity"""
        if len(self.metrics_history) < 2:
            return code_changes
            
        # Get recent metrics
        recent = self.metrics_history[-self.velocity_window:]
        
        if len(recent) < 2:
            return code_changes
            
        # Calculate time difference
        time_diff = (recent[-1].timestamp - recent[0].timestamp).total_seconds() / 3600
        
        if time_diff == 0:
            return code_changes
            
        # Calculate total changes
        total_changes = sum(m.code_changes for m in recent)
        
        return total_changes / time_diff

    def _estimate_completion(
        self,
        coverage: float,
        quality: float,
        velocity: float
    ) -> Optional[datetime]:
        """Estimate when quality targets will be met"""
        
        if velocity <= 0:
            return None
            
        # Calculate remaining work
        coverage_gap = max(0, self.quality_thresholds["test_coverage"] - coverage)
        quality_gap = max(0, self.quality_thresholds["quality"] - quality)
        
        if coverage_gap == 0 and quality_gap == 0:
            return datetime.now()
            
        # Estimate iterations needed
        if len(self.metrics_history) >= 5:
            # Calculate improvement rate
            recent_coverage = [m.coverage_percentage for m in self.metrics_history[-5:]]
            recent_quality = [m.quality_score for m in self.metrics_history[-5:]]
            
            coverage_rate = self._calculate_improvement_rate(recent_coverage)
            quality_rate = self._calculate_improvement_rate(recent_quality)
            
            if coverage_rate > 0 and quality_rate > 0:
                iterations_for_coverage = coverage_gap / coverage_rate
                iterations_for_quality = quality_gap / quality_rate
                
                estimated_iterations = max(iterations_for_coverage, iterations_for_quality)
                
                # Calculate time based on velocity
                hours_per_iteration = 1 / velocity if velocity > 0 else 24
                estimated_hours = estimated_iterations * hours_per_iteration
                
                return datetime.now() + timedelta(hours=estimated_hours)
                
        return None

    def _calculate_improvement_rate(self, values: List[float]) -> float:
        """Calculate rate of improvement"""
        if len(values) < 2:
            return 0.0
            
        # Calculate linear regression slope
        x = list(range(len(values)))
        
        # Simple linear regression
        n = len(values)
        if n == 0:
            return 0.0
            
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        return max(0, slope)  # Return positive improvement rate

    def _save_metrics(self):
        """Save metrics to file"""
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for m in self.metrics_history:
            metric_dict = {
                "iteration": m.iteration,
                "timestamp": m.timestamp.isoformat(),
                "code_changes": m.code_changes,
                "tests_passed": m.tests_passed,
                "tests_failed": m.tests_failed,
                "coverage_percentage": m.coverage_percentage,
                "bugs_fixed": m.bugs_fixed,
                "bugs_introduced": m.bugs_introduced,
                "performance_score": m.performance_score,
                "quality_score": m.quality_score,
                "velocity": m.velocity
            }
            if m.estimated_completion:
                metric_dict["estimated_completion"] = m.estimated_completion.isoformat()
            data.append(metric_dict)
            
        with open(self.metrics_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_current_status(self) -> Dict:
        """Get current progress status"""
        if not self.metrics_history:
            return {
                "status": "not_started",
                "iterations": 0,
                "quality_score": 0,
                "estimated_completion": None
            }
            
        latest = self.metrics_history[-1]
        
        # Determine status
        if latest.quality_score >= self.quality_thresholds["quality"] and \
           latest.coverage_percentage >= self.quality_thresholds["test_coverage"]:
            status = "completed"
        elif latest.velocity > 0:
            status = "in_progress"
        else:
            status = "stalled"
            
        return {
            "status": status,
            "iterations": latest.iteration,
            "quality_score": latest.quality_score,
            "coverage": latest.coverage_percentage,
            "performance": latest.performance_score,
            "velocity": latest.velocity,
            "bugs_remaining": latest.bugs_introduced - latest.bugs_fixed,
            "estimated_completion": latest.estimated_completion.isoformat() 
                if latest.estimated_completion else None
        }

    def get_trend_analysis(self) -> Dict:
        """Analyze trends in metrics"""
        if len(self.metrics_history) < 5:
            return {"status": "insufficient_data"}
            
        recent = self.metrics_history[-10:]
        
        # Calculate trends
        quality_trend = self._calculate_trend([m.quality_score for m in recent])
        coverage_trend = self._calculate_trend([m.coverage_percentage for m in recent])
        performance_trend = self._calculate_trend([m.performance_score for m in recent])
        bug_trend = self._calculate_trend(
            [m.bugs_introduced - m.bugs_fixed for m in recent]
        )
        
        return {
            "quality_trend": quality_trend,
            "coverage_trend": coverage_trend,
            "performance_trend": performance_trend,
            "bug_trend": bug_trend,
            "overall_trend": "improving" if quality_trend == "increasing" else "degrading"
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "stable"
            
        slope = self._calculate_improvement_rate(values)
        
        if slope > 0.5:
            return "increasing"
        elif slope < -0.5:
            return "decreasing"
        else:
            return "stable"

    def get_bottlenecks(self) -> List[str]:
        """Identify current bottlenecks"""
        bottlenecks = []
        
        if not self.metrics_history:
            return ["No metrics available"]
            
        latest = self.metrics_history[-1]
        
        # Check various bottlenecks
        if latest.coverage_percentage < 70:
            bottlenecks.append(f"Low test coverage: {latest.coverage_percentage:.1f}%")
            
        if latest.tests_failed > latest.tests_passed * 0.1:
            bottlenecks.append(f"High test failure rate: {latest.tests_failed} failures")
            
        if latest.bugs_introduced > latest.bugs_fixed:
            bottlenecks.append("Bug introduction rate exceeds fix rate")
            
        if latest.performance_score < 70:
            bottlenecks.append(f"Low performance score: {latest.performance_score:.1f}%")
            
        if latest.velocity < 1:
            bottlenecks.append("Low development velocity")
            
        if len(self.metrics_history) > 5:
            recent_quality = [m.quality_score for m in self.metrics_history[-5:]]
            if self._calculate_improvement_rate(recent_quality) < 0:
                bottlenecks.append("Quality score declining")
                
        return bottlenecks if bottlenecks else ["No significant bottlenecks detected"]

    def generate_report(self) -> str:
        """Generate a progress report"""
        status = self.get_current_status()
        trends = self.get_trend_analysis()
        bottlenecks = self.get_bottlenecks()
        
        report = f"""
========================================
Progress Report: {self.project_name}
========================================

Current Status: {status['status'].upper()}
Iterations Completed: {status.get('iterations', 0)}

Quality Metrics:
- Quality Score: {status.get('quality_score', 0):.1f}%
- Test Coverage: {status.get('coverage', 0):.1f}%
- Performance Score: {status.get('performance', 0):.1f}%
- Development Velocity: {status.get('velocity', 0):.2f} changes/hour

Trends:
- Quality: {trends.get('quality_trend', 'N/A')}
- Coverage: {trends.get('coverage_trend', 'N/A')}
- Performance: {trends.get('performance_trend', 'N/A')}
- Overall: {trends.get('overall_trend', 'N/A')}

Bottlenecks:
{chr(10).join(f'- {b}' for b in bottlenecks)}

Estimated Completion: {status.get('estimated_completion', 'Unable to estimate')}

========================================
        """
        
        return report.strip()

    def should_pivot_strategy(self) -> Tuple[bool, str]:
        """Determine if development strategy should change"""
        if len(self.metrics_history) < 10:
            return False, "Insufficient data"
            
        recent = self.metrics_history[-5:]
        older = self.metrics_history[-10:-5]
        
        # Compare average improvements
        recent_avg_quality = statistics.mean(m.quality_score for m in recent)
        older_avg_quality = statistics.mean(m.quality_score for m in older)
        
        if recent_avg_quality < older_avg_quality * 0.9:
            return True, "Quality declining - consider focusing on refactoring"
            
        recent_avg_velocity = statistics.mean(m.velocity for m in recent)
        if recent_avg_velocity < 0.5:
            return True, "Velocity too low - consider simplifying approach"
            
        recent_bug_rate = sum(m.bugs_introduced for m in recent) / len(recent)
        if recent_bug_rate > 10:
            return True, "High bug rate - focus on debugging and testing"
            
        return False, "Current strategy is effective"