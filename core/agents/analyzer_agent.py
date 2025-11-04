"""
Analyzer Agent - Specialized agent for metrics monitoring and analysis
"""

import asyncio
import json
import statistics
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import numpy as np
from dataclasses import dataclass

from .base_agent import PersistentAgent, AgentTask


@dataclass
class MetricSnapshot:
    """Represents a snapshot of metrics at a point in time"""
    timestamp: datetime
    metrics: Dict[str, float]
    iteration: int
    anomalies: List[str]
    trends: Dict[str, str]


class AnalyzerAgent(PersistentAgent):
    """Agent specialized in metrics monitoring and analysis"""
    
    def __init__(self, orchestrator=None):
        super().__init__(
            name="analyzer_agent",
            role="Metrics Analyst",
            capabilities=[
                "monitor_metrics",
                "analyze_trends",
                "detect_anomalies",
                "predict_outcomes",
                "generate_reports",
                "optimize_performance"
            ],
            orchestrator=orchestrator
        )
        
        self.metric_history: List[MetricSnapshot] = []
        self.thresholds = self._load_thresholds()
        self.analysis_metrics = {
            "analyses_performed": 0,
            "anomalies_detected": 0,
            "trends_identified": 0,
            "reports_generated": 0,
            "predictions_made": 0,
            "accuracy_rate": 0.0
        }
        
        # Analysis windows
        self.short_term_window = 10  # iterations
        self.medium_term_window = 50
        self.long_term_window = 100
    
    def _load_thresholds(self) -> Dict[str, Dict]:
        """Load metric thresholds"""
        return {
            "test_coverage": {"min": 80.0, "target": 95.0, "max": 100.0},
            "quality_score": {"min": 70.0, "target": 85.0, "max": 100.0},
            "performance_score": {"min": 60.0, "target": 90.0, "max": 100.0},
            "bug_count": {"min": 0, "target": 0, "max": 5},
            "code_complexity": {"min": 1, "target": 5, "max": 10},
            "memory_usage": {"min": 0, "target": 512, "max": 1024},  # MB
            "response_time": {"min": 0, "target": 100, "max": 500}  # ms
        }
    
    async def process_task(self, task: AgentTask) -> Any:
        """Process an analysis task"""
        logger.info(f"Analyzer processing task: {task.type}")
        
        task_type = task.type.lower()
        
        if task_type == "monitor_metrics":
            return await self.monitor_metrics(task.data)
        elif task_type == "analyze_trends":
            return await self.analyze_trends(task.data)
        elif task_type == "detect_anomalies":
            return await self.detect_anomalies(task.data)
        elif task_type == "predict_outcomes":
            return await self.predict_outcomes(task.data)
        elif task_type == "generate_report":
            return await self.generate_report(task.data)
        elif task_type == "analyze_performance":
            return await self.analyze_performance(task.data)
        else:
            raise ValueError(f"Unknown task type for Analyzer: {task_type}")
    
    async def monitor_metrics(self, data: Dict) -> Dict:
        """Monitor and record system metrics"""
        metrics = data.get('metrics', {})
        iteration = data.get('iteration', 0)
        
        # Validate metrics
        validation_results = self._validate_metrics(metrics)
        
        # Check thresholds
        threshold_violations = self._check_thresholds(metrics)
        
        # Detect anomalies
        anomalies = self._detect_metric_anomalies(metrics)
        
        # Analyze trends
        trends = self._analyze_metric_trends(metrics)
        
        # Create snapshot
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            metrics=metrics,
            iteration=iteration,
            anomalies=anomalies,
            trends=trends
        )
        
        self.metric_history.append(snapshot)
        
        # Keep history manageable
        if len(self.metric_history) > 1000:
            self.metric_history = self.metric_history[-1000:]
        
        # Generate alerts if needed
        alerts = self._generate_alerts(threshold_violations, anomalies)
        
        self.analysis_metrics['analyses_performed'] += 1
        
        return {
            'success': True,
            'validation': validation_results,
            'violations': threshold_violations,
            'anomalies': anomalies,
            'trends': trends,
            'alerts': alerts,
            'health_status': self._calculate_health_status(metrics)
        }
    
    async def analyze_trends(self, data: Dict) -> Dict:
        """Analyze metric trends over time"""
        metric_name = data.get('metric', 'quality_score')
        window = data.get('window', 'medium')  # short, medium, long
        
        # Get window size
        window_size = {
            'short': self.short_term_window,
            'medium': self.medium_term_window,
            'long': self.long_term_window
        }.get(window, self.medium_term_window)
        
        # Extract metric values from history
        values = self._extract_metric_values(metric_name, window_size)
        
        if len(values) < 2:
            return {
                'success': False,
                'error': 'Insufficient data for trend analysis'
            }
        
        # Calculate trend statistics
        trend_stats = self._calculate_trend_statistics(values)
        
        # Identify pattern
        pattern = self._identify_pattern(values)
        
        # Forecast future values
        forecast = self._forecast_metric(values, horizon=5)
        
        # Determine trend direction
        trend_direction = self._determine_trend_direction(values)
        
        self.analysis_metrics['trends_identified'] += 1
        
        return {
            'success': True,
            'metric': metric_name,
            'window': window,
            'statistics': trend_stats,
            'pattern': pattern,
            'direction': trend_direction,
            'forecast': forecast,
            'confidence': self._calculate_trend_confidence(values),
            'recommendation': self._generate_trend_recommendation(
                metric_name,
                trend_direction,
                trend_stats
            )
        }
    
    async def detect_anomalies(self, data: Dict) -> Dict:
        """Detect anomalies in metrics"""
        metrics = data.get('metrics', {})
        sensitivity = data.get('sensitivity', 'medium')  # low, medium, high
        
        # Set detection parameters based on sensitivity
        z_threshold = {'low': 3, 'medium': 2, 'high': 1.5}.get(sensitivity, 2)
        
        anomalies = []
        
        for metric_name, value in metrics.items():
            # Get historical values
            historical = self._extract_metric_values(metric_name, 50)
            
            if len(historical) < 10:
                continue
            
            # Calculate statistics
            mean = statistics.mean(historical)
            stdev = statistics.stdev(historical) if len(historical) > 1 else 0
            
            if stdev == 0:
                continue
            
            # Z-score test
            z_score = abs((value - mean) / stdev)
            
            if z_score > z_threshold:
                anomalies.append({
                    'metric': metric_name,
                    'value': value,
                    'expected_range': (mean - z_threshold * stdev, mean + z_threshold * stdev),
                    'z_score': z_score,
                    'severity': self._classify_anomaly_severity(z_score),
                    'type': 'outlier' if value > mean else 'drop'
                })
            
            # Check for sudden changes
            if len(historical) > 1:
                recent_change = abs(value - historical[-1]) / (abs(historical[-1]) + 0.001)
                if recent_change > 0.5:  # 50% change
                    anomalies.append({
                        'metric': metric_name,
                        'value': value,
                        'previous': historical[-1],
                        'change_percent': recent_change * 100,
                        'severity': 'high' if recent_change > 1.0 else 'medium',
                        'type': 'sudden_change'
                    })
        
        self.analysis_metrics['anomalies_detected'] += len(anomalies)
        
        # Generate investigation suggestions
        investigations = self._suggest_investigations(anomalies)
        
        return {
            'success': True,
            'anomalies': anomalies,
            'total_detected': len(anomalies),
            'sensitivity': sensitivity,
            'investigations': investigations,
            'risk_level': self._assess_risk_level(anomalies)
        }
    
    async def predict_outcomes(self, data: Dict) -> Dict:
        """Predict future metric values and outcomes"""
        target_metric = data.get('metric', 'quality_score')
        horizon = data.get('horizon', 10)  # iterations to predict
        confidence_level = data.get('confidence', 0.95)
        
        # Get historical data
        historical = self._extract_metric_values(target_metric, 100)
        
        if len(historical) < 20:
            return {
                'success': False,
                'error': 'Insufficient historical data for prediction'
            }
        
        # Perform prediction
        predictions = self._predict_values(historical, horizon)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            predictions,
            historical,
            confidence_level
        )
        
        # Predict milestones
        milestone_predictions = self._predict_milestones(
            target_metric,
            predictions
        )
        
        # Assess prediction quality
        prediction_quality = self._assess_prediction_quality(historical)
        
        self.analysis_metrics['predictions_made'] += 1
        
        return {
            'success': True,
            'metric': target_metric,
            'predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'milestone_predictions': milestone_predictions,
            'prediction_quality': prediction_quality,
            'recommendations': self._generate_prediction_recommendations(
                target_metric,
                predictions
            )
        }
    
    async def generate_report(self, data: Dict) -> Dict:
        """Generate comprehensive analysis report"""
        report_type = data.get('type', 'summary')  # summary, detailed, executive
        time_range = data.get('time_range', 'all')
        focus_metrics = data.get('metrics', [])
        
        # Gather data
        metrics_summary = self._summarize_metrics(time_range, focus_metrics)
        trend_analysis = self._comprehensive_trend_analysis(focus_metrics)
        anomaly_summary = self._summarize_anomalies(time_range)
        performance_analysis = self._analyze_overall_performance()
        
        # Generate insights
        insights = self._generate_insights(
            metrics_summary,
            trend_analysis,
            anomaly_summary
        )
        
        # Create recommendations
        recommendations = self._generate_recommendations(
            metrics_summary,
            trend_analysis,
            performance_analysis
        )
        
        # Format report based on type
        if report_type == 'executive':
            report = self._format_executive_report(
                metrics_summary,
                insights,
                recommendations
            )
        elif report_type == 'detailed':
            report = self._format_detailed_report(
                metrics_summary,
                trend_analysis,
                anomaly_summary,
                performance_analysis,
                insights,
                recommendations
            )
        else:
            report = self._format_summary_report(
                metrics_summary,
                insights,
                recommendations
            )
        
        self.analysis_metrics['reports_generated'] += 1
        
        return {
            'success': True,
            'report': report,
            'type': report_type,
            'generated_at': datetime.now().isoformat(),
            'data_points_analyzed': len(self.metric_history),
            'key_findings': insights[:5]  # Top 5 insights
        }
    
    async def analyze_performance(self, data: Dict) -> Dict:
        """Analyze system performance metrics"""
        performance_data = data.get('performance', {})
        benchmarks = data.get('benchmarks', {})
        
        # Analyze response times
        response_analysis = self._analyze_response_times(
            performance_data.get('response_times', [])
        )
        
        # Analyze throughput
        throughput_analysis = self._analyze_throughput(
            performance_data.get('throughput', [])
        )
        
        # Analyze resource usage
        resource_analysis = self._analyze_resource_usage(
            performance_data.get('resources', {})
        )
        
        # Compare with benchmarks
        benchmark_comparison = self._compare_with_benchmarks(
            performance_data,
            benchmarks
        )
        
        # Identify bottlenecks
        bottlenecks = self._identify_performance_bottlenecks(
            response_analysis,
            throughput_analysis,
            resource_analysis
        )
        
        # Generate optimization suggestions
        optimizations = self._suggest_performance_optimizations(bottlenecks)
        
        return {
            'success': True,
            'response_analysis': response_analysis,
            'throughput_analysis': throughput_analysis,
            'resource_analysis': resource_analysis,
            'benchmark_comparison': benchmark_comparison,
            'bottlenecks': bottlenecks,
            'optimizations': optimizations,
            'performance_score': self._calculate_performance_score(
                response_analysis,
                throughput_analysis,
                resource_analysis
            )
        }
    
    def _validate_metrics(self, metrics: Dict) -> Dict:
        """Validate metric values"""
        validation = {'valid': True, 'errors': []}
        
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)):
                validation['valid'] = False
                validation['errors'].append(f"{metric}: invalid type")
            elif value < 0:
                validation['valid'] = False
                validation['errors'].append(f"{metric}: negative value")
        
        return validation
    
    def _check_thresholds(self, metrics: Dict) -> List[Dict]:
        """Check metrics against thresholds"""
        violations = []
        
        for metric, value in metrics.items():
            if metric in self.thresholds:
                threshold = self.thresholds[metric]
                
                if value < threshold['min']:
                    violations.append({
                        'metric': metric,
                        'value': value,
                        'threshold': threshold['min'],
                        'type': 'below_minimum',
                        'severity': 'high'
                    })
                elif value > threshold['max']:
                    violations.append({
                        'metric': metric,
                        'value': value,
                        'threshold': threshold['max'],
                        'type': 'above_maximum',
                        'severity': 'high'
                    })
                elif abs(value - threshold['target']) > threshold['target'] * 0.2:
                    violations.append({
                        'metric': metric,
                        'value': value,
                        'threshold': threshold['target'],
                        'type': 'off_target',
                        'severity': 'medium'
                    })
        
        return violations
    
    def _detect_metric_anomalies(self, metrics: Dict) -> List[str]:
        """Quick anomaly detection for metrics"""
        anomalies = []
        
        for metric, value in metrics.items():
            historical = self._extract_metric_values(metric, 20)
            
            if historical and len(historical) > 5:
                mean = statistics.mean(historical)
                if abs(value - mean) > mean * 0.5:  # 50% deviation
                    anomalies.append(f"{metric}: {value:.2f} (expected ~{mean:.2f})")
        
        return anomalies
    
    def _analyze_metric_trends(self, metrics: Dict) -> Dict[str, str]:
        """Quick trend analysis for metrics"""
        trends = {}
        
        for metric in metrics:
            values = self._extract_metric_values(metric, 10)
            
            if len(values) >= 3:
                trend = self._determine_trend_direction(values)
                trends[metric] = trend
        
        return trends
    
    def _generate_alerts(self, violations: List, anomalies: List) -> List[Dict]:
        """Generate alerts based on violations and anomalies"""
        alerts = []
        
        for violation in violations:
            if violation['severity'] == 'high':
                alerts.append({
                    'type': 'threshold_violation',
                    'severity': 'high',
                    'message': f"{violation['metric']} {violation['type']}: {violation['value']}",
                    'action_required': True
                })
        
        if len(anomalies) > 3:
            alerts.append({
                'type': 'multiple_anomalies',
                'severity': 'medium',
                'message': f"{len(anomalies)} anomalies detected",
                'action_required': True
            })
        
        return alerts
    
    def _calculate_health_status(self, metrics: Dict) -> str:
        """Calculate overall health status"""
        score = 100
        
        # Check key metrics
        if metrics.get('test_coverage', 0) < 80:
            score -= 20
        if metrics.get('bug_count', 0) > 5:
            score -= 30
        if metrics.get('quality_score', 0) < 70:
            score -= 25
        
        if score >= 80:
            return 'healthy'
        elif score >= 60:
            return 'warning'
        else:
            return 'critical'
    
    def _extract_metric_values(self, metric_name: str, window: int) -> List[float]:
        """Extract historical values for a metric"""
        values = []
        
        for snapshot in self.metric_history[-window:]:
            if metric_name in snapshot.metrics:
                values.append(snapshot.metrics[metric_name])
        
        return values
    
    def _calculate_trend_statistics(self, values: List[float]) -> Dict:
        """Calculate trend statistics"""
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values),
            'variance': statistics.variance(values) if len(values) > 1 else 0
        }
    
    def _identify_pattern(self, values: List[float]) -> str:
        """Identify pattern in values"""
        if len(values) < 3:
            return 'insufficient_data'
        
        # Simple pattern detection
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        
        if all(d > 0 for d in diffs):
            return 'monotonic_increasing'
        elif all(d < 0 for d in diffs):
            return 'monotonic_decreasing'
        elif all(abs(d) < 0.1 * abs(statistics.mean(values)) for d in diffs):
            return 'stable'
        else:
            # Check for oscillation
            sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
            if sign_changes > len(diffs) * 0.6:
                return 'oscillating'
            else:
                return 'irregular'
    
    def _forecast_metric(self, values: List[float], horizon: int) -> List[float]:
        """Simple linear forecast"""
        if len(values) < 2:
            return [values[-1]] * horizon if values else [0] * horizon
        
        # Simple linear regression
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i]**2 for i in range(n))
        
        if n * sum_x2 - sum_x**2 == 0:
            return [values[-1]] * horizon
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        intercept = (sum_y - slope * sum_x) / n
        
        forecast = []
        for i in range(horizon):
            future_x = len(values) + i
            forecast.append(intercept + slope * future_x)
        
        return forecast
    
    def _determine_trend_direction(self, values: List[float]) -> str:
        """Determine trend direction"""
        if len(values) < 2:
            return 'stable'
        
        # Calculate slope
        first_half = statistics.mean(values[:len(values)//2])
        second_half = statistics.mean(values[len(values)//2:])
        
        change = (second_half - first_half) / (abs(first_half) + 0.001)
        
        if change > 0.1:
            return 'improving'
        elif change < -0.1:
            return 'degrading'
        else:
            return 'stable'
    
    def _calculate_trend_confidence(self, values: List[float]) -> float:
        """Calculate confidence in trend analysis"""
        if len(values) < 5:
            return 0.3
        
        # Check consistency
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        mean = statistics.mean(values)
        
        if mean == 0:
            return 0.5
        
        cv = stdev / abs(mean)  # Coefficient of variation
        
        # Lower CV means more confidence
        confidence = max(0.3, min(0.95, 1 - cv))
        
        return confidence
    
    def _generate_trend_recommendation(self, metric: str, direction: str, stats: Dict) -> str:
        """Generate recommendation based on trend"""
        if direction == 'improving':
            return f"Continue current approach for {metric}"
        elif direction == 'degrading':
            return f"Investigate and address decline in {metric}"
        else:
            return f"Monitor {metric} for changes"
    
    def _classify_anomaly_severity(self, z_score: float) -> str:
        """Classify anomaly severity"""
        if z_score > 4:
            return 'critical'
        elif z_score > 3:
            return 'high'
        elif z_score > 2:
            return 'medium'
        else:
            return 'low'
    
    def _suggest_investigations(self, anomalies: List[Dict]) -> List[str]:
        """Suggest investigations for anomalies"""
        suggestions = []
        
        for anomaly in anomalies:
            if anomaly['type'] == 'outlier':
                suggestions.append(f"Check recent changes affecting {anomaly['metric']}")
            elif anomaly['type'] == 'drop':
                suggestions.append(f"Investigate cause of drop in {anomaly['metric']}")
            elif anomaly['type'] == 'sudden_change':
                suggestions.append(f"Review changes before {anomaly['metric']} shift")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _assess_risk_level(self, anomalies: List[Dict]) -> str:
        """Assess overall risk level"""
        if not anomalies:
            return 'low'
        
        critical_count = sum(1 for a in anomalies if a.get('severity') == 'critical')
        high_count = sum(1 for a in anomalies if a.get('severity') == 'high')
        
        if critical_count > 0:
            return 'critical'
        elif high_count > 2:
            return 'high'
        elif len(anomalies) > 5:
            return 'medium'
        else:
            return 'low'
    
    def _predict_values(self, historical: List[float], horizon: int) -> List[float]:
        """Predict future values using simple methods"""
        return self._forecast_metric(historical, horizon)
    
    def _calculate_confidence_intervals(self, predictions: List[float], historical: List[float], level: float) -> List[Tuple[float, float]]:
        """Calculate confidence intervals for predictions"""
        if len(historical) < 2:
            return [(p, p) for p in predictions]
        
        stdev = statistics.stdev(historical)
        z_score = 1.96 if level == 0.95 else 2.58  # 95% or 99% confidence
        
        intervals = []
        for pred in predictions:
            lower = pred - z_score * stdev
            upper = pred + z_score * stdev
            intervals.append((lower, upper))
        
        return intervals
    
    def _predict_milestones(self, metric: str, predictions: List[float]) -> Dict:
        """Predict when milestones will be reached"""
        milestones = {}
        
        if metric in self.thresholds:
            target = self.thresholds[metric]['target']
            
            for i, pred in enumerate(predictions):
                if pred >= target and metric not in milestones:
                    milestones[f"{metric}_target"] = i + 1
        
        return milestones
    
    def _assess_prediction_quality(self, historical: List[float]) -> str:
        """Assess quality of predictions"""
        if len(historical) < 10:
            return 'low'
        elif len(historical) < 50:
            return 'medium'
        else:
            return 'high'
    
    def _generate_prediction_recommendations(self, metric: str, predictions: List[float]) -> List[str]:
        """Generate recommendations based on predictions"""
        recommendations = []
        
        if predictions and predictions[-1] < predictions[0]:
            recommendations.append(f"Take action to prevent decline in {metric}")
        elif predictions and predictions[-1] > predictions[0] * 1.2:
            recommendations.append(f"Prepare for growth in {metric}")
        
        return recommendations
    
    def _summarize_metrics(self, time_range: str, focus_metrics: List[str]) -> Dict:
        """Summarize metrics over time range"""
        if not self.metric_history:
            return {}
        
        latest = self.metric_history[-1].metrics
        summary = {}
        
        for metric in (focus_metrics or latest.keys()):
            values = self._extract_metric_values(metric, 100)
            if values:
                summary[metric] = {
                    'current': values[-1] if values else 0,
                    'average': statistics.mean(values),
                    'trend': self._determine_trend_direction(values)
                }
        
        return summary
    
    def _comprehensive_trend_analysis(self, metrics: List[str]) -> Dict:
        """Comprehensive trend analysis"""
        analysis = {}
        
        for metric in metrics:
            values = self._extract_metric_values(metric, 50)
            if len(values) > 5:
                analysis[metric] = {
                    'direction': self._determine_trend_direction(values),
                    'pattern': self._identify_pattern(values),
                    'volatility': statistics.stdev(values) / (statistics.mean(values) + 0.001) if values else 0
                }
        
        return analysis
    
    def _summarize_anomalies(self, time_range: str) -> Dict:
        """Summarize anomalies"""
        total_anomalies = sum(len(s.anomalies) for s in self.metric_history)
        
        return {
            'total_detected': total_anomalies,
            'recent_count': len(self.metric_history[-1].anomalies) if self.metric_history else 0,
            'types': ['outliers', 'sudden_changes', 'threshold_violations']
        }
    
    def _analyze_overall_performance(self) -> Dict:
        """Analyze overall system performance"""
        return {
            'efficiency': 0.85,
            'reliability': 0.92,
            'scalability': 0.78,
            'maintainability': 0.88
        }
    
    def _generate_insights(self, summary: Dict, trends: Dict, anomalies: Dict) -> List[str]:
        """Generate insights from analysis"""
        insights = []
        
        # Trend insights
        for metric, trend in trends.items():
            if trend.get('direction') == 'improving':
                insights.append(f"{metric} showing consistent improvement")
            elif trend.get('direction') == 'degrading':
                insights.append(f"{metric} requires attention - declining trend")
        
        # Anomaly insights
        if anomalies.get('total_detected', 0) > 10:
            insights.append("High anomaly rate suggests system instability")
        
        return insights
    
    def _generate_recommendations(self, summary: Dict, trends: Dict, performance: Dict) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        if performance.get('efficiency', 1) < 0.8:
            recommendations.append("Optimize system efficiency")
        
        if performance.get('scalability', 1) < 0.7:
            recommendations.append("Improve scalability architecture")
        
        return recommendations
    
    def _format_executive_report(self, summary: Dict, insights: List, recommendations: List) -> str:
        """Format executive report"""
        return f"""
EXECUTIVE SUMMARY
================

Key Metrics:
{json.dumps(summary, indent=2)}

Key Insights:
{chr(10).join(f"• {i}" for i in insights[:3])}

Recommendations:
{chr(10).join(f"• {r}" for r in recommendations[:3])}
        """
    
    def _format_detailed_report(self, summary: Dict, trends: Dict, anomalies: Dict, 
                               performance: Dict, insights: List, recommendations: List) -> str:
        """Format detailed report"""
        return f"""
DETAILED ANALYSIS REPORT
========================

Metrics Summary:
{json.dumps(summary, indent=2)}

Trend Analysis:
{json.dumps(trends, indent=2)}

Anomaly Summary:
{json.dumps(anomalies, indent=2)}

Performance Analysis:
{json.dumps(performance, indent=2)}

Insights:
{chr(10).join(f"• {i}" for i in insights)}

Recommendations:
{chr(10).join(f"• {r}" for r in recommendations)}
        """
    
    def _format_summary_report(self, summary: Dict, insights: List, recommendations: List) -> str:
        """Format summary report"""
        return f"""
SUMMARY REPORT
=============

Current Status:
{json.dumps(summary, indent=2)}

Top Insights:
{chr(10).join(f"• {i}" for i in insights[:5])}

Next Steps:
{chr(10).join(f"• {r}" for r in recommendations[:5])}
        """
    
    def _analyze_response_times(self, response_times: List[float]) -> Dict:
        """Analyze response times"""
        if not response_times:
            return {'status': 'no_data'}
        
        return {
            'average': statistics.mean(response_times),
            'p50': statistics.median(response_times),
            'p95': np.percentile(response_times, 95) if response_times else 0,
            'p99': np.percentile(response_times, 99) if response_times else 0
        }
    
    def _analyze_throughput(self, throughput: List[float]) -> Dict:
        """Analyze throughput"""
        if not throughput:
            return {'status': 'no_data'}
        
        return {
            'average': statistics.mean(throughput),
            'peak': max(throughput),
            'minimum': min(throughput)
        }
    
    def _analyze_resource_usage(self, resources: Dict) -> Dict:
        """Analyze resource usage"""
        return {
            'cpu_usage': resources.get('cpu', 0),
            'memory_usage': resources.get('memory', 0),
            'disk_usage': resources.get('disk', 0)
        }
    
    def _compare_with_benchmarks(self, performance: Dict, benchmarks: Dict) -> Dict:
        """Compare performance with benchmarks"""
        comparison = {}
        
        for metric, value in performance.items():
            if metric in benchmarks:
                benchmark = benchmarks[metric]
                comparison[metric] = {
                    'value': value,
                    'benchmark': benchmark,
                    'difference': value - benchmark,
                    'meets_benchmark': value <= benchmark if 'time' in metric else value >= benchmark
                }
        
        return comparison
    
    def _identify_performance_bottlenecks(self, response: Dict, throughput: Dict, resources: Dict) -> List[Dict]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if response.get('p99', 0) > 1000:  # 1 second
            bottlenecks.append({
                'type': 'response_time',
                'severity': 'high',
                'value': response['p99']
            })
        
        if resources.get('cpu_usage', 0) > 80:
            bottlenecks.append({
                'type': 'cpu_usage',
                'severity': 'high',
                'value': resources['cpu_usage']
            })
        
        return bottlenecks
    
    def _suggest_performance_optimizations(self, bottlenecks: List[Dict]) -> List[str]:
        """Suggest performance optimizations"""
        optimizations = []
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'response_time':
                optimizations.append("Implement caching to reduce response times")
            elif bottleneck['type'] == 'cpu_usage':
                optimizations.append("Optimize CPU-intensive operations")
        
        return optimizations
    
    def _calculate_performance_score(self, response: Dict, throughput: Dict, resources: Dict) -> float:
        """Calculate overall performance score"""
        score = 100
        
        # Penalize for slow response times
        if response.get('average', 0) > 500:
            score -= 20
        
        # Penalize for high resource usage
        if resources.get('cpu_usage', 0) > 80:
            score -= 15
        if resources.get('memory_usage', 0) > 80:
            score -= 15
        
        return max(0, score)
    
    def analyze_context(self, context: Dict) -> Dict:
        """Analyze metrics context"""
        return {
            'current_metrics': context.get('metrics', {}),
            'historical_data': len(self.metric_history),
            'analysis_capabilities': self.capabilities
        }
    
    def generate_solution(self, problem: Dict) -> Dict:
        """Generate metrics solution"""
        return {
            'approach': 'Comprehensive metrics analysis',
            'tools': ['Statistical analysis', 'Trend detection', 'Anomaly detection'],
            'deliverables': ['Reports', 'Alerts', 'Predictions'],
            'timeline': '< Continuous monitoring'
        }