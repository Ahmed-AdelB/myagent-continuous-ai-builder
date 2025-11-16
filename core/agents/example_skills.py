"""
Example Skills Implementation for Modular Agent Skills System

Provides concrete skill implementations to demonstrate the modular skills framework.
"""

import asyncio
import ast
import re
import subprocess
import json
from typing import Dict, Any, Set, List, Optional
from datetime import datetime
import tempfile
import os

from .modular_skills import (
    BaseSkill, SkillType, SkillComplexity, SkillContext, SkillResult
)


class CodeGenerationSkill(BaseSkill):
    """Skill for generating code based on requirements"""

    def __init__(self):
        super().__init__(
            skill_id="code_generation_v1",
            name="Code Generation",
            skill_type=SkillType.CODING,
            complexity=SkillComplexity.ADVANCED
        )
        self.tags.add("generation")
        self.tags.add("programming")

    async def execute(self, context: SkillContext) -> SkillResult:
        """Generate code based on task description"""
        start_time = datetime.now()

        try:
            task_description = context.task_description

            # Simple template-based code generation
            if "function" in task_description.lower():
                code = self._generate_function(task_description)
            elif "class" in task_description.lower():
                code = self._generate_class(task_description)
            elif "api" in task_description.lower():
                code = self._generate_api_endpoint(task_description)
            else:
                code = self._generate_generic_code(task_description)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Validate generated code
            quality_score = self._validate_code_quality(code)

            return SkillResult(
                success=True,
                output=code,
                execution_time=execution_time,
                resource_usage={"cpu": 0.1, "memory": 0.05},
                quality_metrics={"quality_score": quality_score, "lines_generated": len(code.split('\n'))}
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return SkillResult(
                success=False,
                output=None,
                execution_time=execution_time,
                resource_usage={"cpu": 0.1, "memory": 0.05},
                quality_metrics={},
                error_message=str(e)
            )

    def _generate_function(self, description: str) -> str:
        """Generate a function based on description"""
        function_name = self._extract_function_name(description)
        return f"""def {function_name}():
    \"\"\"
    {description}
    \"\"\"
    # TODO: Implement function logic
    pass
"""

    def _generate_class(self, description: str) -> str:
        """Generate a class based on description"""
        class_name = self._extract_class_name(description)
        return f"""class {class_name}:
    \"\"\"
    {description}
    \"\"\"

    def __init__(self):
        # TODO: Initialize class attributes
        pass

    def main_method(self):
        # TODO: Implement main functionality
        pass
"""

    def _generate_api_endpoint(self, description: str) -> str:
        """Generate an API endpoint based on description"""
        endpoint_name = self._extract_endpoint_name(description)
        return f"""from fastapi import APIRouter

router = APIRouter()

@router.get("/{endpoint_name}")
async def {endpoint_name}():
    \"\"\"
    {description}
    \"\"\"
    # TODO: Implement endpoint logic
    return {{"message": "Not implemented yet"}}
"""

    def _generate_generic_code(self, description: str) -> str:
        """Generate generic code structure"""
        return f"""# {description}

def main():
    \"\"\"
    Main function for: {description}
    \"\"\"
    # TODO: Implement logic
    print("Implementation needed for: {description}")

if __name__ == "__main__":
    main()
"""

    def _extract_function_name(self, description: str) -> str:
        """Extract function name from description"""
        words = re.findall(r'\b\w+\b', description.lower())
        return '_'.join(words[:3]) if words else "generated_function"

    def _extract_class_name(self, description: str) -> str:
        """Extract class name from description"""
        words = re.findall(r'\b\w+\b', description)
        return ''.join(word.capitalize() for word in words[:3]) if words else "GeneratedClass"

    def _extract_endpoint_name(self, description: str) -> str:
        """Extract endpoint name from description"""
        words = re.findall(r'\b\w+\b', description.lower())
        return '_'.join(words[:2]) if words else "endpoint"

    def _validate_code_quality(self, code: str) -> float:
        """Validate generated code quality"""
        try:
            # Check if code is valid Python
            ast.parse(code)

            # Basic quality metrics
            lines = code.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            comment_lines = [line for line in lines if line.strip().startswith('#')]

            # Quality score based on structure
            has_docstring = '"""' in code or "'''" in code
            has_comments = len(comment_lines) > 0
            reasonable_length = 5 <= len(non_empty_lines) <= 100

            score = 0.5  # Base score for valid syntax
            if has_docstring:
                score += 0.2
            if has_comments:
                score += 0.1
            if reasonable_length:
                score += 0.2

            return min(score, 1.0)

        except SyntaxError:
            return 0.0

    def get_capabilities(self) -> Set[str]:
        """Return capabilities provided by this skill"""
        return {
            "code_generation",
            "function_generation",
            "class_generation",
            "api_generation",
            "python_coding"
        }

    def get_requirements(self) -> Dict[str, Any]:
        """Return requirements for this skill"""
        return {
            "capabilities": ["text_processing"],
            "resources": {"cpu": 0.1, "memory": 0.05},
            "constraints": {
                "max_execution_time": 30.0,
                "min_quality_score": 0.5
            }
        }


class TestGenerationSkill(BaseSkill):
    """Skill for generating tests for code"""

    def __init__(self):
        super().__init__(
            skill_id="test_generation_v1",
            name="Test Generation",
            skill_type=SkillType.TESTING,
            complexity=SkillComplexity.INTERMEDIATE
        )
        self.tags.add("testing")
        self.tags.add("pytest")

    async def execute(self, context: SkillContext) -> SkillResult:
        """Generate tests for provided code"""
        start_time = datetime.now()

        try:
            code_to_test = context.input_data
            if not code_to_test:
                raise ValueError("No code provided for test generation")

            # Analyze code structure
            functions = self._extract_functions(code_to_test)
            classes = self._extract_classes(code_to_test)

            # Generate test code
            test_code = self._generate_test_file(functions, classes)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Calculate quality metrics
            test_count = test_code.count("def test_")
            quality_score = min(test_count * 0.1, 1.0)

            return SkillResult(
                success=True,
                output=test_code,
                execution_time=execution_time,
                resource_usage={"cpu": 0.05, "memory": 0.03},
                quality_metrics={
                    "quality_score": quality_score,
                    "test_count": test_count,
                    "coverage_estimate": min(test_count * 10, 100)
                }
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return SkillResult(
                success=False,
                output=None,
                execution_time=execution_time,
                resource_usage={"cpu": 0.05, "memory": 0.03},
                quality_metrics={},
                error_message=str(e)
            )

    def _extract_functions(self, code: str) -> List[str]:
        """Extract function definitions from code"""
        functions = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
        except:
            # Fallback to regex if AST parsing fails
            functions = re.findall(r'def\s+(\w+)\s*\(', code)
        return functions

    def _extract_classes(self, code: str) -> List[str]:
        """Extract class definitions from code"""
        classes = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        except:
            # Fallback to regex if AST parsing fails
            classes = re.findall(r'class\s+(\w+)[\s\(]', code)
        return classes

    def _generate_test_file(self, functions: List[str], classes: List[str]) -> str:
        """Generate comprehensive test file"""
        test_code = """import pytest
from unittest.mock import Mock, patch

# Import the module to test
# from your_module import *

"""

        # Generate tests for functions
        for func_name in functions:
            test_code += self._generate_function_test(func_name)

        # Generate tests for classes
        for class_name in classes:
            test_code += self._generate_class_test(class_name)

        return test_code

    def _generate_function_test(self, func_name: str) -> str:
        """Generate test for a specific function"""
        return f"""
def test_{func_name}_success():
    \"\"\"Test successful execution of {func_name}\"\"\"
    # Arrange
    # TODO: Set up test data

    # Act
    result = {func_name}()

    # Assert
    # TODO: Add assertions
    assert result is not None


def test_{func_name}_edge_cases():
    \"\"\"Test edge cases for {func_name}\"\"\"
    # TODO: Test edge cases
    pass


def test_{func_name}_error_handling():
    \"\"\"Test error handling in {func_name}\"\"\"
    # TODO: Test error conditions
    pass

"""

    def _generate_class_test(self, class_name: str) -> str:
        """Generate test for a specific class"""
        return f"""
class Test{class_name}:
    \"\"\"Test class for {class_name}\"\"\"

    def setup_method(self):
        \"\"\"Set up test fixtures\"\"\"
        self.instance = {class_name}()

    def test_initialization(self):
        \"\"\"Test class initialization\"\"\"
        assert self.instance is not None

    def test_main_method(self):
        \"\"\"Test main method functionality\"\"\"
        # TODO: Test main method
        pass

"""

    def get_capabilities(self) -> Set[str]:
        """Return capabilities provided by this skill"""
        return {
            "test_generation",
            "unit_test_creation",
            "pytest_generation",
            "test_analysis"
        }

    def get_requirements(self) -> Dict[str, Any]:
        """Return requirements for this skill"""
        return {
            "capabilities": ["code_analysis"],
            "resources": {"cpu": 0.05, "memory": 0.03},
            "constraints": {
                "max_execution_time": 20.0
            }
        }


class CodeAnalysisSkill(BaseSkill):
    """Skill for analyzing code quality and complexity"""

    def __init__(self):
        super().__init__(
            skill_id="code_analysis_v1",
            name="Code Analysis",
            skill_type=SkillType.ANALYSIS,
            complexity=SkillComplexity.INTERMEDIATE
        )
        self.tags.add("analysis")
        self.tags.add("quality")

    async def execute(self, context: SkillContext) -> SkillResult:
        """Analyze code quality and provide metrics"""
        start_time = datetime.now()

        try:
            code = context.input_data
            if not code:
                raise ValueError("No code provided for analysis")

            # Perform various analyses
            analysis_results = {
                "lines_of_code": self._count_lines(code),
                "cyclomatic_complexity": self._calculate_complexity(code),
                "function_count": len(self._extract_functions(code)),
                "class_count": len(self._extract_classes(code)),
                "comment_ratio": self._calculate_comment_ratio(code),
                "quality_score": self._calculate_quality_score(code)
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return SkillResult(
                success=True,
                output=analysis_results,
                execution_time=execution_time,
                resource_usage={"cpu": 0.1, "memory": 0.05},
                quality_metrics={
                    "quality_score": analysis_results["quality_score"],
                    "complexity": analysis_results["cyclomatic_complexity"]
                }
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return SkillResult(
                success=False,
                output=None,
                execution_time=execution_time,
                resource_usage={"cpu": 0.1, "memory": 0.05},
                quality_metrics={},
                error_message=str(e)
            )

    def _count_lines(self, code: str) -> Dict[str, int]:
        """Count different types of lines"""
        lines = code.split('\n')
        return {
            "total": len(lines),
            "non_empty": len([l for l in lines if l.strip()]),
            "comments": len([l for l in lines if l.strip().startswith('#')])
        }

    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity (simplified)"""
        complexity_keywords = ['if', 'elif', 'for', 'while', 'except', 'and', 'or']
        complexity = 1  # Base complexity

        for keyword in complexity_keywords:
            complexity += code.count(keyword)

        return complexity

    def _extract_functions(self, code: str) -> List[str]:
        """Extract function names from code"""
        return re.findall(r'def\s+(\w+)\s*\(', code)

    def _extract_classes(self, code: str) -> List[str]:
        """Extract class names from code"""
        return re.findall(r'class\s+(\w+)[\s\(]', code)

    def _calculate_comment_ratio(self, code: str) -> float:
        """Calculate ratio of commented lines"""
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        comment_lines = [l for l in lines if l.strip().startswith('#')]

        if not non_empty_lines:
            return 0.0

        return len(comment_lines) / len(non_empty_lines)

    def _calculate_quality_score(self, code: str) -> float:
        """Calculate overall quality score"""
        metrics = {
            "has_docstrings": '"""' in code or "'''" in code,
            "has_comments": '#' in code,
            "reasonable_complexity": self._calculate_complexity(code) < 20,
            "has_functions": 'def ' in code,
            "proper_naming": not re.search(r'\b[a-z][A-Z]', code)  # No camelCase in Python
        }

        return sum(metrics.values()) / len(metrics)

    def get_capabilities(self) -> Set[str]:
        """Return capabilities provided by this skill"""
        return {
            "code_analysis",
            "complexity_analysis",
            "quality_assessment",
            "metrics_calculation"
        }

    def get_requirements(self) -> Dict[str, Any]:
        """Return requirements for this skill"""
        return {
            "capabilities": ["text_processing"],
            "resources": {"cpu": 0.1, "memory": 0.05},
            "constraints": {
                "max_execution_time": 15.0
            }
        }


class DebuggingSkill(BaseSkill):
    """Skill for debugging code and identifying issues"""

    def __init__(self):
        super().__init__(
            skill_id="debugging_v1",
            name="Code Debugging",
            skill_type=SkillType.DEBUGGING,
            complexity=SkillComplexity.ADVANCED
        )
        self.tags.add("debugging")
        self.tags.add("error_analysis")

    async def execute(self, context: SkillContext) -> SkillResult:
        """Debug code and identify issues"""
        start_time = datetime.now()

        try:
            code = context.input_data
            if not code:
                raise ValueError("No code provided for debugging")

            # Check for syntax errors
            syntax_issues = self._check_syntax(code)

            # Check for common issues
            logical_issues = self._check_logical_issues(code)

            # Check for style issues
            style_issues = self._check_style_issues(code)

            issues_found = syntax_issues + logical_issues + style_issues

            debug_report = {
                "total_issues": len(issues_found),
                "syntax_issues": len(syntax_issues),
                "logical_issues": len(logical_issues),
                "style_issues": len(style_issues),
                "issues": issues_found,
                "severity": self._assess_severity(issues_found)
            }

            execution_time = (datetime.now() - start_time).total_seconds()
            quality_score = max(0, 1.0 - (len(issues_found) * 0.1))

            return SkillResult(
                success=True,
                output=debug_report,
                execution_time=execution_time,
                resource_usage={"cpu": 0.15, "memory": 0.08},
                quality_metrics={
                    "quality_score": quality_score,
                    "issues_found": len(issues_found)
                }
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return SkillResult(
                success=False,
                output=None,
                execution_time=execution_time,
                resource_usage={"cpu": 0.15, "memory": 0.08},
                quality_metrics={},
                error_message=str(e)
            )

    def _check_syntax(self, code: str) -> List[Dict[str, Any]]:
        """Check for syntax errors"""
        issues = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append({
                "type": "syntax_error",
                "severity": "high",
                "line": e.lineno,
                "message": str(e),
                "suggestion": "Fix syntax error"
            })
        return issues

    def _check_logical_issues(self, code: str) -> List[Dict[str, Any]]:
        """Check for logical issues"""
        issues = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()

            # Check for common logical issues
            if 'if True:' in line_stripped:
                issues.append({
                    "type": "logical_issue",
                    "severity": "medium",
                    "line": i,
                    "message": "Condition is always True",
                    "suggestion": "Review condition logic"
                })

            if 'if False:' in line_stripped:
                issues.append({
                    "type": "logical_issue",
                    "severity": "medium",
                    "line": i,
                    "message": "Condition is always False",
                    "suggestion": "Remove unreachable code"
                })

            if re.search(r'=\s*=', line_stripped):
                issues.append({
                    "type": "logical_issue",
                    "severity": "high",
                    "line": i,
                    "message": "Possible assignment instead of comparison",
                    "suggestion": "Use == for comparison"
                })

        return issues

    def _check_style_issues(self, code: str) -> List[Dict[str, Any]]:
        """Check for style issues"""
        issues = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 88:
                issues.append({
                    "type": "style_issue",
                    "severity": "low",
                    "line": i,
                    "message": "Line too long",
                    "suggestion": "Break line into multiple lines"
                })

            # Check for trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                issues.append({
                    "type": "style_issue",
                    "severity": "low",
                    "line": i,
                    "message": "Trailing whitespace",
                    "suggestion": "Remove trailing whitespace"
                })

        return issues

    def _assess_severity(self, issues: List[Dict[str, Any]]) -> str:
        """Assess overall severity of issues"""
        if any(issue["severity"] == "high" for issue in issues):
            return "high"
        elif any(issue["severity"] == "medium" for issue in issues):
            return "medium"
        else:
            return "low"

    def get_capabilities(self) -> Set[str]:
        """Return capabilities provided by this skill"""
        return {
            "debugging",
            "error_analysis",
            "syntax_checking",
            "logical_analysis",
            "style_checking"
        }

    def get_requirements(self) -> Dict[str, Any]:
        """Return requirements for this skill"""
        return {
            "capabilities": ["code_analysis"],
            "resources": {"cpu": 0.15, "memory": 0.08},
            "constraints": {
                "max_execution_time": 25.0
            }
        }


class OptimizationSkill(BaseSkill):
    """Skill for optimizing code performance"""

    def __init__(self):
        super().__init__(
            skill_id="optimization_v1",
            name="Code Optimization",
            skill_type=SkillType.OPTIMIZATION,
            complexity=SkillComplexity.EXPERT
        )
        self.tags.add("optimization")
        self.tags.add("performance")

    async def execute(self, context: SkillContext) -> SkillResult:
        """Optimize code for better performance"""
        start_time = datetime.now()

        try:
            code = context.input_data
            if not code:
                raise ValueError("No code provided for optimization")

            # Analyze current code
            analysis = self._analyze_performance_bottlenecks(code)

            # Generate optimization suggestions
            optimizations = self._generate_optimizations(code, analysis)

            # Apply automatic optimizations if requested
            optimized_code = code
            if context.constraints.get("auto_apply", False):
                optimized_code = self._apply_optimizations(code, optimizations)

            optimization_report = {
                "original_analysis": analysis,
                "optimizations": optimizations,
                "optimized_code": optimized_code,
                "performance_improvement": self._estimate_improvement(optimizations)
            }

            execution_time = (datetime.now() - start_time).total_seconds()
            quality_score = min(len(optimizations) * 0.1, 1.0)

            return SkillResult(
                success=True,
                output=optimization_report,
                execution_time=execution_time,
                resource_usage={"cpu": 0.2, "memory": 0.1},
                quality_metrics={
                    "quality_score": quality_score,
                    "optimizations_found": len(optimizations)
                }
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return SkillResult(
                success=False,
                output=None,
                execution_time=execution_time,
                resource_usage={"cpu": 0.2, "memory": 0.1},
                quality_metrics={},
                error_message=str(e)
            )

    def _analyze_performance_bottlenecks(self, code: str) -> Dict[str, Any]:
        """Analyze code for performance bottlenecks"""
        analysis = {
            "nested_loops": 0,
            "string_concatenation": 0,
            "global_variables": 0,
            "inefficient_imports": 0,
            "complexity_score": 0
        }

        lines = code.split('\n')
        indent_level = 0
        max_nesting = 0

        for line in lines:
            stripped = line.strip()

            # Count nesting level
            current_indent = len(line) - len(line.lstrip())
            if current_indent > indent_level:
                indent_level = current_indent
                max_nesting = max(max_nesting, indent_level // 4)

            # Check for nested loops
            if stripped.startswith(('for ', 'while ')) and indent_level > 0:
                analysis["nested_loops"] += 1

            # Check for string concatenation in loops
            if '+=' in stripped and '"' in stripped:
                analysis["string_concatenation"] += 1

            # Check for global variables
            if stripped.startswith('global '):
                analysis["global_variables"] += 1

            # Check for inefficient imports
            if stripped.startswith('from ') and '*' in stripped:
                analysis["inefficient_imports"] += 1

        analysis["complexity_score"] = max_nesting
        return analysis

    def _generate_optimizations(self, code: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization suggestions"""
        optimizations = []

        if analysis["nested_loops"] > 0:
            optimizations.append({
                "type": "loop_optimization",
                "priority": "high",
                "description": "Consider using list comprehensions or vectorized operations",
                "estimated_improvement": "50-80%"
            })

        if analysis["string_concatenation"] > 0:
            optimizations.append({
                "type": "string_optimization",
                "priority": "medium",
                "description": "Use join() instead of += for string concatenation",
                "estimated_improvement": "30-60%"
            })

        if analysis["global_variables"] > 2:
            optimizations.append({
                "type": "scope_optimization",
                "priority": "medium",
                "description": "Minimize global variable usage",
                "estimated_improvement": "10-25%"
            })

        if analysis["inefficient_imports"] > 0:
            optimizations.append({
                "type": "import_optimization",
                "priority": "low",
                "description": "Use specific imports instead of wildcard imports",
                "estimated_improvement": "5-15%"
            })

        return optimizations

    def _apply_optimizations(self, code: str, optimizations: List[Dict[str, Any]]) -> str:
        """Apply automatic optimizations"""
        optimized = code

        for opt in optimizations:
            if opt["type"] == "string_optimization":
                # Simple string concatenation optimization
                optimized = re.sub(
                    r'(\w+)\s*\+=\s*["\']([^"\']*)["\']',
                    r'\1 = "".join([\1, "\2"])',
                    optimized
                )

            elif opt["type"] == "import_optimization":
                # Remove wildcard imports (basic)
                optimized = re.sub(
                    r'from\s+(\w+)\s+import\s+\*',
                    r'# TODO: Use specific imports from \1',
                    optimized
                )

        return optimized

    def _estimate_improvement(self, optimizations: List[Dict[str, Any]]) -> float:
        """Estimate overall performance improvement"""
        if not optimizations:
            return 0.0

        total_improvement = 0
        for opt in optimizations:
            improvement_str = opt.get("estimated_improvement", "0%")
            # Extract average improvement percentage
            percentages = re.findall(r'(\d+)', improvement_str)
            if percentages:
                avg_improvement = sum(int(p) for p in percentages) / len(percentages)
                total_improvement += avg_improvement

        return min(total_improvement / 100, 0.9)  # Cap at 90%

    def get_capabilities(self) -> Set[str]:
        """Return capabilities provided by this skill"""
        return {
            "code_optimization",
            "performance_analysis",
            "bottleneck_detection",
            "efficiency_improvement"
        }

    def get_requirements(self) -> Dict[str, Any]:
        """Return requirements for this skill"""
        return {
            "capabilities": ["code_analysis", "performance_profiling"],
            "resources": {"cpu": 0.2, "memory": 0.1},
            "constraints": {
                "max_execution_time": 30.0
            }
        }