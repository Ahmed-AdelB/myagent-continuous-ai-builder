"""
Tester Agent - Specialized agent for test generation and execution
"""

import asyncio
import subprocess
import json
import ast
import coverage
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from loguru import logger
import pytest
from io import StringIO
import sys

# LangChain imports for AI-powered test generation
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import BaseOutputParser

from .base_agent import PersistentAgent, AgentTask


class CodeOutputParser(BaseOutputParser[Dict]):
    """Parser for code generation output"""

    def parse(self, text: str) -> Dict:
        """Parse the LLM output into structured code data"""
        try:
            # Extract code blocks
            code_blocks = []
            lines = text.split('\n')
            in_code_block = False
            current_block = []
            language = None

            for line in lines:
                if line.startswith('```'):
                    if in_code_block:
                        # End of code block
                        code_blocks.append({
                            'language': language,
                            'code': '\n'.join(current_block)
                        })
                        current_block = []
                        in_code_block = False
                    else:
                        # Start of code block
                        in_code_block = True
                        language = line[3:].strip() or 'python'
                elif in_code_block:
                    current_block.append(line)

            # Extract explanation
            explanation_lines = []
            for line in lines:
                if not line.startswith('```') and not in_code_block:
                    explanation_lines.append(line)

            return {
                'code_blocks': code_blocks,
                'explanation': '\n'.join(explanation_lines).strip(),
                'raw_output': text
            }
        except Exception as e:
            logger.error(f"Failed to parse code output: {e}")
            return {'code_blocks': [], 'explanation': text, 'raw_output': text}


class TesterAgent(PersistentAgent):
    """Agent specialized in test generation and execution"""
    
    def __init__(self, orchestrator=None):
        super().__init__(
            name="tester_agent",
            role="Test Engineer",
            capabilities=[
                "generate_tests",
                "execute_tests",
                "coverage_analysis",
                "regression_testing",
                "performance_testing",
                "integration_testing"
            ],
            orchestrator=orchestrator
        )

        # Initialize LLM for AI-powered test generation
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            max_tokens=2000
        )

        self.code_parser = CodeOutputParser()

        self.test_results = {}
        self.coverage_data = None
        self.test_metrics = {
            "tests_generated": 0,
            "tests_executed": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "coverage_percentage": 0.0,
            "regression_failures": 0
        }

        self.test_templates = {
            "unit": self._create_unit_test_template(),
            "integration": self._create_integration_test_template(),
            "performance": self._create_performance_test_template()
        }
    
    def _create_unit_test_template(self) -> str:
        """Create template for unit tests"""
        return '''import pytest
from unittest.mock import Mock, patch

class Test{class_name}:
    """Unit tests for {class_name}"""
    
    def setup_method(self):
        """Setup test fixtures"""
        {setup_code}
    
    {test_methods}
    
    def teardown_method(self):
        """Cleanup after tests"""
        {teardown_code}
'''
    
    def _create_integration_test_template(self) -> str:
        """Create template for integration tests"""
        return '''import pytest
import asyncio
from pathlib import Path

@pytest.mark.integration
class Test{feature_name}Integration:
    """Integration tests for {feature_name}"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup integration test environment"""
        {setup_code}
        yield
        {teardown_code}
    
    {test_methods}
'''
    
    def _create_performance_test_template(self) -> str:
        """Create template for performance tests"""
        return '''import pytest
import time
import statistics
from memory_profiler import profile

@pytest.mark.performance
class Test{component_name}Performance:
    """Performance tests for {component_name}"""
    
    def test_execution_time(self):
        """Test execution time meets requirements"""
        times = []
        for _ in range({iterations}):
            start = time.time()
            {test_code}
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        assert avg_time < {max_time}, f"Average time {{avg_time}} exceeds limit"
    
    @profile
    def test_memory_usage(self):
        """Test memory usage is within limits"""
        {memory_test_code}
'''
    
    async def process_task(self, task: AgentTask) -> Any:
        """Process a testing task"""
        logger.info(f"Tester processing task: {task.type}")
        
        task_type = task.type.lower()
        
        if task_type == "generate_tests":
            return await self.generate_tests(task.data)
        elif task_type == "execute_tests":
            return await self.execute_tests(task.data)
        elif task_type == "analyze_coverage":
            return await self.analyze_coverage(task.data)
        elif task_type == "regression_test":
            return await self.run_regression_tests(task.data)
        elif task_type == "performance_test":
            return await self.run_performance_tests(task.data)
        else:
            raise ValueError(f"Unknown task type for Tester: {task_type}")
    
    async def generate_tests(self, data: Dict) -> Dict:
        """
        Generates meaningful tests for a given file using an AI model.
        """
        # GEMINI-EDIT - 2025-11-18 - Replaced template-based test generation with an AI-driven approach.
        logger.info("Starting AI-driven test generation...")
        
        file_path = data.get('file_path')
        if not file_path:
            raise ValueError("`file_path` must be provided in the task data for test generation.")

        try:
            # A more advanced agent would have access to tools. For now, we assume it can read files.
            with open(file_path, 'r') as f:
                source_code = f.read()
        except FileNotFoundError:
            logger.error(f"File not found for test generation: {file_path}")
            return {'success': False, 'error': f"File not found: {file_path}"}
        
        # Create a new, more intelligent prompt template for the LLM
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert Senior Software Engineer in Test. Your task is to write a comprehensive suite of `pytest` tests for the provided Python code.

            Follow these principles:
            1.  **Thoroughness:** Cover happy paths, edge cases (e.g., None, empty inputs, zeros), and error conditions.
            2.  **Clarity:** Write clean, readable tests with clear `assert` statements.
            3.  **Independence:** Tests should be independent and not rely on the state of previous tests.
            4.  **Best Practices:** Use `pytest` features like fixtures for setup, `pytest.raises` for expected exceptions, and `mocker` (from `pytest-mock`) for patching dependencies.
            5.  **Meaningful Assertions:** Do not use `assert True` or `assert result is not None` unless it is the only possible check. Assert on specific expected values.
            
            Your output must be a single, complete Python code block containing the full test file, including all necessary imports."""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """Please generate a `pytest` test suite for the following code from the file `{file_path}`:

            ```python
            {source_code}
            ```

            Generate a complete test file.
            """
        )
        
        test_gen_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        # Generate the test code
        # Assuming self.llm and self.code_parser are available from the base class
        response = await self.llm.apredict(test_gen_prompt.format(
            file_path=file_path,
            source_code=source_code
        ))
        
        parsed = self.code_parser.parse(response)
        
        if not parsed['code_blocks']:
            logger.error("AI failed to generate any test code.")
            return {'success': False, 'error': 'AI produced no code.'}
            
        # We expect the LLM to produce a single, complete test file
        test_code = parsed['code_blocks'][0]['code']
        
        # Basic validation
        try:
            ast.parse(test_code)
        except SyntaxError as e:
            logger.error(f"AI generated invalid Python code for tests: {e}")
            return {'success': False, 'error': f"Generated code has syntax errors: {e}", 'generated_code': test_code}

        self.test_metrics['tests_generated'] += test_code.count("def test_")
        
        return {
            'success': True,
            'test_code': test_code,
            'test_file_name': f"test_{Path(file_path).name}",
            'explanation': parsed['explanation']
        }
    
    async def execute_tests(self, data: Dict) -> Dict:
        """Execute tests and return results"""
        test_file = data.get('test_file', '')
        test_directory = data.get('test_directory', 'tests')
        coverage_enabled = data.get('coverage', True)
        
        # Prepare test environment
        test_path = Path(test_directory)
        if test_file:
            test_path = test_path / test_file
        
        # Run tests with coverage
        if coverage_enabled:
            cov = coverage.Coverage()
            cov.start()
        
        # Execute pytest
        result = subprocess.run(
            ['pytest', str(test_path), '-v', '--tb=short', '--json-report'],
            capture_output=True,
            text=True
        )
        
        if coverage_enabled:
            cov.stop()
            cov.save()
            coverage_percentage = cov.report()
            self.test_metrics['coverage_percentage'] = coverage_percentage
        
        # Parse results
        test_results = self._parse_test_results(result.stdout, result.stderr)
        
        self.test_metrics['tests_executed'] += test_results['total']
        self.test_metrics['tests_passed'] += test_results['passed']
        self.test_metrics['tests_failed'] += test_results['failed']
        
        return {
            'success': test_results['failed'] == 0,
            'results': test_results,
            'coverage': coverage_percentage if coverage_enabled else None,
            'output': result.stdout,
            'errors': result.stderr
        }
    
    async def analyze_coverage(self, data: Dict) -> Dict:
        """Analyze test coverage"""
        source_directory = data.get('source_directory', 'core')
        min_coverage = data.get('min_coverage', 80.0)
        
        cov = coverage.Coverage(source=[source_directory])
        cov.load()
        
        # Generate coverage report
        report_buffer = StringIO()
        coverage_percentage = cov.report(file=report_buffer)
        coverage_report = report_buffer.getvalue()
        
        # Identify uncovered code
        uncovered = []
        for filename in cov.get_data().measured_files():
            missing_lines = cov.analysis(filename)[3]
            if missing_lines:
                uncovered.append({
                    'file': filename,
                    'missing_lines': missing_lines
                })
        
        # Generate suggestions for improving coverage
        suggestions = self._generate_coverage_suggestions(uncovered)
        
        return {
            'success': coverage_percentage >= min_coverage,
            'coverage_percentage': coverage_percentage,
            'report': coverage_report,
            'uncovered_code': uncovered,
            'suggestions': suggestions,
            'meets_requirement': coverage_percentage >= min_coverage
        }
    
    async def run_regression_tests(self, data: Dict) -> Dict:
        """Run regression test suite"""
        baseline_results = data.get('baseline_results', {})
        test_suite = data.get('test_suite', 'tests/regression')
        
        # Execute regression tests
        result = subprocess.run(
            ['pytest', test_suite, '--tb=short', '-v'],
            capture_output=True,
            text=True
        )
        
        current_results = self._parse_test_results(result.stdout, result.stderr)
        
        # Compare with baseline
        regressions = []
        if baseline_results:
            for test_name, baseline_status in baseline_results.items():
                current_status = current_results.get('tests', {}).get(test_name)
                if baseline_status == 'passed' and current_status == 'failed':
                    regressions.append(test_name)
        
        self.test_metrics['regression_failures'] = len(regressions)
        
        return {
            'success': len(regressions) == 0,
            'regressions': regressions,
            'results': current_results,
            'baseline_comparison': {
                'new_failures': regressions,
                'fixed_tests': []  # Would identify fixed tests
            }
        }
    
    async def run_performance_tests(self, data: Dict) -> Dict:
        """Run performance test suite"""
        test_suite = data.get('test_suite', 'tests/performance')
        performance_targets = data.get('targets', {})
        
        # Execute performance tests
        result = subprocess.run(
            ['pytest', test_suite, '-v', '--benchmark-only'],
            capture_output=True,
            text=True
        )
        
        # Parse performance results
        performance_results = self._parse_performance_results(result.stdout)
        
        # Check against targets
        violations = []
        for metric, value in performance_results.items():
            target = performance_targets.get(metric)
            if target and value > target:
                violations.append({
                    'metric': metric,
                    'value': value,
                    'target': target,
                    'exceeded_by': value - target
                })
        
        return {
            'success': len(violations) == 0,
            'results': performance_results,
            'violations': violations,
            'meets_targets': len(violations) == 0
        }
    
    def _analyze_code_for_testing(self, code: str) -> Dict:
        """Analyze code to understand testing requirements"""
        try:
            tree = ast.parse(code)
            
            functions = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'returns': bool(node.returns),
                        'decorators': [d.id for d in node.decorator_list if hasattr(d, 'id')]
                    })
                elif isinstance(node, ast.ClassDef):
                    methods = [
                        n.name for n in node.body
                        if isinstance(n, ast.FunctionDef)
                    ]
                    classes.append({
                        'name': node.name,
                        'methods': methods
                    })
            
            return {
                'functions': functions,
                'classes': classes,
                'has_async': any(
                    isinstance(node, ast.AsyncFunctionDef)
                    for node in ast.walk(tree)
                )
            }
        except:
            return {'functions': [], 'classes': [], 'has_async': False}
    
    def _generate_test_method(self, func: Dict) -> str:
        """Generate test method for a function"""
        func_name = func['name']
        has_args = len(func['args']) > 0
        
        test_code = f'''    def test_{func_name}(self):
        """Test {func_name} function"""
'''
        
        if has_args:
            test_code += f'''        # Arrange
        test_input = {self._generate_test_input(func['args'])}
        expected = None  # Define expected output
        
        # Act
        result = {func_name}(*test_input)
        
        # Assert
        assert result == expected
'''
        else:
            test_code += f'''        # Act
        result = {func_name}()
        
        # Assert
        assert result is not None
'''
        
        return test_code
    
    def _generate_test_input(self, args: List[str]) -> str:
        """Generate test input for function arguments"""
        test_values = []
        for arg in args:
            if 'id' in arg.lower():
                test_values.append('"test_id_123"')
            elif 'name' in arg.lower():
                test_values.append('"test_name"')
            elif 'count' in arg.lower() or 'number' in arg.lower():
                test_values.append('42')
            else:
                test_values.append('None')
        
        return '[' + ', '.join(test_values) + ']'
    
    def _generate_setup_code(self, analysis: Dict) -> str:
        """Generate setup code for tests"""
        if analysis.get('has_async'):
            return 'self.loop = asyncio.get_event_loop()'
        return 'pass'
    
    def _generate_teardown_code(self, analysis: Dict) -> str:
        """Generate teardown code for tests"""
        return 'pass'
    
    def _parse_test_results(self, stdout: str, stderr: str) -> Dict:
        """Parse pytest output"""
        results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': [],
            'tests': {}
        }
        
        # Simple parsing - would use pytest-json-report in production
        lines = stdout.split('\n')
        for line in lines:
            if 'passed' in line and 'failed' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'passed' in part and i > 0:
                        results['passed'] = int(parts[i-1])
                    if 'failed' in part and i > 0:
                        results['failed'] = int(parts[i-1])
        
        results['total'] = results['passed'] + results['failed']
        
        return results
    
    def _parse_performance_results(self, stdout: str) -> Dict:
        """Parse performance test results"""
        # Would parse actual benchmark results
        return {
            'average_time': 0.5,
            'min_time': 0.1,
            'max_time': 1.0,
            'memory_usage': 100  # MB
        }
    
    def _generate_coverage_suggestions(self, uncovered: List[Dict]) -> List[str]:
        """Generate suggestions for improving coverage"""
        suggestions = []
        
        for item in uncovered:
            file_name = Path(item['file']).name
            missing_count = len(item['missing_lines'])
            
            suggestions.append(
                f"Add tests for {file_name}: {missing_count} lines uncovered"
            )
        
        return suggestions
    
    def analyze_context(self, context: Dict) -> Dict:
        """Analyze testing context"""
        return {
            'test_framework': 'pytest',
            'coverage_tool': 'coverage.py',
            'test_types': ['unit', 'integration', 'performance'],
            'ci_integration': True
        }
    
    def generate_solution(self, problem: Dict) -> Dict:
        """Generate testing solution for a problem"""
        return {
            'approach': 'Comprehensive test coverage',
            'test_strategy': 'Unit -> Integration -> E2E',
            'coverage_target': 95,
            'automation': 'Full CI/CD integration'
        }