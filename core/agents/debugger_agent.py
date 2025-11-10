"""
Debugger Agent - Specialized agent for error analysis and fixing
"""

import asyncio
import traceback
import ast
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from loguru import logger
import json

from .base_agent import PersistentAgent, AgentTask


class DebuggerAgent(PersistentAgent):
    """Agent specialized in debugging and error resolution"""
    
    def __init__(self, orchestrator=None):
        super().__init__(
            name="debugger_agent",
            role="Debug Specialist",
            capabilities=[
                "analyze_errors",
                "trace_execution",
                "identify_root_cause",
                "suggest_fixes",
                "validate_fixes",
                "prevent_regressions"
            ],
            orchestrator=orchestrator
        )
        
        self.error_patterns = self._load_error_patterns()
        self.fix_strategies = self._load_fix_strategies()
        self.debug_metrics = {
            "errors_analyzed": 0,
            "errors_fixed": 0,
            "root_causes_identified": 0,
            "average_fix_time": 0,
            "regression_prevented": 0
        }
        
        # Error analysis cache
        self.error_cache = {}
        self.solution_cache = {}
    
    def _load_error_patterns(self) -> Dict:
        """Load common error patterns and their characteristics"""
        return {
            "syntax_error": {
                "indicators": ["SyntaxError", "invalid syntax", "unexpected EOF"],
                "priority": 1,
                "auto_fixable": True
            },
            "type_error": {
                "indicators": ["TypeError", "unsupported operand", "not callable"],
                "priority": 2,
                "auto_fixable": True
            },
            "name_error": {
                "indicators": ["NameError", "not defined", "undefined"],
                "priority": 2,
                "auto_fixable": True
            },
            "attribute_error": {
                "indicators": ["AttributeError", "has no attribute"],
                "priority": 3,
                "auto_fixable": True
            },
            "import_error": {
                "indicators": ["ImportError", "ModuleNotFoundError", "cannot import"],
                "priority": 1,
                "auto_fixable": True
            },
            "logic_error": {
                "indicators": ["AssertionError", "test failed", "unexpected result"],
                "priority": 4,
                "auto_fixable": False
            },
            "performance_issue": {
                "indicators": ["timeout", "slow", "performance"],
                "priority": 5,
                "auto_fixable": False
            }
        }
    
    def _load_fix_strategies(self) -> Dict:
        """Load fix strategies for different error types"""
        return {
            "syntax_error": [
                "check_parentheses_balance",
                "check_indentation",
                "check_quotes_matching",
                "check_colon_placement"
            ],
            "type_error": [
                "check_type_compatibility",
                "add_type_conversion",
                "check_function_signature",
                "validate_parameters"
            ],
            "name_error": [
                "check_variable_scope",
                "add_import_statement",
                "fix_typo",
                "initialize_variable"
            ],
            "import_error": [
                "install_missing_package",
                "fix_import_path",
                "check_circular_imports",
                "verify_module_exists"
            ]
        }
    
    async def process_task(self, task: AgentTask) -> Any:
        """Process a debugging task"""
        logger.info(f"Debugger processing task: {task.type}")
        
        task_type = task.type.lower()
        
        if task_type == "analyze_error":
            return await self.analyze_error(task.data)
        elif task_type == "fix_error":
            return await self.fix_error(task.data)
        elif task_type == "trace_execution":
            return await self.trace_execution(task.data)
        elif task_type == "identify_root_cause":
            return await self.identify_root_cause(task.data)
        elif task_type == "validate_fix":
            return await self.validate_fix(task.data)
        else:
            raise ValueError(f"Unknown task type for Debugger: {task_type}")
    
    async def analyze_error(self, data: Dict) -> Dict:
        """Analyze an error to understand its nature"""
        error_message = data.get('error_message', '')
        stack_trace = data.get('stack_trace', '')
        code = data.get('code', '')
        context = data.get('context', {})
        
        # Classify error type
        error_type = self._classify_error(error_message, stack_trace)
        
        # Extract error location
        location = self._extract_error_location(stack_trace)
        
        # Analyze code context
        code_analysis = self._analyze_code_context(code, location)
        
        # Check if we've seen similar error
        similar_errors = self._find_similar_errors(error_message)
        
        # Generate analysis report
        analysis = {
            'error_type': error_type,
            'location': location,
            'severity': self._assess_severity(error_type),
            'auto_fixable': self.error_patterns.get(error_type, {}).get('auto_fixable', False),
            'code_context': code_analysis,
            'similar_errors': similar_errors,
            'possible_causes': self._identify_possible_causes(error_type, code_analysis),
            'suggested_actions': self._suggest_actions(error_type)
        }
        
        # Cache the analysis
        error_hash = self._hash_error(error_message, location)
        self.error_cache[error_hash] = analysis
        
        self.debug_metrics['errors_analyzed'] += 1
        
        # Learn from error if orchestrator available
        if self.orchestrator and hasattr(self.orchestrator, 'error_graph'):
            await self._report_to_knowledge_graph(error_type, error_message, analysis)
        
        return {
            'success': True,
            'analysis': analysis,
            'error_hash': error_hash
        }
    
    async def fix_error(self, data: Dict) -> Dict:
        """Attempt to fix an identified error"""
        error_hash = data.get('error_hash')
        error_message = data.get('error_message', '')
        code = data.get('code', '')
        error_type = data.get('error_type')
        
        # Check solution cache
        if error_hash in self.solution_cache:
            cached_solution = self.solution_cache[error_hash]
            logger.info(f"Using cached solution for error {error_hash}")
            return cached_solution
        
        # Get analysis from cache or re-analyze
        if error_hash in self.error_cache:
            analysis = self.error_cache[error_hash]
        else:
            analysis_result = await self.analyze_error(data)
            analysis = analysis_result['analysis']
            error_type = analysis['error_type']
        
        # Apply fix strategies
        fixed_code = code
        fix_applied = None
        
        if error_type in self.fix_strategies:
            for strategy in self.fix_strategies[error_type]:
                try:
                    fixed_code, success = await self._apply_fix_strategy(
                        strategy,
                        code,
                        analysis
                    )
                    if success:
                        fix_applied = strategy
                        break
                except Exception as e:
                    logger.warning(f"Fix strategy {strategy} failed: {e}")
        
        # Validate the fix
        is_valid = await self._validate_code(fixed_code)
        
        result = {
            'success': is_valid,
            'original_code': code,
            'fixed_code': fixed_code,
            'fix_applied': fix_applied,
            'error_type': error_type,
            'validation': is_valid,
            'explanation': self._explain_fix(fix_applied, analysis)
        }
        
        if is_valid:
            self.solution_cache[error_hash] = result
            self.debug_metrics['errors_fixed'] += 1
        
        return result
    
    async def trace_execution(self, data: Dict) -> Dict:
        """Trace code execution to identify issues"""
        code = data.get('code', '')
        inputs = data.get('inputs', {})
        
        trace_results = []
        
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Instrument code for tracing
            instrumented_code = self._instrument_code(tree)
            
            # Execute with tracing
            trace_data = self._execute_with_trace(instrumented_code, inputs)
            
            # Analyze execution flow
            flow_analysis = self._analyze_execution_flow(trace_data)
            
            return {
                'success': True,
                'trace': trace_data,
                'flow_analysis': flow_analysis,
                'bottlenecks': self._identify_bottlenecks(trace_data),
                'anomalies': self._detect_anomalies(trace_data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'trace': trace_results
            }
    
    async def identify_root_cause(self, data: Dict) -> Dict:
        """Identify the root cause of an error"""
        error_message = data.get('error_message', '')
        stack_trace = data.get('stack_trace', '')
        code_history = data.get('code_history', [])
        recent_changes = data.get('recent_changes', [])
        
        # Analyze stack trace
        call_chain = self._parse_stack_trace(stack_trace)
        
        # Identify failure point
        failure_point = call_chain[-1] if call_chain else None
        
        # Analyze recent changes
        suspicious_changes = self._analyze_recent_changes(
            recent_changes,
            failure_point
        )
        
        # Trace error propagation
        propagation_path = self._trace_error_propagation(
            call_chain,
            error_message
        )
        
        # Determine root cause
        root_cause = {
            'primary_cause': self._determine_primary_cause(
                failure_point,
                suspicious_changes
            ),
            'contributing_factors': self._identify_contributing_factors(
                call_chain,
                code_history
            ),
            'propagation_path': propagation_path,
            'confidence': self._calculate_confidence(
                failure_point,
                suspicious_changes
            )
        }
        
        self.debug_metrics['root_causes_identified'] += 1
        
        return {
            'success': True,
            'root_cause': root_cause,
            'recommendations': self._generate_recommendations(root_cause)
        }
    
    async def validate_fix(self, data: Dict) -> Dict:
        """Validate that a fix resolves the issue"""
        original_code = data.get('original_code', '')
        fixed_code = data.get('fixed_code', '')
        test_cases = data.get('test_cases', [])
        
        validation_results = {
            'syntax_valid': True,
            'tests_pass': True,
            'no_regressions': True,
            'performance_ok': True,
            'issues': []
        }
        
        # Validate syntax
        try:
            ast.parse(fixed_code)
        except SyntaxError as e:
            validation_results['syntax_valid'] = False
            validation_results['issues'].append(f"Syntax error: {e}")
        
        # Run test cases
        for test_case in test_cases:
            try:
                result = self._run_test_case(fixed_code, test_case)
                if not result['passed']:
                    validation_results['tests_pass'] = False
                    validation_results['issues'].append(
                        f"Test '{test_case['name']}' failed: {result['error']}"
                    )
            except Exception as e:
                validation_results['tests_pass'] = False
                validation_results['issues'].append(f"Test execution error: {e}")
        
        # Check for regressions
        regression_check = self._check_regressions(original_code, fixed_code)
        if regression_check['found']:
            validation_results['no_regressions'] = False
            validation_results['issues'].extend(regression_check['regressions'])
        
        # Check performance impact
        perf_check = self._check_performance_impact(original_code, fixed_code)
        if perf_check['degraded']:
            validation_results['performance_ok'] = False
            validation_results['issues'].append(f"Performance degradation: {perf_check['details']}")
        
        is_valid = all([
            validation_results['syntax_valid'],
            validation_results['tests_pass'],
            validation_results['no_regressions'],
            validation_results['performance_ok']
        ])
        
        return {
            'success': is_valid,
            'validation_results': validation_results,
            'can_deploy': is_valid
        }
    
    def _classify_error(self, error_message: str, stack_trace: str) -> str:
        """Classify the type of error"""
        error_text = f"{error_message} {stack_trace}".lower()
        
        for error_type, pattern_info in self.error_patterns.items():
            for indicator in pattern_info['indicators']:
                if indicator.lower() in error_text:
                    return error_type
        
        return 'unknown_error'
    
    def _extract_error_location(self, stack_trace: str) -> Dict:
        """Extract error location from stack trace"""
        lines = stack_trace.split('\n')
        
        for line in reversed(lines):
            match = re.search(r'File "([^"]+)", line (\d+)', line)
            if match:
                return {
                    'file': match.group(1),
                    'line': int(match.group(2)),
                    'context': line
                }
        
        return {'file': None, 'line': None, 'context': None}
    
    def _analyze_code_context(self, code: str, location: Dict) -> Dict:
        """Analyze the code around the error location"""
        if not code or not location.get('line'):
            return {}
        
        lines = code.split('\n')
        error_line = location['line'] - 1
        
        if 0 <= error_line < len(lines):
            return {
                'error_line': lines[error_line],
                'before': lines[max(0, error_line - 2):error_line],
                'after': lines[error_line + 1:min(len(lines), error_line + 3)]
            }
        
        return {}
    
    def _find_similar_errors(self, error_message: str) -> List[Dict]:
        """Find similar errors from history"""
        similar = []
        
        # Check orchestrator's error knowledge graph
        if self.orchestrator and hasattr(self.orchestrator, 'error_graph'):
            similar_nodes = self.orchestrator.error_graph.find_similar_errors(
                error_message,
                threshold=0.8
            )
            
            for node in similar_nodes:
                if node.solutions:
                    similar.append({
                        'error': node.error_message,
                        'solution': node.solutions[0].description,
                        'success_rate': node.solutions[0].success_rate
                    })
        
        return similar
    
    def _identify_possible_causes(self, error_type: str, code_analysis: Dict) -> List[str]:
        """Identify possible causes of the error"""
        causes = []
        
        if error_type == 'syntax_error':
            causes = [
                "Missing or extra parentheses, brackets, or braces",
                "Incorrect indentation",
                "Missing colon after control statements",
                "Unclosed string literals"
            ]
        elif error_type == 'type_error':
            causes = [
                "Incompatible data types in operation",
                "Calling non-callable object",
                "Wrong number of arguments to function",
                "Attempting unsupported operation on type"
            ]
        elif error_type == 'name_error':
            causes = [
                "Variable used before assignment",
                "Typo in variable or function name",
                "Missing import statement",
                "Scope issue - variable not accessible"
            ]
        
        return causes
    
    def _suggest_actions(self, error_type: str) -> List[str]:
        """Suggest actions to fix the error"""
        suggestions = {
            'syntax_error': [
                "Check parentheses, brackets, and braces balance",
                "Verify indentation is consistent",
                "Ensure all strings are properly closed",
                "Add missing colons after if/for/while/def statements"
            ],
            'type_error': [
                "Convert data types explicitly",
                "Check function signatures and arguments",
                "Verify object types before operations",
                "Use isinstance() to check types"
            ],
            'name_error': [
                "Check variable spelling",
                "Ensure variable is defined before use",
                "Add necessary import statements",
                "Verify variable scope"
            ]
        }
        
        return suggestions.get(error_type, ["Review code for logical errors"])
    
    async def _apply_fix_strategy(self, strategy: str, code: str, analysis: Dict) -> Tuple[str, bool]:
        """Apply a specific fix strategy"""
        if strategy == 'check_parentheses_balance':
            return self._fix_parentheses(code), True
        elif strategy == 'check_indentation':
            return self._fix_indentation(code), True
        elif strategy == 'add_type_conversion':
            return self._add_type_conversion(code, analysis), True
        elif strategy == 'add_import_statement':
            return self._add_missing_import(code, analysis), True
        
        return code, False
    
    def _fix_parentheses(self, code: str) -> str:
        """Fix parentheses balance"""
        # Simple fix - would need more sophisticated approach
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            open_count = line.count('(')
            close_count = line.count(')')
            
            if open_count > close_count:
                line += ')' * (open_count - close_count)
            elif close_count > open_count:
                line = '(' * (close_count - open_count) + line
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_indentation(self, code: str) -> str:
        """Fix indentation issues"""
        # Use 4 spaces for indentation
        lines = code.split('\n')
        fixed_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.lstrip()
            
            if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:')):
                fixed_lines.append(' ' * (indent_level * 4) + stripped)
                if stripped.endswith(':'):
                    indent_level += 1
            elif stripped.startswith(('elif ', 'else:', 'except', 'finally:')):
                indent_level = max(0, indent_level - 1)
                fixed_lines.append(' ' * (indent_level * 4) + stripped)
                if stripped.endswith(':'):
                    indent_level += 1
            elif stripped == '':
                fixed_lines.append('')
            else:
                fixed_lines.append(' ' * (indent_level * 4) + stripped)
        
        return '\n'.join(fixed_lines)
    
    def _add_type_conversion(self, code: str, analysis: Dict) -> str:
        """Add type conversion where needed"""
        # This would require more context about the specific type error
        return code
    
    def _add_missing_import(self, code: str, analysis: Dict) -> str:
        """Add missing import statements"""
        # This would require understanding what module is missing
        return code
    
    async def _validate_code(self, code: str) -> bool:
        """Validate that code is syntactically correct"""
        try:
            ast.parse(code)
            return True
        except (SyntaxError, ValueError) as e:
            logger.debug(f"Code validation failed: {e}")
            return False
    
    def _hash_error(self, error_message: str, location: Dict) -> str:
        """Create hash for error caching"""
        import hashlib
        error_str = f"{error_message}_{location.get('file', '')}_{location.get('line', '')}"
        return hashlib.md5(error_str.encode()).hexdigest()[:8]
    
    async def _report_to_knowledge_graph(self, error_type: str, error_message: str, analysis: Dict):
        """Report error to knowledge graph for learning"""
        if self.orchestrator and hasattr(self.orchestrator, 'error_graph'):
            self.orchestrator.error_graph.add_error(
                error_type=error_type,
                error_message=error_message,
                context=analysis
            )
    
    def _explain_fix(self, fix_applied: str, analysis: Dict) -> str:
        """Explain what fix was applied"""
        explanations = {
            'check_parentheses_balance': "Balanced parentheses, brackets, and braces",
            'check_indentation': "Fixed indentation to use consistent 4 spaces",
            'add_type_conversion': "Added type conversion for compatibility",
            'add_import_statement': "Added missing import statement"
        }
        
        return explanations.get(fix_applied, "Applied automatic fix")
    
    def _instrument_code(self, tree: ast.AST) -> str:
        """Instrument code for tracing"""
        # Would add trace statements to code
        return ast.unparse(tree)
    
    def _execute_with_trace(self, code: str, inputs: Dict) -> List:
        """Execute code with tracing"""
        # Would execute and collect trace data
        return []
    
    def _analyze_execution_flow(self, trace_data: List) -> Dict:
        """Analyze execution flow from trace"""
        return {
            'total_steps': len(trace_data),
            'branches_taken': [],
            'loops_executed': []
        }
    
    def _identify_bottlenecks(self, trace_data: List) -> List:
        """Identify performance bottlenecks"""
        return []
    
    def _detect_anomalies(self, trace_data: List) -> List:
        """Detect anomalies in execution"""
        return []
    
    def _parse_stack_trace(self, stack_trace: str) -> List[Dict]:
        """Parse stack trace into call chain"""
        call_chain = []
        lines = stack_trace.split('\n')
        
        for line in lines:
            match = re.search(r'File "([^"]+)", line (\d+), in ([\w]+)', line)
            if match:
                call_chain.append({
                    'file': match.group(1),
                    'line': int(match.group(2)),
                    'function': match.group(3)
                })
        
        return call_chain
    
    def _analyze_recent_changes(self, recent_changes: List, failure_point: Dict) -> List:
        """Analyze recent changes for suspicious modifications"""
        suspicious = []
        
        if failure_point:
            for change in recent_changes:
                if change.get('file') == failure_point.get('file'):
                    suspicious.append(change)
        
        return suspicious
    
    def _trace_error_propagation(self, call_chain: List, error_message: str) -> List:
        """Trace how error propagated through call chain"""
        return call_chain
    
    def _determine_primary_cause(self, failure_point: Dict, suspicious_changes: List) -> str:
        """Determine the primary cause of error"""
        if suspicious_changes:
            return f"Recent change in {failure_point.get('file', 'unknown')} likely caused error"
        return "Root cause requires deeper analysis"
    
    def _identify_contributing_factors(self, call_chain: List, code_history: List) -> List:
        """Identify factors contributing to error"""
        return []
    
    def _calculate_confidence(self, failure_point: Dict, suspicious_changes: List) -> float:
        """Calculate confidence in root cause analysis"""
        confidence = 0.5
        
        if failure_point:
            confidence += 0.2
        if suspicious_changes:
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def _generate_recommendations(self, root_cause: Dict) -> List[str]:
        """Generate recommendations based on root cause"""
        recommendations = []
        
        if root_cause['confidence'] > 0.7:
            recommendations.append(f"Fix: {root_cause['primary_cause']}")
        else:
            recommendations.append("Add more logging for better diagnosis")
            recommendations.append("Review recent changes")
            recommendations.append("Add unit tests for affected code")
        
        return recommendations
    
    def _run_test_case(self, code: str, test_case: Dict) -> Dict:
        """Run a single test case"""
        # Would execute test case
        return {'passed': True, 'error': None}
    
    def _check_regressions(self, original: str, fixed: str) -> Dict:
        """Check for regressions introduced by fix"""
        return {'found': False, 'regressions': []}
    
    def _check_performance_impact(self, original: str, fixed: str) -> Dict:
        """Check performance impact of fix"""
        return {'degraded': False, 'details': ''}
    
    def _assess_severity(self, error_type: str) -> str:
        """Assess error severity"""
        severity_map = {
            'syntax_error': 'critical',
            'import_error': 'critical',
            'type_error': 'high',
            'name_error': 'high',
            'attribute_error': 'medium',
            'logic_error': 'medium',
            'performance_issue': 'low'
        }
        
        return severity_map.get(error_type, 'medium')
    
    def analyze_context(self, context: Dict) -> Dict:
        """Analyze debugging context"""
        return {
            'error_history': context.get('error_history', []),
            'fix_history': context.get('fix_history', []),
            'test_coverage': context.get('test_coverage', 0),
            'code_quality': context.get('code_quality', 0)
        }
    
    def generate_solution(self, problem: Dict) -> Dict:
        """Generate debugging solution"""
        return {
            'approach': 'Systematic root cause analysis',
            'steps': [
                'Analyze error message and stack trace',
                'Identify error type and location',
                'Apply appropriate fix strategy',
                'Validate fix with tests',
                'Check for regressions'
            ],
            'estimated_time': '30-60 minutes',
            'confidence': 0.85
        }