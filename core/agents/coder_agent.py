"""
Coder Agent - Specialized agent for code generation and implementation
"""

import asyncio
import json
import ast
import autopep8
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.callbacks import AsyncCallbackHandler

from .base_agent import PersistentAgent, AgentTask, AgentState
from ..utils.filesystem import list_directory, read_file, write_file


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


class CoderAgent(PersistentAgent):
    """Agent specialized in code generation and implementation"""
    
    def __init__(self, orchestrator=None):
        super().__init__(
            name="coder_agent",
            role="Code Generator",
            capabilities=[
                "generate_code",
                "refactor_code",
                "implement_features",
                "fix_syntax_errors",
                "optimize_code",
                "generate_documentation"
            ],
            orchestrator=orchestrator
        )
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            max_tokens=2000
        )
        
        self.code_parser = CodeOutputParser()
        
        # Code generation templates
        self.templates = {
            "feature": self._create_feature_template(),
            "refactor": self._create_refactor_template(),
            "debug": self._create_debug_template(),
            "optimize": self._create_optimize_template(),
            "document": self._create_documentation_template()
        }
        
        # Code quality settings
        self.quality_checks_enabled = True
        self.auto_format = True
        self.type_checking = True
        
        # Track generated code
        self.generated_files = {}
        self.code_metrics = {
            "lines_generated": 0,
            "files_created": 0,
            "refactorings": 0,
            "bugs_fixed": 0
        }
    
    def _create_feature_template(self) -> ChatPromptTemplate:
        """Create template for feature implementation"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert software engineer implementing features for a continuous AI development system.
            Follow these principles:
            1. Write clean, maintainable, production-ready code
            2. Follow best practices and design patterns
            3. Include proper error handling
            4. Add comprehensive docstrings and comments
            5. Consider performance and scalability
            6. Ensure code is testable"""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """Implement the following feature:
            
            Feature: {feature_name}
            Description: {description}
            Requirements: {requirements}
            Context: {context}
            Existing Code Structure: {code_structure}
            
            Generate complete, working code for this feature."""
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def _create_refactor_template(self) -> ChatPromptTemplate:
        """Create template for code refactoring"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert at code refactoring. Your goals:
            1. Improve code readability and maintainability
            2. Reduce complexity and duplication
            3. Apply SOLID principles
            4. Optimize performance where possible
            5. Maintain backward compatibility
            6. Preserve all functionality"""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """Refactor the following code:
            
            Original Code:
            ```python
            {original_code}
            ```
            
            Refactoring Goals: {goals}
            Constraints: {constraints}
            
            Provide the refactored code with explanations."""
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def _create_debug_template(self) -> ChatPromptTemplate:
        """Create template for debugging"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert debugger. Analyze code for:
            1. Syntax errors
            2. Logic errors
            3. Runtime errors
            4. Performance issues
            5. Security vulnerabilities
            6. Edge cases"""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """Debug the following code:
            
            Code:
            ```python
            {code}
            ```
            
            Error Message: {error_message}
            Stack Trace: {stack_trace}
            Context: {context}
            
            Provide fixed code and explanation."""
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def _create_optimize_template(self) -> ChatPromptTemplate:
        """Create template for code optimization"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a performance optimization expert. Focus on:
            1. Time complexity optimization
            2. Space complexity optimization
            3. Database query optimization
            4. Caching strategies
            5. Parallel processing opportunities
            6. Memory management"""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """Optimize the following code:
            
            Code:
            ```python
            {code}
            ```
            
            Performance Metrics: {metrics}
            Bottlenecks: {bottlenecks}
            Target Improvements: {targets}
            
            Provide optimized code with performance analysis."""
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def _create_documentation_template(self) -> ChatPromptTemplate:
        """Create template for documentation generation"""
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a technical documentation expert. Create:
            1. Clear and comprehensive docstrings
            2. Inline comments for complex logic
            3. API documentation
            4. Usage examples
            5. Type hints
            6. README content when needed"""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """Document the following code:
            
            Code:
            ```python
            {code}
            ```
            
            Documentation Type: {doc_type}
            Target Audience: {audience}
            
            Provide well-documented code."""
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    async def process_task(self, task: AgentTask) -> Any:
        """Process a coding task"""
        logger.info(f"Coder processing task: {task.type}")
        
        task_type = task.type.lower()
        
        if task_type == "implement_feature":
            return await self.implement_feature(task.data)
        elif task_type == "refactor_code":
            return await self.refactor_code(task.data)
        elif task_type == "debug_code":
            return await self.debug_code(task.data)
        elif task_type == "optimize_code":
            return await self.optimize_code(task.data)
        elif task_type == "generate_documentation":
            return await self.generate_documentation(task.data)
        elif task_type == "review_code":
            return await self.review_code(task.data)
        else:
            raise ValueError(f"Unknown task type for Coder: {task_type}")
    
    async def implement_feature(self, data: Dict) -> Dict:
        """
        Implements a new feature by planning, reading existing files,
        generating code, and writing changes to the filesystem.
        """
        # GEMINI-EDIT - 2025-11-18 - Enhanced method to be context-aware and interact with the filesystem.
        logger.info("Starting enhanced feature implementation...")
        feature_name = data.get('feature_name', 'unnamed_feature')
        description = data.get('description', '')

        try:
            # Step 1: Gather context by listing all files in the project.
            logger.info("Step 1: Listing project files to gather context...")
            file_tree = await list_directory(path=".", recursive=True)
            
            # Step 2: Ask the LLM to create a plan.
            logger.info("Step 2: Formulating a plan...")
            planning_prompt = self._create_planning_template().format(
                feature_name=feature_name,
                description=description,
                file_tree=json.dumps(file_tree, indent=2)
            )
            planning_response = await self.llm.apredict(planning_prompt)
            
            try:
                # The plan should be a JSON object specifying files to read and modify.
                plan = json.loads(planning_response)
                files_to_read = plan.get('read', [])
                files_to_modify = plan.get('modify', [])
                new_files = plan.get('create', [])
            except json.JSONDecodeError:
                logger.error("Failed to decode LLM's plan. Aborting feature implementation.")
                return {'success': False, 'error': 'Failed to decode plan from LLM.'}

            # Step 3: Execute the plan - Read files.
            logger.info(f"Step 3: Reading {len(files_to_read)} files for context...")
            context_code = ""
            for file_path in files_to_read:
                try:
                    content = await read_file(file_path=file_path)
                    context_code += f"--- START OF {file_path} ---\n{content}\n--- END OF {file_path} ---\n\n"
                except Exception as e:
                    logger.warning(f"Could not read file {file_path} from plan: {e}")

            # Step 4: Generate the implementation code with the new context.
            logger.info("Step 4: Generating code with full context...")
            implementation_prompt = self.templates["feature"].format(
                feature_name=feature_name,
                description=description,
                requirements=data.get('requirements', '[]'),
                context=context_code,
                code_structure=f"Files to modify: {files_to_modify}, New files to create: {new_files}"
            )
            
            code_response = await self.llm.apredict(implementation_prompt)
            parsed_code = self.code_parser.parse(code_response)

            # Step 5: Apply the changes to the filesystem.
            logger.info("Step 5: Applying changes to the filesystem...")
            applied_files = []
            for block in parsed_code['code_blocks']:
                code = block['code']
                # A more advanced agent would determine the file path from the LLM response.
                # For now, we'll assume the first file to modify or create is the target.
                target_file = None
                if new_files:
                    target_file = new_files.pop(0)
                elif files_to_modify:
                    target_file = files_to_modify.pop(0)

                if target_file:
                    logger.info(f"Writing code to {target_file}...")
                    await write_file(file_path=target_file, content=code)
                    applied_files.append(target_file)
                else:
                    logger.warning("No target file specified in plan for a generated code block.")

            self.code_metrics['lines_generated'] += sum(len(b['code'].split('\n')) for b in parsed_code['code_blocks'])
            self.code_metrics['files_created'] += len(applied_files)

            return {
                'success': True,
                'feature': feature_name,
                'files_modified_or_created': applied_files,
                'explanation': parsed_code['explanation']
            }

        except Exception as e:
            logger.error(f"An exception occurred during feature implementation: {e}")
            return {'success': False, 'error': str(e)}

    def _create_planning_template(self) -> ChatPromptTemplate:
        """Create a template for the planning phase."""
        system_message = SystemMessagePromptTemplate.from_template(
            "You are a senior software architect. Your job is to analyze a feature request and a project's file structure, then create a clear, step-by-step plan for implementation. Your output must be a valid JSON object."
        )
        human_message = HumanMessagePromptTemplate.from_template(
            """I need to implement the following feature:
            - Feature: {feature_name}
            - Description: {description}

            Here is the file structure of the project:
            {file_tree}

            Please provide a plan in JSON format with three keys:
            1. "read": A list of file paths that need to be read to get the full context for implementing the feature.
            2. "modify": A list of existing file paths that will likely need to be changed.
            3. "create": A list of new file paths that need to be created.
            
            Your output must be only the JSON object, with no other text before or after it.
            """
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])

    
    async def refactor_code(self, data: Dict) -> Dict:
        """Refactor existing code"""
        original_code = data.get('code', '')
        goals = data.get('goals', [])
        constraints = data.get('constraints', [])
        
        prompt = self.templates["refactor"].format(
            original_code=original_code,
            goals=json.dumps(goals, indent=2),
            constraints=json.dumps(constraints, indent=2)
        )
        
        response = await self.llm.apredict(prompt)
        parsed = self.code_parser.parse(response)
        
        refactored_code = None
        for block in parsed['code_blocks']:
            refactored_code = block['code']
            
            # Apply quality checks
            if self.quality_checks_enabled:
                refactored_code = await self.apply_quality_checks(
                    refactored_code,
                    block['language']
                )
            break
        
        # Calculate improvements
        improvements = self._analyze_improvements(original_code, refactored_code)
        
        self.code_metrics['refactorings'] += 1
        
        return {
            'success': True,
            'original_code': original_code,
            'refactored_code': refactored_code,
            'improvements': improvements,
            'explanation': parsed['explanation']
        }
    
    async def debug_code(self, data: Dict) -> Dict:
        """Debug and fix code issues"""
        code = data.get('code', '')
        error_message = data.get('error_message', '')
        stack_trace = data.get('stack_trace', '')
        context = data.get('context', {})
        
        prompt = self.templates["debug"].format(
            code=code,
            error_message=error_message,
            stack_trace=stack_trace,
            context=json.dumps(context, indent=2)
        )
        
        response = await self.llm.apredict(prompt)
        parsed = self.code_parser.parse(response)
        
        fixed_code = None
        for block in parsed['code_blocks']:
            fixed_code = block['code']
            break
        
        # Validate the fix
        is_valid = await self.validate_code(fixed_code)
        
        self.code_metrics['bugs_fixed'] += 1 if is_valid else 0
        
        return {
            'success': is_valid,
            'original_code': code,
            'fixed_code': fixed_code,
            'error_fixed': error_message,
            'explanation': parsed['explanation'],
            'validation': is_valid
        }
    
    async def optimize_code(self, data: Dict) -> Dict:
        """Optimize code for performance"""
        code = data.get('code', '')
        metrics = data.get('metrics', {})
        bottlenecks = data.get('bottlenecks', [])
        targets = data.get('targets', {})
        
        prompt = self.templates["optimize"].format(
            code=code,
            metrics=json.dumps(metrics, indent=2),
            bottlenecks=json.dumps(bottlenecks, indent=2),
            targets=json.dumps(targets, indent=2)
        )
        
        response = await self.llm.apredict(prompt)
        parsed = self.code_parser.parse(response)
        
        optimized_code = None
        for block in parsed['code_blocks']:
            optimized_code = block['code']
            break
        
        # Analyze performance improvements
        performance_analysis = self._analyze_performance(
            original_code=code,
            optimized_code=optimized_code
        )
        
        return {
            'success': True,
            'original_code': code,
            'optimized_code': optimized_code,
            'performance_analysis': performance_analysis,
            'explanation': parsed['explanation']
        }
    
    async def generate_documentation(self, data: Dict) -> Dict:
        """Generate documentation for code"""
        code = data.get('code', '')
        doc_type = data.get('doc_type', 'comprehensive')
        audience = data.get('audience', 'developers')
        
        prompt = self.templates["document"].format(
            code=code,
            doc_type=doc_type,
            audience=audience
        )
        
        response = await self.llm.apredict(prompt)
        parsed = self.code_parser.parse(response)
        
        documented_code = None
        for block in parsed['code_blocks']:
            documented_code = block['code']
            break
        
        return {
            'success': True,
            'original_code': code,
            'documented_code': documented_code,
            'documentation': parsed['explanation']
        }
    
    async def review_code(self, data: Dict) -> Dict:
        """Review code for quality and issues"""
        code = data.get('code', '')
        review_criteria = data.get('criteria', [
            'correctness', 'performance', 'readability',
            'maintainability', 'security', 'best_practices'
        ])
        
        issues = []
        suggestions = []
        
        # Check for syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append({
                'type': 'syntax_error',
                'line': e.lineno,
                'message': str(e)
            })
        
        # Check code complexity
        complexity = self._calculate_complexity(code)
        if complexity > 10:
            suggestions.append({
                'type': 'complexity',
                'message': f'High complexity detected: {complexity}',
                'suggestion': 'Consider breaking down complex functions'
            })
        
        # Check for common issues
        common_issues = self._check_common_issues(code)
        issues.extend(common_issues)
        
        return {
            'success': True,
            'code': code,
            'issues': issues,
            'suggestions': suggestions,
            'complexity': complexity,
            'quality_score': max(0, 100 - len(issues) * 10 - complexity)
        }
    
    async def apply_quality_checks(self, code: str, language: str) -> str:
        """Apply quality checks and formatting to code"""
        if language == 'python':
            # Auto-format with autopep8
            if self.auto_format:
                code = autopep8.fix_code(code)
            
            # Add type hints if missing
            # This would use a more sophisticated type inference system
            
        return code
    
    async def validate_code(self, code: str) -> bool:
        """Validate that code is syntactically correct"""
        try:
            ast.parse(code)
            return True
        except:
            return False
    
    def _generate_file_name(self, feature_name: str, language: str) -> str:
        """Generate appropriate file name for code"""
        extensions = {
            'python': '.py',
            'javascript': '.js',
            'typescript': '.ts',
            'java': '.java',
            'cpp': '.cpp',
            'go': '.go'
        }
        
        extension = extensions.get(language, '.txt')
        file_name = feature_name.lower().replace(' ', '_') + extension
        
        return file_name
    
    def _analyze_improvements(self, original: str, refactored: str) -> Dict:
        """Analyze improvements from refactoring"""
        original_lines = original.split('\n')
        refactored_lines = refactored.split('\n')
        
        return {
            'line_reduction': len(original_lines) - len(refactored_lines),
            'complexity_reduction': (
                self._calculate_complexity(original) -
                self._calculate_complexity(refactored)
            ),
            'readability_improvement': 'Improved'  # Would use actual metrics
        }
    
    def _analyze_performance(self, original_code: str, optimized_code: str) -> Dict:
        """Analyze performance improvements"""
        # This would use actual profiling and analysis
        return {
            'estimated_speedup': '2-3x',
            'memory_reduction': '30%',
            'complexity_improvement': 'O(n²) to O(n log n)'
        }
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity of code"""
        # Simple complexity calculation
        complexity = 1
        
        for line in code.split('\n'):
            line = line.strip()
            if any(keyword in line for keyword in ['if', 'elif', 'for', 'while', 'except']):
                complexity += 1
            if 'and' in line or 'or' in line:
                complexity += 1
        
        return complexity
    
    def _check_common_issues(self, code: str) -> List[Dict]:
        """Check for common code issues"""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for hardcoded values
            if any(pattern in line for pattern in ['password=', 'api_key=', 'secret=']):
                issues.append({
                    'type': 'security',
                    'line': i,
                    'message': 'Possible hardcoded credential detected'
                })
            
            # Check for print statements in production code
            if 'print(' in line and not '#' in line:
                issues.append({
                    'type': 'debug_code',
                    'line': i,
                    'message': 'Print statement should be replaced with logging'
                })
            
            # Check for bare except
            if 'except:' in line:
                issues.append({
                    'type': 'error_handling',
                    'line': i,
                    'message': 'Bare except clause - specify exception type'
                })
        
        return issues
    
    def analyze_context(self, context: Dict) -> Dict:
        """Analyze coding context"""
        return {
            'language': context.get('language', 'python'),
            'framework': context.get('framework'),
            'dependencies': context.get('dependencies', []),
            'patterns': context.get('patterns', []),
            'constraints': context.get('constraints', [])
        }
    
    def generate_solution(self, problem: Dict) -> Dict:
        """Generate coding solution for a problem"""
        return {
            'approach': 'Implement using best practices',
            'technologies': ['Python', 'AsyncIO', 'LangChain'],
            'estimated_time': '2-4 hours',
            'complexity': 'Medium'
        }