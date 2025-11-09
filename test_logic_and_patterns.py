#!/usr/bin/env python3
"""
Logic and Pattern Tests
Tests business logic, algorithms, and design patterns without requiring dependencies
"""

import sys
import ast
import json
from pathlib import Path
from typing import Dict, List, Set

results = {
    "total_checks": 0,
    "passed": 0,
    "warnings": 0,
    "issues": []
}


def analyze_function_complexity(func_node: ast.FunctionDef) -> int:
    """Calculate cyclomatic complexity of a function"""
    complexity = 1
    for node in ast.walk(func_node):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
    return complexity


def check_function_length(func_node: ast.FunctionDef) -> Dict:
    """Check if function is too long"""
    if hasattr(func_node, 'end_lineno'):
        length = func_node.end_lineno - func_node.lineno
        if length > 50:
            return {
                "issue": f"Long function '{func_node.name}' ({length} lines)",
                "severity": "medium" if length < 100 else "high"
            }
    return None


def check_too_many_parameters(func_node: ast.FunctionDef) -> Dict:
    """Check if function has too many parameters"""
    param_count = len(func_node.args.args)
    if param_count > 5:
        return {
            "issue": f"Too many parameters in '{func_node.name}' ({param_count})",
            "severity": "low"
        }
    return None


def check_nested_loops(tree: ast.AST) -> List[Dict]:
    """Find deeply nested loops"""
    issues = []

    def get_nesting_level(node, level=0):
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                if level >= 2:
                    issues.append({
                        "issue": f"Deeply nested loop at line {getattr(child, 'lineno', 'unknown')}",
                        "severity": "medium",
                        "depth": level + 1
                    })
                get_nesting_level(child, level + 1)

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            get_nesting_level(node, 1)

    return issues


def check_error_handling_patterns(tree: ast.AST) -> List[Dict]:
    """Check error handling patterns"""
    issues = []

    for node in ast.walk(tree):
        # Bare except
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append({
                "issue": f"Bare except clause at line {node.lineno}",
                "severity": "high",
                "pattern": "bare_except"
            })

        # Empty except block
        if isinstance(node, ast.ExceptHandler):
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                issues.append({
                    "issue": f"Empty except block at line {node.lineno}",
                    "severity": "medium",
                    "pattern": "empty_except"
                })

    return issues


def check_magic_numbers(tree: ast.AST) -> List[Dict]:
    """Find magic numbers in code"""
    issues = []
    acceptable_numbers = {0, 1, -1, 2, 10, 100, 1000}

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value not in acceptable_numbers and abs(node.value) > 1:
                # Skip if it's in a constant assignment
                parent_is_assignment = False
                # This is simplified - real check would need parent tracking

                if not parent_is_assignment:
                    issues.append({
                        "issue": f"Magic number {node.value} at line {node.lineno}",
                        "severity": "low",
                        "value": node.value
                    })

    return issues


def check_naming_conventions(tree: ast.AST, file_path: Path) -> List[Dict]:
    """Check naming conventions"""
    issues = []

    for node in ast.walk(tree):
        # Class names should be PascalCase
        if isinstance(node, ast.ClassDef):
            if not node.name[0].isupper():
                issues.append({
                    "issue": f"Class '{node.name}' should use PascalCase",
                    "severity": "low",
                    "line": node.lineno
                })

        # Function names should be snake_case
        if isinstance(node, ast.FunctionDef):
            if any(c.isupper() for c in node.name) and not node.name.startswith('test_'):
                issues.append({
                    "issue": f"Function '{node.name}' should use snake_case",
                    "severity": "low",
                    "line": node.lineno
                })

        # Constants should be UPPER_CASE
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    # If all uppercase, check if it's actually constant
                    if name.isupper() and len(name) > 1:
                        # This is actually good
                        pass

    return issues


def check_code_smells(tree: ast.AST) -> List[Dict]:
    """Detect code smells"""
    issues = []

    # Check for long parameter lists in function calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if len(node.args) > 5:
                issues.append({
                    "issue": f"Function call with {len(node.args)} arguments at line {node.lineno}",
                    "severity": "low",
                    "smell": "long_parameter_list"
                })

    # Check for deep nesting
    def get_depth(node, depth=0):
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                child_depth = get_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)
        return max_depth

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            depth = get_depth(node)
            if depth > 4:
                issues.append({
                    "issue": f"Deep nesting in '{node.name}' (depth: {depth})",
                    "severity": "medium",
                    "smell": "deep_nesting"
                })

    return issues


def analyze_file(file_path: Path) -> Dict:
    """Analyze a Python file for logic issues"""
    try:
        with open(file_path, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        file_issues = {
            "file": str(file_path),
            "functions": 0,
            "classes": 0,
            "complex_functions": [],
            "issues": []
        }

        # Count structures
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

        file_issues["classes"] = len(classes)
        file_issues["functions"] = len(functions)

        # Analyze functions
        for func in functions:
            complexity = analyze_function_complexity(func)
            if complexity > 10:
                file_issues["complex_functions"].append({
                    "name": func.name,
                    "complexity": complexity,
                    "line": func.lineno
                })

            # Check function length
            length_issue = check_function_length(func)
            if length_issue:
                file_issues["issues"].append(length_issue)

            # Check parameters
            param_issue = check_too_many_parameters(func)
            if param_issue:
                file_issues["issues"].append(param_issue)

        # Check patterns
        file_issues["issues"].extend(check_error_handling_patterns(tree))
        file_issues["issues"].extend(check_nested_loops(tree))
        # Magic numbers create too much noise
        # file_issues["issues"].extend(check_magic_numbers(tree))
        file_issues["issues"].extend(check_code_smells(tree))

        return file_issues

    except Exception as e:
        return {
            "file": str(file_path),
            "error": str(e),
            "issues": []
        }


def main():
    print("=" * 80)
    print("🔍 LOGIC AND PATTERN ANALYSIS")
    print("=" * 80)

    # Analyze core files
    core_files = list(Path("core").rglob("*.py"))
    api_files = list(Path("api").rglob("*.py"))
    all_files = core_files + api_files

    print(f"\nAnalyzing {len(all_files)} files...\n")

    all_results = []
    total_issues = 0
    high_severity = 0
    medium_severity = 0

    for file_path in all_files:
        results["total_checks"] += 1
        file_result = analyze_file(file_path)
        all_results.append(file_result)

        if file_result.get("error"):
            print(f"❌ ERROR: {file_path}")
            print(f"   {file_result['error']}")
            continue

        # Count issues
        issue_count = len(file_result["issues"])
        complex_count = len(file_result["complex_functions"])

        if issue_count > 0 or complex_count > 0:
            total_issues += issue_count
            print(f"⚠️  {file_path}")

            if complex_count > 0:
                print(f"   Complex functions: {complex_count}")
                for func in file_result["complex_functions"]:
                    print(f"      - {func['name']} (complexity: {func['complexity']}, line: {func['line']})")

            for issue in file_result["issues"]:
                severity = issue.get("severity", "low")
                if severity == "high":
                    high_severity += 1
                    icon = "🔴"
                elif severity == "medium":
                    medium_severity += 1
                    icon = "🟡"
                else:
                    icon = "🟢"

                print(f"      {icon} {issue['issue']}")

            results["warnings"] += 1
        else:
            print(f"✅ {file_path}")
            results["passed"] += 1

    # Summary
    print("\n" + "=" * 80)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Files analyzed:        {results['total_checks']}")
    print(f"✅ Clean files:         {results['passed']}")
    print(f"⚠️  Files with issues:  {results['warnings']}")
    print(f"Total issues found:    {total_issues}")
    print(f"   🔴 High severity:    {high_severity}")
    print(f"   🟡 Medium severity:  {medium_severity}")
    print(f"   🟢 Low severity:     {total_issues - high_severity - medium_severity}")

    # Function complexity summary
    all_complex = [f for r in all_results for f in r.get("complex_functions", [])]
    if all_complex:
        print("\n" + "=" * 80)
        print("🧮 MOST COMPLEX FUNCTIONS")
        print("=" * 80)
        sorted_complex = sorted(all_complex, key=lambda x: x['complexity'], reverse=True)[:10]
        for func in sorted_complex:
            print(f"   Complexity {func['complexity']:2d} - {func['name']} (line {func['line']})")

    # Save results
    with open('test_results_logic.json', 'w') as f:
        json.dump({
            "summary": results,
            "files": all_results,
            "high_severity_count": high_severity,
            "medium_severity_count": medium_severity
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("📄 Detailed results saved to: test_results_logic.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
