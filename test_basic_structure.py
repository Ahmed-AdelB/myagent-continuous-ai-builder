#!/usr/bin/env python3
"""
Basic Structure and Syntax Tests
Tests code structure, imports, and basic functionality without heavy dependencies
"""

import sys
import os
import ast
import json
from pathlib import Path
from typing import List, Dict

# Test results
results = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "issues": []
}


def test_file_syntax(file_path: Path) -> Dict:
    """Test if a Python file has valid syntax"""
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        ast.parse(code)
        return {"status": "PASS", "file": str(file_path), "error": None}
    except SyntaxError as e:
        return {"status": "FAIL", "file": str(file_path), "error": f"Syntax error at line {e.lineno}: {e.msg}"}
    except Exception as e:
        return {"status": "FAIL", "file": str(file_path), "error": str(e)}


def test_file_structure(file_path: Path) -> Dict:
    """Analyze file structure for issues"""
    issues = []

    try:
        with open(file_path, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Check for docstrings
        if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
            pass  # Has module docstring
        else:
            issues.append("Missing module docstring")

        # Count classes and functions
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

        # Check for long functions (>100 lines)
        for func in functions:
            if hasattr(func, 'end_lineno') and func.end_lineno - func.lineno > 100:
                issues.append(f"Long function: {func.name} ({func.end_lineno - func.lineno} lines)")

        # Check for bare except clauses
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append(f"Bare except clause at line {node.lineno}")

        # Check for hardcoded strings that might be secrets
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lower = node.value.lower()
                if any(keyword in lower for keyword in ['password', 'api_key', 'secret', 'token']):
                    if len(node.value) > 10 and not node.value.startswith('{'):
                        issues.append(f"Possible hardcoded secret at line {node.lineno}")

        return {
            "status": "PASS" if not issues else "WARNING",
            "file": str(file_path),
            "classes": len(classes),
            "functions": len(functions),
            "issues": issues
        }

    except Exception as e:
        return {"status": "FAIL", "file": str(file_path), "error": str(e), "issues": []}


def find_python_files(directory: str) -> List[Path]:
    """Find all Python files in directory"""
    path = Path(directory)
    return list(path.rglob("*.py"))


def main():
    print("=" * 80)
    print("🔍 BASIC STRUCTURE AND SYNTAX TESTS")
    print("=" * 80)

    # Test core files
    core_dirs = ['core', 'api', 'config']

    all_files = []
    for dir_name in core_dirs:
        if Path(dir_name).exists():
            files = find_python_files(dir_name)
            all_files.extend(files)
            print(f"\nFound {len(files)} Python files in {dir_name}/")

    print(f"\nTotal Python files to test: {len(all_files)}\n")
    print("=" * 80)

    syntax_results = []
    structure_results = []

    for file_path in all_files:
        # Test syntax
        syntax_result = test_file_syntax(file_path)
        syntax_results.append(syntax_result)
        results["total"] += 1

        if syntax_result["status"] == "PASS":
            results["passed"] += 1
            print(f"✅ SYNTAX OK: {file_path}")

            # Test structure
            struct_result = test_file_structure(file_path)
            structure_results.append(struct_result)

            if struct_result["issues"]:
                print(f"   ⚠️  Issues found:")
                for issue in struct_result["issues"]:
                    print(f"      - {issue}")
                    results["issues"].append({"file": str(file_path), "issue": issue})
        else:
            results["failed"] += 1
            print(f"❌ SYNTAX ERROR: {file_path}")
            print(f"   Error: {syntax_result['error']}")
            results["issues"].append({"file": str(file_path), "issue": syntax_result["error"]})

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Total files tested:     {results['total']}")
    print(f"✅ Valid syntax:         {results['passed']}")
    print(f"❌ Syntax errors:        {results['failed']}")
    print(f"⚠️  Code quality issues: {len(results['issues'])}")

    # Categorize issues
    issue_types = {}
    for issue in results['issues']:
        issue_text = issue['issue']
        if 'bare except' in issue_text.lower():
            issue_types['Bare except clauses'] = issue_types.get('Bare except clauses', 0) + 1
        elif 'long function' in issue_text.lower():
            issue_types['Long functions'] = issue_types.get('Long functions', 0) + 1
        elif 'docstring' in issue_text.lower():
            issue_types['Missing docstrings'] = issue_types.get('Missing docstrings', 0) + 1
        elif 'secret' in issue_text.lower():
            issue_types['Possible secrets'] = issue_types.get('Possible secrets', 0) + 1

    if issue_types:
        print("\n" + "=" * 80)
        print("⚠️  ISSUE BREAKDOWN")
        print("=" * 80)
        for issue_type, count in sorted(issue_types.items()):
            print(f"{issue_type}: {count}")

    # Save results
    with open('test_results_structure.json', 'w') as f:
        json.dump({
            "summary": {
                "total": results['total'],
                "passed": results['passed'],
                "failed": results['failed'],
                "issues": len(results['issues'])
            },
            "issue_types": issue_types,
            "details": results['issues']
        }, f, indent=2)

    print("\n" + "=" * 80)
    print(f"📄 Detailed results saved to: test_results_structure.json")
    print("=" * 80)

    return results['failed'] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
