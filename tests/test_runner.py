#!/usr/bin/env python3
"""
Main test runner for MyAgent comprehensive testing suite
Orchestrates different test categories and provides detailed reporting
"""

import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import json
from datetime import datetime
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from loguru import logger


class TestRunner:
    """Comprehensive test runner for MyAgent system"""

    def __init__(self):
        self.project_root = project_root
        self.test_results = {}
        self.start_time = None
        self.total_duration = 0

    def run_test_category(self, category: str, markers: Optional[List[str]] = None, verbose: bool = True) -> Dict[str, Any]:
        """Run a specific category of tests"""
        logger.info(f"Running {category} tests...")

        # Build pytest command
# GEMINI-EDIT - 2025-11-18 - Replaced hardcoded 'python' with 'sys.executable'.
        cmd = [sys.executable, "-m", "pytest", f"tests/{category}/"]

        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])

        if verbose:
            cmd.append("-v")

        # Add coverage for unit tests
        if category == "unit":
            cmd.extend([
                "--cov=core",
                "--cov=api",
                "--cov-report=term-missing"
            ])

        # Add performance reporting for performance tests
        if category == "performance":
            cmd.extend(["--benchmark-only", "--benchmark-sort=mean"])

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=1800  # 30 minute timeout
            )

            duration = time.time() - start_time

            test_result = {
                "category": category,
                "status": "PASSED" if result.returncode == 0 else "FAILED",
                "duration": duration,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }

            # Extract test statistics from output
            if "=" in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "passed" in line and "failed" in line:
                        test_result["summary"] = line.strip()
                        break

            logger.info(f"{category} tests completed in {duration:.2f}s - {test_result['status']}")
            return test_result

        except subprocess.TimeoutExpired:
            logger.error(f"{category} tests timed out after 30 minutes")
            return {
                "category": category,
                "status": "TIMEOUT",
                "duration": 1800,
                "error": "Test execution timed out"
            }
        except Exception as e:
            logger.error(f"Error running {category} tests: {e}")
            return {
                "category": category,
                "status": "ERROR",
                "duration": 0,
                "error": str(e)
            }

    def run_all_tests(self, quick: bool = False) -> Dict[str, Any]:
        """Run all test categories"""
        self.start_time = datetime.now()
        logger.info("Starting comprehensive test suite...")

        test_categories = [
            ("unit", ["unit"]),
            ("integration", ["integration"]),
            ("system", ["system"])
        ]

        # Add additional categories for full test run
        if not quick:
            test_categories.extend([
                ("e2e", ["e2e"]),
                ("performance", ["performance"]),
                ("usability", ["usability"])
            ])

        results = {}
        total_start = time.time()

        for category, markers in test_categories:
            if Path(self.project_root / "tests" / category).exists():
                results[category] = self.run_test_category(category, markers)
            else:
                logger.warning(f"Test directory tests/{category} does not exist, skipping...")
                results[category] = {
                    "category": category,
                    "status": "SKIPPED",
                    "reason": "Directory not found"
                }

        self.total_duration = time.time() - total_start

        # Generate comprehensive report
        report = self.generate_report(results)
        return report

    def run_specific_tests(self, test_path: str, markers: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run specific tests by path or pattern"""
        logger.info(f"Running specific tests: {test_path}")

# GEMINI-EDIT - 2025-11-18 - Replaced hardcoded 'python' with 'sys.executable'.
        cmd = [sys.executable, "-m", "pytest", test_path, "-v"]

        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            return {
                "test_path": test_path,
                "status": "PASSED" if result.returncode == 0 else "FAILED",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        except Exception as e:
            return {
                "test_path": test_path,
                "status": "ERROR",
                "error": str(e)
            }

    def run_gpt5_priority_tests(self) -> Dict[str, Any]:
        """Run tests specifically for GPT-5 priorities"""
        logger.info("Running GPT-5 priority system tests...")

        gpt5_tests = [
            "tests/unit/test_gpt5_p4_memory.py",
            "tests/unit/test_gpt5_p5_security.py",
            "tests/unit/test_gpt5_p6_healing.py",
            "tests/unit/test_gpt5_p7_knowledge.py",
            "tests/unit/test_gpt5_p9_deployment.py",
            "tests/unit/test_gpt5_p10_causal.py"
        ]

        results = {}
        for test_file in gpt5_tests:
            if Path(self.project_root / test_file).exists():
                results[test_file] = self.run_specific_tests(test_file, ["gpt5"])
            else:
                logger.warning(f"GPT-5 test file {test_file} not found")

        return results

    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test report"""

        total_tests = sum(1 for r in results.values() if r.get("status") not in ["SKIPPED", "ERROR"])
        passed_tests = sum(1 for r in results.values() if r.get("status") == "PASSED")
        failed_tests = sum(1 for r in results.values() if r.get("status") == "FAILED")

        report = {
            "timestamp": self.start_time.isoformat(),
            "total_duration": self.total_duration,
            "summary": {
                "total_categories": len(results),
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "results": results,
            "recommendations": self.generate_recommendations(results)
        }

        # Log summary
        logger.info(f"Test Summary: {passed_tests}/{total_tests} categories passed ({report['summary']['success_rate']:.1f}%)")

        if failed_tests > 0:
            logger.error(f"Failed test categories: {failed_tests}")
            for category, result in results.items():
                if result.get("status") == "FAILED":
                    logger.error(f"  - {category}: {result.get('error', 'See output for details')}")

        return report

    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        # Check for failed categories
        failed_categories = [cat for cat, result in results.items() if result.get("status") == "FAILED"]

        if "unit" in failed_categories:
            recommendations.append("Fix unit test failures before proceeding with integration testing")

        if "integration" in failed_categories:
            recommendations.append("Review agent coordination and API integration issues")

        if "performance" in failed_categories:
            recommendations.append("Optimize performance bottlenecks identified in performance tests")

        if "usability" in failed_categories:
            recommendations.append("Address UI/UX issues found in usability testing")

        # Check for missing test categories
        expected_categories = ["unit", "integration", "system", "e2e", "performance", "usability"]
        missing_categories = [cat for cat in expected_categories if cat not in results]

        if missing_categories:
            recommendations.append(f"Implement missing test categories: {', '.join(missing_categories)}")

        # Coverage recommendations
        for category, result in results.items():
            if "cov" in result.get("stdout", "") and "100%" not in result.get("stdout", ""):
                recommendations.append(f"Improve test coverage in {category} tests")

        return recommendations

    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Save test report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.json"

        report_path = self.project_root / "tests" / "reports" / filename
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Test report saved to: {report_path}")
        return str(report_path)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="MyAgent Comprehensive Test Runner")
    parser.add_argument("--category", "-c", help="Run specific test category",
                       choices=["unit", "integration", "system", "e2e", "performance", "usability"])
    parser.add_argument("--gpt5", action="store_true", help="Run GPT-5 priority tests only")
    parser.add_argument("--quick", "-q", action="store_true", help="Run quick test suite (unit, integration, system)")
    parser.add_argument("--markers", "-m", nargs="+", help="Pytest markers to filter tests")
    parser.add_argument("--test-path", "-t", help="Run specific test file or pattern")
    parser.add_argument("--save-report", "-s", action="store_true", help="Save detailed report to file")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Verbose output")

    args = parser.parse_args()

    runner = TestRunner()

    try:
        if args.gpt5:
            results = runner.run_gpt5_priority_tests()
            print(json.dumps(results, indent=2, default=str))
        elif args.category:
            result = runner.run_test_category(args.category, args.markers, args.verbose)
            print(json.dumps(result, indent=2, default=str))
        elif args.test_path:
            result = runner.run_specific_tests(args.test_path, args.markers)
            print(json.dumps(result, indent=2, default=str))
        else:
            report = runner.run_all_tests(quick=args.quick)
            print(json.dumps(report, indent=2, default=str))

            if args.save_report:
                runner.save_report(report)

        # Exit with error code if tests failed
        if isinstance(results if args.gpt5 or args.test_path else
                     result if args.category else report, dict):
            test_results = results if args.gpt5 else (result if args.category else report)

            if args.category:
                sys.exit(0 if test_results.get("status") == "PASSED" else 1)
            elif not args.gpt5:
                failed_count = test_results.get("summary", {}).get("failed", 0)
                sys.exit(0 if failed_count == 0 else 1)

    except KeyboardInterrupt:
        logger.info("Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()