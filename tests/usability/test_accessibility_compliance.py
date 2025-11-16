"""
Accessibility and Cross-Browser Compatibility Tests for MyAgent
Tests WCAG compliance, screen reader compatibility, keyboard navigation, and cross-browser support
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pytest
import logging
from dataclasses import dataclass
from enum import Enum

# Accessibility configuration
ACCESSIBILITY_CONFIG = {
    "wcag_compliance_level": "AA",
    "supported_browsers": ["chrome", "firefox", "safari", "edge"],
    "screen_readers": ["nvda", "jaws", "voiceover"],
    "keyboard_navigation_timeout": 2.0,
    "color_contrast_ratio": 4.5,  # WCAG AA standard
    "font_size_minimum": 12,
    "touch_target_minimum": 44  # pixels
}

class AccessibilityLevel(Enum):
    A = "A"
    AA = "AA"
    AAA = "AAA"

class BrowserType(Enum):
    CHROME = "chrome"
    FIREFOX = "firefox"
    SAFARI = "safari"
    EDGE = "edge"

@dataclass
class AccessibilityIssue:
    """Accessibility compliance issue"""
    rule_id: str
    severity: str
    element_selector: str
    description: str
    wcag_guideline: str
    impact: str
    help_url: Optional[str] = None

@dataclass
class AccessibilityTestResult:
    """Accessibility test result"""
    page_url: str
    compliance_level: AccessibilityLevel
    total_issues: int
    critical_issues: int
    serious_issues: int
    moderate_issues: int
    minor_issues: int
    issues: List[AccessibilityIssue]
    timestamp: datetime
    browser: BrowserType
    success: bool

@dataclass
class KeyboardNavigationResult:
    """Keyboard navigation test result"""
    element_selector: str
    focusable: bool
    tab_order: int
    keyboard_accessible: bool
    aria_labels: Dict[str, str]
    navigation_time: float
    success: bool
    error_message: Optional[str] = None

class MockAccessibilityScanner:
    """Mock accessibility scanner for testing"""

    def __init__(self, browser_type: BrowserType = BrowserType.CHROME):
        self.browser_type = browser_type
        self.scan_results: List[AccessibilityTestResult] = []

    async def scan_page(self, url: str, compliance_level: AccessibilityLevel = AccessibilityLevel.AA) -> AccessibilityTestResult:
        """Scan page for accessibility issues"""
        start_time = time.time()

        # Simulate scan time
        await asyncio.sleep(0.5)

        # Generate mock accessibility issues based on page complexity
        issues = await self._generate_mock_issues(url)

        # Categorize issues by severity
        critical_issues = [i for i in issues if i.severity == "critical"]
        serious_issues = [i for i in issues if i.severity == "serious"]
        moderate_issues = [i for i in issues if i.severity == "moderate"]
        minor_issues = [i for i in issues if i.severity == "minor"]

        result = AccessibilityTestResult(
            page_url=url,
            compliance_level=compliance_level,
            total_issues=len(issues),
            critical_issues=len(critical_issues),
            serious_issues=len(serious_issues),
            moderate_issues=len(moderate_issues),
            minor_issues=len(minor_issues),
            issues=issues,
            timestamp=datetime.now(),
            browser=self.browser_type,
            success=True
        )

        self.scan_results.append(result)
        return result

    async def _generate_mock_issues(self, url: str) -> List[AccessibilityIssue]:
        """Generate mock accessibility issues based on page type"""
        issues = []

        # Common issues across pages
        common_issues = [
            AccessibilityIssue(
                rule_id="color-contrast",
                severity="serious",
                element_selector=".text-muted",
                description="Text elements have insufficient color contrast",
                wcag_guideline="1.4.3 Contrast (Minimum)",
                impact="serious",
                help_url="https://dequeuniversity.com/rules/axe/4.7/color-contrast"
            ),
            AccessibilityIssue(
                rule_id="link-name",
                severity="serious",
                element_selector="a[href]",
                description="Links must have discernible text",
                wcag_guideline="2.4.4 Link Purpose (In Context)",
                impact="serious",
                help_url="https://dequeuniversity.com/rules/axe/4.7/link-name"
            )
        ]

        # Page-specific issues
        if url.endswith('/dashboard'):
            dashboard_issues = [
                AccessibilityIssue(
                    rule_id="aria-valid-attr-value",
                    severity="critical",
                    element_selector=".chart-container",
                    description="ARIA attribute values must be valid",
                    wcag_guideline="4.1.2 Name, Role, Value",
                    impact="critical"
                ),
                AccessibilityIssue(
                    rule_id="button-name",
                    severity="serious",
                    element_selector="button.icon-only",
                    description="Buttons must have discernible text",
                    wcag_guideline="4.1.2 Name, Role, Value",
                    impact="serious"
                ),
                AccessibilityIssue(
                    rule_id="image-alt",
                    severity="critical",
                    element_selector="img.chart-icon",
                    description="Images must have alternate text",
                    wcag_guideline="1.1.1 Non-text Content",
                    impact="critical"
                )
            ]
            issues.extend(dashboard_issues)

        elif url.endswith('/projects'):
            project_issues = [
                AccessibilityIssue(
                    rule_id="label",
                    severity="critical",
                    element_selector="input#search",
                    description="Form elements must have labels",
                    wcag_guideline="3.3.2 Labels or Instructions",
                    impact="critical"
                ),
                AccessibilityIssue(
                    rule_id="focusable-element",
                    severity="serious",
                    element_selector=".project-card",
                    description="Interactive elements must be focusable",
                    wcag_guideline="2.1.1 Keyboard",
                    impact="serious"
                )
            ]
            issues.extend(project_issues)

        elif url.endswith('/agents'):
            agent_issues = [
                AccessibilityIssue(
                    rule_id="role-required-aria-attrs",
                    severity="serious",
                    element_selector="[role='grid']",
                    description="ARIA roles must have required attributes",
                    wcag_guideline="4.1.2 Name, Role, Value",
                    impact="serious"
                ),
                AccessibilityIssue(
                    rule_id="region",
                    severity="moderate",
                    element_selector=".agent-status",
                    description="All page content must be contained by landmarks",
                    wcag_guideline="2.4.1 Bypass Blocks",
                    impact="moderate"
                )
            ]
            issues.extend(agent_issues)

        # Browser-specific adjustments
        if self.browser_type in [BrowserType.FIREFOX, BrowserType.SAFARI]:
            # Some browsers have better built-in accessibility
            issues = issues[:-1]  # Remove one issue

        # Add some minor issues
        minor_issues = [
            AccessibilityIssue(
                rule_id="landmark-one-main",
                severity="minor",
                element_selector="main",
                description="Page should have one main landmark",
                wcag_guideline="2.4.1 Bypass Blocks",
                impact="minor"
            ),
            AccessibilityIssue(
                rule_id="page-has-heading-one",
                severity="minor",
                element_selector="h1",
                description="Page should have a level-one heading",
                wcag_guideline="2.4.6 Headings and Labels",
                impact="minor"
            )
        ]

        issues.extend(common_issues)
        if len(issues) < 3:  # Ensure we have some issues
            issues.extend(minor_issues[:2])

        return issues

class MockKeyboardNavigator:
    """Mock keyboard navigation tester"""

    def __init__(self):
        self.focus_order: List[str] = []
        self.current_focus_index = -1

    async def test_keyboard_navigation(self, url: str) -> List[KeyboardNavigationResult]:
        """Test keyboard navigation on a page"""
        await asyncio.sleep(0.2)  # Simulate page load

        # Generate focusable elements based on page
        focusable_elements = self._get_focusable_elements(url)

        results = []

        for i, element_info in enumerate(focusable_elements):
            selector = element_info["selector"]
            element_type = element_info["type"]

            # Simulate tab navigation
            start_time = time.time()
            await asyncio.sleep(0.05)  # Simulate focus time
            navigation_time = time.time() - start_time

            # Check if element is properly accessible
            keyboard_accessible = element_info.get("keyboard_accessible", True)
            focusable = element_info.get("focusable", True)

            # Generate ARIA labels
            aria_labels = {}
            if element_type == "button":
                aria_labels = {
                    "aria-label": element_info.get("aria_label", f"Button {i+1}"),
                    "role": "button"
                }
            elif element_type == "input":
                aria_labels = {
                    "aria-label": element_info.get("aria_label", f"Input field {i+1}"),
                    "aria-required": "true" if element_info.get("required", False) else "false"
                }
            elif element_type == "link":
                aria_labels = {
                    "aria-label": element_info.get("aria_label", f"Link {i+1}"),
                    "role": "link"
                }

            result = KeyboardNavigationResult(
                element_selector=selector,
                focusable=focusable,
                tab_order=i + 1,
                keyboard_accessible=keyboard_accessible,
                aria_labels=aria_labels,
                navigation_time=navigation_time,
                success=keyboard_accessible and focusable
            )

            results.append(result)

        return results

    def _get_focusable_elements(self, url: str) -> List[Dict[str, Any]]:
        """Get focusable elements for a page"""
        common_elements = [
            {"selector": "button.menu-toggle", "type": "button", "keyboard_accessible": True, "focusable": True},
            {"selector": "input.search", "type": "input", "keyboard_accessible": True, "focusable": True},
            {"selector": "a.nav-link", "type": "link", "keyboard_accessible": True, "focusable": True}
        ]

        if url.endswith('/dashboard'):
            page_elements = [
                {"selector": "button.refresh-btn", "type": "button", "keyboard_accessible": True, "aria_label": "Refresh dashboard data"},
                {"selector": ".chart-container", "type": "div", "keyboard_accessible": False, "focusable": False},
                {"selector": "select.time-range", "type": "select", "keyboard_accessible": True, "focusable": True},
                {"selector": "button.export-data", "type": "button", "keyboard_accessible": True, "aria_label": "Export dashboard data"}
            ]
        elif url.endswith('/projects'):
            page_elements = [
                {"selector": "button.new-project", "type": "button", "keyboard_accessible": True, "aria_label": "Create new project"},
                {"selector": "input.project-search", "type": "input", "keyboard_accessible": True, "aria_label": "Search projects", "required": False},
                {"selector": ".project-card", "type": "div", "keyboard_accessible": True, "focusable": True},
                {"selector": "select.status-filter", "type": "select", "keyboard_accessible": True, "focusable": True}
            ]
        elif url.endswith('/agents'):
            page_elements = [
                {"selector": "button.add-agent", "type": "button", "keyboard_accessible": True, "aria_label": "Add new agent"},
                {"selector": ".agent-card", "type": "div", "keyboard_accessible": True, "focusable": True},
                {"selector": "button.agent-config", "type": "button", "keyboard_accessible": True, "aria_label": "Configure agent"}
            ]
        else:  # Home page
            page_elements = [
                {"selector": "button.cta-primary", "type": "button", "keyboard_accessible": True, "aria_label": "Get started with MyAgent"},
                {"selector": "a.feature-link", "type": "link", "keyboard_accessible": True, "aria_label": "Learn more about features"}
            ]

        return common_elements + page_elements

class AccessibilityTester:
    """Comprehensive accessibility testing framework"""

    def __init__(self):
        self.scanner = MockAccessibilityScanner()
        self.navigator = MockKeyboardNavigator()
        self.test_results: Dict[str, List] = {
            "accessibility_scans": [],
            "keyboard_navigation": [],
            "screen_reader_tests": [],
            "cross_browser_tests": []
        }

    async def test_wcag_compliance(self, urls: List[str],
                                  compliance_level: AccessibilityLevel = AccessibilityLevel.AA) -> List[AccessibilityTestResult]:
        """Test WCAG compliance for multiple URLs"""
        results = []

        for url in urls:
            result = await self.scanner.scan_page(url, compliance_level)
            results.append(result)
            self.test_results["accessibility_scans"].append(result)

        return results

    async def test_keyboard_navigation(self, urls: List[str]) -> Dict[str, List[KeyboardNavigationResult]]:
        """Test keyboard navigation for multiple URLs"""
        results = {}

        for url in urls:
            navigation_results = await self.navigator.test_keyboard_navigation(url)
            results[url] = navigation_results
            self.test_results["keyboard_navigation"].extend(navigation_results)

        return results

    async def test_cross_browser_accessibility(self, url: str, browsers: List[BrowserType]) -> Dict[BrowserType, AccessibilityTestResult]:
        """Test accessibility across different browsers"""
        results = {}

        for browser in browsers:
            scanner = MockAccessibilityScanner(browser)
            result = await scanner.scan_page(url)
            results[browser] = result
            self.test_results["cross_browser_tests"].append(result)

        return results

    async def test_screen_reader_compatibility(self, urls: List[str]) -> Dict[str, Dict[str, Any]]:
        """Test screen reader compatibility"""
        results = {}

        for url in urls:
            await asyncio.sleep(0.3)  # Simulate screen reader test time

            # Mock screen reader test results
            screen_reader_result = {
                "heading_structure": {
                    "logical_order": True,
                    "missing_levels": [],
                    "multiple_h1": False
                },
                "landmark_navigation": {
                    "main_present": True,
                    "nav_present": True,
                    "complementary_present": True,
                    "contentinfo_present": True
                },
                "image_descriptions": {
                    "images_with_alt": 85,  # percentage
                    "decorative_images_marked": 70,
                    "complex_images_described": 60
                },
                "form_accessibility": {
                    "labels_associated": 90,
                    "required_fields_indicated": 85,
                    "error_messages_accessible": 80
                },
                "interactive_elements": {
                    "focus_visible": 95,
                    "keyboard_accessible": 88,
                    "aria_labels_present": 75
                }
            }

            results[url] = screen_reader_result
            self.test_results["screen_reader_tests"].append({
                "url": url,
                "result": screen_reader_result,
                "timestamp": datetime.now()
            })

        return results

    def generate_accessibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive accessibility report"""
        report = {
            "summary": {},
            "detailed_results": self.test_results,
            "recommendations": [],
            "compliance_status": {}
        }

        # Analyze accessibility scan results
        if self.test_results["accessibility_scans"]:
            scans = self.test_results["accessibility_scans"]

            total_issues = sum(scan.total_issues for scan in scans)
            critical_issues = sum(scan.critical_issues for scan in scans)
            serious_issues = sum(scan.serious_issues for scan in scans)

            report["summary"]["accessibility"] = {
                "total_scans": len(scans),
                "total_issues": total_issues,
                "critical_issues": critical_issues,
                "serious_issues": serious_issues,
                "avg_issues_per_page": total_issues / len(scans) if scans else 0
            }

            # Determine compliance status
            if critical_issues == 0 and serious_issues <= 2:
                compliance_status = "WCAG AA Compliant"
            elif critical_issues <= 1 and serious_issues <= 5:
                compliance_status = "Minor Issues Present"
            else:
                compliance_status = "Major Issues Present"

            report["compliance_status"]["wcag_aa"] = compliance_status

        # Analyze keyboard navigation results
        if self.test_results["keyboard_navigation"]:
            nav_results = self.test_results["keyboard_navigation"]
            successful_nav = sum(1 for result in nav_results if result.success)

            report["summary"]["keyboard_navigation"] = {
                "total_elements_tested": len(nav_results),
                "successful_navigation": successful_nav,
                "success_rate": successful_nav / len(nav_results) if nav_results else 0,
                "avg_navigation_time": sum(r.navigation_time for r in nav_results) / len(nav_results) if nav_results else 0
            }

        # Generate recommendations
        if critical_issues > 0:
            report["recommendations"].append("Address critical accessibility issues immediately")
        if serious_issues > 3:
            report["recommendations"].append("Review and fix serious accessibility violations")

        keyboard_success_rate = report.get("summary", {}).get("keyboard_navigation", {}).get("success_rate", 0)
        if keyboard_success_rate < 0.9:
            report["recommendations"].append("Improve keyboard navigation accessibility")

        return report

# Test Fixtures

@pytest.fixture
def accessibility_tester():
    """Accessibility tester fixture"""
    return AccessibilityTester()

@pytest.fixture
def test_urls():
    """Test URLs for accessibility testing"""
    return [
        "http://localhost:3000/",
        "http://localhost:3000/dashboard",
        "http://localhost:3000/projects",
        "http://localhost:3000/agents",
        "http://localhost:3000/settings"
    ]

@pytest.fixture
def browser_list():
    """List of browsers for cross-browser testing"""
    return [BrowserType.CHROME, BrowserType.FIREFOX, BrowserType.SAFARI, BrowserType.EDGE]

# Accessibility Tests

@pytest.mark.asyncio
@pytest.mark.usability
@pytest.mark.accessibility
class TestAccessibilityCompliance:
    """Accessibility compliance test suite"""

    async def test_wcag_aa_compliance(self, accessibility_tester, test_urls):
        """Test WCAG AA compliance for all pages"""
        results = await accessibility_tester.test_wcag_compliance(test_urls, AccessibilityLevel.AA)

        for result in results:
            assert result.success, f"Accessibility scan failed for {result.page_url}"

            # Critical issues should be minimal for AA compliance
            assert result.critical_issues <= 1, \
                f"Too many critical accessibility issues on {result.page_url}: {result.critical_issues}"

            # Serious issues should be limited
            assert result.serious_issues <= 3, \
                f"Too many serious accessibility issues on {result.page_url}: {result.serious_issues}"

            logging.info(f"Accessibility scan for {result.page_url}: "
                        f"{result.critical_issues} critical, {result.serious_issues} serious issues")

    async def test_keyboard_navigation_accessibility(self, accessibility_tester, test_urls):
        """Test keyboard navigation accessibility"""
        navigation_results = await accessibility_tester.test_keyboard_navigation(test_urls)

        for url, nav_results in navigation_results.items():
            successful_navigation = sum(1 for result in nav_results if result.success)
            total_elements = len(nav_results)

            success_rate = successful_navigation / total_elements if total_elements > 0 else 0

            assert success_rate >= 0.85, \
                f"Keyboard navigation success rate too low for {url}: {success_rate:.2%}"

            # Check that tab order is logical
            tab_orders = [result.tab_order for result in nav_results if result.focusable]
            if tab_orders:
                assert tab_orders == sorted(tab_orders), \
                    f"Tab order not logical for {url}: {tab_orders}"

            logging.info(f"Keyboard navigation for {url}: {success_rate:.1%} success rate")

    @pytest.mark.parametrize("browser", [BrowserType.CHROME, BrowserType.FIREFOX])
    async def test_cross_browser_accessibility(self, accessibility_tester, browser):
        """Test accessibility across different browsers"""
        test_url = "http://localhost:3000/dashboard"

        results = await accessibility_tester.test_cross_browser_accessibility(
            test_url, [browser]
        )

        result = results[browser]
        assert result.success, f"Accessibility test failed in {browser.value}"

        # Browser-specific thresholds
        if browser == BrowserType.CHROME:
            max_critical = 1
            max_serious = 3
        else:  # Firefox and others
            max_critical = 2
            max_serious = 4

        assert result.critical_issues <= max_critical, \
            f"Too many critical issues in {browser.value}: {result.critical_issues}"
        assert result.serious_issues <= max_serious, \
            f"Too many serious issues in {browser.value}: {result.serious_issues}"

    async def test_screen_reader_compatibility(self, accessibility_tester, test_urls):
        """Test screen reader compatibility"""
        screen_reader_results = await accessibility_tester.test_screen_reader_compatibility(test_urls)

        for url, result in screen_reader_results.items():
            # Check heading structure
            assert result["heading_structure"]["logical_order"], \
                f"Illogical heading structure on {url}"
            assert not result["heading_structure"]["multiple_h1"], \
                f"Multiple H1 elements found on {url}"

            # Check landmark navigation
            landmarks = result["landmark_navigation"]
            assert landmarks["main_present"], f"Main landmark missing on {url}"
            assert landmarks["nav_present"], f"Navigation landmark missing on {url}"

            # Check image descriptions
            images = result["image_descriptions"]
            assert images["images_with_alt"] >= 80, \
                f"Too few images have alt text on {url}: {images['images_with_alt']}%"

            # Check form accessibility
            forms = result["form_accessibility"]
            assert forms["labels_associated"] >= 85, \
                f"Form labels not properly associated on {url}: {forms['labels_associated']}%"

            logging.info(f"Screen reader compatibility for {url}: "
                        f"{images['images_with_alt']}% images with alt text")

    async def test_color_contrast_compliance(self, accessibility_tester):
        """Test color contrast compliance"""
        # This would typically use a color contrast analyzer
        # For mock purposes, we'll simulate the test

        test_elements = [
            {"selector": ".text-primary", "contrast_ratio": 7.1, "expected_min": 4.5},
            {"selector": ".text-secondary", "contrast_ratio": 4.2, "expected_min": 4.5},
            {"selector": ".text-muted", "contrast_ratio": 3.8, "expected_min": 4.5},
            {"selector": ".btn-primary", "contrast_ratio": 5.2, "expected_min": 4.5},
            {"selector": ".link-text", "contrast_ratio": 6.1, "expected_min": 4.5}
        ]

        failed_elements = []

        for element in test_elements:
            if element["contrast_ratio"] < element["expected_min"]:
                failed_elements.append(element["selector"])

        assert len(failed_elements) <= 1, \
            f"Too many elements fail color contrast requirements: {failed_elements}"

        if failed_elements:
            logging.warning(f"Color contrast issues: {failed_elements}")

    async def test_touch_target_accessibility(self, accessibility_tester):
        """Test touch target size for mobile accessibility"""
        # Simulate touch target size testing
        touch_targets = [
            {"selector": "button.primary", "size": 48},
            {"selector": "button.secondary", "size": 42},
            {"selector": ".nav-link", "size": 44},
            {"selector": ".close-btn", "size": 32},
            {"selector": ".checkbox", "size": 24}
        ]

        small_targets = []

        for target in touch_targets:
            if target["size"] < ACCESSIBILITY_CONFIG["touch_target_minimum"]:
                small_targets.append(target["selector"])

        assert len(small_targets) <= 2, \
            f"Too many touch targets below minimum size: {small_targets}"

        if small_targets:
            logging.warning(f"Small touch targets: {small_targets}")

    async def test_focus_management(self, accessibility_tester):
        """Test focus management and visibility"""
        test_url = "http://localhost:3000/projects"

        navigation_results = await accessibility_tester.test_keyboard_navigation([test_url])
        results = navigation_results[test_url]

        # Check that all interactive elements are focusable
        interactive_elements = [r for r in results if "button" in r.element_selector or "input" in r.element_selector]
        focusable_interactive = [r for r in interactive_elements if r.focusable]

        focusable_rate = len(focusable_interactive) / len(interactive_elements) if interactive_elements else 1
        assert focusable_rate >= 0.95, \
            f"Not enough interactive elements are focusable: {focusable_rate:.2%}"

        # Check navigation timing
        avg_navigation_time = sum(r.navigation_time for r in results) / len(results) if results else 0
        assert avg_navigation_time < ACCESSIBILITY_CONFIG["keyboard_navigation_timeout"], \
            f"Keyboard navigation too slow: {avg_navigation_time:.3f}s"

    async def test_comprehensive_accessibility_report(self, accessibility_tester, test_urls):
        """Generate and validate comprehensive accessibility report"""
        # Run all accessibility tests
        await accessibility_tester.test_wcag_compliance(test_urls)
        await accessibility_tester.test_keyboard_navigation(test_urls)
        await accessibility_tester.test_screen_reader_compatibility(test_urls)

        # Generate report
        report = accessibility_tester.generate_accessibility_report()

        assert "summary" in report
        assert "detailed_results" in report
        assert "recommendations" in report
        assert "compliance_status" in report

        # Check that we have data for all test types
        assert len(report["detailed_results"]["accessibility_scans"]) > 0
        assert len(report["detailed_results"]["keyboard_navigation"]) > 0
        assert len(report["detailed_results"]["screen_reader_tests"]) > 0

        # Compliance status should be reasonable
        wcag_status = report["compliance_status"].get("wcag_aa", "Unknown")
        assert wcag_status in ["WCAG AA Compliant", "Minor Issues Present", "Major Issues Present"]

        logging.info(f"Accessibility report generated. Status: {wcag_status}")
        logging.info(f"Recommendations: {len(report['recommendations'])}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v", "--tb=short"])