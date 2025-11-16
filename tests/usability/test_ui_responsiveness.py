"""
UI Responsiveness and User Experience Tests for MyAgent
Tests user interface performance, responsiveness, and usability metrics
"""

import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pytest
import json
import logging
from dataclasses import dataclass
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException

# UI responsiveness configuration
UI_CONFIG = {
    "base_url": "http://localhost:3000",
    "page_load_timeout": 10,
    "element_load_timeout": 5,
    "acceptable_load_time": 3.0,
    "acceptable_interaction_time": 0.5,
    "viewport_sizes": [
        (1920, 1080),  # Desktop
        (1366, 768),   # Laptop
        (768, 1024),   # Tablet
        (375, 667),    # Mobile
    ],
    "test_pages": [
        "/",
        "/dashboard",
        "/projects",
        "/agents",
        "/settings"
    ]
}

@dataclass
class UIMetric:
    """UI performance measurement"""
    page_url: str
    action_type: str
    load_time: float
    viewport_size: Tuple[int, int]
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    dom_elements_count: Optional[int] = None
    network_requests: Optional[int] = None

@dataclass
class InteractionMetric:
    """User interaction performance measurement"""
    interaction_type: str
    element_selector: str
    response_time: float
    success: bool
    viewport_size: Tuple[int, int]
    timestamp: datetime
    error_message: Optional[str] = None

class MockWebDriver:
    """Mock web driver for UI testing"""

    def __init__(self, viewport_size: Tuple[int, int] = (1920, 1080)):
        self.viewport_size = viewport_size
        self.current_url = ""
        self.page_source = ""
        self.title = ""
        self.dom_elements = {}
        self.network_requests = 0
        self.javascript_errors = []

    def set_window_size(self, width: int, height: int):
        """Set browser window size"""
        self.viewport_size = (width, height)

    def get_window_size(self) -> Dict[str, int]:
        """Get current window size"""
        return {"width": self.viewport_size[0], "height": self.viewport_size[1]}

    async def get(self, url: str) -> float:
        """Navigate to URL and measure load time"""
        start_time = time.time()

        # Simulate network latency based on viewport (mobile = slower)
        if self.viewport_size[0] < 768:  # Mobile
            base_delay = 0.8
        elif self.viewport_size[0] < 1366:  # Tablet
            base_delay = 0.5
        else:  # Desktop
            base_delay = 0.3

        # Simulate page complexity based on URL
        if url.endswith('/dashboard'):
            complexity_delay = 1.2
            self.network_requests = 15
        elif url.endswith('/projects'):
            complexity_delay = 0.8
            self.network_requests = 10
        elif url.endswith('/agents'):
            complexity_delay = 0.6
            self.network_requests = 8
        elif url.endswith('/settings'):
            complexity_delay = 0.4
            self.network_requests = 5
        else:  # Home page
            complexity_delay = 0.2
            self.network_requests = 3

        # Simulate random variance
        import random
        variance = random.uniform(0.8, 1.2)

        total_delay = (base_delay + complexity_delay) * variance
        await asyncio.sleep(total_delay)

        self.current_url = url
        self.title = f"MyAgent - {url.split('/')[-1].title() or 'Home'}"

        # Generate mock DOM structure
        self._generate_mock_dom(url)

        return time.time() - start_time

    async def find_element(self, by: By, value: str) -> 'MockWebElement':
        """Find element by selector"""
        start_time = time.time()

        # Simulate element search time
        search_delay = 0.01 + (0.005 * len(value))
        await asyncio.sleep(search_delay)

        # Check if element exists in mock DOM
        element_key = f"{by}:{value}"
        if element_key in self.dom_elements:
            element = MockWebElement(value, self.dom_elements[element_key])
            element.load_time = time.time() - start_time
            return element
        else:
            raise TimeoutException(f"Element not found: {value}")

    async def find_elements(self, by: By, value: str) -> List['MockWebElement']:
        """Find multiple elements by selector"""
        start_time = time.time()

        # Simulate search time
        await asyncio.sleep(0.02)

        elements = []
        for key, props in self.dom_elements.items():
            if value in key:
                element = MockWebElement(value, props)
                element.load_time = time.time() - start_time
                elements.append(element)

        return elements

    def execute_script(self, script: str) -> Any:
        """Execute JavaScript and return result"""
        # Mock common JavaScript operations
        if "performance.timing" in script:
            return {
                "navigationStart": int(time.time() * 1000) - 2000,
                "domContentLoadedEventEnd": int(time.time() * 1000) - 500,
                "loadEventEnd": int(time.time() * 1000)
            }
        elif "document.readyState" in script:
            return "complete"
        elif "window.innerWidth" in script:
            return self.viewport_size[0]
        elif "window.innerHeight" in script:
            return self.viewport_size[1]
        else:
            return None

    def _generate_mock_dom(self, url: str):
        """Generate mock DOM elements for the page"""
        self.dom_elements = {}

        # Common elements on all pages
        common_elements = {
            "tag:body": {"visible": True, "size": {"width": self.viewport_size[0], "height": self.viewport_size[1]}},
            "class:header": {"visible": True, "size": {"width": self.viewport_size[0], "height": 80}},
            "class:navigation": {"visible": True, "size": {"width": 250, "height": self.viewport_size[1] - 80}},
            "id:main-content": {"visible": True, "size": {"width": self.viewport_size[0] - 250, "height": self.viewport_size[1] - 80}},
            "class:footer": {"visible": True, "size": {"width": self.viewport_size[0], "height": 60}}
        }

        # Page-specific elements
        if url.endswith('/dashboard'):
            page_elements = {
                "class:dashboard-cards": {"visible": True, "size": {"width": 800, "height": 400}},
                "class:metrics-panel": {"visible": True, "size": {"width": 600, "height": 300}},
                "class:recent-activity": {"visible": True, "size": {"width": 400, "height": 500}},
                "button:refresh-data": {"visible": True, "size": {"width": 120, "height": 40}},
                "class:chart-container": {"visible": True, "size": {"width": 700, "height": 350}}
            }
        elif url.endswith('/projects'):
            page_elements = {
                "class:project-list": {"visible": True, "size": {"width": 900, "height": 600}},
                "button:new-project": {"visible": True, "size": {"width": 150, "height": 40}},
                "input:search-projects": {"visible": True, "size": {"width": 300, "height": 35}},
                "class:project-card": {"visible": True, "size": {"width": 280, "height": 200}},
                "select:filter-status": {"visible": True, "size": {"width": 200, "height": 35}}
            }
        elif url.endswith('/agents'):
            page_elements = {
                "class:agent-grid": {"visible": True, "size": {"width": 1000, "height": 700}},
                "button:add-agent": {"visible": True, "size": {"width": 130, "height": 40}},
                "class:agent-status": {"visible": True, "size": {"width": 100, "height": 25}},
                "class:agent-metrics": {"visible": True, "size": {"width": 250, "height": 150}}
            }
        elif url.endswith('/settings'):
            page_elements = {
                "form:settings-form": {"visible": True, "size": {"width": 600, "height": 500}},
                "input:api-key": {"visible": True, "size": {"width": 400, "height": 35}},
                "button:save-settings": {"visible": True, "size": {"width": 100, "height": 40}},
                "class:settings-tabs": {"visible": True, "size": {"width": 600, "height": 50}}
            }
        else:  # Home page
            page_elements = {
                "class:hero-section": {"visible": True, "size": {"width": self.viewport_size[0], "height": 400}},
                "button:get-started": {"visible": True, "size": {"width": 180, "height": 50}},
                "class:feature-grid": {"visible": True, "size": {"width": 1000, "height": 300}},
                "class:testimonials": {"visible": True, "size": {"width": 800, "height": 250}}
            }

        # Merge common and page-specific elements
        self.dom_elements.update(common_elements)
        self.dom_elements.update(page_elements)

        # Adjust for mobile responsiveness
        if self.viewport_size[0] < 768:
            for key, props in self.dom_elements.items():
                if props["size"]["width"] > self.viewport_size[0]:
                    props["size"]["width"] = self.viewport_size[0] - 20
                if "navigation" in key:
                    props["visible"] = False  # Hidden on mobile

    def quit(self):
        """Close the browser"""
        pass

class MockWebElement:
    """Mock web element for interaction testing"""

    def __init__(self, selector: str, properties: Dict[str, Any]):
        self.selector = selector
        self.properties = properties
        self.load_time = 0

    async def click(self) -> float:
        """Click the element and measure response time"""
        start_time = time.time()

        # Simulate click processing time based on element type
        if "button" in self.selector:
            await asyncio.sleep(0.05)  # Button clicks are quick
        elif "input" in self.selector:
            await asyncio.sleep(0.02)  # Input focus is very quick
        elif "select" in self.selector:
            await asyncio.sleep(0.1)   # Dropdowns take a bit longer
        else:
            await asyncio.sleep(0.03)  # Generic click

        return time.time() - start_time

    async def send_keys(self, text: str) -> float:
        """Type text into the element and measure response time"""
        start_time = time.time()

        # Simulate typing delay (faster for shorter text)
        typing_delay = len(text) * 0.01  # 10ms per character
        await asyncio.sleep(typing_delay)

        return time.time() - start_time

    def is_displayed(self) -> bool:
        """Check if element is visible"""
        return self.properties.get("visible", True)

    def get_attribute(self, name: str) -> Optional[str]:
        """Get element attribute"""
        return self.properties.get(name)

    @property
    def size(self) -> Dict[str, int]:
        """Get element size"""
        return self.properties.get("size", {"width": 100, "height": 30})

    @property
    def text(self) -> str:
        """Get element text"""
        return self.properties.get("text", "Sample Text")

class UIResponsivenessTester:
    """UI responsiveness testing framework"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.driver: Optional[MockWebDriver] = None
        self.metrics: List[UIMetric] = []
        self.interaction_metrics: List[InteractionMetric] = []

    async def setup_driver(self, viewport_size: Tuple[int, int] = (1920, 1080)):
        """Setup mock web driver"""
        self.driver = MockWebDriver(viewport_size)

    async def cleanup_driver(self):
        """Cleanup web driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None

    async def measure_page_load(self, page_path: str, viewport_size: Tuple[int, int]) -> UIMetric:
        """Measure page load performance"""
        if not self.driver:
            await self.setup_driver(viewport_size)

        self.driver.set_window_size(*viewport_size)
        url = f"{self.base_url}{page_path}"

        try:
            load_time = await self.driver.get(url)

            metric = UIMetric(
                page_url=url,
                action_type="page_load",
                load_time=load_time,
                viewport_size=viewport_size,
                timestamp=datetime.now(),
                success=True,
                dom_elements_count=len(self.driver.dom_elements),
                network_requests=self.driver.network_requests
            )

        except Exception as e:
            metric = UIMetric(
                page_url=url,
                action_type="page_load",
                load_time=float('inf'),
                viewport_size=viewport_size,
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )

        self.metrics.append(metric)
        return metric

    async def measure_user_interaction(self, element_selector: str, interaction_type: str,
                                     viewport_size: Tuple[int, int], interaction_data: Optional[str] = None) -> InteractionMetric:
        """Measure user interaction responsiveness"""
        if not self.driver:
            await self.setup_driver(viewport_size)

        try:
            element = await self.driver.find_element(By.CSS_SELECTOR, element_selector)

            if interaction_type == "click":
                response_time = await element.click()
            elif interaction_type == "type" and interaction_data:
                response_time = await element.send_keys(interaction_data)
            else:
                response_time = 0.01  # Default minimal interaction

            metric = InteractionMetric(
                interaction_type=interaction_type,
                element_selector=element_selector,
                response_time=response_time,
                success=True,
                viewport_size=viewport_size,
                timestamp=datetime.now()
            )

        except Exception as e:
            metric = InteractionMetric(
                interaction_type=interaction_type,
                element_selector=element_selector,
                response_time=float('inf'),
                success=False,
                viewport_size=viewport_size,
                timestamp=datetime.now(),
                error_message=str(e)
            )

        self.interaction_metrics.append(metric)
        return metric

    async def test_responsive_design(self, page_path: str) -> List[UIMetric]:
        """Test page responsiveness across different viewport sizes"""
        results = []

        for viewport_size in UI_CONFIG["viewport_sizes"]:
            metric = await self.measure_page_load(page_path, viewport_size)
            results.append(metric)

            # Small delay between viewport changes
            await asyncio.sleep(0.1)

        return results

    async def test_critical_user_flows(self, viewport_size: Tuple[int, int] = (1920, 1080)) -> List[InteractionMetric]:
        """Test critical user interaction flows"""
        await self.setup_driver(viewport_size)

        interactions = [
            ("button:new-project", "click"),
            ("input:search-projects", "type", "test project"),
            ("select:filter-status", "click"),
            ("button:save-settings", "click"),
            ("button:refresh-data", "click")
        ]

        results = []

        for interaction in interactions:
            element_selector = interaction[0]
            interaction_type = interaction[1]
            interaction_data = interaction[2] if len(interaction) > 2 else None

            # Navigate to appropriate page first
            if "project" in element_selector:
                await self.driver.get(f"{self.base_url}/projects")
            elif "settings" in element_selector:
                await self.driver.get(f"{self.base_url}/settings")
            elif "refresh" in element_selector:
                await self.driver.get(f"{self.base_url}/dashboard")

            try:
                metric = await self.measure_user_interaction(
                    element_selector, interaction_type, viewport_size, interaction_data
                )
                results.append(metric)
            except Exception as e:
                logging.error(f"Error testing interaction {element_selector}: {e}")

            await asyncio.sleep(0.1)  # Small delay between interactions

        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get UI performance summary"""
        if not self.metrics and not self.interaction_metrics:
            return {"error": "No metrics collected"}

        summary = {"page_loads": {}, "interactions": {}}

        # Analyze page load metrics
        if self.metrics:
            successful_loads = [m for m in self.metrics if m.success]
            load_times = [m.load_time for m in successful_loads]

            if load_times:
                summary["page_loads"] = {
                    "total_pages_tested": len(self.metrics),
                    "successful_loads": len(successful_loads),
                    "avg_load_time": statistics.mean(load_times),
                    "median_load_time": statistics.median(load_times),
                    "max_load_time": max(load_times),
                    "min_load_time": min(load_times),
                    "success_rate": len(successful_loads) / len(self.metrics)
                }

                # Breakdown by viewport size
                viewport_breakdown = {}
                for metric in successful_loads:
                    viewport_key = f"{metric.viewport_size[0]}x{metric.viewport_size[1]}"
                    if viewport_key not in viewport_breakdown:
                        viewport_breakdown[viewport_key] = []
                    viewport_breakdown[viewport_key].append(metric.load_time)

                for viewport, times in viewport_breakdown.items():
                    summary["page_loads"][viewport] = {
                        "avg_load_time": statistics.mean(times),
                        "max_load_time": max(times),
                        "min_load_time": min(times)
                    }

        # Analyze interaction metrics
        if self.interaction_metrics:
            successful_interactions = [m for m in self.interaction_metrics if m.success]
            interaction_times = [m.response_time for m in successful_interactions]

            if interaction_times:
                summary["interactions"] = {
                    "total_interactions_tested": len(self.interaction_metrics),
                    "successful_interactions": len(successful_interactions),
                    "avg_response_time": statistics.mean(interaction_times),
                    "median_response_time": statistics.median(interaction_times),
                    "max_response_time": max(interaction_times),
                    "min_response_time": min(interaction_times),
                    "success_rate": len(successful_interactions) / len(self.interaction_metrics)
                }

        return summary

# Test Fixtures

@pytest.fixture
def ui_tester():
    """UI responsiveness tester fixture"""
    return UIResponsivenessTester(UI_CONFIG["base_url"])

@pytest.fixture
def test_scenarios():
    """Test scenarios for UI testing"""
    return {
        "critical_pages": ["/", "/dashboard", "/projects"],
        "interaction_flows": [
            {"name": "project_creation", "steps": ["navigate_to_projects", "click_new_project", "fill_form", "submit"]},
            {"name": "search_functionality", "steps": ["navigate_to_projects", "focus_search", "type_query", "view_results"]},
            {"name": "settings_update", "steps": ["navigate_to_settings", "modify_setting", "save_changes", "verify_update"]}
        ],
        "viewport_priorities": ["desktop", "tablet", "mobile"]
    }

# UI Responsiveness Tests

@pytest.mark.asyncio
@pytest.mark.usability
@pytest.mark.ui
class TestUIResponsiveness:
    """UI responsiveness test suite"""

    async def test_page_load_performance(self, ui_tester):
        """Test page load performance across different pages"""
        for page_path in UI_CONFIG["test_pages"]:
            metric = await ui_tester.measure_page_load(page_path, (1920, 1080))

            assert metric.success, f"Failed to load page {page_path}: {metric.error_message}"
            assert metric.load_time < UI_CONFIG["acceptable_load_time"], \
                f"Page {page_path} loaded too slowly: {metric.load_time:.2f}s"

            logging.info(f"Page {page_path} load time: {metric.load_time:.3f}s")

        await ui_tester.cleanup_driver()

    @pytest.mark.parametrize("viewport_size", UI_CONFIG["viewport_sizes"])
    async def test_responsive_design(self, ui_tester, viewport_size):
        """Test responsive design across different viewport sizes"""
        critical_pages = ["/", "/dashboard", "/projects"]

        for page_path in critical_pages:
            metric = await ui_tester.measure_page_load(page_path, viewport_size)

            assert metric.success, f"Failed to load page {page_path} at {viewport_size}"

            # Mobile pages may load slightly slower
            timeout_multiplier = 1.5 if viewport_size[0] < 768 else 1.0
            max_load_time = UI_CONFIG["acceptable_load_time"] * timeout_multiplier

            assert metric.load_time < max_load_time, \
                f"Page {page_path} at {viewport_size} loaded too slowly: {metric.load_time:.2f}s"

            logging.info(f"Page {page_path} at {viewport_size}: {metric.load_time:.3f}s")

        await ui_tester.cleanup_driver()

    async def test_user_interaction_responsiveness(self, ui_tester):
        """Test user interaction responsiveness"""
        interactions = await ui_tester.test_critical_user_flows()

        for interaction in interactions:
            if interaction.success:
                assert interaction.response_time < UI_CONFIG["acceptable_interaction_time"], \
                    f"Interaction {interaction.interaction_type} on {interaction.element_selector} " \
                    f"was too slow: {interaction.response_time:.3f}s"

                logging.info(f"Interaction {interaction.interaction_type}: {interaction.response_time:.3f}s")

        await ui_tester.cleanup_driver()

    async def test_mobile_usability(self, ui_tester):
        """Test mobile-specific usability"""
        mobile_viewport = (375, 667)

        # Test critical mobile interactions
        interactions = await ui_tester.test_critical_user_flows(mobile_viewport)

        successful_interactions = [i for i in interactions if i.success]
        success_rate = len(successful_interactions) / len(interactions) if interactions else 0

        assert success_rate > 0.8, f"Mobile interaction success rate too low: {success_rate:.2%}"

        if successful_interactions:
            avg_response_time = statistics.mean(i.response_time for i in successful_interactions)
            assert avg_response_time < UI_CONFIG["acceptable_interaction_time"] * 1.5, \
                f"Mobile interactions too slow: {avg_response_time:.3f}s"

        await ui_tester.cleanup_driver()

    async def test_load_time_consistency(self, ui_tester):
        """Test load time consistency across multiple attempts"""
        page_path = "/dashboard"  # Most complex page
        load_times = []

        # Test multiple loads
        for _ in range(5):
            metric = await ui_tester.measure_page_load(page_path, (1920, 1080))
            if metric.success:
                load_times.append(metric.load_time)
            await asyncio.sleep(0.5)

        assert len(load_times) >= 3, "Not enough successful loads to test consistency"

        # Check consistency (low variance)
        avg_load_time = statistics.mean(load_times)
        std_deviation = statistics.stdev(load_times) if len(load_times) > 1 else 0

        # Standard deviation should be less than 30% of mean
        assert std_deviation < avg_load_time * 0.3, \
            f"Load time too inconsistent: {avg_load_time:.3f}s ± {std_deviation:.3f}s"

        logging.info(f"Load time consistency: {avg_load_time:.3f}s ± {std_deviation:.3f}s")

        await ui_tester.cleanup_driver()

    async def test_ui_performance_degradation(self, ui_tester):
        """Test UI performance under prolonged usage"""
        # Simulate extended session
        initial_metric = await ui_tester.measure_page_load("/dashboard", (1920, 1080))

        # Perform many interactions
        for _ in range(20):
            await ui_tester.measure_user_interaction("button:refresh-data", "click", (1920, 1080))
            await asyncio.sleep(0.1)

        # Test final performance
        final_metric = await ui_tester.measure_page_load("/dashboard", (1920, 1080))

        assert initial_metric.success and final_metric.success, "Failed to complete degradation test"

        # Performance shouldn't degrade significantly
        performance_degradation = (final_metric.load_time - initial_metric.load_time) / initial_metric.load_time
        assert performance_degradation < 0.5, \
            f"Performance degraded too much: {performance_degradation:.2%}"

        logging.info(f"Performance degradation: {performance_degradation:.2%}")

        await ui_tester.cleanup_driver()

    async def test_cross_viewport_consistency(self, ui_tester):
        """Test that functionality works consistently across viewports"""
        test_page = "/projects"
        results = await ui_tester.test_responsive_design(test_page)

        successful_results = [r for r in results if r.success]
        assert len(successful_results) == len(UI_CONFIG["viewport_sizes"]), \
            "Not all viewports loaded successfully"

        # All viewports should load within reasonable time
        for result in successful_results:
            viewport_name = "mobile" if result.viewport_size[0] < 768 else \
                          "tablet" if result.viewport_size[0] < 1366 else "desktop"

            timeout_multiplier = {"mobile": 2.0, "tablet": 1.5, "desktop": 1.0}[viewport_name]
            max_time = UI_CONFIG["acceptable_load_time"] * timeout_multiplier

            assert result.load_time < max_time, \
                f"{viewport_name} load time too slow: {result.load_time:.3f}s"

        await ui_tester.cleanup_driver()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v", "--tb=short"])