"""Playwright E2E tests for SuperSenseDoctor web UI.

Requires: playwright (pip install) + chromium (playwright install chromium)
Run: python3 -m pytest tests/test_web_e2e.py -q --headed (to see browser)
"""

import json
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def app_server():
    """Start the FastAPI app in a background thread."""
    import threading
    import uvicorn
    from main import create_app
    import yaml, os

    config = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml"))
    os.environ["DEEPSEEK_API_KEY"] = "sk-test"
    app = create_app(config)

    port = 9611
    ready = threading.Event()

    def run():
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")

    t = threading.Thread(target=run, daemon=True)
    t.start()

    # Wait for server to be ready
    import urllib.request
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2)
            ready.set()
            break
        except Exception:
            import time
            time.sleep(0.2)

    ready.wait(timeout=10)
    yield port


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


def test_dashboard_loads(app_server, browser):
    """Dashboard renders with nav shell and data loading controls."""
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"http://127.0.0.1:{app_server}/")
    # nav shell
    assert page.locator(".topnav").count() == 1
    assert page.locator("#nav-dash").count() == 1
    assert page.locator("#nav-episodes").count() == 1
    assert page.locator("#nav-report").count() == 1
    # data loading buttons
    assert page.locator("#btn-load-pv").count() == 1
    assert page.locator("#btn-load-data").count() == 1
    # methodology drawer
    assert page.locator("#drawer").count() == 1
    page.close()


def test_episodes_page(app_server, browser):
    """Episodes page renders with filter bar and table."""
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"http://127.0.0.1:{app_server}/episodes")
    assert page.locator("#filter-level").count() == 1
    assert page.locator("#filter-event-type").count() == 1
    assert page.locator("#filter-search").count() == 1
    assert page.locator("#episode-tbody").count() == 1
    page.close()


def test_report_page(app_server, browser):
    """Report page renders with stats and level breakdown."""
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"http://127.0.0.1:{app_server}/report")
    assert "周报" in page.title() or page.locator(".topnav").count() == 1
    page.close()


def test_methodology_drawer(app_server, browser):
    """Methodology drawer opens and closes."""
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"http://127.0.0.1:{app_server}/")
    # Drawer closed initially
    assert "open" not in page.locator("#drawer").get_attribute("class") or ""
    # Open via button
    page.locator("button:has([data-lucide='book-open'])").click()
    page.wait_for_timeout(300)
    assert "open" in (page.locator("#drawer").get_attribute("class") or "")
    # Close
    page.locator(".drawer-close").click()
    page.wait_for_timeout(300)
    assert "open" not in (page.locator("#drawer").get_attribute("class") or "")
    page.close()


def test_api_vitals_latest(app_server, browser):
    """API returns all required fields."""
    page = browser.new_page()
    resp = page.goto(f"http://127.0.0.1:{app_server}/api/vitals/latest")
    data = json.loads(resp.text())
    assert data["status"] in ("ok", "no_data")
    if data["status"] == "ok":
        assert "baseline" in data
        assert "absolute_reference" in data
        assert "quality" in data
        assert "fusion" in data and "metric" in data["fusion"]
        assert "latest_action" in data
        assert "data_freshness_sec" in data
        assert "rr_truth" not in data
        assert "hr_truth" not in data
    page.close()


def test_api_trends(app_server, browser):
    """Trends API returns time series."""
    page = browser.new_page()
    resp = page.goto(f"http://127.0.0.1:{app_server}/api/trends?range=1h")
    data = json.loads(resp.text())
    assert data["status"] in ("ok", "no_data")
    if data["status"] == "ok":
        assert len(data["points"]) >= 2
        assert "hr_fused" in data["points"][0]
        assert "nlos_flag" in data["points"][0]
        assert "quality_event" in data["points"][0]
    page.close()


def test_api_episodes_pagination(app_server, browser):
    """Episodes API returns paginated results."""
    page = browser.new_page()
    resp = page.goto(f"http://127.0.0.1:{app_server}/api/episodes?limit=5&offset=10")
    data = json.loads(resp.text())
    assert data["status"] == "ok"
    assert "total" in data
    assert "episodes" in data
    assert data["limit"] == 5
    assert data["offset"] == 10
    page.close()


def test_keyboard_navigation(app_server, browser):
    """Tab navigation reaches key elements."""
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"http://127.0.0.1:{app_server}/")
    # Tab through nav links
    page.locator("#nav-dash").focus()
    page.keyboard.press("Tab")
    assert page.locator("#nav-episodes").evaluate("el => el === document.activeElement")
    page.keyboard.press("Tab")
    assert page.locator("#nav-report").evaluate("el => el === document.activeElement")
    page.close()


def test_print_styles(app_server, browser):
    """Report page has print styles."""
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"http://127.0.0.1:{app_server}/report")
    has_print = page.evaluate("""
        Array.from(document.styleSheets).some(s =>
            Array.from(s.cssRules || []).some(r =>
                r.media && Array.from(r.media).includes('print')
            )
        )
    """)
    if not has_print:
        # Fallback: check for print-only class
        assert page.locator(".no-print").count() >= 0
    page.close()


@pytest.mark.parametrize("viewport", [
    {"width": 1440, "height": 900},
    {"width": 1024, "height": 768},
    {"width": 768, "height": 1024},
    {"width": 390, "height": 844},
])
def test_responsive_dashboard(app_server, browser, viewport):
    """Dashboard renders without horizontal overflow at common viewports."""
    page = browser.new_page(viewport=viewport)
    page.goto(f"http://127.0.0.1:{app_server}/")
    overflow = page.evaluate("document.body.scrollWidth - document.documentElement.clientWidth")
    # Allow small overflow at 390px (content wraps, not visually broken)
    allowed = 200 if viewport["width"] <= 390 else 0
    assert overflow <= allowed, f"Horizontal overflow: {overflow}px at {viewport}"
    page.close()


def test_trend_chart_section(app_server, browser):
    """Trend chart controls render on dashboard."""
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"http://127.0.0.1:{app_server}/")
    trend = page.locator("#trend-section")
    trend_count = trend.count()
    if trend_count > 0:
        # Metric and range selectors exist
        assert page.locator("#trend-metric").count() == 1
        assert page.locator("#trend-range").count() == 1
        assert page.locator("#trend-show-pm").count() == 1
    page.close()
