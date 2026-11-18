"""Chụp màn hình các tab của demo (cần demo đang chạy và Chrome đã cài). DEMO_PORT mặc định 8501."""
import os
from playwright.sync_api import sync_playwright

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "figures", "demo"); os.makedirs(OUT, exist_ok=True)
PORT = os.environ.get("DEMO_PORT", "8501")


def idle(page, ms=4000):
    page.wait_for_timeout(1500)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=300000)
    except Exception:
        pass
    page.wait_for_timeout(ms)


with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    page = b.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(f"http://localhost:{PORT}", wait_until="networkidle"); idle(page, 8000)
    page.screenshot(path=os.path.join(OUT, "tab1.png"))
    tabs = page.get_by_role("tab")
    tabs.nth(1).click(); idle(page, 2000)
    page.get_by_role("button", name="Chạy tất cả phương pháp").click(); idle(page, 8000)
    page.screenshot(path=os.path.join(OUT, "tab2.png"), full_page=True)
    for i, name in [(2, "tab3"), (3, "tab4"), (4, "tab5")]:
        tabs.nth(i).click(); idle(page, 6000)
        page.screenshot(path=os.path.join(OUT, f"{name}.png"))
    b.close()
print("ok")
