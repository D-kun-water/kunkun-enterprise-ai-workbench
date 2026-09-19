"""Opt-in browser regression: python tests/browser_identity_check.py --output-dir <path>."""

import argparse
from pathlib import Path


def main() -> None:
    from playwright.sync_api import expect, sync_playwright

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(args.url)
        card_container = page.locator(".st-key-sidebar-user-card")
        card = page.get_by_test_id("stPopoverButton")
        expect(card).to_be_enabled()
        # Computed browser styles catch selectors that exist in CSS but match no DOM.
        style = card_container.evaluate("""el => ({
            avatar: el.querySelector('.sidebar-avatar')?.textContent.trim(),
            avatarWidth: getComputedStyle(el.querySelector('.sidebar-avatar')).width,
            border: getComputedStyle(el.querySelector('.sidebar-user-card-content')).borderTopWidth,
            triggerOpacity: getComputedStyle(el.querySelector('[data-testid="stPopoverButton"]')).opacity
        })""")
        assert style == {"avatar": "林", "avatarWidth": "42px", "border": "1px", "triggerOpacity": "0"}, style
        page.locator("label").filter(has=page.get_by_role("radio", name="个人中心", exact=True)).click()
        for name, initial, todo, privileged in (
            ("苏闻", "苏", "复核供应商合同续约条款", True),
            ("陈知远", "陈", "审批季度经营复盘材料", True),
            ("周予安", "周", "确认客户合同付款节点", True),
            ("林知夏", "林", "补充差旅报销材料", True),
        ):
            card.click()
            menu = page.get_by_test_id("stPopoverBody")
            expect(menu).to_be_visible()
            expect(menu.get_by_role("radio").first).to_be_enabled()
            assert menu.inner_text().split() == ["林知夏", "周予安", "苏闻", "陈知远"]
            menu.screenshot(path=str(args.output_dir / "identity-menu.png"))
            page.locator("label").filter(has=page.get_by_role("radio", name=name, exact=True)).click()
            expect(card_container.locator(".sidebar-user strong")).to_have_text(name)
            expect(menu).not_to_be_visible()
            expect(page.get_by_role("heading", name=name, exact=True)).to_be_visible()
            expect(page.get_by_text(todo, exact=True)).to_be_visible()
            entry = page.get_by_role("radiogroup", name="员工工作台", exact=True).get_by_role("radio", name="合同评审", exact=True)
            assert entry.count() == int(privileged)
            assert page.get_by_role("radiogroup", name="知识库运营", exact=True).get_by_role("radio", name="合同评审", exact=True).count() == 0
            assert card_container.locator(".sidebar-avatar").inner_text().strip() == initial
            if name == "苏闻":
                card_container.screenshot(path=str(args.output_dir / "identity-legal.png"))
                page.screenshot(path=str(args.output_dir / "identity-desktop.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        expect(page.get_by_test_id("stSidebar")).to_have_attribute("aria-expanded", "false")
        page.get_by_test_id("stSidebarCollapseButton").get_by_role("button").click()
        expect(page.get_by_test_id("stSidebar")).to_have_attribute("aria-expanded", "true")
        expect(card_container).to_be_visible()
        card.click()
        box = card_container.bounding_box()
        assert box and box["x"] >= 0 and box["x"] + box["width"] <= 390
        assert box["y"] >= 0 and box["y"] + box["height"] <= 844
        page.locator("label").filter(has=page.get_by_role("radio", name="苏闻", exact=True)).click()
        expect(card_container.locator(".sidebar-user strong")).to_have_text("苏闻")
        contract = page.get_by_role("radiogroup", name="员工工作台", exact=True).get_by_role("radio", name="合同评审", exact=True)
        page.locator("label").filter(has=page.get_by_role("radio", name="合同评审", exact=True)).click()
        expect(page.get_by_role("heading", name="合同评审", exact=True)).to_be_visible()
        page.screenshot(path=str(args.output_dir / "identity-mobile.png"))
        card.click()
        page.locator("label").filter(has=page.get_by_role("radio", name="林知夏", exact=True)).click()
        expect(contract).to_have_count(1)
        expect(page.get_by_role("heading", name="合同评审", exact=True)).to_be_visible()
        expect(page.get_by_text("当前角色没有可审阅的合同。", exact=True)).to_be_visible()
        browser.close()
    print("PASS: rendered avatar, card styles, name-only menu, four identities, profile/todos, contract entry, mobile switching")


if __name__ == "__main__":
    main()
