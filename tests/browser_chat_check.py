"""Opt-in browser regression: python tests/browser_chat_check.py --output-dir <path>."""

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
        for width, height in ((1440, 1000), (390, 844)):
            page = browser.new_page(viewport={"width": width, "height": height})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            # Reproduce an unavailable lazy ChatInput module without depending on cache state.
            page.route("**/static/js/ChatInput.*.js", lambda route: route.abort("failed"))
            page.goto(args.url)
            page.locator(".st-key-assistant-launcher button:visible").click()
            dialog = page.get_by_role("dialog")
            field = dialog.get_by_role("textbox", name="输入问题", exact=True)
            expect(field).to_be_visible()
            expect(dialog.get_by_test_id("stException")).to_have_count(0)
            initial_dialog_box = dialog.bounding_box()
            initial_field_box = field.bounding_box()
            assert initial_dialog_box and initial_field_box
            assert initial_field_box["y"] > initial_dialog_box["y"] + initial_dialog_box["height"] * 0.65
            field.fill("   ")
            field.press("Enter")
            expect(field).to_have_value("")
            expect(dialog.get_by_test_id("stChatMessage")).to_have_count(0)
            for index, question in enumerate(("忘记打卡后几天内可以补卡？", "需要提交哪些材料？")):
                field.fill(question)
                if index == 0:
                    field.press("Enter")
                else:
                    dialog.locator('[data-testid="stFormSubmitButton"] button:visible').click()
                expect(dialog.get_by_test_id("stChatMessage")).to_have_count((index + 1) * 2, timeout=120000)
                expect(dialog.locator(".answer-panel")).to_have_count(index + 1, timeout=120000)
                expect(field).to_have_value("")
                expect(dialog.get_by_text("问答服务暂时不可用", exact=False)).to_have_count(0)
                if index == 1:
                    # The completed rerun must put the latest user turn just
                    # below the dialog header; earlier turns remain above it.
                    page.wait_for_timeout(1800)
                    newest_user = dialog.get_by_test_id("stChatMessage").nth(2)
                    newest_box = newest_user.bounding_box()
                    dialog_box = dialog.bounding_box()
                    assert newest_box and dialog_box
                    assert newest_box["y"] - dialog_box["y"] < 210, (newest_box, dialog_box)
            # Close and reopen to check that submitted messages persist.
            dialog.locator('button[aria-label="Close"]').click()
            page.locator(".st-key-assistant-launcher button:visible").click()
            expect(dialog.get_by_test_id("stChatMessage")).to_have_count(4)
            expect(field).to_be_visible()
            assert not errors, errors
            field.scroll_into_view_if_needed()
            box = field.bounding_box()
            assert box and box["x"] >= 0 and box["x"] + box["width"] <= width
            send = dialog.locator('[data-testid="stFormSubmitButton"] button:visible')
            send_box = send.bounding_box()
            assert send_box and abs(send_box["y"] - box["y"]) < 8
            assert send_box["x"] >= box["x"] + box["width"]
            assert "Material" in send.get_by_test_id("stIconMaterial").evaluate("el => getComputedStyle(el).fontFamily")
            page.screenshot(path=str(args.output_dir / f"chat-{width}.png"))
            page.close()
        browser.close()
    print("PASS: blocked ChatInput module, empty input, Enter/button send, follow-up answers, reopen history, desktop/mobile")


if __name__ == "__main__":
    main()
