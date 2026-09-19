import pytest

from app.models import OpenAICompatibleChatClient


@pytest.mark.parametrize("content", [None, "", " \n "])
def test_empty_chat_response_is_not_reported_as_success(monkeypatch, content):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr("app.models.requests.post", lambda *args, **kwargs: Response())
    client = OpenAICompatibleChatClient("https://example.invalid", "", "test")
    with pytest.raises(RuntimeError, match="有效回答"):
        client.generate("规则", "问题")
