#!/usr/bin/env python3
"""Tests for multimodal attachment handling in channel integrations."""

import asyncio
import base64
import json
import os
from unittest.mock import patch

from integrations import (
    _attachment_url_to_base64_data_url,
    _build_user_message_content,
    _build_user_message_content_async,
    _build_user_message_content_from_normalized,
    _extract_agent_response_payload,
    NVCF_MODEL_IDS,
)


def test_multimodal_message_blocks() -> None:
    """Build image_url blocks when a multimodal model is selected."""
    with patch("integrations._model_needs_nvcf_assets", return_value=False), \
         patch("integrations._model_supports_multimodal", return_value=True):
        content = _build_user_message_content(
            "Please describe these.",
            ["https://img.example/a.png", "https://img.example/b.png"],
        )

    assert isinstance(content, list)
    assert content[0] == {
        "type": "text",
        "text": (
            "IMPORTANT: You can view and analyze the attached images in this "
            "message. Do not claim you cannot view images.\n\n"
            "User message: Please describe these."
        ),
    }
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://img.example/a.png"},
    }
    assert content[2] == {
        "type": "image_url",
        "image_url": {"url": "https://img.example/b.png"},
    }


def test_non_multimodal_fallback_text() -> None:
    """Fallback to plain text attachment list on non-multimodal models."""
    with patch("integrations._model_needs_nvcf_assets", return_value=False), \
         patch("integrations._model_supports_multimodal", return_value=False):
        content = _build_user_message_content(
            "Summarize these images.",
            ["https://img.example/only.png"],
        )

    assert isinstance(content, str)
    assert "Summarize these images." in content
    assert "[Image attachments]" in content
    assert "https://img.example/only.png" in content


def test_multimodal_message_blocks_respect_attachment_limit() -> None:
    """Cap inbound image blocks to configured MAX_IMAGE_ATTACHMENTS_PER_MESSAGE."""
    with patch("integrations._model_needs_nvcf_assets", return_value=False), \
         patch("integrations._model_supports_multimodal", return_value=True), \
         patch.dict(os.environ, {"MAX_IMAGE_ATTACHMENTS_PER_MESSAGE": "2"}):
        content = _build_user_message_content(
            "Please describe all images.",
            [
                "https://img.example/a.png",
                "https://img.example/b.png",
                "https://img.example/c.png",
            ],
        )

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "included 2 of 3 images" in content[0]["text"]
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://img.example/a.png"},
    }
    assert content[2] == {
        "type": "image_url",
        "image_url": {"url": "https://img.example/b.png"},
    }
    assert len(content) == 3


def test_extract_structured_response_payload() -> None:
    """Parse text and multiple attachments from JSON assistant output."""
    raw = json.dumps(
        {
            "text": "Here you go",
            "attachments": ["https://img.example/1.png", "https://img.example/2.png"],
        }
    )

    text, attachments = _extract_agent_response_payload(raw)

    assert text == "Here you go"
    assert attachments == [
        "https://img.example/1.png",
        "https://img.example/2.png",
    ]


class _FakeResponse:
    def __init__(self, *, status: int, body: bytes = b"", headers: dict | None = None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def read(self) -> bytes:
        return self._body


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.requested_urls: list[str] = []

    def get(self, url: str) -> _FakeResponse:
        self.requested_urls.append(url)
        return self.response


def test_attachment_url_to_base64_data_url_converts_to_jpeg() -> None:
    """Always convert remote image URLs to JPEG base64 data URLs."""
    session = _FakeSession(
        _FakeResponse(
            status=200,
            body=b"fake-image-bytes",
            headers={"Content-Type": "image/png"},
        )
    )

    with patch(
        "integrations._convert_image_with_imagemagick_sync",
        return_value=b"jpeg-converted-bytes",
    ) as convert_mock:
        result = asyncio.run(
            _attachment_url_to_base64_data_url(session, "https://img.example/one.png")
        )

    assert result == (
        "data:image/jpeg;base64,"
        + base64.b64encode(b"jpeg-converted-bytes").decode("ascii")
    )
    assert session.requested_urls == ["https://img.example/one.png"]
    assert convert_mock.call_count == 1
    args = convert_mock.call_args.args
    assert args[0] == b"fake-image-bytes"
    assert args[1] == ".png"
    assert args[2] == ".jpg"


def test_attachment_data_url_to_base64_data_url_converts_to_jpeg() -> None:
    """Always re-encode image data URLs as JPEG base64 data URLs."""
    class _NoNetworkSession:
        def get(self, _url: str):
            raise AssertionError("Network fetch should not run for data URLs")

    encoded_png = base64.b64encode(b"fake-png").decode("ascii")
    source_data_url = f"data:image/png;base64,{encoded_png}"

    with patch(
        "integrations._convert_image_with_imagemagick_sync",
        return_value=b"jpeg-from-data-url",
    ) as convert_mock:
        result = asyncio.run(
            _attachment_url_to_base64_data_url(_NoNetworkSession(), source_data_url)
        )

    assert result == (
        "data:image/jpeg;base64,"
        + base64.b64encode(b"jpeg-from-data-url").decode("ascii")
    )
    assert convert_mock.call_count == 1
    args = convert_mock.call_args.args
    assert args[0] == b"fake-png"
    assert args[1] == ".png"
    assert args[2] == ".jpg"


def test_build_user_message_content_async_uses_base64_and_drops_failed() -> None:
    """Async builder should keep only successful base64 JPEG conversions."""
    async def _fake_convert(_session, url: str) -> str | None:
        if url.endswith("a.png"):
            return "data:image/jpeg;base64,AAAA"
        return None

    with patch("integrations._model_needs_nvcf_assets", return_value=False), \
         patch("integrations._model_supports_multimodal", return_value=True), \
         patch("integrations._attachment_url_to_base64_data_url", side_effect=_fake_convert):
        content = asyncio.run(
            _build_user_message_content_async(
                "Please analyze these images.",
                ["https://img.example/a.png", "https://img.example/b.png"],
            )
        )

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "Image conversion warning: dropped 1 image" in content[0]["text"]
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,AAAA"},
    }
    assert len(content) == 2


def test_build_user_message_content_from_normalized_nvcf_returns_text() -> None:
    """NVCF models inline asset refs in plain text instead of image_url blocks."""
    import integrations
    saved = integrations.NVCF_MODEL_IDS.copy()
    integrations.NVCF_MODEL_IDS = {"test/nvcf-model"}
    try:
        with patch("integrations._model_needs_nvcf_assets", return_value=True):
            content = _build_user_message_content_from_normalized(
                "What is this?",
                ['<img src="data:image/jpeg;asset_id,abc123" />'],
            )
        assert isinstance(content, str)
        assert "What is this?" in content
        assert 'asset_id,abc123' in content
    finally:
        integrations.NVCF_MODEL_IDS = saved


def test_build_user_message_content_from_normalized_multimodal_returns_list() -> None:
    """Non-NVCF multimodal models return image_url block lists."""
    with patch("integrations._model_needs_nvcf_assets", return_value=False), \
         patch("integrations._model_supports_multimodal", return_value=True):
        content = _build_user_message_content_from_normalized(
            "Please describe these.",
            ["https://img.example/a.png", "https://img.example/b.png"],
        )
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://img.example/a.png"},
    }


def main() -> int:
    test_multimodal_message_blocks()
    test_non_multimodal_fallback_text()
    test_multimodal_message_blocks_respect_attachment_limit()
    test_extract_structured_response_payload()
    test_attachment_url_to_base64_data_url_converts_to_jpeg()
    test_attachment_data_url_to_base64_data_url_converts_to_jpeg()
    test_build_user_message_content_async_uses_base64_and_drops_failed()
    test_build_user_message_content_from_normalized_nvcf_returns_text()
    test_build_user_message_content_from_normalized_multimodal_returns_list()
    print("All multimodal integration tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
