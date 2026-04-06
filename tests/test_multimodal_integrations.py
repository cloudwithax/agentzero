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
    _extract_agent_response_payload,
)


def test_multimodal_message_blocks() -> None:
    """Build image_url blocks when a multimodal model is selected."""
    previous_model = os.environ.get("MODEL_ID")
    os.environ["MODEL_ID"] = "meta/llama-3.2-11b-vision-instruct"

    try:
        content = _build_user_message_content(
            "Please describe these.",
            ["https://img.example/a.png", "https://img.example/b.png"],
        )
    finally:
        if previous_model is None:
            os.environ.pop("MODEL_ID", None)
        else:
            os.environ["MODEL_ID"] = previous_model

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
    previous_model = os.environ.get("MODEL_ID")
    os.environ["MODEL_ID"] = "moonshotai/kimi-k2-instruct-0905"

    try:
        content = _build_user_message_content(
            "Summarize these images.",
            ["https://img.example/only.png"],
        )
    finally:
        if previous_model is None:
            os.environ.pop("MODEL_ID", None)
        else:
            os.environ["MODEL_ID"] = previous_model

    assert isinstance(content, str)
    assert "Summarize these images." in content
    assert "[Image attachments]" in content
    assert "https://img.example/only.png" in content


def test_multimodal_message_blocks_respect_attachment_limit() -> None:
    """Cap inbound image blocks to configured MAX_IMAGE_ATTACHMENTS_PER_MESSAGE."""
    previous_model = os.environ.get("MODEL_ID")
    previous_limit = os.environ.get("MAX_IMAGE_ATTACHMENTS_PER_MESSAGE")
    os.environ["MODEL_ID"] = "meta/llama-3.2-11b-vision-instruct"
    os.environ["MAX_IMAGE_ATTACHMENTS_PER_MESSAGE"] = "2"

    try:
        content = _build_user_message_content(
            "Please describe all images.",
            [
                "https://img.example/a.png",
                "https://img.example/b.png",
                "https://img.example/c.png",
            ],
        )
    finally:
        if previous_model is None:
            os.environ.pop("MODEL_ID", None)
        else:
            os.environ["MODEL_ID"] = previous_model
        if previous_limit is None:
            os.environ.pop("MAX_IMAGE_ATTACHMENTS_PER_MESSAGE", None)
        else:
            os.environ["MAX_IMAGE_ATTACHMENTS_PER_MESSAGE"] = previous_limit

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
    previous_model = os.environ.get("MODEL_ID")
    os.environ["MODEL_ID"] = "meta/llama-3.2-11b-vision-instruct"

    async def _fake_convert(_session, url: str) -> str | None:
        if url.endswith("a.png"):
            return "data:image/jpeg;base64,AAAA"
        return None

    try:
        with patch(
            "integrations._attachment_url_to_base64_data_url",
            side_effect=_fake_convert,
        ):
            content = asyncio.run(
                _build_user_message_content_async(
                    "Please analyze these images.",
                    ["https://img.example/a.png", "https://img.example/b.png"],
                )
            )
    finally:
        if previous_model is None:
            os.environ.pop("MODEL_ID", None)
        else:
            os.environ["MODEL_ID"] = previous_model

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "Image conversion warning: dropped 1 image" in content[0]["text"]
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,AAAA"},
    }
    assert len(content) == 2


def main() -> int:
    test_multimodal_message_blocks()
    test_non_multimodal_fallback_text()
    test_multimodal_message_blocks_respect_attachment_limit()
    test_extract_structured_response_payload()
    test_attachment_url_to_base64_data_url_converts_to_jpeg()
    test_attachment_data_url_to_base64_data_url_converts_to_jpeg()
    test_build_user_message_content_async_uses_base64_and_drops_failed()
    print("All multimodal integration tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
