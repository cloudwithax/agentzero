#!/usr/bin/env python3
"""Tests for multimodal attachment handling in channel integrations."""

import asyncio
import base64
import json
import os
import struct
import zlib
from unittest.mock import patch, AsyncMock

from integrations import (
    _attachment_url_to_base64_data_url,
    _build_user_message_content,
    _build_user_message_content_async,
    _build_user_message_content_from_normalized,
    _extract_agent_response_payload,
    NVCF_MODEL_IDS,
)


def test_build_user_message_content_returns_text_with_attachments() -> None:
    """Build user message content returns text (images are described by describer model)."""
    with (
        patch("integrations._model_needs_nvcf_assets", return_value=False),
        patch("integrations._model_supports_multimodal", return_value=True),
    ):
        content = _build_user_message_content(
            "Please describe these.",
            ["https://img.example/a.png", "https://img.example/b.png"],
        )

    assert isinstance(content, str)
    assert "Please describe these." in content
    assert "https://img.example/a.png" in content
    assert "https://img.example/b.png" in content


def test_non_multimodal_fallback_text() -> None:
    """Fallback to plain text attachment list on non-multimodal models."""
    with (
        patch("integrations._model_needs_nvcf_assets", return_value=False),
        patch("integrations._model_supports_multimodal", return_value=False),
    ):
        content = _build_user_message_content(
            "Summarize these images.",
            ["https://img.example/only.png"],
        )

    assert isinstance(content, str)
    assert "Summarize these images." in content
    assert "https://img.example/only.png" in content


def test_build_user_message_content_respect_attachment_limit() -> None:
    """Cap inbound image attachments to configured MAX_IMAGE_ATTACHMENTS_PER_MESSAGE."""
    with (
        patch("integrations._model_needs_nvcf_assets", return_value=False),
        patch("integrations._model_supports_multimodal", return_value=True),
        patch.dict(os.environ, {"MAX_IMAGE_ATTACHMENTS_PER_MESSAGE": "2"}),
    ):
        content = _build_user_message_content(
            "Please describe all images.",
            [
                "https://img.example/a.png",
                "https://img.example/b.png",
                "https://img.example/c.png",
            ],
        )

    assert isinstance(content, str)
    assert "included 2 of 3 images" in content
    assert "https://img.example/a.png" in content
    assert "https://img.example/b.png" in content
    assert "https://img.example/c.png" not in content


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


def test_attachment_url_to_base64_data_url_preserves_original() -> None:
    """Remote image URLs are base64-encoded unaltered (no ImageMagick compression)."""
    session = _FakeSession(
        _FakeResponse(
            status=200,
            body=b"fake-image-bytes",
            headers={"Content-Type": "image/png"},
        )
    )

    result = asyncio.run(
        _attachment_url_to_base64_data_url(session, "https://img.example/one.png")  # type: ignore[arg-type]
    )

    assert result == (
        "data:image/png;base64," + base64.b64encode(b"fake-image-bytes").decode("ascii")
    )
    assert session.requested_urls == ["https://img.example/one.png"]


def test_attachment_data_url_to_base64_data_url_preserves_original() -> None:
    """Image data URLs are re-encoded unaltered with their original content type."""

    class _NoNetworkSession:
        def get(self, _url: str):
            raise AssertionError("Network fetch should not run for data URLs")

    encoded_png = base64.b64encode(b"fake-png").decode("ascii")
    source_data_url = f"data:image/png;base64,{encoded_png}"

    result = asyncio.run(
        _attachment_url_to_base64_data_url(_NoNetworkSession(), source_data_url)  # type: ignore[arg-type]
    )

    assert result == (
        "data:image/png;base64," + base64.b64encode(b"fake-png").decode("ascii")
    )


def test_build_user_message_content_async_describes_images_non_multimodal() -> None:
    """Non-multimodal models describe images via the Mistral describer model."""

    async def _fake_download(session, url):
        if url.endswith("a.png"):
            return b"fake-image-bytes-a", "image/png"
        elif url.endswith("b.png"):
            return b"fake-image-bytes-b", "image/png"
        return None, None

    fake_description = "A beautiful sunset over the ocean"

    async def _fake_describe_images(session, image_data):
        return [fake_description] * len(image_data)

    with (
        patch("integrations._model_needs_nvcf_assets", return_value=False),
        patch("integrations._model_supports_multimodal", return_value=False),
        patch("integrations._download_image_bytes", side_effect=_fake_download),
        patch(
            "integrations._describe_images_with_mistral",
            side_effect=_fake_describe_images,
        ),
    ):
        content = asyncio.run(
            _build_user_message_content_async(
                "Please analyze these images.",
                ["https://img.example/a.png", "https://img.example/b.png"],
            )
        )

    assert isinstance(content, str)
    assert "Please analyze these images." in content
    assert "[Image 1 description]" in content
    assert "[Image 2 description]" in content
    assert fake_description in content


def test_build_user_message_content_async_handles_failed_images() -> None:
    """Non-multimodal describer path should handle failed image downloads gracefully."""

    async def _fake_download(session, url):
        if url.endswith("a.png"):
            return b"fake-image-bytes-a", "image/png"
        return None, None

    fake_description = "A beautiful sunset"

    async def _fake_describe_images(session, image_data):
        return [fake_description]

    with (
        patch("integrations._model_needs_nvcf_assets", return_value=False),
        patch("integrations._model_supports_multimodal", return_value=False),
        patch("integrations._download_image_bytes", side_effect=_fake_download),
        patch(
            "integrations._describe_images_with_mistral",
            side_effect=_fake_describe_images,
        ),
    ):
        content = asyncio.run(
            _build_user_message_content_async(
                "Please analyze.",
                ["https://img.example/a.png", "https://img.example/b.png"],
            )
        )

    assert isinstance(content, str)
    assert "Please analyze." in content
    assert "1 image could not be processed" in content


def test_build_user_message_content_async_multimodal_returns_image_blocks() -> None:
    """Native multimodal models receive image_url blocks, not Mistral descriptions."""

    async def _fake_data_url(session, url):
        return f"data:image/jpeg;base64,FAKE-{url[-5:]}"

    async def _should_not_run(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("describer must not run for multimodal models")

    with (
        patch("integrations._model_needs_nvcf_assets", return_value=False),
        patch("integrations._model_supports_multimodal", return_value=True),
        patch(
            "integrations._attachment_url_to_base64_data_url",
            side_effect=_fake_data_url,
        ),
        patch(
            "integrations._describe_images_with_mistral",
            side_effect=_should_not_run,
        ),
    ):
        content = asyncio.run(
            _build_user_message_content_async(
                "What do you see?",
                ["https://img.example/a.png", "https://img.example/b.png"],
            )
        )

    assert isinstance(content, list), f"Expected multimodal block list, got {type(content)}"
    text_blocks = [b for b in content if b.get("type") == "text"]
    image_blocks = [b for b in content if b.get("type") == "image_url"]
    assert any("What do you see?" in b["text"] for b in text_blocks)
    assert len(image_blocks) == 2
    assert all(
        b["image_url"]["url"].startswith("data:image/jpeg;base64,") for b in image_blocks
    )


def test_build_user_message_content_async_no_attachments() -> None:
    """Async builder should return text-only when no attachments."""
    content = asyncio.run(
        _build_user_message_content_async(
            "Hello, how are you?",
            [],
        )
    )

    assert isinstance(content, str)
    assert content == "Hello, how are you?"


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
        assert "asset_id,abc123" in content
    finally:
        integrations.NVCF_MODEL_IDS = saved


def test_build_user_message_content_from_normalized_multimodal_returns_text() -> None:
    """Multimodal models now return text only (images are described by describer)."""
    with (
        patch("integrations._model_needs_nvcf_assets", return_value=False),
        patch("integrations._model_supports_multimodal", return_value=True),
    ):
        content = _build_user_message_content_from_normalized(
            "Please describe these.",
            ["https://img.example/a.png", "https://img.example/b.png"],
        )
    assert isinstance(content, str)
    assert "Please describe these." in content
    assert "https://img.example/a.png" in content


def test_nvcf_model_ids_empty_by_default() -> None:
    """NVCF_MODEL_IDS should be empty so no models hit the broken NVCF upload path."""
    assert NVCF_MODEL_IDS == set(), (
        "NVCF_MODEL_IDS must be empty until a model genuinely requires NVCF asset refs. "
        "Otherwise multimodal models fall back to base64 data URLs embedded as plain text."
    )


def test_async_builder_falls_back_to_text_when_conversion_fails() -> None:
    """Multimodal path degrades to text when every image conversion fails."""

    async def _fail_data_url(session, url):
        return None

    with (
        patch("integrations._model_needs_nvcf_assets", return_value=False),
        patch("integrations._model_supports_multimodal", return_value=True),
        patch(
            "integrations._attachment_url_to_base64_data_url",
            side_effect=_fail_data_url,
        ),
    ):
        content = asyncio.run(
            _build_user_message_content_async(
                "What do you see?",
                ["https://img.example/photo.jpg"],
            )
        )

    assert isinstance(content, str)
    assert "What do you see?" in content
    assert "could not be processed" in content


def test_resize_image_to_jpeg_sync_converts_png_to_jpeg() -> None:
    """_resize_image_to_jpeg_sync converts any image format to resized JPEG."""
    from integrations import _resize_image_to_jpeg_sync

    def _make_minimal_png(width: int, height: int) -> bytes:
        def _chunk(chunk_type: bytes, data: bytes) -> bytes:
            c = chunk_type + data
            return (
                struct.pack(">I", len(data))
                + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            )

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        raw = b""
        for y in range(height):
            raw += b"\x00" + b"\xff\x00\x00" * width
        idat = _chunk(b"IDAT", zlib.compress(raw))
        iend = _chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    png_bytes = _make_minimal_png(200, 100)
    result = _resize_image_to_jpeg_sync(png_bytes)
    assert result is not None
    assert result[:3] == b"\xff\xd8\xff"


def main() -> int:
    test_build_user_message_content_returns_text_with_attachments()
    test_non_multimodal_fallback_text()
    test_build_user_message_content_respect_attachment_limit()
    test_extract_structured_response_payload()
    test_attachment_url_to_base64_data_url_preserves_original()
    test_attachment_data_url_to_base64_data_url_preserves_original()
    test_build_user_message_content_async_describes_images_non_multimodal()
    test_build_user_message_content_async_handles_failed_images()
    test_build_user_message_content_async_multimodal_returns_image_blocks()
    test_build_user_message_content_async_no_attachments()
    test_build_user_message_content_from_normalized_nvcf_returns_text()
    test_build_user_message_content_from_normalized_multimodal_returns_text()
    test_nvcf_model_ids_empty_by_default()
    test_async_builder_falls_back_to_text_when_conversion_fails()
    test_resize_image_to_jpeg_sync_converts_png_to_jpeg()
    print("All multimodal integration tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
