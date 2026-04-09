#!/usr/bin/env python3
"""Tests for Telegram voice-note extraction and transcription preprocessing."""

import asyncio
import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

from integrations import (
    _build_telegram_user_content,
    _extract_telegram_attachment_urls,
    _transcribe_audio_bytes_with_whisper,
)


async def test_extract_telegram_attachment_urls_includes_voice_audio_and_images() -> None:
    """Telegram attachment extraction should include image and audio payloads."""
    message = SimpleNamespace(
        photo=[SimpleNamespace(file_id="photo-small"), SimpleNamespace(file_id="photo-large")],
        voice=SimpleNamespace(file_id="voice-note"),
        audio=SimpleNamespace(file_id="audio-track"),
        document=SimpleNamespace(file_id="audio-doc", mime_type="audio/ogg"),
    )

    async def _fake_file_url(_bot: Any, file_id: str) -> str | None:
        return f"https://cdn.telegram.test/{file_id}"

    with patch("integrations._telegram_file_url", new=_fake_file_url):
        urls = await _extract_telegram_attachment_urls(message, bot=cast(Any, object()))

    assert urls == [
        "https://cdn.telegram.test/photo-large",
        "https://cdn.telegram.test/voice-note",
        "https://cdn.telegram.test/audio-track",
        "https://cdn.telegram.test/audio-doc",
    ]


async def test_build_telegram_user_content_transcribes_voice_notes_and_keeps_images() -> None:
    """Telegram voice notes should be transcribed before multimodal image handling."""
    previous_enabled = os.environ.get("TELEGRAM_VOICE_MEMO_TRANSCRIPTION_ENABLED")
    os.environ["TELEGRAM_VOICE_MEMO_TRANSCRIPTION_ENABLED"] = "1"

    transcribe_mock = AsyncMock(return_value="telegram transcript")
    build_content_mock = AsyncMock(return_value="compiled telegram content")
    try:
        with (
            patch(
                "integrations._transcribe_voice_memo_attachment_url",
                new=transcribe_mock,
            ),
            patch(
                "integrations._build_user_message_content_async",
                new=build_content_mock,
            ),
        ):
            result = await _build_telegram_user_content(
                "Please summarize this voice note",
                [
                    "https://cdn.telegram.test/voice-note.ogg",
                    "https://cdn.telegram.test/photo.png",
                ],
            )
    finally:
        if previous_enabled is None:
            os.environ.pop("TELEGRAM_VOICE_MEMO_TRANSCRIPTION_ENABLED", None)
        else:
            os.environ["TELEGRAM_VOICE_MEMO_TRANSCRIPTION_ENABLED"] = previous_enabled

    assert result == "compiled telegram content"
    assert transcribe_mock.await_count == 1
    assert transcribe_mock.await_args.kwargs.get("config_prefix") == "TELEGRAM"
    assert build_content_mock.await_count == 1

    build_args = build_content_mock.await_args.args
    assert "Please summarize this voice note" in build_args[0]
    assert "[Voice memo transcript]" in build_args[0]
    assert "telegram transcript" in build_args[0]
    assert build_args[1] == ["https://cdn.telegram.test/photo.png"]


async def test_transcribe_audio_bytes_uses_telegram_voice_memo_overrides() -> None:
    """Telegram transcription should honor TELEGRAM_* voice-memo overrides."""
    previous_api_key = os.environ.get("NVIDIA_API_KEY")
    previous_function_id = os.environ.get("TELEGRAM_VOICE_MEMO_FUNCTION_ID")
    previous_language = os.environ.get("TELEGRAM_VOICE_MEMO_LANGUAGE")
    os.environ["NVIDIA_API_KEY"] = "test-key"
    os.environ["TELEGRAM_VOICE_MEMO_FUNCTION_ID"] = "telegram-function"
    os.environ["TELEGRAM_VOICE_MEMO_LANGUAGE"] = "es-US"

    transcribe_mock = Mock(return_value="transcript")
    try:
        with patch(
            "integrations._transcribe_audio_bytes_with_whisper_sync",
            new=transcribe_mock,
        ):
            transcript = await _transcribe_audio_bytes_with_whisper(
                cast(Any, None),
                b"audio-bytes",
                "voice.ogg",
                "audio/ogg",
                config_prefix="TELEGRAM",
            )
    finally:
        if previous_api_key is None:
            os.environ.pop("NVIDIA_API_KEY", None)
        else:
            os.environ["NVIDIA_API_KEY"] = previous_api_key

        if previous_function_id is None:
            os.environ.pop("TELEGRAM_VOICE_MEMO_FUNCTION_ID", None)
        else:
            os.environ["TELEGRAM_VOICE_MEMO_FUNCTION_ID"] = previous_function_id

        if previous_language is None:
            os.environ.pop("TELEGRAM_VOICE_MEMO_LANGUAGE", None)
        else:
            os.environ["TELEGRAM_VOICE_MEMO_LANGUAGE"] = previous_language

    assert transcript == "transcript"
    transcribe_mock.assert_called_once_with(
        b"audio-bytes",
        "test-key",
        "grpc.nvcf.nvidia.com:443",
        "telegram-function",
        "es-US",
        "",
    )


async def main() -> int:
    await test_extract_telegram_attachment_urls_includes_voice_audio_and_images()
    await test_build_telegram_user_content_transcribes_voice_notes_and_keeps_images()
    await test_transcribe_audio_bytes_uses_telegram_voice_memo_overrides()
    print("All Telegram voice memo tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
