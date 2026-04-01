#!/usr/bin/env python3
"""Tests for Sendblue voice memo transcription preprocessing."""

import asyncio
import os
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

from integrations import (
    _append_voice_memo_transcripts,
    _is_native_imessage_m4a,
    _split_voice_memo_attachments,
    _transcribe_audio_bytes_with_whisper,
    _transcribe_sendblue_voice_memos,
)


def test_split_voice_memo_attachments_detects_audio_urls() -> None:
    """Classify typical iMessage voice memo URLs as audio attachments."""
    voice_urls, passthrough_urls = _split_voice_memo_attachments(
        [
            "https://cdn.example/voice-1.opus",
            "https://cdn.example/photo.png",
            "https://cdn.example/voice-2.m4a?token=abc",
        ]
    )

    assert voice_urls == [
        "https://cdn.example/voice-1.opus",
        "https://cdn.example/voice-2.m4a?token=abc",
    ]
    assert passthrough_urls == ["https://cdn.example/photo.png"]


def test_append_voice_memo_transcripts_with_failures() -> None:
    """Append transcript blocks and failed attachment references to user text."""
    merged = _append_voice_memo_transcripts(
        "Please summarize this.",
        ["hello from memo one", "hello from memo two"],
        ["https://cdn.example/voice-3.opus"],
    )

    assert "Please summarize this." in merged
    assert "[Voice memo transcripts]" in merged
    assert "1. hello from memo one" in merged
    assert "2. hello from memo two" in merged
    assert "[Voice memo attachments not transcribed]" in merged
    assert "https://cdn.example/voice-3.opus" in merged


def test_is_native_imessage_m4a_detects_common_signatures() -> None:
    """Treat iMessage m4a/caf signals as conversion candidates."""
    assert _is_native_imessage_m4a("voice.m4a", None)
    assert _is_native_imessage_m4a("voice.caf", None)
    assert _is_native_imessage_m4a("voice", "audio/x-m4a")
    assert _is_native_imessage_m4a("voice", "audio/x-caf")
    assert _is_native_imessage_m4a("voice.mp4", "audio/mp4")
    assert not _is_native_imessage_m4a("voice.opus", "audio/ogg")


async def test_transcribe_sendblue_voice_memos_merges_and_filters() -> None:
    """Transcribe audio URLs and remove them from downstream image attachments."""
    previous_enabled = os.environ.get("SENDBLUE_VOICE_MEMO_TRANSCRIPTION_ENABLED")
    os.environ["SENDBLUE_VOICE_MEMO_TRANSCRIPTION_ENABLED"] = "1"

    mock_transcribe = AsyncMock(side_effect=["transcript text", None])
    try:
        with patch(
            "integrations._transcribe_voice_memo_attachment_url", new=mock_transcribe
        ):
            merged_text, passthrough_urls = await _transcribe_sendblue_voice_memos(
                "Original message",
                [
                    "https://cdn.example/voice-1.opus",
                    "https://cdn.example/image-1.png",
                    "https://cdn.example/voice-2.m4a",
                ],
            )
    finally:
        if previous_enabled is None:
            os.environ.pop("SENDBLUE_VOICE_MEMO_TRANSCRIPTION_ENABLED", None)
        else:
            os.environ["SENDBLUE_VOICE_MEMO_TRANSCRIPTION_ENABLED"] = previous_enabled

    assert mock_transcribe.await_count == 2
    assert "Original message" in merged_text
    assert "[Voice memo transcript]" in merged_text
    assert "transcript text" in merged_text
    assert "[Voice memo attachments not transcribed]" in merged_text
    assert "https://cdn.example/voice-2.m4a" in merged_text
    assert passthrough_urls == ["https://cdn.example/image-1.png"]


async def test_transcribe_sendblue_voice_memos_respects_disable_flag() -> None:
    """Skip transcription pipeline when disabled by env."""
    previous_enabled = os.environ.get("SENDBLUE_VOICE_MEMO_TRANSCRIPTION_ENABLED")
    os.environ["SENDBLUE_VOICE_MEMO_TRANSCRIPTION_ENABLED"] = "0"

    mock_transcribe = AsyncMock(return_value="should-not-run")
    try:
        with patch(
            "integrations._transcribe_voice_memo_attachment_url", new=mock_transcribe
        ):
            merged_text, passthrough_urls = await _transcribe_sendblue_voice_memos(
                "",
                ["https://cdn.example/voice-1.opus", "https://cdn.example/photo.png"],
            )
    finally:
        if previous_enabled is None:
            os.environ.pop("SENDBLUE_VOICE_MEMO_TRANSCRIPTION_ENABLED", None)
        else:
            os.environ["SENDBLUE_VOICE_MEMO_TRANSCRIPTION_ENABLED"] = previous_enabled

    assert mock_transcribe.await_count == 0
    assert merged_text == ""
    assert passthrough_urls == [
        "https://cdn.example/voice-1.opus",
        "https://cdn.example/photo.png",
    ]


async def test_transcribe_audio_bytes_retries_m4a_with_ffmpeg() -> None:
    """Retry m4a voice memos with ffmpeg-converted bytes when first pass fails."""
    previous_api_key = os.environ.get("NVIDIA_API_KEY")
    os.environ["NVIDIA_API_KEY"] = "test-key"

    transcribe_mock = Mock(side_effect=[None, "converted transcript"])
    convert_mock = Mock(
        return_value=(b"wav-bytes", "voice-memo-converted.wav", "audio/wav")
    )

    try:
        with patch(
            "integrations._transcribe_audio_bytes_with_whisper_sync",
            new=transcribe_mock,
        ), patch(
            "integrations._convert_m4a_audio_with_ffmpeg_sync",
            new=convert_mock,
        ):
            transcript = await _transcribe_audio_bytes_with_whisper(
                cast(Any, None),
                b"m4a-bytes",
                "voice.m4a",
                "audio/mp4",
            )
    finally:
        if previous_api_key is None:
            os.environ.pop("NVIDIA_API_KEY", None)
        else:
            os.environ["NVIDIA_API_KEY"] = previous_api_key

    assert transcript == "converted transcript"
    assert transcribe_mock.call_count == 2
    assert convert_mock.call_count == 1


async def test_transcribe_audio_bytes_retries_caf_with_ffmpeg() -> None:
    """Retry caf voice memos with ffmpeg-converted bytes when first pass fails."""
    previous_api_key = os.environ.get("NVIDIA_API_KEY")
    os.environ["NVIDIA_API_KEY"] = "test-key"

    transcribe_mock = Mock(side_effect=[None, "caf converted transcript"])
    convert_mock = Mock(
        return_value=(b"wav-bytes", "voice-memo-converted.wav", "audio/wav")
    )

    try:
        with patch(
            "integrations._transcribe_audio_bytes_with_whisper_sync",
            new=transcribe_mock,
        ), patch(
            "integrations._convert_m4a_audio_with_ffmpeg_sync",
            new=convert_mock,
        ):
            transcript = await _transcribe_audio_bytes_with_whisper(
                cast(Any, None),
                b"caf-bytes",
                "voice.caf",
                "audio/x-caf",
            )
    finally:
        if previous_api_key is None:
            os.environ.pop("NVIDIA_API_KEY", None)
        else:
            os.environ["NVIDIA_API_KEY"] = previous_api_key

    assert transcript == "caf converted transcript"
    assert transcribe_mock.call_count == 2
    assert convert_mock.call_count == 1


async def main() -> int:
    test_split_voice_memo_attachments_detects_audio_urls()
    test_append_voice_memo_transcripts_with_failures()
    test_is_native_imessage_m4a_detects_common_signatures()
    await test_transcribe_sendblue_voice_memos_merges_and_filters()
    await test_transcribe_sendblue_voice_memos_respects_disable_flag()
    await test_transcribe_audio_bytes_retries_m4a_with_ffmpeg()
    await test_transcribe_audio_bytes_retries_caf_with_ffmpeg()
    print("All Sendblue voice memo tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
