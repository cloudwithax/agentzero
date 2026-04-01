#!/usr/bin/env python3
"""Tests for Sendblue voice memo transcription preprocessing."""

import asyncio
import os
from unittest.mock import AsyncMock, patch

from integrations import (
    _append_voice_memo_transcripts,
    _split_voice_memo_attachments,
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


async def main() -> int:
    test_split_voice_memo_attachments_detects_audio_urls()
    test_append_voice_memo_transcripts_with_failures()
    await test_transcribe_sendblue_voice_memos_merges_and_filters()
    await test_transcribe_sendblue_voice_memos_respects_disable_flag()
    print("All Sendblue voice memo tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
