#!/usr/bin/env python3
"""Tests for multimodal attachment handling in channel integrations."""

import json
import os

from integrations import (
    _build_user_message_content,
    _extract_agent_response_payload,
    _extract_nvcf_asset_fields,
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
    assert content[0] == {"type": "text", "text": "Please describe these."}
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


def test_multimodal_asset_data_url_block() -> None:
    """Keep prebuilt NVCF asset references intact in multimodal content blocks."""
    previous_model = os.environ.get("MODEL_ID")
    os.environ["MODEL_ID"] = "qwen/qwen3.5-397b-a17b"

    try:
        content = _build_user_message_content(
            "Analyze this image.",
            [
                "data:image/jpeg;asset_id,123e4567-e89b-12d3-a456-426614174000",
            ],
        )
    finally:
        if previous_model is None:
            os.environ.pop("MODEL_ID", None)
        else:
            os.environ["MODEL_ID"] = previous_model

    assert isinstance(content, list)
    assert content[1] == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;asset_id,123e4567-e89b-12d3-a456-426614174000"
        },
    }


def test_extract_nvcf_asset_fields_with_nested_payload() -> None:
    """Accept common create-asset response shapes with nested upload URL fields."""
    payload = {
        "assetId": "11111111-2222-3333-4444-555555555555",
        "uploadDetails": {
            "uploadUrl": "https://bucket.example/upload",
        },
    }

    asset_id, upload_url = _extract_nvcf_asset_fields(payload)

    assert asset_id == "11111111-2222-3333-4444-555555555555"
    assert upload_url == "https://bucket.example/upload"


def main() -> int:
    test_multimodal_message_blocks()
    test_non_multimodal_fallback_text()
    test_extract_structured_response_payload()
    test_multimodal_asset_data_url_block()
    test_extract_nvcf_asset_fields_with_nested_payload()
    print("All multimodal integration tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
