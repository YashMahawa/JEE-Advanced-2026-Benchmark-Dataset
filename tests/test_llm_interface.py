"""Tests for src/llm_interface.py — unit-testable functions only (no API calls)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import base64
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from llm_interface import encode_image_to_base64, construct_initial_prompt


def _make_test_image(mode="RGB"):
    return Image.new(mode, (10, 10), color="white")


def test_encode_returns_valid_base64():
    img = _make_test_image()
    result = encode_image_to_base64(img)
    # Should decode without error
    decoded = base64.b64decode(result)
    assert len(decoded) > 0


def test_encode_produces_png_not_jpeg():
    img = _make_test_image()
    result = encode_image_to_base64(img)
    raw = base64.b64decode(result)
    # PNG magic bytes: \x89PNG
    assert raw[:4] == b'\x89PNG', f"Expected PNG header, got {raw[:4]!r}"


def test_encode_preserves_rgba_mode():
    """RGBA images (transparency) should NOT be force-converted to RGB."""
    img = _make_test_image(mode="RGBA")
    result = encode_image_to_base64(img)
    raw = base64.b64decode(result)
    assert raw[:4] == b'\x89PNG'


# --- Claude extended thinking: request payload ---

def test_claude_model_gets_thinking_param():
    """anthropic/* models should have thinking enabled and temperature=1."""
    import requests
    captured = {}
    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        raise RuntimeError("stop after capture")

    with patch("llm_interface.requests.post", side_effect=fake_post):
        try:
            from llm_interface import get_openrouter_prediction
            get_openrouter_prediction(
                model_identifier="anthropic/claude-sonnet-4.6",
                api_key="test",
                image=_make_test_image(),
                exam_name="NEET", exam_year="2025",
                question_type="MCQ_SINGLE_CORRECT",
                max_tokens=10000,
            )
        except RuntimeError:
            pass

    assert captured["json"]["thinking"] == {"type": "enabled", "budget_tokens": 9000}
    assert captured["json"]["temperature"] == 1


def test_non_claude_model_no_thinking_param():
    """Non-anthropic models should NOT have a thinking param."""
    captured = {}
    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        raise RuntimeError("stop after capture")

    with patch("llm_interface.requests.post", side_effect=fake_post):
        try:
            from llm_interface import get_openrouter_prediction
            get_openrouter_prediction(
                model_identifier="openai/gpt-5.5",
                api_key="test",
                image=_make_test_image(),
                exam_name="NEET", exam_year="2025",
                question_type="MCQ_SINGLE_CORRECT",
                max_tokens=10000,
                temperature=0.0,
            )
        except RuntimeError:
            pass

    assert "thinking" not in captured["json"]
    assert captured["json"]["temperature"] == 0.0
