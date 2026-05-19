"""Tests for src/benchmark_runner.py — pure functions only."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from benchmark_runner import log_question_result


def _result(evaluation_status, marks_awarded, api_ok=True, parse_ok=True, pred=["A"]):
    return {
        "question_id": "TEST001",
        "api_call_successful": api_ok,
        "parse_successful": parse_ok,
        "predicted_answer": pred,
        "marks_awarded": marks_awarded,
        "evaluation_status": evaluation_status,
        "attempt": 1,
    }


def test_correct_returns_correct():
    assert log_question_result(_result("correct", 4)) == "correct"


def test_incorrect_returns_incorrect():
    assert log_question_result(_result("incorrect", -1)) == "incorrect"


def test_partial_3_of_4_returns_partial():
    assert log_question_result(_result("partial_3_of_4", 3)) == "partial"


def test_partial_2_of_3_plus_returns_partial():
    assert log_question_result(_result("partial_2_of_3_plus", 2)) == "partial"


def test_partial_1_of_2_plus_returns_partial():
    assert log_question_result(_result("partial_1_of_2_plus", 1)) == "partial"


def test_skip_returns_skipped():
    assert log_question_result(_result("skipped", 0, pred="SKIP")) == "skipped"


def test_api_fail_returns_api_fail():
    assert log_question_result(_result("failure_api_or_parse", 0, api_ok=False, parse_ok=False, pred=None)) == "api_fail"


def test_parse_fail_returns_parse_fail():
    r = _result("failure_api_or_parse", 0, api_ok=True, parse_ok=False, pred=None)
    assert log_question_result(r) == "parse_fail"
