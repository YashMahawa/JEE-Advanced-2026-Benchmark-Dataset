"""Tests for scripts/generate_leaderboard.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import json
import tempfile
import pytest
from generate_leaderboard import parse_result_dirname, scan_results, compute_stats


# --- parse_result_dirname ---

def test_parse_full_dirname():
    result = parse_result_dirname("google_gemini-3.1-pro-preview_NEET_2026_20260503_120000")
    assert result == {"model": "google/gemini-3.1-pro-preview", "exam": "NEET", "year": "2026", "timestamp": "20260503_120000"}


def test_parse_jee_advanced_dirname():
    result = parse_result_dirname("openai_o3_JEE_ADVANCED_2025_20260101_090000")
    assert result == {"model": "openai/o3", "exam": "JEE_ADVANCED", "year": "2025", "timestamp": "20260101_090000"}


def test_parse_no_year_dirname():
    """Dir without year (--exam_year all) should still parse exam."""
    result = parse_result_dirname("openai_o3_NEET_20260503_120000")
    assert result is not None
    assert result["exam"] == "NEET"
    assert result["year"] == "AllYears"


def test_parse_no_exam_no_year_dirname():
    """Dir without exam or year (--exam_name all --exam_year all) returns None from name alone."""
    result = parse_result_dirname("openai_o3_20260503_120000")
    assert result is None  # Cannot determine exam from name — fallback needed


def test_parse_invalid_dirname():
    assert parse_result_dirname("notaresult") is None


# --- scan_results fallback ---

def _write_summary_jsonl(dirpath, records):
    with open(os.path.join(dirpath, "summary.jsonl"), "w") as f:
        for r in records:
            json.dump(r, f); f.write("\n")


def test_scan_results_fallback_reads_exam_from_summary():
    """When dirname has no exam/year, scan_results reads them from summary.jsonl records."""
    with tempfile.TemporaryDirectory() as results_dir:
        run_dir = os.path.join(results_dir, "openai_o3_20260503_120000")
        os.makedirs(run_dir)
        records = [
            {"question_id": f"N26{i:03d}", "exam_name": "NEET", "exam_year": 2026,
             "marks_awarded": 4, "evaluation_status": "correct",
             "predicted_answer": ["1"], "ground_truth": ["1"]}
            for i in range(15)
        ]
        _write_summary_jsonl(run_dir, records)

        entries = scan_results(results_dir, min_questions=10)
        assert len(entries) == 1
        assert entries[0]["exam"] == "NEET"
        assert entries[0]["year"] == "2026"
