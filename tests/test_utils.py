"""Tests for src/utils.py - parse_llm_answer function."""

import sys
import os

# Add src to path so we can import utils directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from utils import parse_llm_answer


# --- MCQ_SINGLE_CORRECT ---

class TestMCQSingleCorrect:
    def test_numeric_option(self):
        assert parse_llm_answer("<answer>2</answer>", "MCQ_SINGLE_CORRECT") == ["2"]

    def test_letter_option(self):
        assert parse_llm_answer("<answer>B</answer>", "MCQ_SINGLE_CORRECT") == ["B"]

    def test_lowercase_letter_uppercased(self):
        assert parse_llm_answer("<answer> c </answer>", "MCQ_SINGLE_CORRECT") == ["C"]

    def test_multiple_numeric_rejected(self):
        assert parse_llm_answer("<answer>1,3</answer>", "MCQ_SINGLE_CORRECT") is None

    def test_multiple_letter_rejected(self):
        assert parse_llm_answer("<answer>A,C</answer>", "MCQ_SINGLE_CORRECT") is None

    def test_invalid_letter_rejected(self):
        """Letters beyond A-D should be rejected."""
        assert parse_llm_answer("<answer>X</answer>", "MCQ_SINGLE_CORRECT") is None

    def test_single_digit_5_accepted(self):
        """Single digits 1-9 are accepted as valid options."""
        assert parse_llm_answer("<answer>5</answer>", "MCQ_SINGLE_CORRECT") == ["5"]

    def test_multi_digit_rejected(self):
        """Multi-digit numbers like 99 should be rejected."""
        assert parse_llm_answer("<answer>99</answer>", "MCQ_SINGLE_CORRECT") is None

    def test_zero_rejected(self):
        """0 is not a valid MCQ option."""
        assert parse_llm_answer("<answer>0</answer>", "MCQ_SINGLE_CORRECT") is None


# --- INTEGER ---

class TestInteger:
    def test_positive_integer(self):
        assert parse_llm_answer("<answer>42</answer>", "INTEGER") == ["42"]

    def test_zero(self):
        assert parse_llm_answer("<answer>0</answer>", "INTEGER") == ["0"]

    def test_decimal(self):
        assert parse_llm_answer("<answer>12.75</answer>", "INTEGER") == ["12.75"]

    def test_small_decimal(self):
        assert parse_llm_answer("<answer>0.5</answer>", "INTEGER") == ["0.5"]

    def test_negative_integer(self):
        assert parse_llm_answer("<answer>-5</answer>", "INTEGER") == ["-5"]

    def test_trailing_zeros(self):
        assert parse_llm_answer("<answer>5.00</answer>", "INTEGER") == ["5.00"]

    def test_text_rejected(self):
        assert parse_llm_answer("<answer>abc</answer>", "INTEGER") is None

    def test_multiple_values_rejected(self):
        assert parse_llm_answer("<answer>1,2</answer>", "INTEGER") is None


# --- MCQ_MULTIPLE_CORRECT ---

class TestMCQMultipleCorrect:
    def test_two_numeric_options(self):
        assert parse_llm_answer("<answer>1,3</answer>", "MCQ_MULTIPLE_CORRECT") == ["1", "3"]

    def test_two_letter_options(self):
        assert parse_llm_answer("<answer>A,C</answer>", "MCQ_MULTIPLE_CORRECT") == ["A", "C"]

    def test_lowercase_uppercased_with_spaces(self):
        assert parse_llm_answer("<answer> b , d </answer>", "MCQ_MULTIPLE_CORRECT") == ["B", "D"]

    def test_single_numeric_valid(self):
        assert parse_llm_answer("<answer>2</answer>", "MCQ_MULTIPLE_CORRECT") == ["2"]

    def test_single_letter_valid(self):
        assert parse_llm_answer("<answer>D</answer>", "MCQ_MULTIPLE_CORRECT") == ["D"]

    def test_mixed_sorted(self):
        assert parse_llm_answer("<answer>1,C,3</answer>", "MCQ_MULTIPLE_CORRECT") == ["1", "3", "C"]

    def test_duplicates_removed(self):
        assert parse_llm_answer("<answer>C,1,A,1</answer>", "MCQ_MULTIPLE_CORRECT") == ["1", "A", "C"]

    def test_invalid_letter_rejected(self):
        """X is not in A-D, so entire parse fails."""
        assert parse_llm_answer("<answer>1,X,3</answer>", "MCQ_MULTIPLE_CORRECT") is None

    def test_double_comma_handled(self):
        assert parse_llm_answer("<answer>1,,2</answer>", "MCQ_MULTIPLE_CORRECT") == ["1", "2"]


# --- SKIP and edge cases ---

class TestSkipAndEdgeCases:
    def test_skip_uppercase_tag(self):
        assert parse_llm_answer("<ANSWER>SKIP</ANSWER>", "MCQ_SINGLE_CORRECT") == "SKIP"

    def test_skip_with_whitespace(self):
        assert parse_llm_answer("  <answer>  SKIP  </answer>  ", "INTEGER") == "SKIP"

    def test_no_tag(self):
        assert parse_llm_answer("No answer tag here.", "MCQ_SINGLE_CORRECT") is None

    def test_empty_tag(self):
        assert parse_llm_answer("<answer></answer>", "MCQ_SINGLE_CORRECT") is None

    def test_whitespace_only_tag(self):
        assert parse_llm_answer("<answer>  </answer>", "MCQ_SINGLE_CORRECT") is None

    def test_none_input(self):
        assert parse_llm_answer(None, "MCQ_SINGLE_CORRECT") is None

    def test_empty_string_input(self):
        assert parse_llm_answer("", "MCQ_SINGLE_CORRECT") is None

    def test_unknown_type(self):
        assert parse_llm_answer("<answer>5</answer>", "UNKNOWN_TYPE") is None
