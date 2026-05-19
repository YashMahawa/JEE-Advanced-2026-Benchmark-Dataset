"""Tests for src/evaluation.py - scoring and evaluation functions."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from evaluation import (
    calculate_single_question_score_details,
    calculate_exam_scores,
    calculate_max_score_for_question,
    is_within_range,
)


# --- calculate_single_question_score_details ---

class TestSingleQuestionScoring:
    """Tests for individual question scoring across exam types."""

    # NEET MCQ
    def test_neet_correct(self):
        result = calculate_single_question_score_details({
            "question_id": "N001", "exam_name": "NEET",
            "question_type": "MCQ_SINGLE_CORRECT",
            "ground_truth": ["1"], "predicted_answer": ["1"],
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == 4
        assert result["evaluation_status"] == "correct"

    def test_neet_incorrect(self):
        result = calculate_single_question_score_details({
            "question_id": "N002", "exam_name": "NEET",
            "question_type": "MCQ_SINGLE_CORRECT",
            "ground_truth": ["D"], "predicted_answer": ["B"],
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == -1
        assert result["evaluation_status"] == "incorrect"

    def test_neet_skipped(self):
        result = calculate_single_question_score_details({
            "question_id": "N003", "exam_name": "NEET",
            "question_type": "MCQ_SINGLE_CORRECT",
            "ground_truth": ["4"], "predicted_answer": "SKIP",
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == 0
        assert result["evaluation_status"] == "skipped"

    def test_api_failure_no_penalty(self):
        """API/parse failures should score 0, not -1."""
        result = calculate_single_question_score_details({
            "question_id": "N004", "exam_name": "NEET",
            "question_type": "MCQ_SINGLE_CORRECT",
            "ground_truth": ["C"], "predicted_answer": None,
            "api_call_successful": False,
        })
        assert result["marks_awarded"] == 0
        assert result["evaluation_status"] == "failure_api_or_parse"

    def test_parse_failure_no_penalty(self):
        """Parse failure (api_success=True but pred=None) should score 0."""
        result = calculate_single_question_score_details({
            "question_id": "N005", "exam_name": "NEET",
            "question_type": "MCQ_SINGLE_CORRECT",
            "ground_truth": ["A"], "predicted_answer": None,
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == 0
        assert result["evaluation_status"] == "failure_api_or_parse"

    # JEE Main
    def test_jee_main_mcq_correct(self):
        result = calculate_single_question_score_details({
            "question_id": "JM001", "exam_name": "JEE_MAIN",
            "question_type": "MCQ_SINGLE_CORRECT",
            "ground_truth": ["2"], "predicted_answer": ["2"],
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == 4

    def test_jee_main_integer_correct(self):
        result = calculate_single_question_score_details({
            "question_id": "JM003", "exam_name": "JEE_MAIN",
            "question_type": "INTEGER",
            "ground_truth": ["5"], "predicted_answer": ["5"],
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == 4

    def test_jee_main_integer_incorrect_no_penalty(self):
        result = calculate_single_question_score_details({
            "question_id": "JM004", "exam_name": "JEE_MAIN",
            "question_type": "INTEGER",
            "ground_truth": ["10"], "predicted_answer": ["8"],
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == 0

    # JEE Advanced
    def test_jee_advanced_mcq_single_correct(self):
        result = calculate_single_question_score_details({
            "question_id": "JA001", "exam_name": "JEE_ADVANCED",
            "question_type": "MCQ_SINGLE_CORRECT",
            "ground_truth": ["A"], "predicted_answer": ["a"],
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == 3  # JEE Advanced MCQ single = +3

    def test_jee_advanced_mcq_multiple_all_correct(self):
        result = calculate_single_question_score_details({
            "question_id": "JA005", "exam_name": "JEE_ADVANCED",
            "question_type": "MCQ_MULTIPLE_CORRECT",
            "ground_truth": ["A", "C"], "predicted_answer": ["a", "c"],
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == 4
        assert result["evaluation_status"] == "correct_full"

    def test_jee_advanced_mcq_multiple_partial_2_of_3(self):
        result = calculate_single_question_score_details({
            "question_id": "JA006", "exam_name": "JEE_ADVANCED",
            "question_type": "MCQ_MULTIPLE_CORRECT",
            "ground_truth": ["A", "B", "C"], "predicted_answer": ["a", "b"],
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == 2
        assert result["evaluation_status"] == "partial_2_of_3_plus"

    def test_jee_advanced_mcq_multiple_partial_3_of_4(self):
        result = calculate_single_question_score_details({
            "question_id": "JA007", "exam_name": "JEE_ADVANCED",
            "question_type": "MCQ_MULTIPLE_CORRECT",
            "ground_truth": ["A", "B", "C", "D"], "predicted_answer": ["a", "b", "c"],
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == 3
        assert result["evaluation_status"] == "partial_3_of_4"

    def test_jee_advanced_mcq_multiple_partial_1_of_2(self):
        result = calculate_single_question_score_details({
            "question_id": "JA008", "exam_name": "JEE_ADVANCED",
            "question_type": "MCQ_MULTIPLE_CORRECT",
            "ground_truth": ["A", "B"], "predicted_answer": ["a"],
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == 1
        assert result["evaluation_status"] == "partial_1_of_2_plus"

    def test_jee_advanced_mcq_multiple_incorrect_chosen(self):
        result = calculate_single_question_score_details({
            "question_id": "JA009", "exam_name": "JEE_ADVANCED",
            "question_type": "MCQ_MULTIPLE_CORRECT",
            "ground_truth": ["A", "C"], "predicted_answer": ["a", "b"],
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == -2
        assert result["evaluation_status"] == "incorrect_negative"

    def test_jee_advanced_integer_with_range(self):
        result = calculate_single_question_score_details({
            "question_id": "JA_RANGE", "exam_name": "JEE_ADVANCED",
            "question_type": "INTEGER",
            "ground_truth": [["0.7", "0.8"]],
            "predicted_answer": ["0.75"],
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == 4
        assert result["evaluation_status"] == "correct"

    def test_jee_advanced_integer_outside_range(self):
        result = calculate_single_question_score_details({
            "question_id": "JA_RANGE2", "exam_name": "JEE_ADVANCED",
            "question_type": "INTEGER",
            "ground_truth": [["0.7", "0.8"]],
            "predicted_answer": ["0.9"],
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == 0
        assert result["evaluation_status"] == "incorrect"

    def test_unexpected_prediction_type_no_penalty(self):
        """Unexpected pred type is a code/data bug, not a deliberate wrong answer — no penalty."""
        result = calculate_single_question_score_details({
            "question_id": "X001", "exam_name": "NEET",
            "question_type": "MCQ_SINGLE_CORRECT",
            "ground_truth": ["A"], "predicted_answer": 42,  # int, not list or str
            "api_call_successful": True,
        })
        assert result["marks_awarded"] == 0
        assert result["evaluation_status"] == "failure_unexpected_type"


# --- is_within_range ---

class TestIsWithinRange:
    def test_within_range(self):
        assert is_within_range("0.75", "0.7", "0.8") is True

    def test_at_lower_bound(self):
        assert is_within_range("0.7", "0.7", "0.8") is True

    def test_at_upper_bound(self):
        assert is_within_range("0.8", "0.7", "0.8") is True

    def test_below_range(self):
        assert is_within_range("0.5", "0.7", "0.8") is False

    def test_above_range(self):
        assert is_within_range("0.9", "0.7", "0.8") is False

    def test_invalid_value(self):
        assert is_within_range("abc", "0.7", "0.8") is False


# --- calculate_max_score_for_question ---

class TestMaxScore:
    def test_neet_mcq(self):
        assert calculate_max_score_for_question("NEET", "MCQ_SINGLE_CORRECT") == 4

    def test_jee_main_mcq(self):
        assert calculate_max_score_for_question("JEE_MAIN", "MCQ_SINGLE_CORRECT") == 4

    def test_jee_main_integer(self):
        assert calculate_max_score_for_question("JEE_MAIN", "INTEGER") == 4

    def test_jee_advanced_mcq_single(self):
        assert calculate_max_score_for_question("JEE_ADVANCED", "MCQ_SINGLE_CORRECT") == 3

    def test_jee_advanced_mcq_multiple(self):
        assert calculate_max_score_for_question("JEE_ADVANCED", "MCQ_MULTIPLE_CORRECT") == 4

    def test_jee_advanced_integer(self):
        assert calculate_max_score_for_question("JEE_ADVANCED", "INTEGER") == 4

    def test_unknown(self):
        assert calculate_max_score_for_question("UNKNOWN", "UNKNOWN") == 0


# --- calculate_exam_scores (integration) ---

class TestExamScoresIntegration:
    """Integration test using the full test dataset from evaluation.py __main__."""

    @pytest.fixture
    def test_results(self):
        return [
            {"question_id": "N001", "subject": "Physics", "exam_name": "NEET", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["1"], "predicted_answer": ["1"], "api_call_successful": True},
            {"question_id": "N002", "subject": "Physics", "exam_name": "NEET", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["D"], "predicted_answer": ["B"], "api_call_successful": True},
            {"question_id": "N003", "subject": "Chemistry", "exam_name": "NEET", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["4"], "predicted_answer": "SKIP", "api_call_successful": True},
            {"question_id": "N004", "subject": "Chemistry", "exam_name": "NEET", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["C"], "predicted_answer": None, "api_call_successful": False},
            {"question_id": "N005", "subject": "Botany", "exam_name": "NEET", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["A"], "predicted_answer": None, "api_call_successful": True},
            {"question_id": "JM001", "subject": "Maths", "exam_name": "JEE_MAIN", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["2"], "predicted_answer": ["2"], "api_call_successful": True},
            {"question_id": "JM002", "subject": "Maths", "exam_name": "JEE_MAIN", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["C"], "predicted_answer": ["a"], "api_call_successful": True},
            {"question_id": "JM003", "subject": "Physics", "exam_name": "JEE_MAIN", "question_type": "INTEGER", "ground_truth": ["5"], "predicted_answer": ["5"], "api_call_successful": True},
            {"question_id": "JM004", "subject": "Physics", "exam_name": "JEE_MAIN", "question_type": "INTEGER", "ground_truth": ["10"], "predicted_answer": ["8"], "api_call_successful": True},
            {"question_id": "JM005", "subject": "Chemistry", "exam_name": "JEE_MAIN", "question_type": "INTEGER", "ground_truth": ["7"], "predicted_answer": None, "api_call_successful": True},
            {"question_id": "JA001", "subject": "Maths", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["A"], "predicted_answer": ["a"], "api_call_successful": True},
            {"question_id": "JA002", "subject": "Maths", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["B"], "predicted_answer": ["C"], "api_call_successful": True},
            {"question_id": "JA003", "subject": "Physics", "exam_name": "JEE_ADVANCED", "question_type": "INTEGER", "ground_truth": ["12"], "predicted_answer": ["12"], "api_call_successful": True},
            {"question_id": "JA004", "subject": "Physics", "exam_name": "JEE_ADVANCED", "question_type": "INTEGER", "ground_truth": ["0"], "predicted_answer": ["1"], "api_call_successful": True},
            {"question_id": "JA005", "subject": "Chemistry", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "C"], "predicted_answer": ["a", "c"], "api_call_successful": True},
            {"question_id": "JA006", "subject": "Chemistry", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "B", "C"], "predicted_answer": ["a", "b"], "api_call_successful": True},
            {"question_id": "JA007", "subject": "Chemistry", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "B", "C", "D"], "predicted_answer": ["a", "b", "c"], "api_call_successful": True},
            {"question_id": "JA008", "subject": "Chemistry", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "B"], "predicted_answer": ["a"], "api_call_successful": True},
            {"question_id": "JA009", "subject": "Chemistry", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "C"], "predicted_answer": ["a", "b"], "api_call_successful": True},
            {"question_id": "JA010", "subject": "Chemistry", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "C"], "predicted_answer": ["b", "d"], "api_call_successful": True},
            {"question_id": "JA011", "subject": "Chemistry", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "C"], "predicted_answer": "SKIP", "api_call_successful": True},
            {"question_id": "JA012", "subject": "Maths", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A"], "predicted_answer": ["a"], "api_call_successful": True},
            {"question_id": "JA013", "subject": "Physics", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "B", "C"], "predicted_answer": ["a", "d"], "api_call_successful": True},
        ]

    def test_overall_score(self, test_results):
        summary = calculate_exam_scores(test_results)
        # NEET: 4-1+0+0+0=3, JEE Main MCQ: 4-1=3, JEE Main INT: 4+0+0=4,
        # JEE Adv MCQ Single: 3-1=2, JEE Adv INT: 4+0=4,
        # JEE Adv MCQ Multi: 4+2+3+1-2-2+0+4-2=8 => Total=24
        assert summary["overall_score"] == 24

    def test_overall_counts(self, test_results):
        summary = calculate_exam_scores(test_results)
        assert summary["overall_correct_full"] == 7
        assert summary["overall_partial_correct"] == 3
        assert summary["overall_incorrect_choice"] == 8
        assert summary["overall_skipped"] == 2
        assert summary["overall_api_parse_failures"] == 3

    def test_section_scores(self, test_results):
        summary = calculate_exam_scores(test_results)
        sections = summary["section_breakdown"]
        assert sections["Physics"]["score"] == 9    # (4-1)+(4+0)+(4+0)-2
        assert sections["Chemistry"]["score"] == 6  # (0+0)+(0)+(4+2+3+1-2-2+0)
        assert sections["Botany"]["score"] == 0     # parse fail = 0
        assert sections["Maths"]["score"] == 9      # (4-1)+(3-1)+4

    def test_empty_results(self):
        summary = calculate_exam_scores([])
        assert "error" in summary
