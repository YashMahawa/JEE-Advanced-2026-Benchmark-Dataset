import argparse
import subprocess
import threading
import yaml
import os
import json
import logging
import datetime # Added for timestamp
from typing import Dict, Any, List, Set, Tuple # Added for type hinting
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datasets import Dataset, Features, Value, Image as HFImage
from tqdm import tqdm
from PIL import Image as PILImage # Import PIL for type hinting

# ANSI escape codes for colors
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
YELLOW = '\033[93m' # For skipped
CYAN = '\033[96m' # For parse failures
MAGENTA = '\033[95m' # For API failures

# Import local modules
from utils import load_api_key
from llm_interface import get_openrouter_prediction
# Import evaluation functions
from evaluation import calculate_exam_scores, calculate_single_question_score_details

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_git_sha() -> str | None:
    """Returns the current git commit SHA, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_available_models(config_path: str) -> List[str]:
    """Loads models from the benchmark configuration YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        models = config.get("openrouter_models", [])
        if not models:
            logging.warning(f"No models found in {config_path} under 'openrouter_models'.")
        return models
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path} for model retrieval.")
        return []
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {config_path} for model retrieval: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error retrieving models from {config_path}: {e}")
        return []

def get_available_exam_details(metadata_path: str) -> Tuple[List[str], List[str]]:
    """Reads metadata.jsonl to get unique exam names and years."""
    exam_names: Set[str] = set()
    exam_years: Set[str] = set()
    try:
        with open(metadata_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'exam_name' in data:
                        exam_names.add(data['exam_name'])
                    if 'exam_year' in data:
                        exam_years.add(str(data['exam_year']))
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSON line in {metadata_path}: {line.strip()}")
        
        sorted_exam_names = sorted(list(exam_names))
        sorted_exam_years = sorted(list(exam_years))
        
        if not sorted_exam_names:
            logging.warning(f"No exam names found in {metadata_path}.")
        if not sorted_exam_years:
            logging.warning(f"No exam years found in {metadata_path}.")
            
        return sorted_exam_names, sorted_exam_years
    except FileNotFoundError:
        logging.error(f"Metadata file not found at {metadata_path}.")
        return [], []
    except Exception as e:
        logging.error(f"Unexpected error reading or parsing {metadata_path}: {e}")
        return [], []

def load_config(config_path: str) -> dict:
    """Loads the benchmark configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {config_path}: {e}")
        raise

def append_prediction(result: Dict[str, Any], filepath: str, lock=None):
    """Appends a single prediction result to a JSONL file."""
    # Create a copy to avoid modifying the original dict that might be used elsewhere
    # and remove evaluation-specific fields before saving to predictions.jsonl
    prediction_data = result.copy()
    prediction_data.pop('marks_awarded', None)
    prediction_data.pop('evaluation_status', None)
    prediction_data.pop('predicted_answer', None) # Remove predicted_answer
    prediction_data.pop('ground_truth', None) # Remove ground_truth
    ctx = lock if lock is not None else nullcontext()
    try:
        with ctx:
            with open(filepath, 'a') as f:
                json.dump(prediction_data, f)
                f.write('\n')
    except IOError as e:
        logging.error(f"Failed to append prediction to {filepath}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error appending prediction to {filepath}: {e}")

def sort_jsonl_by_question_id(filepath: str):
    """Rewrites a JSONL file in question_id order. No-op if the file is missing or empty."""
    if not os.path.exists(filepath):
        return
    try:
        with open(filepath) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows.sort(key=lambda r: r.get("question_id", ""))
        with open(filepath, 'w') as f:
            for row in rows:
                json.dump(row, f)
                f.write('\n')
    except Exception as e:
        logging.error(f"Failed to sort {filepath} by question_id: {e}")


def append_summary_detail(result_detail: Dict[str, Any], filepath: str, lock=None):
    """Appends a single question's summary details (evaluation status, marks, predicted, truth) to a JSONL file."""
    ctx = lock if lock is not None else nullcontext()
    try:
        with ctx:
            with open(filepath, 'a') as f:
                json.dump(result_detail, f)
                f.write('\n')
    except IOError as e:
        logging.error(f"Failed to append summary detail to {filepath}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error appending summary detail to {filepath}: {e}")


# Removed save_summary function as summary.json is no longer needed.

def generate_markdown_summary(summary: Dict[str, Any], filepath: str):
    """Generates a human-readable Markdown summary from the results dictionary."""
    try:
        md_content = []
        model_name = summary.get("model_name", "N/A")
        exam_name = summary.get("exam_name", "N/A")
        exam_year = summary.get("exam_year", "N/A")
        timestamp = summary.get("timestamp", "N/A")
        total_questions_in_dataset = summary.get("total_questions_in_dataset", 0)
        total_questions_processed_in_run = summary.get("total_questions_processed_in_run", 0)

        filtered_questions_count = 0
        if total_questions_in_dataset > 0 and total_questions_processed_in_run > 0:
             filtered_questions_count = total_questions_in_dataset - total_questions_processed_in_run


        md_content.append(f"# Benchmark Results: {model_name}")
        if exam_name and exam_name not in ["N/A", "All_Exams"]: # Only display if a specific exam was targeted
            md_content.append(f"**Exam Name:** {exam_name}")
        if exam_year and exam_year not in ["N/A", "All_Years"]: # Only display if a specific year was targeted
            md_content.append(f"**Exam Year:** {exam_year}")
        md_content.append(f"**Timestamp:** {timestamp}")
        md_content.append(f"**Total Questions in Dataset:** {total_questions_in_dataset if total_questions_in_dataset > 0 else 'N/A'}")
        if filtered_questions_count > 0:
            md_content.append(f"**Questions Filtered Out:** {filtered_questions_count}")
        md_content.append(f"**Total Questions Processed in this Run:** {total_questions_processed_in_run}")

        # API usage stats
        total_cost_usd = summary.get("total_cost_usd")
        total_tokens = summary.get("total_tokens")
        avg_latency = summary.get("avg_response_latency_ms")
        median_latency = summary.get("median_response_latency_ms")
        if total_tokens or total_cost_usd or avg_latency:
            md_content.append("")
            md_content.append("### API Usage")
            if summary.get("total_prompt_tokens"):
                md_content.append(f"- **Prompt Tokens:** {summary['total_prompt_tokens']:,}")
            if summary.get("total_completion_tokens"):
                md_content.append(f"- **Completion Tokens:** {summary['total_completion_tokens']:,}")
            if total_tokens:
                md_content.append(f"- **Total Tokens:** {total_tokens:,}")
            if total_cost_usd:
                md_content.append(f"- **Total Cost:** ${total_cost_usd:.4f}")
            if avg_latency:
                md_content.append(f"- **Avg Response Latency:** {avg_latency:,} ms")
            if median_latency:
                md_content.append(f"- **Median Response Latency:** {median_latency:,} ms")

        md_content.append("\n---\n")

        # Check if NEET results are present (or any dataset with overall_score and section_breakdown)
        if "overall_score" in summary and "section_breakdown" in summary: # Generic check for score-based summary
            total_processed = summary.get("total_questions_processed", 0)
            
            overall_score = summary.get('overall_score', 'N/A')
            total_possible_score = summary.get('total_possible_score_for_processed_questions', 'N/A')
            correct_full_count = summary.get('overall_correct_full', 'N/A')
            partial_correct_count = summary.get('overall_partial_correct', 'N/A')
            incorrect_choice_count = summary.get('overall_incorrect_choice', 'N/A')
            skipped_count = summary.get('overall_skipped', 'N/A')
            failures_count = summary.get('overall_api_parse_failures', 'N/A')
            unmapped_count = summary.get('unmapped_section_questions', 'N/A')

            md_content.append("## Exam Scoring Results")
            md_content.append(f"**Overall Score:** **{overall_score}** / **{total_possible_score}**")
            md_content.append(f"- **Fully Correct Answers:** {correct_full_count}")
            if partial_correct_count != 'N/A' and partial_correct_count > 0 :
                md_content.append(f"- **Partially Correct Answers:** {partial_correct_count}")
            md_content.append(f"- **Incorrectly Answered (Choice Made):** {incorrect_choice_count}")
            md_content.append(f"- **Skipped Questions:** {skipped_count}")
            md_content.append(f"- **API/Parse Failures:** {failures_count}")
            md_content.append(f"- **Total Questions Processed:** {total_processed}")
            if unmapped_count > 0:
                md_content.append(f"- **Unmapped Section Questions:** {unmapped_count} *(Not included in section breakdown)*")

            md_content.append("\n### Detailed Score Calculation by Question Type")
            question_type_breakdown = summary.get("question_type_breakdown", {})
            if question_type_breakdown:
                sorted_q_types = sorted(question_type_breakdown.keys())
                for q_type in sorted_q_types:
                    stats = question_type_breakdown[q_type]
                    q_type_display = q_type.replace('_', ' ').title()

                    correct_count_q = stats.get('correct_full', 0)
                    partial_count_q = stats.get('partial_correct', 0)
                    incorrect_count_q = stats.get('incorrect_choice', 0)
                    skipped_count_q = stats.get('skipped', 0)
                    api_fail_count_q = stats.get('api_parse_failures', 0)
                    score_q = stats.get('score', 0)

                    breakdown_parts = []
                    if correct_count_q > 0:
                        breakdown_parts.append(f"{correct_count_q} Correct")
                    if partial_count_q > 0:
                        breakdown_parts.append(f"{partial_count_q} Partial")
                    if incorrect_count_q > 0:
                        breakdown_parts.append(f"{incorrect_count_q} Incorrect")
                    if skipped_count_q > 0:
                        breakdown_parts.append(f"{skipped_count_q} Skipped")
                    if api_fail_count_q > 0:
                        breakdown_parts.append(f"{api_fail_count_q} API/Parse Fail")

                    breakdown_str = ", ".join(breakdown_parts) if breakdown_parts else "No questions processed"

                    md_content.append(f"**{q_type_display} ({stats.get('count', 0)} questions):** {score_q} marks")
                    md_content.append(f"  *Breakdown:* {breakdown_str}")
            else:
                md_content.append("No question type breakdown available.")

            md_content.append("\n### Section Breakdown")
            md_content.append("| Section       | Score | Fully Correct | Partially Correct | Incorrect Choice | Skipped | API/Parse Failures |")
            md_content.append("|---------------|-------|---------------|-------------------|------------------|---------|--------------------|")
            section_breakdown = summary.get("section_breakdown", {})
            
            sorted_section_names = sorted(section_breakdown.keys())
            if not sorted_section_names and section_breakdown:
                logging.warning("Could not sort section names for Markdown summary; using unsorted.")
                sorted_section_names = list(section_breakdown.keys())

            for section_name in sorted_section_names:
                stats = section_breakdown.get(section_name, {})
                score = stats.get('score', 'N/A')
                s_correct = stats.get('correct', 'N/A')
                s_partial = stats.get('partial_correct', 'N/A')
                s_incorrect = stats.get('incorrect', 'N/A')
                s_skipped = stats.get('skipped', 'N/A')
                s_failures = stats.get('api_parse_failures', 'N/A')
                display_section_name = section_name.replace('_', ' ')
                md_content.append(f"| {display_section_name:<13} | {score:<5} | {s_correct:<13} | {s_partial:<17} | {s_incorrect:<16} | {s_skipped:<7} | {s_failures:<18} |")
            if not sorted_section_names:
                md_content.append("| No section data available | N/A   | N/A           | N/A               | N/A              | N/A     | N/A                |")
        
        # Fallback for simple accuracy (if exam scoring wasn't applicable or failed)
        elif "accuracy_on_parsed" in summary:
            md_content.append("## Simple Accuracy Results (Fallback)")
            md_content.append(f"- **Accuracy (on successfully parsed non-skipped):** {summary.get('accuracy_on_parsed', 'N/A'):.4f}")
            md_content.append(f"- **Total Processed Attempts:** {summary.get('total_processed_attempts', 'N/A')}")
            # Add other relevant simple stats if available
        else:
            md_content.append("## Summary")
            md_content.append("*(No specific Exam Scoring or Accuracy metrics found in summary)*")


        with open(filepath, 'w') as f:
            f.write("\n".join(md_content))
        logging.info(f"Markdown summary saved to {filepath}")

    except IOError as e:
        logging.error(f"Failed to save markdown summary to {filepath}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error generating or saving markdown summary to {filepath}: {e}")


def process_question(
    model_id: str,
    api_key: str,
    config: dict,
    example: dict,
    image: PILImage.Image,
    predictions_path: str,
    summary_details_path: str,
    attempt: int = 1,
    lock=None,
) -> Dict[str, Any]:
    """Handles a single question: API call, re-prompt on parse failure, scoring, writing to disk.

    Returns the result_data dict. On API exception, raises so the caller can handle retry queueing.
    """
    question_id = example["question_id"]
    subject = example["subject"]
    exam_name_from_data = example.get("exam_name", "UNKNOWN_EXAM")
    exam_year_from_data = example.get("exam_year")
    question_type_from_data = example.get("question_type", "MCQ_SINGLE_CORRECT")
    truth = json.loads(example["correct_answer"])

    result_data = {
        "question_id": question_id,
        "subject": subject,
        "exam_name": exam_name_from_data,
        "question_type": question_type_from_data,
        "ground_truth": truth,
        "predicted_answer": None,
        "raw_response": None,
        "parse_successful": False,
        "api_call_successful": False,
        "error": None,
        "attempt": attempt,
        "previous_raw_response_on_reprompt": None,
        "response_metadata": None,
    }

    # --- API Call ---
    logging.info(f"Attempting API call for question: {question_id} with model: {model_id}")
    parsed_answer, raw_response, response_metadata = get_openrouter_prediction(
        model_identifier=model_id,
        api_key=api_key,
        image=image,
        exam_name=exam_name_from_data,
        exam_year=str(example.get("exam_year", "UNKNOWN_YEAR")),
        question_type=question_type_from_data,
        max_tokens=config.get("max_tokens", 100),
        request_timeout=config.get("request_timeout", 60),
        temperature=config.get("temperature", 0),
        reasoning_effort=config.get("reasoning_effort", "high"),
    )

    api_success = raw_response is not None  # False if API returned empty content
    parse_success = parsed_answer is not None

    # --- Re-prompt Logic ---
    if api_success and not parse_success and raw_response is not None:
        logging.warning(f"Question {question_id}: Initial parse failed. Attempting re-prompt.")
        result_data["previous_raw_response_on_reprompt"] = raw_response
        try:
            parsed_answer_rp, raw_response_rp, rp_metadata = get_openrouter_prediction(
                model_identifier=model_id,
                api_key=api_key,
                previous_raw_response=raw_response,
                question_type=question_type_from_data,
                max_tokens=config.get("max_tokens", 100),
                request_timeout=config.get("request_timeout", 60),
                temperature=config.get("temperature", 0),
                reasoning_effort=config.get("reasoning_effort", "high"),
            )
            if isinstance(parsed_answer_rp, list):
                parsed_answer_rp = [str(item) for item in parsed_answer_rp]

            # Accumulate tokens/cost across both calls so the stored total is complete
            if rp_metadata and response_metadata:
                rp_metadata = {
                    **rp_metadata,
                    "prompt_tokens": (response_metadata.get("prompt_tokens") or 0) + (rp_metadata.get("prompt_tokens") or 0),
                    "completion_tokens": (response_metadata.get("completion_tokens") or 0) + (rp_metadata.get("completion_tokens") or 0),
                    "cost": (response_metadata.get("cost") or 0.0) + (rp_metadata.get("cost") or 0.0),
                    "response_latency_ms": (response_metadata.get("response_latency_ms") or 0) + (rp_metadata.get("response_latency_ms") or 0),
                }

            result_data.update({
                "predicted_answer": parsed_answer_rp,
                "raw_response": raw_response_rp,
                "parse_successful": parsed_answer_rp is not None,
                "api_call_successful": True,
                "attempt": attempt + 1,
                "response_metadata": rp_metadata,
            })
            logging.info(f"Question {question_id}: Re-prompt {'succeeded' if result_data['parse_successful'] else 'failed to parse'}.")
        except Exception as e_rp:
            logging.error(f"Re-prompt API call failed for question {question_id}: {e_rp}")
            result_data.update({
                "predicted_answer": None,
                "raw_response": raw_response,
                "parse_successful": False,
                "api_call_successful": True,
                "error": f"Initial parse failed. Re-prompt API call failed: {str(e_rp)}",
                "attempt": attempt,
                "response_metadata": response_metadata,
            })
    else:
        current_error = result_data.get("error")
        if not api_success:
            current_error = "API call returned empty content."

        if isinstance(parsed_answer, list):
            parsed_answer = [str(item) for item in parsed_answer]

        result_data.update({
            "predicted_answer": parsed_answer,
            "raw_response": raw_response,
            "parse_successful": parse_success,
            "api_call_successful": api_success,
            "error": current_error,
            "attempt": attempt,
            "response_metadata": response_metadata,
        })

    # --- Score and write to disk ---
    # JEE Advanced 2026 metadata intentionally uses [] until the official key is released.
    # Preserve model outputs now; compute marks later with --score-only after filling answers.
    if truth:
        score_details = calculate_single_question_score_details(result_data)
        result_data["marks_awarded"] = score_details.get("marks_awarded")
        result_data["evaluation_status"] = score_details.get("evaluation_status")
    else:
        result_data["marks_awarded"] = None
        result_data["evaluation_status"] = "pending_official_answer_key"

    metadata = result_data.get("response_metadata") or {}
    append_summary_detail(
        {
            "question_id": question_id,
            "exam_name": exam_name_from_data,
            "exam_year": exam_year_from_data,
            "marks_awarded": result_data["marks_awarded"],
            "evaluation_status": result_data["evaluation_status"],
            "predicted_answer": result_data["predicted_answer"],
            "ground_truth": result_data["ground_truth"],
            "attempt": result_data["attempt"],
            "prompt_tokens": metadata.get("prompt_tokens"),
            "completion_tokens": metadata.get("completion_tokens"),
            "cost": metadata.get("cost"),
            "response_latency_ms": metadata.get("response_latency_ms"),
        },
        summary_details_path,
        lock=lock,
    )
    append_prediction(result_data, predictions_path, lock=lock)

    return result_data


def log_question_result(result_data: Dict[str, Any], prefix: str = "") -> str:
    """Logs a color-coded result for a single question.

    Returns a category string: 'correct', 'partial', 'incorrect', 'skipped', 'parse_fail', or 'api_fail'.
    """
    question_id = result_data["question_id"]
    log_message_prefix = f"{prefix}Question {question_id}:"
    log_message_suffix = f"(Attempt {result_data['attempt']})"

    if not result_data["api_call_successful"]:
        print(f"{MAGENTA}{log_message_prefix} API Call Failed {log_message_suffix}{RESET}")
        return "api_fail"

    if not result_data["parse_successful"]:
        print(f"{CYAN}{log_message_prefix} Failed to parse answer {log_message_suffix}{RESET}")
        return "parse_fail"

    if result_data.get("evaluation_status") == "pending_official_answer_key":
        print(f"{CYAN}{log_message_prefix} Answer saved - scoring pending official key {log_message_suffix}{RESET}")
        return "pending"

    if result_data["predicted_answer"] == "SKIP":
        print(f"{YELLOW}{log_message_prefix} Skipped {log_message_suffix}{RESET}")
        return "skipped"

    marks_awarded = result_data.get("marks_awarded", 0)
    evaluation_status_value = result_data.get("evaluation_status")

    is_considered_correct = False
    is_partial = False
    log_display_status = "N/A"
    status_check_string = ""

    if evaluation_status_value is True:
        is_considered_correct = True
        log_display_status = "True (Boolean)"
        status_check_string = "CORRECT_TRUE_BOOLEAN"
    elif isinstance(evaluation_status_value, str):
        log_display_status = evaluation_status_value
        status_check_string = evaluation_status_value.strip().upper()
        if status_check_string.startswith("CORRECT"):
            is_considered_correct = True
        elif status_check_string.startswith("PARTIAL_"):
            is_partial = True
    elif evaluation_status_value is None:
        log_display_status = "None"
        status_check_string = "NONE_STATUS"
    else:
        log_display_status = str(evaluation_status_value)
        status_check_string = str(evaluation_status_value).strip().upper()

    known_eval_skip_statuses = ["SKIPPED_BY_EVAL", "SKIPPED"]

    if is_considered_correct:
        print(f"{GREEN}{log_message_prefix} Correct - Marks: {marks_awarded}, Status: {log_display_status} {log_message_suffix}{RESET}")
        return "correct"
    elif is_partial:
        print(f"{YELLOW}{log_message_prefix} Partial - Marks: {marks_awarded}, Status: {log_display_status} {log_message_suffix}{RESET}")
        return "partial"
    elif status_check_string in known_eval_skip_statuses:
        print(f"{YELLOW}{log_message_prefix} Skipped by Eval - Marks: {marks_awarded}, Status: {log_display_status} {log_message_suffix}{RESET}")
        return "skipped"
    else:
        print(f"{RED}{log_message_prefix} Incorrect - Marks: {marks_awarded}, Status: {log_display_status} {log_message_suffix}{RESET}")
        return "incorrect"


def load_completed_question_ids(summary_details_path: str) -> set:
    """Reads summary.jsonl and returns a set of question_ids that have already been processed."""
    completed_ids = set()
    if not os.path.exists(summary_details_path):
        return completed_ids
    try:
        with open(summary_details_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    qid = data.get("question_id")
                    if qid:
                        completed_ids.add(qid)
                except json.JSONDecodeError:
                    continue
    except IOError as e:
        logging.warning(f"Could not read {summary_details_path} for resume: {e}")
    return completed_ids


def rescore_result_dir(result_dir: str, config: dict) -> None:
    """Re-scores an existing result directory against the current metadata.jsonl answer key.

    Reads predicted answers from summary.jsonl, loads updated ground truths from
    metadata.jsonl, re-scores every question, then overwrites summary.jsonl and summary.md.
    No API calls are made.
    """
    summary_details_path = os.path.join(result_dir, "summary.jsonl")
    markdown_summary_path = os.path.join(result_dir, "summary.md")

    if not os.path.exists(summary_details_path):
        logging.error(f"No summary.jsonl found in {result_dir}. Cannot re-score.")
        return

    # Load updated metadata (ground truths + question details)
    metadata_path = config.get("metadata_path", "images/metadata.jsonl")
    metadata_by_qid: Dict[str, Dict] = {}
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                qid = row.get("question_id")
                if qid:
                    metadata_by_qid[qid] = row
    except Exception as e:
        logging.error(f"Failed to load metadata from {metadata_path}: {e}")
        return

    # Load existing per-question records (predicted answers + old scores)
    existing_records: List[Dict[str, Any]] = []
    try:
        with open(summary_details_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except IOError as e:
        logging.error(f"Failed to read {summary_details_path}: {e}")
        return

    if not existing_records:
        logging.error(f"No records found in {summary_details_path}.")
        return

    logging.info(f"Re-scoring {len(existing_records)} questions in {result_dir}")

    # Build result dicts with updated ground truths from metadata
    results_for_scoring: List[Dict[str, Any]] = []
    missing_qids: List[str] = []
    for rec in existing_records:
        qid = rec.get("question_id")
        meta = metadata_by_qid.get(qid)
        if not meta:
            missing_qids.append(qid)
            continue
        pred = rec.get("predicted_answer")
        results_for_scoring.append({
            "question_id": qid,
            "subject": meta.get("subject"),
            "exam_name": meta.get("exam_name", ""),
            "question_type": meta.get("question_type", "MCQ_SINGLE_CORRECT"),
            "ground_truth": json.loads(meta["correct_answer"]),
            "predicted_answer": pred,
            "api_call_successful": pred is not None,
            "response_metadata": {
                "prompt_tokens": rec.get("prompt_tokens"),
                "completion_tokens": rec.get("completion_tokens"),
                "cost": rec.get("cost"),
                "response_latency_ms": rec.get("response_latency_ms"),
            },
        })

    if missing_qids:
        preview = missing_qids[:5]
        suffix = "..." if len(missing_qids) > 5 else ""
        logging.warning(f"{len(missing_qids)} question(s) not found in metadata (skipped): {preview}{suffix}")

    if not results_for_scoring:
        logging.error("No scoreable results after metadata lookup.")
        return

    evaluation_summary = calculate_exam_scores(results_for_scoring)
    updated_by_qid = {r["question_id"]: r for r in results_for_scoring}

    # Overwrite summary.jsonl with updated scores and ground truths
    with open(summary_details_path, 'w') as f:
        for rec in existing_records:
            qid = rec.get("question_id")
            updated = updated_by_qid.get(qid)
            if updated:
                new_rec = {
                    **rec,
                    "marks_awarded": updated["marks_awarded"],
                    "evaluation_status": updated["evaluation_status"],
                    "ground_truth": updated["ground_truth"],
                }
            else:
                new_rec = rec
            json.dump(new_rec, f)
            f.write('\n')
    logging.info(f"Updated {summary_details_path}")

    # Aggregate token/cost/latency across all scored questions
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    latencies: List[float] = []
    for r in results_for_scoring:
        meta_r = r.get("response_metadata") or {}
        if meta_r.get("prompt_tokens"):
            total_prompt_tokens += meta_r["prompt_tokens"]
        if meta_r.get("completion_tokens"):
            total_completion_tokens += meta_r["completion_tokens"]
        if meta_r.get("cost"):
            total_cost += meta_r["cost"]
        if meta_r.get("response_latency_ms"):
            latencies.append(meta_r["response_latency_ms"])

    # Derive model name from the result directory name (format: provider_model_EXAM_YEAR_TS)
    dirname = os.path.basename(os.path.abspath(result_dir))
    first_us = dirname.find("_")
    if first_us > 0:
        model_from_dir = dirname[:first_us] + "/" + dirname[first_us + 1:].split("_")[0]
    else:
        model_from_dir = dirname

    # Derive exam name and year from the scored data
    exam_names_seen = list({r["exam_name"] for r in results_for_scoring if r.get("exam_name")})
    exam_years_seen = list({
        metadata_by_qid[r["question_id"]].get("exam_year")
        for r in results_for_scoring
        if r["question_id"] in metadata_by_qid
    })
    exam_display = exam_names_seen[0] if len(exam_names_seen) == 1 else "Mixed"
    year_display = str(exam_years_seen[0]) if len(exam_years_seen) == 1 else "Mixed"

    summary = {
        "model_name": model_from_dir,
        "exam_name": exam_display,
        "exam_year": year_display,
        "question_ids_filter": "None",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_rescored",
        "git_sha": get_git_sha(),
        "temperature": config.get("temperature", 0),
        "max_tokens": config.get("max_tokens", 10000),
        "total_questions_in_dataset": len(metadata_by_qid),
        "total_questions_processed_in_run": len(existing_records),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "total_cost_usd": round(total_cost, 6),
        "avg_response_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        "median_response_latency_ms": round(sorted(latencies)[len(latencies) // 2]) if latencies else None,
        **evaluation_summary,
    }
    generate_markdown_summary(summary, markdown_summary_path)
    logging.info(
        f"Re-scored summary written to {markdown_summary_path}. "
        f"Score: {summary.get('overall_score')} / {summary.get('total_possible_score_for_processed_questions')}"
    )


def run_benchmark(
    config: dict,
    api_key: str,
    model_to_run: str,
    output_dir_override: str | None = None,
    exam_name_choice: str | None = None,
    exam_year_choice: str | None = None,
    question_ids_str: str | None = None,
    resume_dir: str | None = None
):
    """Runs the benchmark evaluation loop with incremental saving and retries."""

    # Determine models to run - now it's a single model
    models_to_run = [model_to_run] # Benchmark will run for the single specified model
    logging.info(f"Target model for this run: {model_to_run}")

     # Determine base output directory
    base_output_dir = output_dir_override if output_dir_override else config.get("results_base_dir", "results")
    os.makedirs(base_output_dir, exist_ok=True)

     # Load dataset directly from metadata.jsonl and images/
    metadata_path = config.get("metadata_path", "images/metadata.jsonl")
    images_base_dir = config.get("images_base_dir", "images")
    try:
        records = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                row["image"] = os.path.join(images_base_dir, row.pop("file_name"))
                row.setdefault("paper_id", None)
                records.append(row)

        features = Features({
            "image": HFImage(decode=True),
            "question_id": Value("string"),
            "exam_name": Value("string"),
            "exam_year": Value("int32"),
            "subject": Value("string"),
            "question_type": Value("string"),
            "correct_answer": Value("string"),
            "paper_id": Value("int64"),
        })
        dataset = Dataset.from_list(records, features=features)
        logging.info(f"Dataset loaded successfully from {metadata_path}. Original number of questions: {len(dataset)}")
    except Exception as e:
        logging.error(f"Failed to load dataset from '{metadata_path}': {e}")
        logging.error("Ensure 'metadata.jsonl' exists and image paths are valid.")
        return

    # Filter dataset based on choices
    original_dataset_size = len(dataset)
    
    # Filter by exam_name
    if exam_name_choice and exam_name_choice.lower() != "all":
        logging.info(f"Filtering dataset for exam_name: '{exam_name_choice}'")
        dataset = dataset.filter(lambda example: example.get('exam_name') == exam_name_choice)
        logging.info(f"Dataset size after exam_name filter: {len(dataset)} questions.")

    # Filter by exam_year
    if exam_year_choice and exam_year_choice.lower() != "all":
        try:
            filter_year_int = int(exam_year_choice)
            logging.info(f"Filtering dataset for exam_year: {filter_year_int}")
            dataset = dataset.filter(lambda example: example.get('exam_year') == filter_year_int)
            logging.info(f"Dataset size after exam_year filter: {len(dataset)} questions.")
        except ValueError:
            logging.error(f"Invalid exam_year provided: '{exam_year_choice}'. Must be an integer or 'all'. Year filtering skipped.")

    # Filter by specific question IDs if provided
    if question_ids_str:
        try:
            target_question_ids = {q_id.strip() for q_id in question_ids_str.split(',') if q_id.strip()}
            if target_question_ids:
                logging.info(f"Filtering dataset for specific question IDs: {target_question_ids}")
                dataset = dataset.filter(lambda example: example.get('question_id') in target_question_ids)
                logging.info(f"Dataset size after question_id filter: {len(dataset)} questions.")
            else:
                logging.warning("Empty or invalid question_ids string provided. No question ID filtering applied.")
        except Exception as e:
            logging.error(f"Error processing question_ids_str '{question_ids_str}': {e}. No question ID filtering applied.")
            
    if len(dataset) < original_dataset_size:
        logging.info(f"Final dataset size after all filters: {len(dataset)} (originally {original_dataset_size}).")
    
    if len(dataset) == 0:
        logging.warning("No questions to process after filtering. Skipping model benchmark.")
        return

     # --- Main Loop: Iterate through models ---
    for model_id in models_to_run:
        logging.info(f"--- Starting benchmark for model: {model_id} ---")

        # Set up output directory (resume existing or create new)
        if resume_dir:
            model_output_dir = resume_dir
            timestamp = os.path.basename(resume_dir).rsplit('_', 2)[-2] + '_' + os.path.basename(resume_dir).rsplit('_', 1)[-1] if '_' in os.path.basename(resume_dir) else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_name = model_id.replace('/', '_')
            dir_name_parts = [safe_model_name]

            current_exam_name_for_dir = exam_name_choice if exam_name_choice and exam_name_choice.lower() != "all" else "AllExams"
            current_exam_year_for_dir = exam_year_choice if exam_year_choice and exam_year_choice.lower() != "all" else "AllYears"

            if current_exam_name_for_dir != "AllExams":
                dir_name_parts.append(current_exam_name_for_dir.replace('/', '_'))
            if current_exam_year_for_dir != "AllYears":
                dir_name_parts.append(str(current_exam_year_for_dir))

            dir_name_parts.append(timestamp)

            model_output_dir_name = "_".join(filter(None, dir_name_parts))
            model_output_dir = os.path.join(base_output_dir, model_output_dir_name)

        os.makedirs(model_output_dir, exist_ok=True)
        predictions_path = os.path.join(model_output_dir, "predictions.jsonl")
        summary_details_path = os.path.join(model_output_dir, "summary.jsonl")
        markdown_summary_path = os.path.join(model_output_dir, "summary.md")
        logging.info(f"Results for {model_id} will be saved to: {model_output_dir}")

        # Resume: skip already-completed questions
        previous_results: List[Dict[str, Any]] = []
        if resume_dir:
            completed_ids = load_completed_question_ids(summary_details_path)
            if completed_ids:
                logging.info(f"Resuming: found {len(completed_ids)} already-completed questions. Skipping them.")

                # Build metadata lookup from the full filtered dataset before removing completed questions
                completed_metadata = {
                    ex['question_id']: ex
                    for ex in dataset
                    if ex['question_id'] in completed_ids
                }

                # Load per-question scores/answers already written to summary.jsonl
                existing_records: Dict[str, Dict] = {}
                if os.path.exists(summary_details_path):
                    try:
                        with open(summary_details_path, 'r') as f:
                            for line in f:
                                try:
                                    rec = json.loads(line)
                                    qid = rec.get('question_id')
                                    if qid:
                                        existing_records[qid] = rec
                                except json.JSONDecodeError:
                                    continue
                    except IOError as e:
                        logging.warning(f"Could not read {summary_details_path} for resume reconstruction: {e}")

                # Reconstruct full result dicts so the final summary covers all questions
                for qid, meta in completed_metadata.items():
                    rec = existing_records.get(qid, {})
                    pred = rec.get('predicted_answer')
                    previous_results.append({
                        'question_id': qid,
                        'subject': meta.get('subject'),
                        'exam_name': meta.get('exam_name', ''),
                        'question_type': meta.get('question_type', 'MCQ_SINGLE_CORRECT'),
                        'ground_truth': rec.get('ground_truth'),
                        'predicted_answer': pred,
                        'api_call_successful': pred is not None,
                        'response_metadata': {
                            'prompt_tokens': rec.get('prompt_tokens'),
                            'completion_tokens': rec.get('completion_tokens'),
                            'cost': rec.get('cost'),
                            'response_latency_ms': rec.get('response_latency_ms'),
                        },
                    })

                dataset = dataset.filter(lambda example: example.get('question_id') not in completed_ids)
                logging.info(f"Remaining questions to process: {len(dataset)}")
                if len(dataset) == 0:
                    logging.info("All questions already completed. Nothing to resume.")
                    return

        current_total_questions = len(dataset)
        logging.info(f"Processing {current_total_questions} questions for model: {model_id}")

        model_results = [] # Stores results in memory for final calculation
        failed_questions_data = [] # Stores data needed to retry failed questions
        write_lock = threading.Lock()
        max_workers = config.get("max_concurrent_requests", 4)

        # Counters for tqdm postfix
        initial_correct_count = 0
        initial_partial_count = 0
        initial_incorrect_count = 0
        initial_skipped_count = 0
        initial_parse_fail_count = 0
        initial_api_fail_count = 0

        # --- Initial Pass: Iterate through questions ---
        pbar_initial = tqdm(total=current_total_questions, desc=f"Processing {model_id} (Initial Pass)")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_question,
                    model_id=model_id, api_key=api_key, config=config,
                    example=example, image=example["image"],
                    predictions_path=predictions_path,
                    summary_details_path=summary_details_path,
                    attempt=1, lock=write_lock,
                ): example
                for example in dataset
            }
            for future in as_completed(futures):
                example = futures[future]
                try:
                    result_data = future.result()
                    model_results.append(result_data)
                    category = log_question_result(result_data)
                except Exception as e:
                    logging.error(f"Initial API call failed for question {example['question_id']} (Attempt 1): {e}")
                    with write_lock:
                        failed_questions_data.append(example)
                    category = "api_fail"

                if category == "correct": initial_correct_count += 1
                elif category == "partial": initial_partial_count += 1
                elif category == "incorrect": initial_incorrect_count += 1
                elif category == "skipped": initial_skipped_count += 1
                elif category == "parse_fail": initial_parse_fail_count += 1
                elif category == "api_fail": initial_api_fail_count += 1
                pbar_initial.update(1)
                pbar_initial.set_postfix_str(
                    f"✓:{initial_correct_count} ~:{initial_partial_count} ✗:{initial_incorrect_count} "
                    f"S:{initial_skipped_count} P!:{initial_parse_fail_count} A!:{initial_api_fail_count}"
                )
        pbar_initial.close()

        # --- Retry Pass for questions with initial API failures ---
        if failed_questions_data:
            logging.info(f"--- Retrying {len(failed_questions_data)} questions with initial API failures for model: {model_id} ---")

            retry_correct_count = 0
            retry_partial_count = 0
            retry_incorrect_count = 0
            retry_skipped_count = 0
            retry_parse_fail_count = 0
            retry_api_fail_count = 0

            pbar_retry = tqdm(total=len(failed_questions_data), desc=f"Processing {model_id} (API Retry Pass)")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                retry_futures = {
                    executor.submit(
                        process_question,
                        model_id=model_id, api_key=api_key, config=config,
                        example=ex, image=ex["image"],
                        predictions_path=predictions_path,
                        summary_details_path=summary_details_path,
                        attempt=2, lock=write_lock,
                    ): ex
                    for ex in failed_questions_data
                }
                for future in as_completed(retry_futures):
                    ex = retry_futures[future]
                    try:
                        result_data_retry = future.result()
                        model_results.append(result_data_retry)
                        category = log_question_result(result_data_retry, prefix="(Retry) ")
                    except Exception as e_retry_api:
                        logging.error(f"API call failed permanently for question {ex['question_id']} (Attempt 2 API Retry): {e_retry_api}")
                        fail_data = {
                            "question_id": ex["question_id"],
                            "subject": ex["subject"],
                            "exam_name": ex.get("exam_name", "UNKNOWN_EXAM"),
                            "question_type": ex.get("question_type", "MCQ_SINGLE_CORRECT"),
                            "ground_truth": json.loads(ex["correct_answer"]),
                            "predicted_answer": None, "raw_response": None,
                            "parse_successful": False, "api_call_successful": False,
                            "error": f"Initial API fail. Retry API call also failed: {str(e_retry_api)}",
                            "attempt": 2, "previous_raw_response_on_reprompt": None,
                        }
                        if fail_data["ground_truth"]:
                            score_details = calculate_single_question_score_details(fail_data)
                            fail_data["marks_awarded"] = score_details.get("marks_awarded")
                            fail_data["evaluation_status"] = score_details.get("evaluation_status")
                        else:
                            fail_data["marks_awarded"] = None
                            fail_data["evaluation_status"] = "pending_official_answer_key"
                        append_summary_detail(
                            {"question_id": fail_data["question_id"], "marks_awarded": fail_data["marks_awarded"],
                             "evaluation_status": fail_data["evaluation_status"], "predicted_answer": None,
                             "ground_truth": fail_data["ground_truth"], "attempt": 2},
                            summary_details_path, lock=write_lock,
                        )
                        append_prediction(fail_data, predictions_path, lock=write_lock)
                        model_results.append(fail_data)
                        category = "api_fail"

                    if category == "correct": retry_correct_count += 1
                    elif category == "partial": retry_partial_count += 1
                    elif category == "incorrect": retry_incorrect_count += 1
                    elif category == "skipped": retry_skipped_count += 1
                    elif category == "parse_fail": retry_parse_fail_count += 1
                    elif category == "api_fail": retry_api_fail_count += 1
                    pbar_retry.update(1)
                    pbar_retry.set_postfix_str(
                        f"✓:{retry_correct_count} ~:{retry_partial_count} ✗:{retry_incorrect_count} "
                        f"S:{retry_skipped_count} P!:{retry_parse_fail_count} A!:{retry_api_fail_count}"
                    )
            pbar_retry.close()

        # --- Final Evaluation for the current model ---
        logging.info(f"--- Calculating final results for model: {model_id} ---")
        
        # Combine results from previous sessions (if resuming) with this session's results
        all_results = previous_results + model_results
        has_answer_key = any(r.get("ground_truth") for r in all_results)
        if has_answer_key:
            evaluation_summary = calculate_exam_scores(all_results)
        else:
            evaluation_summary = {
                "overall_score": None,
                "overall_correct_full": 0,
                "overall_partial_correct": 0,
                "overall_incorrect_choice": 0,
                "overall_skipped": 0,
                "overall_api_parse_failures": 0,
                "section_breakdown": {},
                "question_type_breakdown": {},
                "total_possible_score_for_processed_questions": None,
                "evaluation_status": "pending_official_answer_key",
            }

        # Use the actual choices for the summary, defaulting to "All" if not specified or "all"
        summary_exam_name_display = exam_name_choice if exam_name_choice and exam_name_choice.lower() != "all" else "All_Exams"
        summary_exam_year_display = exam_year_choice if exam_year_choice and exam_year_choice.lower() != "all" else "All_Years"

        # Aggregate response metadata across all questions
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        latencies = []
        for r in all_results:
            meta = r.get("response_metadata") or {}
            if meta.get("prompt_tokens"):
                total_prompt_tokens += meta["prompt_tokens"]
            if meta.get("completion_tokens"):
                total_completion_tokens += meta["completion_tokens"]
            if meta.get("cost"):
                total_cost += meta["cost"]
            if meta.get("response_latency_ms"):
                latencies.append(meta["response_latency_ms"])

        summary = {
            "model_name": model_id, # This is model_to_run
            "exam_name": summary_exam_name_display,
            "exam_year": summary_exam_year_display,
            "question_ids_filter": question_ids_str if question_ids_str else "None", # Add question ID filter info
            "timestamp": timestamp,
            "git_sha": get_git_sha(),
            "temperature": config.get("temperature", 0),
            "max_tokens": config.get("max_tokens", 100),
            "total_questions_in_dataset": original_dataset_size,
            "total_questions_processed_in_run": len(previous_results) + len(model_results),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_response_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
            "median_response_latency_ms": round(sorted(latencies)[len(latencies) // 2]) if latencies else None,
            **evaluation_summary
        }
        logging.info(f"Overall Score: {summary.get('overall_score')}")
        logging.info(f"Full Correct: {summary.get('overall_correct_full')}, Partial Correct: {summary.get('overall_partial_correct')}, Incorrect Choice: {summary.get('overall_incorrect_choice')}, Skipped: {summary.get('overall_skipped')}, API/Parse Failures: {summary.get('overall_api_parse_failures')}")
        
        logging.info(f"--- Results Summary for model: {model_id} ---")
        logging.info(json.dumps(summary, indent=2, sort_keys=True))
        logging.info("-------------------------------------")

        sort_jsonl_by_question_id(predictions_path)
        sort_jsonl_by_question_id(summary_details_path)

        generate_markdown_summary(summary, markdown_summary_path)

    logging.info("Benchmark run completed.")


if __name__ == "__main__":
    # Get available choices for arguments
    # Assuming benchmark_config.yaml is in a 'configs' directory relative to script or a fixed path
    default_config_path = "configs/benchmark_config.yaml"
    default_metadata_path = "images/metadata.jsonl"

    available_models = get_available_models(default_config_path)
    available_exam_names, available_exam_years = get_available_exam_details(default_metadata_path)

    # Add "all" option for exams and years
    exam_name_choices = ["all"] + available_exam_names
    exam_year_choices = ["all"] + available_exam_years

    parser = argparse.ArgumentParser(description="Run JEE/NEET LLM Benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help=f"Path to the benchmark configuration YAML file (default: {default_config_path})."
    )
    parser.add_argument(
        "--score-only",
        type=str,
        default=None,
        metavar="RESULT_DIR",
        help="Re-score an existing result directory against the current metadata.jsonl answer key. "
             "No API calls are made. Overwrites summary.jsonl and summary.md in RESULT_DIR."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        choices=available_models if available_models else None,
        help="Select the model to run." + (f" Available: {', '.join(available_models)}." if available_models else " (No models found in config)")
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override the base output directory specified in the config file."
    )
    parser.add_argument(
        "--exam_name",
        type=str,
        default="all",
        choices=exam_name_choices if exam_name_choices else ["all"],
        help="Select the exam name to run, or 'all' for all exams." + (f" Available: {', '.join(available_exam_names)}." if available_exam_names else "")
    )
    parser.add_argument(
        "--exam_year",
        type=str, 
        default="all",
        choices=exam_year_choices if exam_year_choices else ["all"],
        help="Select the exam year to run, or 'all' for all years." + (f" Available: {', '.join(available_exam_years)}." if available_exam_years else "")
    )
    parser.add_argument(
        "--question_ids",
        type=str,
        default=None,
        help="Optional: Comma-separated list of specific question IDs to run (e.g., ID1,ID2,ID3)."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional: Path to an existing results directory to resume an interrupted run."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override sampling temperature (default: from config, typically 0 for deterministic output)."
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of independent runs per model (default: 1). Use 3+ for publication-grade variance analysis."
    )
    args = parser.parse_args()

    # Dynamically update config path if user provides a different one
    if args.config != default_config_path:
        logging.info(f"User provided config path: {args.config}. Re-fetching models if necessary.")
        # If models were not found with default, or if user specified a different config, try to load models from it.
        if not available_models or args.model not in available_models: 
            user_config_models = get_available_models(args.config)
            if args.model not in user_config_models:
                logging.error(f"Selected model '{args.model}' not found in the specified config '{args.config}'. Exiting.")
                exit(1) # Or handle more gracefully
            # Potentially update choices if parser allowed any string due to no initial models
            # This is complex with argparse after parsing. For now, we rely on the initial check or error out.

    try:
        config = load_config(args.config)

        if args.score_only:
            # Re-score mode: no API key needed
            if not os.path.isdir(args.score_only):
                logging.error(f"--score-only path is not a directory: {args.score_only}")
                exit(1)
            rescore_result_dir(args.score_only, config)
        else:
            # Normal benchmark run
            if not args.model:
                parser.error("--model is required unless --score-only is used.")

            # Load API key first - fail fast if not set
            api_key = load_api_key()

            # Apply CLI temperature override
            if args.temperature is not None:
                config["temperature"] = args.temperature

            if args.model not in config.get("openrouter_models", []):
                logging.error(f"The model '{args.model}' is not listed in '{args.config}'. Please check the model name or the config file.")
                exit(1)

            for run_num in range(1, args.num_runs + 1):
                if args.num_runs > 1:
                    logging.info(f"=== Starting run {run_num}/{args.num_runs} ===")
                run_benchmark(
                    config=config,
                    api_key=api_key,
                    model_to_run=args.model,
                    output_dir_override=args.output_dir,
                    exam_name_choice=args.exam_name,
                    exam_year_choice=args.exam_year,
                    question_ids_str=args.question_ids,
                    resume_dir=args.resume if run_num == 1 else None,
                )
    except (ValueError, FileNotFoundError, yaml.YAMLError) as e:
        logging.error(f"Setup failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during benchmark execution: {e}", exc_info=True)
