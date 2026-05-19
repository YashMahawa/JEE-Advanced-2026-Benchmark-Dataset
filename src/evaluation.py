import logging
import re
from typing import List, Optional, Union, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_subject_as_section(subject: str, question_num_for_log: int) -> Optional[str]:
    """
    Returns the subject name directly as the section identifier.
    question_num_for_log is only used for logging context if subject is invalid.
    """
    if subject and isinstance(subject, str) and subject.strip():
        return subject.strip()
    else:
        logging.warning(f"Invalid or missing subject ('{subject}') for question_num '{question_num_for_log}'. Cannot determine section.")
        return None

def is_within_range(predicted_value_str: str, lower_bound_str: str, upper_bound_str: str) -> bool:
    """
    Checks if a predicted numerical value (as a string) falls within a specified range.
    The comparison is inclusive.
    """
    try:
        predicted_value = float(predicted_value_str)
        lower_bound = float(lower_bound_str)
        upper_bound = float(upper_bound_str)
    except ValueError:
        logging.debug(f"Could not convert predicted value '{predicted_value_str}' or bounds ('{lower_bound_str}', '{upper_bound_str}') to numbers.")
        return False

    return lower_bound <= predicted_value <= upper_bound


def calculate_single_question_score_details(result_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates marks_awarded and evaluation_status for a single question result.

    Args:
        result_item (Dict[str, Any]): A dictionary for a single question, must contain:
            'question_id' (str)
            'exam_name' (str)
            'question_type' (str)
            'ground_truth' (List[str] | str)  # Changed to string
            'predicted_answer' (List[str] | str | None) # Changed to string
            'api_call_successful' (bool)

    Returns:
        Dict[str, Any]: A dictionary with 'marks_awarded' (int) and 'evaluation_status' (str).
    """
    question_id = result_item.get("question_id", "UNKNOWN_QID") # Provide default for logging
    exam_name = result_item.get("exam_name", "").upper()
    question_type = result_item.get("question_type", "").upper()
    pred = result_item.get("predicted_answer")
    truth = result_item.get("ground_truth")
    api_success = result_item.get("api_call_successful", False)

    current_score_change = 0
    evaluation_status = "unknown"

    # Ensure truth is a set of uppercase strings for consistent processing.
    # Ground truth from metadata.jsonl is expected to be a list of strings.
    # e.g., ["1"], ["A"], ["12.75"], ["A", "C"]
    # For integer ranges, it will be like [["0.7", "0.8"]]
    truth_processed: List[Union[str, List[str]]] = []
    if isinstance(truth, list):
        for t_item in truth:
            if isinstance(t_item, str):
                truth_processed.append(t_item.upper())
            elif isinstance(t_item, list) and len(t_item) == 2 and all(isinstance(x, str) for x in t_item):
                truth_processed.append([x.upper() for x in t_item]) # Store range as list of uppercase strings
            else:
                logging.error(f"Invalid item in ground_truth list for {question_id}: {t_item}. Skipping.")
    elif isinstance(truth, str):
        truth_processed.append(truth.upper())
    else:
        logging.error(f"Invalid ground_truth format for {question_id}: {truth} (type: {type(truth)}). Assigning 0 marks.")
        return {"marks_awarded": 0, "evaluation_status": "error_bad_ground_truth"}


    if not api_success or pred is None: # pred is None means our internal parsing failed
        evaluation_status = "failure_api_or_parse"
        current_score_change = 0  # No penalty: parse/API failure is not a deliberate wrong choice
    elif isinstance(pred, str) and pred.upper() == "SKIP": # Standardize SKIP comparison
        current_score_change = 0
        evaluation_status = "skipped"
    elif isinstance(pred, list) and all(isinstance(p, str) for p in pred):
        pred_set = {p.upper() for p in pred} # Convert to uppercase strings
        
        # Handle MCQ_SINGLE_CORRECT first, as it has special logic for multiple truths.
        # The parser (`parse_llm_answer`) returns `pred` as `list[str]` with one element for valid single answers.
        if question_type == "MCQ_SINGLE_CORRECT":
            # A prediction is correct if its single element is present in the truth_set.
            # This accommodates metadata entries where `correct_answer` for an MCQ_SINGLE_CORRECT
            # might list multiple acceptable options (e.g., if a question had two official correct answers).
            is_correct = False
            if len(pred_set) == 1: # Ensure prediction is indeed a single option
                single_pred_answer = list(pred_set)[0] # Get the single predicted option
                # Check against all processed truths (which are single strings for MCQ)
                if single_pred_answer in truth_processed: 
                    is_correct = True
            
            if is_correct:
                evaluation_status = "correct"
                if exam_name == "NEET": current_score_change = 4
                elif exam_name == "JEE_MAIN": current_score_change = 4
                elif exam_name == "JEE_ADVANCED": current_score_change = 3
                else: current_score_change = 1 # Default positive score for unknown exam
            else:
                evaluation_status = "incorrect"
                if exam_name == "NEET": current_score_change = -1
                elif exam_name == "JEE_MAIN": current_score_change = -1
                elif exam_name == "JEE_ADVANCED": current_score_change = -1
                else: current_score_change = 0 # Default no penalty

        elif exam_name == "JEE_MAIN" and question_type == "INTEGER": # Integer answers are now strings in a list e.g. ["14"]
            # For JEE_MAIN INTEGER, we expect truth_processed to contain single strings
            is_correct = False
            if len(pred_set) == 1:
                predicted_answer_str = list(pred_set)[0]
                if predicted_answer_str in truth_processed: # Check against single string truths
                    is_correct = True
            
            if is_correct:
                current_score_change = 4; evaluation_status = "correct"
            else:
                current_score_change = 0; evaluation_status = "incorrect"
        
        elif exam_name == "JEE_ADVANCED":
            # Note: MCQ_SINGLE_CORRECT for JEE_ADVANCED is handled by the common block above
            if question_type == "INTEGER":
                is_correct = False
                if len(pred_set) == 1:
                    predicted_answer_str = list(pred_set)[0] # Get the single predicted string

                    # Iterate through each ground truth entry in the 'truth_processed' list
                    for gt_entry in truth_processed:
                        if isinstance(gt_entry, list) and len(gt_entry) == 2: # This is a range [lower, upper]
                            lower_bound_str, upper_bound_str = gt_entry[0], gt_entry[1]
                            if is_within_range(predicted_answer_str, lower_bound_str, upper_bound_str):
                                is_correct = True
                                break # Found a matching range, no need to check others
                        elif isinstance(gt_entry, str): # This is an exact integer match
                            if predicted_answer_str == gt_entry: # gt_entry is already uppercase
                                is_correct = True
                                break # Found an exact match, no need to check others
                
                if is_correct:
                    current_score_change = 4; evaluation_status = "correct"
                else:
                    current_score_change = 0; evaluation_status = "incorrect"
            elif question_type == "MCQ_MULTIPLE_CORRECT":
                # For MCQ_MULTIPLE_CORRECT, truth_processed contains single strings
                truth_set_mcq = set(truth_processed) # Convert to set for intersection operations

                num_correct_options_in_truth = len(truth_set_mcq)
                num_chosen_options = len(pred_set)
                correct_chosen_options = pred_set.intersection(truth_set_mcq)
                incorrect_chosen_options = pred_set.difference(truth_set_mcq)
                num_correct_chosen = len(correct_chosen_options)
                num_incorrect_chosen = len(incorrect_chosen_options)

                if num_incorrect_chosen > 0:
                    current_score_change = -2; evaluation_status = "incorrect_negative"
                elif num_correct_chosen == num_correct_options_in_truth and num_chosen_options == num_correct_options_in_truth:
                    current_score_change = 4; evaluation_status = "correct_full"
                elif num_correct_options_in_truth == 4 and num_correct_chosen == 3 and num_chosen_options == 3:
                    current_score_change = 3; evaluation_status = "partial_3_of_4"
                elif num_correct_options_in_truth >= 3 and num_correct_chosen == 2 and num_chosen_options == 2:
                    current_score_change = 2; evaluation_status = "partial_2_of_3_plus"
                elif num_correct_options_in_truth >= 2 and num_correct_chosen == 1 and num_chosen_options == 1:
                    current_score_change = 1; evaluation_status = "partial_1_of_2_plus"
                else:
                    current_score_change = 0; evaluation_status = "no_marks_no_penalty"
        else:
            logging.warning(f"Unknown exam_name/question_type combination for scoring: {exam_name}/{question_type} for QID {question_id}. Assigning 0 marks.")
            current_score_change = 0
            evaluation_status = "unknown_exam_type"
    else:
        logging.error(f"Unexpected prediction type for {question_id}: {pred}. Treating as API/Parse Failure.")
        current_score_change = 0
        evaluation_status = "failure_unexpected_type"

    return {"marks_awarded": current_score_change, "evaluation_status": evaluation_status}


def calculate_max_score_for_question(exam_name: str, question_type: str) -> int:
    """
    Returns the maximum possible score for a given exam and question type.
    """
    exam_name = exam_name.upper()
    question_type = question_type.upper()

    if exam_name == "NEET" and question_type == "MCQ_SINGLE_CORRECT":
        return 4
    elif exam_name == "JEE_MAIN":
        if question_type == "MCQ_SINGLE_CORRECT":
            return 4
        elif question_type == "INTEGER":
            return 4
    elif exam_name == "JEE_ADVANCED":
        if question_type == "MCQ_SINGLE_CORRECT":
            return 3
        elif question_type == "INTEGER":
            return 4
        elif question_type == "MCQ_MULTIPLE_CORRECT":
            return 4 # Max score for multiple correct is 4
    return 0 # Default for unknown types


def calculate_exam_scores(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates exam scores based on exam_name and question_type, providing section-wise breakdown
    and detailed question type statistics.

    Args:
        results (List[Dict[str, Any]]): A list of result dictionaries. Each dict must contain:
            'question_id' (str)
            'subject' (str)
            'exam_name' (str) e.g., "NEET", "JEE_MAIN", "JEE_ADVANCED"
            'question_type' (str) e.g., "MCQ_SINGLE_CORRECT", "MCQ_MULTIPLE_CORRECT", "INTEGER"
            'ground_truth' (List[str] | str): Correct answer(s). For INTEGER, it's a single str.
            'predicted_answer' (List[str] | str | None): Model's prediction.
            'api_call_successful' (bool): Whether the API call succeeded.
            This list will be modified in-place to add 'evaluation_status' and 'marks_awarded' by calling
            calculate_single_question_score_details for each item.
    Returns:
        Dict[str, Any]: A dictionary containing overall and section-wise scores and counts,
                        plus question type breakdowns and total possible score.
    """
    if not results:
        return {"error": "No results provided."}

    overall_stats = {
        "score": 0,
        "correct": 0,
        "incorrect": 0,
        "skipped": 0,
        "api_parse_failures": 0,
        "partial_correct": 0,
        "total_possible_score": 0 # New field
    }
    
    # Initialize question type breakdown
    question_type_breakdown: Dict[str, Dict[str, Any]] = {}

    valid_subjects_from_data = [r.get("subject") for r in results if r.get("subject") and isinstance(r.get("subject"), str) and r.get("subject").strip()]
    if not valid_subjects_from_data and results:
        logging.warning("No valid subjects found in results data to initialize section_stats.")
    
    unique_subjects = sorted(list(set(s.strip() for s in valid_subjects_from_data)))
    section_stats = {
        subj: {"score": 0, "correct": 0, "incorrect": 0, "skipped": 0, "api_parse_failures": 0, "partial_correct": 0}
        for subj in unique_subjects
    }
    if not unique_subjects and results:
        logging.warning("section_stats is empty because no unique, valid subjects were found.")
    
    unmapped_section_questions = 0

    for result in results:
        question_id = result.get("question_id") # For logging within loop
        subject = result.get("subject") # For section mapping
        exam_name = result.get("exam_name", "").upper()
        question_type = result.get("question_type", "").upper()

        # Calculate score details for the single question
        score_details = calculate_single_question_score_details(result)
        current_score_change = score_details.get("marks_awarded", 0)
        evaluation_status = score_details.get("evaluation_status", "unknown_error_in_scoring")

        # Update the result dictionary in-place (as original function did)
        result['evaluation_status'] = evaluation_status
        result['marks_awarded'] = current_score_change
        
        # Accumulate total possible score
        overall_stats["total_possible_score"] += calculate_max_score_for_question(exam_name, question_type)

        # Determine boolean flags based on evaluation_status for aggregation
        is_correct_full = evaluation_status in ["correct", "correct_full"]
        is_partial_correct = evaluation_status.startswith("partial_")
        is_incorrect_choice = evaluation_status in ["incorrect", "incorrect_negative"]
        is_skipped = evaluation_status == "skipped"
        is_api_parse_failure = evaluation_status in ["failure_api_or_parse", "failure_unexpected_type", "error_bad_ground_truth"]


        overall_stats["score"] += current_score_change
        if is_correct_full: overall_stats["correct"] += 1
        if is_incorrect_choice: overall_stats["incorrect"] += 1
        if is_skipped: overall_stats["skipped"] += 1
        if is_api_parse_failure: overall_stats["api_parse_failures"] += 1
        if is_partial_correct: overall_stats["partial_correct"] +=1
        
        # Aggregate by question type
        if question_type not in question_type_breakdown:
            question_type_breakdown[question_type] = {
                "count": 0,
                "score": 0,
                "correct_full": 0,
                "partial_correct": 0,
                "incorrect_choice": 0,
                "skipped": 0,
                "api_parse_failures": 0,
                "max_score_per_question": calculate_max_score_for_question(exam_name, question_type)
            }
        
        q_type_stats = question_type_breakdown[question_type]
        q_type_stats["count"] += 1
        q_type_stats["score"] += current_score_change
        if is_correct_full: q_type_stats["correct_full"] += 1
        if is_incorrect_choice: q_type_stats["incorrect_choice"] += 1
        if is_skipped: q_type_stats["skipped"] += 1
        if is_api_parse_failure: q_type_stats["api_parse_failures"] += 1
        if is_partial_correct: q_type_stats["partial_correct"] += 1

        section = None
        if subject:
            question_num_for_log = -1 # Placeholder, as QID might not have number
            if question_id:
                match_num = re.search(r'_(\d+)$', question_id)
                if match_num:
                    try: question_num_for_log = int(match_num.group(1))
                    except ValueError: pass
            section = get_subject_as_section(subject, question_num_for_log)

        if section and section in section_stats:
            section_stats[section]["score"] += current_score_change
            if is_correct_full: section_stats[section]["correct"] += 1
            if is_incorrect_choice: section_stats[section]["incorrect"] += 1
            if is_skipped: section_stats[section]["skipped"] += 1
            if is_api_parse_failure: section_stats[section]["api_parse_failures"] += 1
            if is_partial_correct: section_stats[section]["partial_correct"] +=1
        elif section is None and not is_api_parse_failure : # only count as unmapped if not already an API/parse failure (which might lack subject)
            unmapped_section_questions += 1
            # logging.warning(f"Could not map question to section: ID={question_id}, Subject={subject}") # Already logged by get_subject_as_section if subject is bad

    logging.info(f"Exam Score Calculation Complete. Overall Score: {overall_stats['score']}")
    if unmapped_section_questions > 0:
        logging.warning(f"{unmapped_section_questions} questions could not be mapped to a section.")

    return {
        "overall_score": overall_stats["score"],
        "overall_correct_full": overall_stats["correct"],
        "overall_partial_correct": overall_stats["partial_correct"],
        "overall_incorrect_choice": overall_stats["incorrect"],
        "overall_skipped": overall_stats["skipped"],
        "overall_api_parse_failures": overall_stats["api_parse_failures"],
        "total_questions_processed": len(results),
        "total_possible_score_for_processed_questions": overall_stats["total_possible_score"], # New field
        "unmapped_section_questions": unmapped_section_questions,
        "section_breakdown": section_stats,
        "question_type_breakdown": question_type_breakdown # New field
    }


# Example Usage (for testing)
if __name__ == '__main__':
    print("Running evaluation tests...")

    # --- Test calculate_exam_scores (now with strings) ---
    print("\n--- Testing calculate_exam_scores ---")
    test_results_exam = [
        # NEET - Answers as strings "1", "2", "A", "B" etc.
        {"question_id": "N001", "subject": "Physics", "exam_name": "NEET", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["1"], "predicted_answer": ["1"], "api_call_successful": True}, # Correct +4
        {"question_id": "N002", "subject": "Physics", "exam_name": "NEET", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["D"], "predicted_answer": ["B"], "api_call_successful": True}, # Incorrect -1
        {"question_id": "N003", "subject": "Chemistry", "exam_name": "NEET", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["4"], "predicted_answer": "SKIP", "api_call_successful": True}, # Skipped 0
        {"question_id": "N004", "subject": "Chemistry", "exam_name": "NEET", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["C"], "predicted_answer": None, "api_call_successful": False}, # API Fail -1
        {"question_id": "N005", "subject": "Botany", "exam_name": "NEET", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["A"], "predicted_answer": None, "api_call_successful": True}, # Parse Fail -1

        # JEE Main - MCQ - Answers as strings "1", "2", "A", "B" etc.
        {"question_id": "JM001", "subject": "Maths", "exam_name": "JEE_MAIN", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["2"], "predicted_answer": ["2"], "api_call_successful": True}, # Correct +4
        {"question_id": "JM002", "subject": "Maths", "exam_name": "JEE_MAIN", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["C"], "predicted_answer": ["a"], "api_call_successful": True}, # Incorrect -1 (C vs a)
        # JEE Main - Integer - Answers as strings e.g., "5", "10"
        {"question_id": "JM003", "subject": "Physics", "exam_name": "JEE_MAIN", "question_type": "INTEGER", "ground_truth": ["5"], "predicted_answer": ["5"], "api_call_successful": True}, # Correct +4
        {"question_id": "JM004", "subject": "Physics", "exam_name": "JEE_MAIN", "question_type": "INTEGER", "ground_truth": ["10"], "predicted_answer": ["8"], "api_call_successful": True}, # Incorrect 0
        {"question_id": "JM005", "subject": "Chemistry", "exam_name": "JEE_MAIN", "question_type": "INTEGER", "ground_truth": ["7"], "predicted_answer": None, "api_call_successful": True}, # Parse Fail 0

        # JEE Advanced - MCQ Single Correct - Answers as strings "A", "B" etc.
        {"question_id": "JA001", "subject": "Maths", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["A"], "predicted_answer": ["a"], "api_call_successful": True}, # Correct +3 (case-insensitive)
        {"question_id": "JA002", "subject": "Maths", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_SINGLE_CORRECT", "ground_truth": ["B"], "predicted_answer": ["C"], "api_call_successful": True}, # Incorrect -1
        # JEE Advanced - Integer - Answers as strings e.g., "12", "0"
        {"question_id": "JA003", "subject": "Physics", "exam_name": "JEE_ADVANCED", "question_type": "INTEGER", "ground_truth": ["12"], "predicted_answer": ["12"], "api_call_successful": True}, # Correct +4
        {"question_id": "JA004", "subject": "Physics", "exam_name": "JEE_ADVANCED", "question_type": "INTEGER", "ground_truth": ["0"], "predicted_answer": ["1"], "api_call_successful": True}, # Incorrect 0
        # JEE Advanced - MCQ Multiple Correct - Answers as strings "A", "C" etc.
        {"question_id": "JA005", "subject": "Chemistry", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "C"], "predicted_answer": ["a", "c"], "api_call_successful": True}, # All Correct +4
        {"question_id": "JA006", "subject": "Chemistry", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "B", "C"], "predicted_answer": ["a", "b"], "api_call_successful": True}, # Partial +2 (3 correct, 2 chosen)
        {"question_id": "JA007", "subject": "Chemistry", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "B", "C", "D"], "predicted_answer": ["a", "b", "c"], "api_call_successful": True}, # Partial +3 (4 correct, 3 chosen)
        {"question_id": "JA008", "subject": "Chemistry", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "B"], "predicted_answer": ["a"], "api_call_successful": True}, # Partial +1 (2 correct, 1 chosen)
        {"question_id": "JA009", "subject": "Chemistry", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "C"], "predicted_answer": ["a", "b"], "api_call_successful": True}, # Incorrect option chosen -2
        {"question_id": "JA010", "subject": "Chemistry", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "C"], "predicted_answer": ["b", "d"], "api_call_successful": True}, # All incorrect options chosen -2
        {"question_id": "JA011", "subject": "Chemistry", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A", "C"], "predicted_answer": "SKIP", "api_call_successful": True}, # Skipped 0
        {"question_id": "JA012", "subject": "Maths", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A"], "predicted_answer": ["a"], "api_call_successful": True}, # Single correct in multi-choice, full marks +4
        {"question_id": "JA013", "subject": "Physics", "exam_name": "JEE_ADVANCED", "question_type": "MCQ_MULTIPLE_CORRECT", "ground_truth": ["A","B","C"], "predicted_answer": ["a","d"], "api_call_successful": True}, # One correct, one incorrect -> -2
    ]

    exam_summary = calculate_exam_scores(test_results_exam)
    print("\nExam Score Summary:")
    import json
    print(json.dumps(exam_summary, indent=2, sort_keys=True))

    # Basic assertions - can be expanded
    assert exam_summary["overall_score"] == (4-1+0+0+0) + (4-1) + (4+0+0) + (3-1) + (4+0) + (4+2+3+1-2-2+0+4-2)
    assert exam_summary["overall_correct_full"] == 7
    assert exam_summary["overall_partial_correct"] == 3
    assert exam_summary["overall_incorrect_choice"] == 8
    assert exam_summary["overall_skipped"] == 2
    assert exam_summary["overall_api_parse_failures"] == 3 # N004, N005, JM005

    assert exam_summary["section_breakdown"]["Physics"]["score"] == (4-1) + (4+0) + (4+0) - 2 # N001,N002 + JM003,JM004 + JA003,JA004 + JA013
    assert exam_summary["section_breakdown"]["Chemistry"]["score"] == (0+0) + (0) + (4+2+3+1-2-2+0) # N003,N004 + JM005 + JA005-JA011
    assert exam_summary["section_breakdown"]["Botany"]["score"] == 0 # N005 (parse fail = 0 penalty)
    assert exam_summary["section_breakdown"]["Maths"]["score"] == (4-1) + (3-1) + 4 # JM001,JM002 + JA001,JA002 + JA012

    print("\nEvaluation tests completed.")
