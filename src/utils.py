import os
import re
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_api_key(key_name="OPENROUTER_API_KEY"):
    """
    Loads the specified API key from environment variables.
    Loads .env file first if it exists.

    Args:
        key_name (str): The name of the environment variable holding the API key.

    Returns:
        str: The API key.

    Raises:
        ValueError: If the API key environment variable is not set.
    """
    load_dotenv() # Load environment variables from .env file if it exists
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(
            f"'{key_name}' environment variable not set. Please create a .env file or set the variable."
        )
    return api_key


def parse_llm_answer(response_text: str, question_type: str = "MCQ_SINGLE_CORRECT") -> list[str] | str | None:
    """
    Parses the LLM response text to extract answers within <answer> tags.
    The parsing logic adapts based on the question_type.

    Handles:
    - MCQ_SINGLE_CORRECT: Single option identifier (integer like "1", "2" or letter like "A", "B").
    - INTEGER: Single numerical value, which can be an integer or a decimal (e.g., "5", "12.75", "0.5").
    - MCQ_MULTIPLE_CORRECT: Multiple option identifiers (integers or letters), comma-separated.
    - The specific string "SKIP" for skipped questions (case-insensitive content within tags).
    - Potential newlines and varied spacing within the tags.

    Args:
        response_text (str): The raw text response from the LLM.
        question_type (str): The type of question, e.g., "MCQ_SINGLE_CORRECT",
                             "MCQ_MULTIPLE_CORRECT", "INTEGER".
                             Defaults to "MCQ_SINGLE_CORRECT".

    Returns:
        list[str] | str | None:
            - A list containing string answer(s) if found and valid.
              (e.g., ["1"], ["A"], ["12.75"], ["A", "C"])
            - The string "SKIP" if the response indicates a skip.
            - None if parsing fails (no tag, invalid content, type mismatch, etc.).
    """
    if not response_text:
        return None

    # Strip provider-specific wrapper tokens (e.g. GLM's <|begin_of_box|>...<|end_of_box|>,
    # Qwen's <|im_*|>, etc.) so the <answer> tag matches even when nested inside them.
    cleaned_text = re.sub(r"<\|[^|]*\|>", "", response_text)

    # Check for exact SKIP response first (case-insensitive for the tag and content)
    # Using regex to be more flexible with whitespace around SKIP
    skip_match = re.search(r"<answer>\s*SKIP\s*</answer>", cleaned_text, re.IGNORECASE)
    if skip_match or re.fullmatch(r"\s*SKIP\s*", cleaned_text, re.IGNORECASE):
        logging.info(f"Parsed answer as SKIP for question_type: {question_type}.")
        return "SKIP"

    match = re.search(r"<answer>(.*?)</answer>", cleaned_text, re.DOTALL | re.IGNORECASE)

    # Fallback: model emitted a bare answer with no <answer> tag — accept it only when
    # the (cleaned) response is short and looks like nothing but the answer itself.
    # Prevents false positives from long explanations that happen to contain a digit.
    if not match:
        bare = cleaned_text.strip()
        if bare and len(bare) <= 30:
            extracted_content = bare
        else:
            logging.warning(f"Could not find <answer> tag in response for question_type: {question_type} in response: '{response_text[:200]}...'")
            return None
    else:
        extracted_content = match.group(1).strip()
    if not extracted_content:
        logging.warning(f"Found <answer> tag but content is empty for question_type: {question_type}.")
        return None

    potential_answers_str_list = [item.strip() for item in extracted_content.split(',')]
    parsed_answers_final = []

    for ans_str_raw in potential_answers_str_list:
        ans_str = ans_str_raw.strip()
        if not ans_str: # Skip empty strings that might result from "1, ,2" or trailing commas
            continue

        if question_type == "INTEGER":
            try:
                # Try to parse as float to validate it's a number (integer or decimal).
                # The value is kept as a string to preserve original formatting (e.g., "0.50", "5.0")
                # for exact string comparison with ground truth if needed, and for consistent type handling.
                float(ans_str) 
                parsed_answers_final.append(ans_str)
            except ValueError:
                logging.warning(f"Could not parse '{ans_str}' as a valid number (integer or decimal) for INTEGER type. Full content: '{extracted_content}'")
                return None # If any part is not a valid number for INTEGER type, fail parsing.
        
        elif question_type in ["MCQ_SINGLE_CORRECT", "MCQ_MULTIPLE_CORRECT"]:
            # For MCQs, accept single-digit numbers (1-9) or letters A-D.
            if re.fullmatch(r"[1-9]", ans_str): # Single digit options
                parsed_answers_final.append(ans_str)
            elif re.fullmatch(r"[a-dA-D]", ans_str): # Letter options A-D
                parsed_answers_final.append(ans_str.upper())
            else:
                logging.warning(f"Could not parse '{ans_str}' as a valid MCQ option (expected 1-9 or A-D). Full content: '{extracted_content}'")
                return None
        else: # Should not happen if question_type is validated before calling
            logging.error(f"Unknown question_type '{question_type}' encountered in parse_llm_answer logic.")
            return None


    if not parsed_answers_final: # If list is empty after processing (e.g. content was just commas)
        logging.warning(f"No valid answer items found after parsing content: '{extracted_content}' for question_type: {question_type}.")
        return None

    # Apply rules based on question_type for number of answers
    if question_type in ["MCQ_SINGLE_CORRECT", "INTEGER"]:
        if len(parsed_answers_final) == 1:
            return parsed_answers_final # Returns list[str] with one element
        else:
            logging.warning(f"Expected single answer for {question_type}, but found {len(parsed_answers_final)} items: {parsed_answers_final}. Content: '{extracted_content}'")
            return None
    elif question_type == "MCQ_MULTIPLE_CORRECT":
        # For multiple correct, any number of valid items is acceptable.
        # Return them sorted and unique.
        return sorted(list(set(parsed_answers_final)))
    else:
        # This case should ideally be caught by earlier checks or input validation.
        logging.error(f"Unknown question_type '{question_type}' provided to parse_llm_answer at final stage.")
        return None

# Example Usage (for testing)
if __name__ == '__main__':
    test_cases = [
        # MCQ_SINGLE_CORRECT (can be number or letter)
        {"resp": "<answer>2</answer>", "type": "MCQ_SINGLE_CORRECT", "expected": ["2"]},
        {"resp": "<answer>B</answer>", "type": "MCQ_SINGLE_CORRECT", "expected": ["B"]},
        {"resp": "<answer> c </answer>", "type": "MCQ_SINGLE_CORRECT", "expected": ["C"]},
        {"resp": "<answer>1,3</answer>", "type": "MCQ_SINGLE_CORRECT", "expected": None}, # Fail: multiple for single
        {"resp": "<answer>A,C</answer>", "type": "MCQ_SINGLE_CORRECT", "expected": None}, # Fail: multiple for single
        {"resp": "<answer>X</answer>", "type": "MCQ_SINGLE_CORRECT", "expected": None}, # Fail: not A-D
        {"resp": "<answer>5</answer>", "type": "MCQ_SINGLE_CORRECT", "expected": ["5"]}, # Accept single digits 1-9

        # INTEGER (can be int or decimal string)
        {"resp": "<answer>42</answer>", "type": "INTEGER", "expected": ["42"]},
        {"resp": "<answer>0</answer>", "type": "INTEGER", "expected": ["0"]},
        {"resp": "<answer>12.75</answer>", "type": "INTEGER", "expected": ["12.75"]},
        {"resp": "<answer>0.5</answer>", "type": "INTEGER", "expected": ["0.5"]},
        {"resp": "<answer>-5</answer>", "type": "INTEGER", "expected": ["-5"]}, # Assuming negative integers are valid if problem allows
        {"resp": "<answer>5.00</answer>", "type": "INTEGER", "expected": ["5.00"]},
        {"resp": "<answer>abc</answer>", "type": "INTEGER", "expected": None}, # Fail: not a number
        {"resp": "<answer>1,2</answer>", "type": "INTEGER", "expected": None}, # Fail: multiple for single int

        # MCQ_MULTIPLE_CORRECT (can be numbers or letters, mixed is not typical but parser allows if items are valid individually)
        {"resp": "<answer>1,3</answer>", "type": "MCQ_MULTIPLE_CORRECT", "expected": ["1", "3"]},
        {"resp": "<answer>A,C</answer>", "type": "MCQ_MULTIPLE_CORRECT", "expected": ["A", "C"]},
        {"resp": "<answer> b , d </answer>", "type": "MCQ_MULTIPLE_CORRECT", "expected": ["B", "D"]},
        {"resp": "<answer>2</answer>", "type": "MCQ_MULTIPLE_CORRECT", "expected": ["2"]}, # Single is valid
        {"resp": "<answer>D</answer>", "type": "MCQ_MULTIPLE_CORRECT", "expected": ["D"]}, # Single is valid
        {"resp": "<answer>1,C,3</answer>", "type": "MCQ_MULTIPLE_CORRECT", "expected": ["1", "3", "C"]}, # Mixed, sorted
        {"resp": "<answer>C,1,A,1</answer>", "type": "MCQ_MULTIPLE_CORRECT", "expected": ["1", "A", "C"]}, # Unique and sorted
        {"resp": "<answer>1,X,3</answer>", "type": "MCQ_MULTIPLE_CORRECT", "expected": None}, # Fail: X not in A-D

        # SKIP and general failures
        {"resp": "<ANSWER>SKIP</ANSWER>", "type": "MCQ_SINGLE_CORRECT", "expected": "SKIP"},
        {"resp": "  <answer>  SKIP  </answer>  ", "type": "INTEGER", "expected": "SKIP"},
        {"resp": "No answer tag here.", "type": "MCQ_SINGLE_CORRECT", "expected": None},
        {"resp": "<answer></answer>", "type": "MCQ_SINGLE_CORRECT", "expected": None},
        {"resp": "<answer>  </answer>", "type": "MCQ_SINGLE_CORRECT", "expected": None},
        {"resp": None, "type": "MCQ_SINGLE_CORRECT", "expected": None},
        {"resp": "", "type": "MCQ_SINGLE_CORRECT", "expected": None},
        {"resp": "<answer>5</answer>", "type": "UNKNOWN_TYPE", "expected": None}, # Unknown type
        {"resp": "<answer>1,,2</answer>", "type": "MCQ_MULTIPLE_CORRECT", "expected": ["1", "2"]}, # Handles empty item from double comma
    ]

    print("\n--- Testing parse_llm_answer (revised) ---")
    all_passed = True
    for i, case in enumerate(test_cases):
        parsed = parse_llm_answer(case["resp"], case["type"])
        if parsed == case["expected"]:
            print(f"Test {i+1} PASSED: Response: '{str(case['resp'])[:50]}...', Type: {case['type']} -> Parsed: {parsed}")
        else:
            print(f"Test {i+1} FAILED: Response: '{str(case['resp'])[:50]}...', Type: {case['type']} -> Parsed: {parsed} (Expected: {case['expected']})")
            all_passed = False
    
    if all_passed:
        print("\nAll revised parse_llm_answer tests passed!")
    else:
        print("\nSome revised parse_llm_answer tests FAILED.")

    # Test API key loading (will raise error if .env or env var not set)
    # try:
    #     key = load_api_key()
    #     print(f"\nAPI Key loaded successfully (partially hidden): {key[:4]}...{key[-4:]}")
    # except ValueError as e:
    #     print(f"\nError loading API key: {e}")
