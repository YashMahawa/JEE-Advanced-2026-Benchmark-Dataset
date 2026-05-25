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
    Robustly parses the LLM response text to extract answers.
    Prioritizes <answer> tags but has several fallbacks for advanced models.
    """
    if not response_text:
        return None

    # 1. Pre-cleaning
    cleaned_text = re.sub(r"<\|[^|]*\|>", "", response_text) # Provider tokens
    cleaned_text = cleaned_text.replace("\u200b", "").strip()

    # 2. Check for SKIP
    if re.search(r"<answer>\s*SKIP\s*</answer>", cleaned_text, re.I) or \
       re.search(r"(?:Final Answer|Answer):\s*SKIP", cleaned_text, re.I) or \
       re.fullmatch(r"\s*SKIP\s*", cleaned_text, re.I):
        return "SKIP"

    # 3. Extract content using multiple strategies
    extracted_content = None
    
    # Strategy A: Standard <answer> tags
    tag_match = re.search(r"<answer>(.*?)</answer>", cleaned_text, re.DOTALL | re.I)
    if tag_match:
        extracted_content = tag_match.group(1).strip()
    
    # Strategy B: "Final Answer: ..." or "Answer: ..." labels
    if not extracted_content:
        label_match = re.search(r"(?:Final\s+)?Answer\s*:\s*(.*)$", cleaned_text, re.I | re.MULTILINE)
        if label_match:
            extracted_content = label_match.group(1).strip()
            # If it's too long, it might just be the start of an explanation
            if len(extracted_content) > 100:
                extracted_content = None

    # Strategy C: Short responses (bare answers)
    if not extracted_content and len(cleaned_text) <= 50:
        extracted_content = cleaned_text

    # Strategy D: Bolded text at the very end (e.g., **(A)** or **A, B**)
    if not extracted_content:
        bold_matches = re.findall(r"\*\*\(?([A-D](?:\s*,\s*[A-D])*)\)?\*\*", cleaned_text)
        if bold_matches:
            extracted_content = bold_matches[-1] # Take the last one

    if not extracted_content:
        logging.warning(f"Could not extract answer content for {question_type} from: {response_text[:100]}...")
        return None

    # 4. Clean the extracted content
    # Remove prefix text like "Option ", "(", ")", "."
    extracted_content = re.sub(r"^(?:Option|The\s+correct\s+answer\s+is|Answer\s+is)\s*:?\s*", "", extracted_content, flags=re.I)
    extracted_content = extracted_content.strip("()[] \t\n\r.")

    # 5. Question-type specific parsing
    parsed_items = []
    
    if question_type == "INTEGER":
        # Handle ranges like "10 to 12" or "10-12"
        range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:to|TO|-)\s*(\d+(?:\.\d+)?)", extracted_content)
        if range_match:
            # For simplicity, we just return the first number, 
            # the evaluator handles ranges if ground truth is a range.
            parsed_items = [range_match.group(1)]
        else:
            # Extract all possible numbers
            nums = re.findall(r"-?\d+(?:\.\d+)?", extracted_content)
            if nums:
                parsed_items = [nums[0]] # Take the first number found

    elif question_type in ["MCQ_SINGLE_CORRECT", "MCQ_MULTIPLE_CORRECT"]:
        # Extract letters A-D or numbers 1-9
        # Handle comma or space separation
        items = re.split(r"[,/;\s]+", extracted_content)
        for item in items:
            clean_item = item.strip("()[] .")
            if re.fullmatch(r"[A-D]", clean_item, re.I):
                parsed_items.append(clean_item.upper())
            elif re.fullmatch(r"[1-9]", clean_item):
                parsed_items.append(clean_item)

    if not parsed_items:
        return None

    # 6. Final Validation
    if question_type in ["MCQ_SINGLE_CORRECT", "INTEGER"]:
        return [parsed_items[0]]
    else:
        return sorted(list(set(parsed_items)))


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
