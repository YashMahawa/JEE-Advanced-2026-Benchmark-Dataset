import base64
import requests
import time
import logging
from io import BytesIO
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils import parse_llm_answer
from prompts import (
    INITIAL_PROMPT_TEMPLATE,
    REPROMPT_PROMPT_TEMPLATE,
    get_answer_format_instruction,
    get_example_instruction,
    get_specific_instructions_reprompt,
    get_reprompt_example_instruction
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenRouter API endpoint
OPENROUTER_API_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Define exceptions for retry logic
RETRYABLE_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.RequestException # Catch broader request errors for retries
)

# Define status codes that warrant a retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Retry decorator configuration
retry_config = dict(
    stop=stop_after_attempt(3),  # Retry up to 3 times
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff: 2s, 4s, 8s...
    retry=(retry_if_exception_type(RETRYABLE_EXCEPTIONS)) # Retry on specific exceptions
    # We will handle status code retries manually within the function for more control
)

def encode_image_to_base64(image: Image.Image) -> str:
    """Encodes a PIL Image to a base64 PNG string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def construct_reprompt_prompt(previous_raw_response: str, question_type: str) -> list:
    """Constructs the message list for a re-prompt API call based on question_type."""
    specific_instructions = get_specific_instructions_reprompt(question_type)
    reprompt_example_instruction = get_reprompt_example_instruction(question_type)

    prompt_text = REPROMPT_PROMPT_TEMPLATE.format(
        previous_raw_response=previous_raw_response,
        question_type=question_type,
        specific_instructions=specific_instructions,
        reprompt_example_instruction=reprompt_example_instruction
    )
    messages = [{"role": "user", "content": prompt_text}]
    return messages


def construct_initial_prompt(base64_image: str, exam_name: str, exam_year: str, question_type: str) -> list:
    """Constructs the initial message list with image for the OpenRouter API call, tailored by question_type."""
    
    answer_format_instruction = get_answer_format_instruction(question_type)
    example_instruction = get_example_instruction(question_type)

    prompt_text = INITIAL_PROMPT_TEMPLATE.format(
        exam_name=exam_name,
        exam_year=exam_year,
        question_type=question_type,
        answer_format_instruction=answer_format_instruction,
        example_instruction=example_instruction
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    return messages

@retry(**retry_config)
def get_openrouter_prediction(
    model_identifier: str,
    api_key: str,
    image: Image.Image | None = None, # Image is now optional
    previous_raw_response: str | None = None, # Added for re-prompting
    exam_name: str | None = None,
    exam_year: str | None = None,
    question_type: str = "MCQ_SINGLE_CORRECT", # New parameter with default
    max_tokens: int = 100,
    request_timeout: int = 60,
    temperature: float = 0.0,
    reasoning_effort: str = "high",
) -> tuple[list[str] | str | None, str | None, dict | None]:
    """
    Gets a prediction from an OpenRouter model. Handles initial image prompts and text-only re-prompts.

    Args:
        model_identifier (str): The OpenRouter model identifier (e.g., "openai/gpt-5.5").
        api_key (str): The OpenRouter API key.
        image (Image.Image | None): The question image (for initial prompt). Default None.
        previous_raw_response (str | None): The raw response from a previous failed parse attempt (for re-prompt). Default None.
        exam_name (str | None): The name of the exam (e.g., "NEET", "JEE"). Required if 'image' is provided for initial prompt.
        exam_year (str | None): The year of the exam. Required if 'image' is provided for initial prompt.
        question_type (str): Type of question, e.g., "MCQ_SINGLE_CORRECT", "MCQ_MULTIPLE_CORRECT", "INTEGER".
        max_tokens (int): Max tokens for the response.
        request_timeout (int): Timeout for the API request in seconds.
        temperature (float): Sampling temperature. 0.0 for deterministic output.

    Returns:
        tuple[list[str] | str | None, str | None, dict | None]: A tuple containing:
            - The parsed answer as a list of strings, the string "SKIP", or None if failed.
            - The raw response text from the LLM (or None if API call failed).
            - Response metadata dict (tokens, latency, model version, cost) or None.

    Raises:
        ValueError: If arguments are inconsistent (e.g., image provided without exam details for initial prompt).
        requests.exceptions.RequestException: If the API call fails after retries.
    """
    logging.info(f"Requesting prediction from model: {model_identifier} for question_type: {question_type}")

    if image is not None and previous_raw_response is None:
        # Initial prompt with image
        if not exam_name or not exam_year: # exam_name and exam_year are crucial for initial prompt context
            raise ValueError("'exam_name' and 'exam_year' must be provided when 'image' is specified for an initial prompt.")
        logging.debug(f"Constructing initial prompt with image for {exam_name} {exam_year}, type: {question_type}.")
        base64_image = encode_image_to_base64(image)
        messages = construct_initial_prompt(base64_image, exam_name, exam_year, question_type)
    elif image is None and previous_raw_response is not None:
        # Re-prompt based on previous response
        logging.debug(f"Constructing re-prompt based on previous response for type: {question_type}.")
        messages = construct_reprompt_prompt(previous_raw_response, question_type)
    else:
        # This condition means either both image and previous_raw_response are None, or both are provided.
        # The latter (both provided) is ambiguous for which prompt to use.
        # The former (both None) means no input to act on.
        raise ValueError("Provide 'image' (with 'exam_name' and 'exam_year') for an initial call, OR 'previous_raw_response' for a re-prompt. Not neither or both.")


    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        is_claude = model_identifier.startswith("anthropic/")
        supports_reasoning_effort = model_identifier.startswith(("google/", "qwen/", "z-ai/"))
        data = {
            "model": model_identifier,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 1 if is_claude else temperature,
            "usage": {"include": True},
        }
        if is_claude:
            # Extended thinking requires temperature=1 and an explicit budget.
            # Reserve 1000 tokens for the answer; the rest goes to thinking.
            data["thinking"] = {"type": "enabled", "budget_tokens": max(1000, max_tokens - 1000)}
        elif supports_reasoning_effort:
            # OpenRouter's reasoning.effort, mapped per provider:
            # - Google Gemini 3: thinking_level (minimal/low/medium/high)
            # - Qwen3-VL Thinking: reasoning toggle
            # - z-ai GLM-V: reasoning toggle
            data["reasoning"] = {"effort": reasoning_effort}

        request_start_time = time.time()
        response = requests.post(
            OPENROUTER_API_ENDPOINT,
            headers=headers,
            json=data,
            timeout=request_timeout
        )
        response_latency_ms = round((time.time() - request_start_time) * 1000)

        if response.status_code in RETRYABLE_STATUS_CODES:
            logging.warning(f"Received retryable status code {response.status_code} from {model_identifier} for {question_type}. Retrying might occur.")
            response.raise_for_status()

        if not response.ok:
            logging.error(f"API Error for model {model_identifier} ({question_type}): Status {response.status_code} - {response.text}")
            return None, None, None

        response_json = response.json()
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content")
        # Claude with extended thinking returns a list of blocks; extract the text block.
        if isinstance(content, list):
            text_parts = [b.get("text", "") for b in content if b.get("type") == "text"]
            raw_response_text = "\n".join(text_parts) if text_parts else None
        else:
            raw_response_text = content

        # Extract response metadata
        usage = response_json.get("usage", {})
        response_metadata = {
            "generation_id": response_json.get("id"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost": usage.get("cost"),
            "response_latency_ms": response_latency_ms,
            "model_version": response_json.get("model"),
            "temperature": temperature,
        }

        if not raw_response_text:
            logging.warning(f"Empty response content received from model: {model_identifier} for {question_type}")
            return None, None, response_metadata

        logging.info(f"Raw response received from {model_identifier} ({question_type}): '{raw_response_text[:100]}...'")
        # Pass question_type to parse_llm_answer
        parsed_answer = parse_llm_answer(raw_response_text, question_type=question_type)

        if parsed_answer is None:
             logging.warning(f"Failed to parse answer from model {model_identifier} for {question_type}.")

        return parsed_answer, raw_response_text, response_metadata

    except requests.exceptions.Timeout as e:
        logging.error(f"Request timed out for model {model_identifier} ({question_type}): {e}")
        raise
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for model {model_identifier} ({question_type}): {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred for model {model_identifier} ({question_type}): {e}")
        return None, None, None

# Example Usage (requires a valid API key in .env and Pillow/requests/tenacity installed)
if __name__ == '__main__':
    from src.utils import load_api_key
    try:
        dummy_image = Image.new('RGB', (60, 30), color = 'black')
        api_key = load_api_key()
        test_model = "anthropic/claude-sonnet-4.6"

        print(f"\n--- Testing with model: {test_model} ---")

        # Test Case 1: Initial call - MCQ_SINGLE_CORRECT
        print("\nTest Case 1: Initial - MCQ_SINGLE_CORRECT")
        parsed_ans_1, raw_resp_1, meta_1 = get_openrouter_prediction(
            model_identifier=test_model, api_key=api_key, image=dummy_image,
            exam_name="DUMMY_EXAM", exam_year="2024", question_type="MCQ_SINGLE_CORRECT"
        )
        print(f"Parsed: {parsed_ans_1}, Raw: {raw_resp_1[:60] if raw_resp_1 else None}...")
        print(f"Metadata: {meta_1}")

        # Test Case 2: Initial call - MCQ_MULTIPLE_CORRECT
        print("\nTest Case 2: Initial - MCQ_MULTIPLE_CORRECT")
        parsed_ans_2, raw_resp_2, meta_2 = get_openrouter_prediction(
            model_identifier=test_model, api_key=api_key, image=dummy_image,
            exam_name="DUMMY_EXAM", exam_year="2024", question_type="MCQ_MULTIPLE_CORRECT"
        )
        print(f"Parsed: {parsed_ans_2}, Raw: {raw_resp_2[:60] if raw_resp_2 else None}...")

        # Test Case 3: Initial call - INTEGER
        print("\nTest Case 3: Initial - INTEGER")
        parsed_ans_3, raw_resp_3, meta_3 = get_openrouter_prediction(
            model_identifier=test_model, api_key=api_key, image=dummy_image,
            exam_name="DUMMY_EXAM", exam_year="2024", question_type="INTEGER"
        )
        print(f"Parsed: {parsed_ans_3}, Raw: {raw_resp_3[:60] if raw_resp_3 else None}...")


        # Test Case 4: Re-prompt - MCQ_SINGLE_CORRECT (simulating bad initial response)
        print("\nTest Case 4: Re-prompt - MCQ_SINGLE_CORRECT")
        bad_initial_resp_mcq_single = "<answer>1 2</answer> This is some extra text."
        reprompt_ans_4, reprompt_raw_4, meta_4 = get_openrouter_prediction(
            model_identifier=test_model, api_key=api_key,
            previous_raw_response=bad_initial_resp_mcq_single, question_type="MCQ_SINGLE_CORRECT"
        )
        print(f"Parsed: {reprompt_ans_4}, Raw: {reprompt_raw_4[:60] if reprompt_raw_4 else None}...")

        # Test Case 5: Re-prompt - MCQ_MULTIPLE_CORRECT (simulating bad initial response)
        print("\nTest Case 5: Re-prompt - MCQ_MULTIPLE_CORRECT")
        bad_initial_resp_mcq_multi = "The answer is <answer>option 1 and 4</answer> because reasons."
        reprompt_ans_5, reprompt_raw_5, meta_5 = get_openrouter_prediction(
            model_identifier=test_model, api_key=api_key,
            previous_raw_response=bad_initial_resp_mcq_multi, question_type="MCQ_MULTIPLE_CORRECT"
        )
        print(f"Parsed: {reprompt_ans_5}, Raw: {reprompt_raw_5[:60] if reprompt_raw_5 else None}...")

    except ValueError as e:
        print(f"Setup or Argument Error: {e}")
    except Exception as e:
        print(f"Runtime Error during example execution: {e}")
