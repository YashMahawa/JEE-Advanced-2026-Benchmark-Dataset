# System prompts for LLM interaction

# --- Initial Prompt Components ---

ANSWER_FORMAT_INSTRUCTIONS = {
    "MCQ_SINGLE_CORRECT": "Pick the single correct option (e.g., A, B, 1, 2).",
    "INTEGER": "Find the numerical answer (integer or decimal).",
    "MCQ_MULTIPLE_CORRECT": "Pick all correct options (e.g., A,C or 1,3), comma-separated.",
    "DEFAULT": "Answer based on the question's format."
}

EXAMPLE_INSTRUCTIONS = {
    "MCQ_SINGLE_CORRECT": "- <answer>A</answer>\n- <answer>2</answer>",
    "INTEGER": "- <answer>5</answer>\n- <answer>12.75</answer>",
    "MCQ_MULTIPLE_CORRECT": "- <answer>A,C</answer>\n- <answer>1,3</answer>\n- <answer>B</answer>",
    "DEFAULT": "- <answer>Your Answer</answer>"
}

INITIAL_PROMPT_TEMPLATE = """Look at this {exam_name} {exam_year} exam question ({question_type}).

{answer_format_instruction}

Respond with ONLY an <answer> tag. No other text. Do not wrap the answer in any other tokens (e.g. no <|begin_of_box|>, no markdown code fences).

{example_instruction}
- Unsure: <answer>SKIP</answer>"""


# --- Reprompt Components ---

SPECIFIC_INSTRUCTIONS_REPROMPT = {
    "MCQ_SINGLE_CORRECT": "provide the correct option (e.g., A or 2)",
    "INTEGER": "provide the numerical answer",
    "MCQ_MULTIPLE_CORRECT": "provide all correct options comma-separated (e.g., A,C)",
    "DEFAULT": "provide the answer"
}

REPROMPT_EXAMPLE_INSTRUCTIONS = {
    "MCQ_SINGLE_CORRECT": "<answer>A</answer>",
    "MCQ_MULTIPLE_CORRECT": "<answer>A,C</answer>",
    "INTEGER": "<answer>42</answer>",
    "DEFAULT": "<answer>Your Answer</answer>",
}

REPROMPT_PROMPT_TEMPLATE = """Your previous response was not in the correct format.

Previous response:
{previous_raw_response}

This is a '{question_type}' question. Please {specific_instructions} in an <answer> tag.

Example: {reprompt_example_instruction}
Unsure: <answer>SKIP</answer>

Respond with ONLY the <answer> tag. No other text."""

# --- Helper functions to get instructions ---

def get_answer_format_instruction(question_type: str) -> str:
    return ANSWER_FORMAT_INSTRUCTIONS.get(question_type, ANSWER_FORMAT_INSTRUCTIONS["DEFAULT"])

def get_example_instruction(question_type: str) -> str:
    return EXAMPLE_INSTRUCTIONS.get(question_type, EXAMPLE_INSTRUCTIONS["DEFAULT"])

def get_specific_instructions_reprompt(question_type: str) -> str:
    return SPECIFIC_INSTRUCTIONS_REPROMPT.get(question_type, SPECIFIC_INSTRUCTIONS_REPROMPT["DEFAULT"])

def get_reprompt_example_instruction(question_type: str) -> str:
    return REPROMPT_EXAMPLE_INSTRUCTIONS.get(question_type, REPROMPT_EXAMPLE_INSTRUCTIONS["DEFAULT"])
