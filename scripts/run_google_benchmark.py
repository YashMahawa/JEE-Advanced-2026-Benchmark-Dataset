#!/usr/bin/env python3
import argparse
import base64
import collections
import concurrent.futures
import datetime as dt
import json
import os
import re
import threading
import time
from io import BytesIO
from pathlib import Path

import requests
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from prompts import INITIAL_PROMPT_TEMPLATE, get_answer_format_instruction, get_example_instruction
from utils import parse_llm_answer


def encode_image(path: Path, max_image_side: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max(image.size) > max_image_side:
            image.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=92, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_prompt(row: dict) -> str:
    question_type = row.get("question_type", "MCQ_SINGLE_CORRECT")
    return INITIAL_PROMPT_TEMPLATE.format(
        exam_name=row.get("exam_name", "JEE_ADVANCED"),
        exam_year=str(row.get("exam_year", "2026")),
        question_type=question_type,
        answer_format_instruction=get_answer_format_instruction(question_type),
        example_instruction=get_example_instruction(question_type),
    )


def generate_once(api_key: str, model: str, row: dict, image_path: Path, timeout: int, temperature: float, thinking_level: str, max_image_side: int) -> dict:
    prompt = build_prompt(row)
    image_b64 = encode_image(image_path, max_image_side)
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    generation_config = {
        "temperature": temperature,
        "mediaResolution": "MEDIA_RESOLUTION_HIGH",
    }
    if thinking_level and thinking_level.lower() != "off":
        generation_config["thinkingConfig"] = {
            "thinkingLevel": thinking_level,
            "includeThoughts": False,
        }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}},
                ],
            }
        ],
        "generationConfig": generation_config,
    }
    started = time.time()
    print(f"DEBUG: Starting API call for {row['question_id']} to {model}", flush=True)
    response = requests.post(endpoint, params={"key": api_key}, json=payload, timeout=timeout)
    latency_ms = round((time.time() - started) * 1000)
    print(f"DEBUG: API call finished for {row['question_id']} (status: {response.status_code})", flush=True)
    response.raise_for_status()
    data = response.json()
    if not data.get("candidates"):
        print(f"DEBUG: No candidates returned for {row['question_id']}. Full data: {data}", flush=True)
    
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    
    # Collect both text and thought parts. Some models (Gemma 4) might put the answer in thought.
    text_segments = []
    for part in parts:
        if part.get("text"):
            text_segments.append(part["text"])
        if part.get("thought"):
            text_segments.append(f"\n<thought>\n{part['thought']}\n</thought>\n")
            
    raw_response = "\n".join(text_segments).strip() or None
    
    if raw_response is None:
        print(f"DEBUG: raw_response is None for {row['question_id']}. Full data: {data}", flush=True)
    
    predicted = parse_llm_answer(raw_response or "", question_type=row.get("question_type", "MCQ_SINGLE_CORRECT"))
    if predicted is None and raw_response is not None:
        print(f"DEBUG: Parse failed for {row['question_id']}. raw_response: {raw_response}", flush=True)
    usage = data.get("usageMetadata", {})
    return {
        "question_id": row["question_id"],
        "subject": row["subject"],
        "exam_name": row["exam_name"],
        "exam_year": row["exam_year"],
        "question_type": row["question_type"],
        "raw_response": raw_response,
        "predicted_answer": predicted,
        "parse_successful": predicted is not None,
        "api_call_successful": raw_response is not None,
        "error": None,
        "attempt": 1,
        "previous_raw_response_on_reprompt": None,
        "response_metadata": {
            "prompt_tokens": usage.get("promptTokenCount"),
            "completion_tokens": usage.get("candidatesTokenCount"),
            "total_tokens": usage.get("totalTokenCount"),
            "response_latency_ms": latency_ms,
            "model_version": model,
            "temperature": temperature,
        },
    }


def generate(api_key: str, model: str, row: dict, image_path: Path, timeout: int, temperature: float, thinking_level: str, max_image_side: int, retries: int) -> dict:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            result = generate_once(api_key, model, row, image_path, timeout, temperature, thinking_level, max_image_side)
            result["attempt"] = attempt
            return result
        except requests.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            if status not in {429, 500, 502, 503, 504} or attempt == retries:
                raise
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt == retries:
                raise
        time.sleep(min(90, 10 * attempt))
    raise last_exc or RuntimeError("generation failed")


class RollingRateLimiter:
    def __init__(self, max_starts: int, window_seconds: int):
        self.max_starts = max_starts
        self.window_seconds = window_seconds
        self.min_interval = window_seconds / max_starts
        self.starts = collections.deque()
        self.last_start = 0.0
        self.lock = threading.Lock()

    def wait_for_slot(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.starts and now - self.starts[0] >= self.window_seconds:
                    self.starts.popleft()
                interval_remaining = self.min_interval - (now - self.last_start)
                if len(self.starts) < self.max_starts and interval_remaining <= 0:
                    self.starts.append(now)
                    self.last_start = now
                    return
                window_sleep = self.window_seconds - (now - self.starts[0]) + 0.1 if self.starts else 0.1
                sleep_for = max(interval_remaining, window_sleep if len(self.starts) >= self.max_starts else 0.1)
            time.sleep(max(0.1, sleep_for))


def load_rows(metadata_path: Path, images_base_dir: Path, exam_name: str, exam_year: int, completed: set[str]) -> list[dict]:
    rows = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("exam_name") != exam_name or int(row.get("exam_year")) != exam_year:
                continue
            if row["question_id"] in completed:
                continue
            row["image_path"] = str(images_base_dir / row["file_name"])
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict, lock: threading.Lock):
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def completed_ids(summary_path: Path) -> set[str]:
    ids = set()
    if not summary_path.exists():
        return ids
    with summary_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("question_id"):
                ids.add(row["question_id"])
    return ids


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_")


def public_error(exc: Exception) -> str:
    text = str(exc)
    text = re.sub(r"key=[^&\s]+", "key=REDACTED", text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_config.yaml")
    parser.add_argument("--model", default="gemma-4-31b-it")
    parser.add_argument("--exam-name", default="JEE_ADVANCED")
    parser.add_argument("--exam-year", type=int, default=2026)
    parser.add_argument("--api-key-env", default="GOOGLE_API_KEY")
    parser.add_argument("--rpm", type=int, default=10)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--thinking-level", default="high")
    parser.add_argument("--max-image-side", type=int, default=1536)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-ids", default=None, help="Comma-separated list of specific question IDs to run")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing {args.api_key_env}")

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    metadata_path = ROOT / config.get("metadata_path", "images/metadata.jsonl")
    images_base_dir = ROOT / config.get("images_base_dir", "images")
    results_base_dir = ROOT / config.get("results_base_dir", "results")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.resume) if args.resume else results_base_dir / f"google_{safe_model_name(args.model)}_{args.exam_name}_{args.exam_year}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.jsonl"
    summary_md_path = output_dir / "summary.md"

    rows = load_rows(metadata_path, images_base_dir, args.exam_name, args.exam_year, completed_ids(summary_path))
    if args.question_ids:
        wanted = {qid.strip() for qid in args.question_ids.split(",") if qid.strip()}
        rows = [row for row in rows if row["question_id"] in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    limiter = RollingRateLimiter(args.rpm, 60)
    write_lock = threading.Lock()

    print(f"Output: {output_dir}")
    print(f"Remaining questions: {len(rows)}")
    print(f"Rate limit: {args.rpm} starts/min, max concurrent: {args.max_concurrent}")

    def worker(row: dict) -> dict:
        limiter.wait_for_slot()
        try:
            return generate(
                api_key,
                args.model,
                row,
                Path(row["image_path"]),
                args.timeout,
                args.temperature,
                args.thinking_level,
                args.max_image_side,
                args.retries,
            )
        except Exception as exc:
            return {
                "question_id": row["question_id"],
                "subject": row["subject"],
                "exam_name": row["exam_name"],
                "exam_year": row["exam_year"],
                "question_type": row["question_type"],
                "raw_response": None,
                "predicted_answer": None,
                "parse_successful": False,
                "api_call_successful": False,
                "error": public_error(exc),
                "attempt": 1,
                "previous_raw_response_on_reprompt": None,
                "response_metadata": None,
            }

    completed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        future_to_row = {executor.submit(worker, row): row for row in rows}
        for future in concurrent.futures.as_completed(future_to_row):
            result = future.result()
            append_jsonl(predictions_path, result, write_lock)
            summary_row = {
                "question_id": result["question_id"],
                "exam_name": result["exam_name"],
                "exam_year": result["exam_year"],
                "marks_awarded": None,
                "evaluation_status": "pending_official_answer_key",
                "predicted_answer": result["predicted_answer"],
                "ground_truth": [],
                "attempt": result["attempt"],
            }
            metadata = result.get("response_metadata") or {}
            summary_row.update({
                "prompt_tokens": metadata.get("prompt_tokens"),
                "completion_tokens": metadata.get("completion_tokens"),
                "total_tokens": metadata.get("total_tokens"),
                "response_latency_ms": metadata.get("response_latency_ms"),
            })
            append_jsonl(summary_path, summary_row, write_lock)
            completed_count += 1
            status = "ok" if result["api_call_successful"] else "fail"
            print(f"[{completed_count}/{len(rows)}] {result['question_id']} {status} parsed={result['parse_successful']}", flush=True)

    summary_md_path.write_text(
        "# Google Gemma JEE Advanced 2026 Run\n\n"
        f"- Model: `{args.model}`\n"
        f"- Questions processed this run: `{len(rows)}`\n"
        "- Scoring: pending official answer key\n"
        f"- Full responses: `{predictions_path.name}`\n"
        f"- Extracted answers: `{summary_path.name}`\n",
        encoding="utf-8",
    )
    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
