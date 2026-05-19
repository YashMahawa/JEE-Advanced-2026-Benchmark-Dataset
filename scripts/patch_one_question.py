"""One-off patch for a single question that hit a provider-side refusal.

Re-runs a single question via OpenRouter with the *exact* benchmark prompt,
pinned to a chosen provider (e.g. 'Anthropic' to bypass Vertex's safety
classifier), then appends the result to an existing run's jsonl files and
triggers rescore_result_dir() so summary.md is regenerated.
"""

import argparse
import base64
import json
import os
import sys
import time

import requests
import yaml
from dotenv import load_dotenv
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_interface import construct_initial_prompt
from utils import parse_llm_answer
from benchmark_runner import rescore_result_dir

load_dotenv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--question_id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True, help="OpenRouter provider name to pin (e.g. Anthropic)")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    with open("images/metadata.jsonl") as f:
        meta = next(json.loads(l) for l in f if json.loads(l)["question_id"] == args.question_id)

    img_path = os.path.join("images", meta["file_name"])
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    messages = construct_initial_prompt(b64, meta["exam_name"], str(meta["exam_year"]), meta["question_type"])

    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "provider": {"order": [args.provider], "allow_fallbacks": False},
    }
    key = os.environ["OPENROUTER_API_KEY"]
    t0 = time.time()
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=payload,
        timeout=180,
    )
    latency_ms = int((time.time() - t0) * 1000)
    r.raise_for_status()
    j = r.json()

    choice = j["choices"][0]
    content = choice["message"].get("content")
    if isinstance(content, list):
        content = "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
    print(f"finish={choice.get('finish_reason')}/{choice.get('native_finish_reason')} provider={j.get('provider')}")
    print(f"raw: {content!r}")

    parsed = parse_llm_answer(content, question_type=meta["question_type"])
    print(f"parsed: {parsed}")
    if parsed is None:
        sys.exit("parse failed; aborting before patch")

    usage = j.get("usage", {})
    pred_record = {
        "question_id": args.question_id,
        "subject": meta.get("subject"),
        "exam_name": meta["exam_name"],
        "question_type": meta["question_type"],
        "raw_response": content,
        "parse_successful": True,
        "api_call_successful": True,
        "error": None,
        "attempt": 1,
        "previous_raw_response_on_reprompt": None,
        "response_metadata": {
            "generation_id": j.get("id"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost": usage.get("cost"),
            "response_latency_ms": latency_ms,
            "model_version": j.get("model"),
            "temperature": args.temperature,
            "provider_pinned": args.provider,
        },
    }

    summary_record = {
        "question_id": args.question_id,
        "exam_name": meta["exam_name"],
        "exam_year": meta["exam_year"],
        "marks_awarded": 0,  # rescore will overwrite
        "evaluation_status": "pending_rescore",
        "predicted_answer": parsed,
        "ground_truth": json.loads(meta["correct_answer"]) if isinstance(meta["correct_answer"], str) else meta["correct_answer"],
        "attempt": 1,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost": usage.get("cost"),
        "response_latency_ms": latency_ms,
    }

    with open(os.path.join(args.result_dir, "predictions.jsonl"), "a") as f:
        f.write(json.dumps(pred_record) + "\n")
    with open(os.path.join(args.result_dir, "summary.jsonl"), "a") as f:
        f.write(json.dumps(summary_record) + "\n")
    print(f"appended records to {args.result_dir}")

    with open("configs/benchmark_config.yaml") as f:
        config = yaml.safe_load(f)
    rescore_result_dir(args.result_dir, config)
    print("rescore done")


if __name__ == "__main__":
    main()
