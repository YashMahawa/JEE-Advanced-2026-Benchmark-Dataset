import json
import os
from pathlib import Path

def generate_leaderboard():
    results_dir = Path("results")
    if not results_dir.exists():
        print("Results directory not found.")
        return

    model_scores = {}

    for folder in results_dir.glob("google_*_JEE_ADVANCED_2026_*"):
        if not folder.is_dir():
            continue

        model_name = folder.name.split("_JEE_ADVANCED")[0].replace("google_", "")
        summary_file = folder / "summary.jsonl"

        if not summary_file.exists():
            continue

        total_score = 0
        total_questions = 0
        correct_full = 0
        partial_correct = 0
        incorrect = 0
        api_parse_fail = 0
        prompt_tokens = 0
        completion_tokens = 0

        with open(summary_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                # Token counts
                metadata = data.get("response_metadata", {})
                if metadata:
                    prompt_tokens += metadata.get("prompt_tokens", 0)
                    completion_tokens += metadata.get("completion_tokens", 0) or 0
                
                # Check for patch script marks
                marks = data.get("marks_awarded")
                if marks is None:
                     marks = 0
                
                total_score += marks
                total_questions += 1
                
                status = data.get("evaluation_status", "")
                if status in ["correct", "correct_full"]:
                    correct_full += 1
                elif status.startswith("partial_"):
                    partial_correct += 1
                elif status in ["incorrect", "incorrect_negative"]:
                    incorrect += 1
                elif status in ["failure_api_or_parse", "failure_unexpected_type", "error_bad_ground_truth"]:
                     api_parse_fail += 1

        if model_name not in model_scores or total_questions > model_scores[model_name]["total_questions"]:
            # Store the best run (most questions processed)
             model_scores[model_name] = {
                "score": total_score,
                "total_questions": total_questions,
                "correct_full": correct_full,
                "partial_correct": partial_correct,
                "incorrect": incorrect,
                "api_parse_fail": api_parse_fail,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
             }

    print("\n--- JEE Advanced 2026 Benchmark Leaderboard ---")
    print(f"{'Model Name':<30} | {'Score':<6} | {'Questions':<10} | {'Correct':<8} | {'Partial':<8} | {'Incorrect':<10} | {'Failures':<8} | {'Prompt T':<10} | {'Comp T':<10}")
    print("-" * 120)
    
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    
    for model, stats in sorted_models:
        print(f"{model:<30} | {stats['score']:<6} | {stats['total_questions']:<10} | {stats['correct_full']:<8} | {stats['partial_correct']:<8} | {stats['incorrect']:<10} | {stats['api_parse_fail']:<8} | {stats['prompt_tokens']:<10} | {stats['completion_tokens']:<10}")

if __name__ == "__main__":
    generate_leaderboard()