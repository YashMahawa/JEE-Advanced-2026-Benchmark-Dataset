"""Aggregate results from multiple benchmark runs of the same model/exam/year.

Computes mean, std dev, 95% CI, min/max scores, and per-question stability.

Usage:
    uv run python scripts/aggregate_runs.py results/model_EXAM_YEAR_*/
    uv run python scripts/aggregate_runs.py --pattern "openai_o3_JEE_ADVANCED_2025"
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict


def load_summary_jsonl(path: str) -> list[dict]:
    """Load all records from a summary.jsonl file."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_run_stats(records: list[dict]) -> dict:
    """Compute aggregate stats for a single run."""
    total_score = sum(r.get("marks_awarded", 0) for r in records)
    correct = sum(1 for r in records if r.get("evaluation_status") in ("correct", "correct_full"))
    incorrect = sum(1 for r in records if r.get("evaluation_status") in ("incorrect", "incorrect_negative"))
    skipped = sum(1 for r in records if r.get("evaluation_status") == "skipped")
    failures = sum(1 for r in records if r.get("evaluation_status") in ("failure_api_or_parse", "failure_unexpected_type"))
    partial = sum(1 for r in records if r.get("evaluation_status", "").startswith("partial_"))
    return {
        "score": total_score,
        "correct": correct,
        "partial": partial,
        "incorrect": incorrect,
        "skipped": skipped,
        "failures": failures,
        "num_questions": len(records),
    }


def aggregate_runs(run_dirs: list[str]) -> dict:
    """Aggregate stats across multiple runs."""
    if not run_dirs:
        return {"error": "No run directories provided."}

    all_run_stats = []
    per_question_answers: dict[str, list] = defaultdict(list)

    for run_dir in run_dirs:
        summary_path = os.path.join(run_dir, "summary.jsonl")
        if not os.path.exists(summary_path):
            print(f"Warning: {summary_path} not found, skipping.", file=sys.stderr)
            continue

        records = load_summary_jsonl(summary_path)
        stats = compute_run_stats(records)
        stats["run_dir"] = run_dir
        all_run_stats.append(stats)

        for r in records:
            qid = r.get("question_id")
            if qid:
                per_question_answers[qid].append({
                    "predicted_answer": r.get("predicted_answer"),
                    "evaluation_status": r.get("evaluation_status"),
                    "marks_awarded": r.get("marks_awarded", 0),
                })

    if not all_run_stats:
        return {"error": "No valid runs found."}

    scores = [s["score"] for s in all_run_stats]
    n = len(scores)
    mean_score = sum(scores) / n
    if n > 1:
        variance = sum((s - mean_score) ** 2 for s in scores) / (n - 1)
        std_dev = math.sqrt(variance)
        # 95% CI using t-distribution approximation (for small n, use 2.0 as rough multiplier)
        t_multiplier = 2.0 if n < 30 else 1.96
        ci_half = t_multiplier * std_dev / math.sqrt(n)
    else:
        std_dev = 0.0
        ci_half = 0.0

    # Per-question stability
    question_stability = {}
    for qid, answers in per_question_answers.items():
        statuses = [a["evaluation_status"] for a in answers]
        most_common = max(set(statuses), key=statuses.count)
        agreement_rate = statuses.count(most_common) / len(statuses)
        question_stability[qid] = {
            "agreement_rate": round(agreement_rate, 3),
            "dominant_status": most_common,
            "all_statuses": statuses,
            "scores": [a["marks_awarded"] for a in answers],
        }

    # Unstable questions (agreement < 100%)
    unstable = {qid: v for qid, v in question_stability.items() if v["agreement_rate"] < 1.0}

    return {
        "num_runs": n,
        "scores": scores,
        "mean_score": round(mean_score, 2),
        "std_dev": round(std_dev, 2),
        "ci_95_lower": round(mean_score - ci_half, 2),
        "ci_95_upper": round(mean_score + ci_half, 2),
        "min_score": min(scores),
        "max_score": max(scores),
        "per_run": all_run_stats,
        "num_questions": all_run_stats[0]["num_questions"] if all_run_stats else 0,
        "unstable_questions": len(unstable),
        "total_questions": len(question_stability),
        "question_stability": question_stability,
    }


def print_report(result: dict):
    """Print a human-readable report."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"# Multi-Run Aggregation Report")
    print(f"\n**Runs:** {result['num_runs']}")
    print(f"**Questions per run:** {result['num_questions']}")
    print(f"\n## Score Summary")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Mean | {result['mean_score']} |")
    print(f"| Std Dev | {result['std_dev']} |")
    print(f"| 95% CI | [{result['ci_95_lower']}, {result['ci_95_upper']}] |")
    print(f"| Min | {result['min_score']} |")
    print(f"| Max | {result['max_score']} |")
    print(f"| Scores | {result['scores']} |")

    print(f"\n## Per-Question Stability")
    print(f"- **Stable questions:** {result['total_questions'] - result['unstable_questions']}/{result['total_questions']}")
    print(f"- **Unstable questions:** {result['unstable_questions']}/{result['total_questions']}")

    if result["unstable_questions"] > 0:
        print(f"\n### Unstable Questions (different answers across runs)")
        print(f"| Question ID | Agreement | Dominant Status | Scores |")
        print(f"|-------------|-----------|-----------------|--------|")
        for qid, info in sorted(result["question_stability"].items()):
            if info["agreement_rate"] < 1.0:
                print(f"| {qid} | {info['agreement_rate']:.0%} | {info['dominant_status']} | {info['scores']} |")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results from multiple benchmark runs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help="Paths to result directories to aggregate.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Glob pattern to match result directories (e.g., 'openai_o3_JEE_ADVANCED_2025').",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base results directory (default: results).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path for aggregated results.",
    )
    args = parser.parse_args()

    run_dirs = list(args.run_dirs) if args.run_dirs else []

    if args.pattern:
        pattern = os.path.join(args.results_dir, f"*{args.pattern}*")
        matched = sorted(glob.glob(pattern))
        run_dirs.extend(d for d in matched if os.path.isdir(d))

    if not run_dirs:
        parser.error("No run directories specified. Provide paths or use --pattern.")

    run_dirs = sorted(set(run_dirs))
    print(f"Aggregating {len(run_dirs)} runs...", file=sys.stderr)

    result = aggregate_runs(run_dirs)
    print_report(result)

    if args.output:
        # Remove per-question detail for compact output unless needed
        output_data = {k: v for k, v in result.items() if k != "question_stability"}
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
