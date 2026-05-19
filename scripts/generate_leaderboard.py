"""Generate a cross-model leaderboard from benchmark results.

Reads all result directories, computes scores, and outputs a Markdown
leaderboard table grouped by exam+year.

Usage:
    uv run python scripts/generate_leaderboard.py
    uv run python scripts/generate_leaderboard.py --results-dir results --min-questions 10
    uv run python scripts/generate_leaderboard.py --output leaderboard.json
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict


# Known exam names that may contain underscores
KNOWN_EXAMS = {"JEE_ADVANCED", "JEE_MAIN", "NEET"}


def parse_result_dirname(dirname: str) -> dict | None:
    """Parse a result directory name into model, exam, year, timestamp.

    Handles formats:
      {model}_{EXAM}_{YEAR}_{YYYYMMDD}_{HHMMSS}   (full)
      {model}_{EXAM}_{YYYYMMDD}_{HHMMSS}           (no year — AllYears run)
      {model}_{YYYYMMDD}_{HHMMSS}                  (no exam/year — returns None, caller falls back)
    """
    parts = dirname.split("_")
    if len(parts) < 4:
        return None

    # Last 2 parts are always the timestamp
    if not (len(parts[-2]) == 8 and parts[-2].isdigit() and
            len(parts[-1]) == 6 and parts[-1].isdigit()):
        return None
    timestamp = "_".join(parts[-2:])
    remaining = parts[:-2]  # everything before the timestamp

    # Try to find a known exam (2-part like JEE_ADVANCED, or 1-part like NEET)
    exam = None
    exam_end_idx = None  # index in remaining[] just after the exam tokens

    for i in range(len(remaining) - 1, 0, -1):
        candidate2 = remaining[i - 1] + "_" + remaining[i] if i >= 1 else None
        candidate1 = remaining[i]
        if candidate2 in KNOWN_EXAMS:
            exam = candidate2
            exam_end_idx = i + 1
            break
        if candidate1 in KNOWN_EXAMS:
            exam = candidate1
            exam_end_idx = i + 1
            break

    if exam is None:
        return None  # caller must fall back to summary.jsonl

    # Token immediately after the exam (if present and 4-digit) is the year
    exam_start_idx = exam_end_idx - (2 if "_" in exam else 1)
    year = "AllYears"
    if exam_end_idx < len(remaining):
        candidate_year = remaining[exam_end_idx]
        if len(candidate_year) == 4 and candidate_year.isdigit():
            year = candidate_year

    model_parts = remaining[:exam_start_idx]
    if not model_parts:
        return None

    model_name = "_".join(model_parts)
    first_underscore = model_name.find("_")
    if first_underscore > 0:
        model_name = model_name[:first_underscore] + "/" + model_name[first_underscore + 1:]

    return {"model": model_name, "exam": exam, "year": year, "timestamp": timestamp}


def load_summary_jsonl(filepath: str) -> list[dict]:
    """Load records from a summary.jsonl file."""
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_max_score_from_md(filepath: str) -> int | None:
    """Extract max possible score from summary.md's Overall Score line.

    Handles both formats:
    - New: **Overall Score:** **315** / **360**
    - Old: **Overall Score:** **727 / 800**
    """
    try:
        with open(filepath, "r") as f:
            for line in f:
                # New format: **315** / **360**
                match = re.search(r"\*\*Overall Score:\*\*\s+\*\*\d+\*\*\s*/\s*\*\*(\d+)\*\*", line)
                if match:
                    return int(match.group(1))
                # Old format: **727 / 800**
                match = re.search(r"\*\*Overall Score:\*\*\s+\*\*\d+\s*/\s*(\d+)\*\*", line)
                if match:
                    return int(match.group(1))
                # Oldest format: **322** (Max score is 360)
                match = re.search(r"\(Max score is (\d+)\)", line)
                if match:
                    return int(match.group(1))
    except (OSError, ValueError):
        pass
    return None


def compute_stats(records: list[dict]) -> dict:
    """Compute aggregate stats from summary.jsonl records."""
    total_score = sum(r.get("marks_awarded", 0) for r in records)
    correct = sum(1 for r in records if r.get("evaluation_status") in ("correct", "correct_full"))
    partial = sum(1 for r in records if r.get("evaluation_status", "").startswith("partial_"))
    incorrect = sum(1 for r in records if r.get("evaluation_status") in ("incorrect", "incorrect_negative"))
    skipped = sum(1 for r in records if r.get("evaluation_status") == "skipped")
    failures = sum(1 for r in records if r.get("evaluation_status") in (
        "failure_api_or_parse", "failure_unexpected_type", "error_bad_ground_truth"))

    return {
        "score": total_score,
        "correct": correct,
        "partial": partial,
        "incorrect": incorrect,
        "skipped": skipped,
        "failures": failures,
        "num_questions": len(records),
    }


def load_summary_json(filepath: str) -> dict | None:
    """Load stats from an old-format summary.json file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return {
            "score": data.get("overall_score", 0),
            "correct": data.get("overall_correct", data.get("overall_correct_full", 0)),
            "partial": data.get("overall_partial_correct", 0),
            "incorrect": data.get("overall_incorrect", data.get("overall_incorrect_choice", 0)),
            "skipped": data.get("overall_skipped", 0),
            "failures": data.get("overall_api_parse_failures", 0),
            "num_questions": data.get("total_questions_processed", 0),
        }
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def scan_results(results_dir: str, min_questions: int) -> list[dict]:
    """Scan result directories and collect stats."""
    entries = []

    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        return entries

    for dirname in sorted(os.listdir(results_dir)):
        dirpath = os.path.join(results_dir, dirname)
        if not os.path.isdir(dirpath):
            continue

        parsed = parse_result_dirname(dirname)

        # Try summary.jsonl first (new format), then summary.json (old format)
        stats = None
        records = []
        summary_jsonl_path = os.path.join(dirpath, "summary.jsonl")
        summary_json_path = os.path.join(dirpath, "summary.json")

        if os.path.exists(summary_jsonl_path):
            records = load_summary_jsonl(summary_jsonl_path)
            if records:
                stats = compute_stats(records)
        elif os.path.exists(summary_json_path):
            stats = load_summary_json(summary_json_path)

        # If dirname didn't yield exam/year, infer from summary.jsonl records
        if parsed is None and records:
            exam_names = list({r.get("exam_name") for r in records if r.get("exam_name")})
            exam_years = list({str(r.get("exam_year")) for r in records if r.get("exam_year")})
            if len(exam_names) == 1 and len(exam_years) == 1:
                # Derive model from dirname (provider/rest pattern)
                first_us = dirname.find("_")
                if first_us > 0:
                    model_from_dir = (dirname[:first_us] + "/" +
                                      dirname[first_us + 1:].split("_")[0])
                else:
                    model_from_dir = dirname
                parsed = {
                    "model": model_from_dir,
                    "exam": exam_names[0],
                    "year": exam_years[0],
                    "timestamp": "_".join(dirname.rsplit("_", 2)[-2:]),
                }
            else:
                print(f"Skipping {dirname}: cannot determine exam/year from dirname or data.", file=sys.stderr)
                continue

        if not parsed:
            continue

        if not stats or stats.get("num_questions", 0) < min_questions:
            continue

        # Try to get max score from summary.md
        md_path = os.path.join(dirpath, "summary.md")
        max_score = extract_max_score_from_md(md_path)

        entry = {
            **parsed,
            **stats,
            "max_score": max_score,
            "result_dir": dirname,
        }
        entries.append(entry)

    return entries


def generate_markdown(entries: list[dict]) -> str:
    """Generate a Markdown leaderboard table grouped by exam+year."""
    if not entries:
        return "# Benchmark Leaderboard\n\nNo results found.\n"

    # Group by exam+year
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        key = f"{e['exam']}_{e['year']}"
        groups[key].append(e)

    lines = ["# Benchmark Leaderboard\n"]

    for group_key in sorted(groups.keys()):
        group_entries = groups[group_key]
        # Sort by score descending
        group_entries.sort(key=lambda x: x["score"], reverse=True)

        exam_display = group_key.replace("_", " ").replace("JEE ADVANCED", "JEE Advanced").replace("JEE MAIN", "JEE Main")
        lines.append(f"\n## {exam_display}\n")
        lines.append("| Rank | Model | Score | Max | % | Correct | Partial | Incorrect | Skipped | Failures |")
        lines.append("|------|-------|-------|-----|---|---------|---------|-----------|---------|----------|")

        for rank, e in enumerate(group_entries, 1):
            max_score = e.get("max_score") or "?"
            if isinstance(max_score, int) and max_score > 0:
                pct = f"{e['score'] / max_score * 100:.1f}%"
            else:
                pct = "?"
            lines.append(
                f"| {rank} | {e['model']} | {e['score']} | {max_score} | {pct} "
                f"| {e['correct']} | {e['partial']} | {e['incorrect']} | {e['skipped']} | {e['failures']} |"
            )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark leaderboard.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to the results directory (default: results).",
    )
    parser.add_argument(
        "--min-questions",
        type=int,
        default=10,
        help="Minimum questions to include a run (default: 10, filters incomplete runs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for leaderboard.json.",
    )
    args = parser.parse_args()

    entries = scan_results(args.results_dir, args.min_questions)

    if not entries:
        print("No valid results found.", file=sys.stderr)
        sys.exit(1)

    md = generate_markdown(entries)
    print(md)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"\nJSON output saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
