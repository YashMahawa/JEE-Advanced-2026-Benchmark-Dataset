import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_latest_results(base_dir):
    results_dir = base_dir / "results"
    
    # Gemini 3.1 Flash Lite
    gemini_folders = sorted(results_dir.glob("google_gemini-3.1-flash-lite_JEE_ADVANCED_2026_*"), key=lambda x: x.name, reverse=True)
    # Gemma 4 31b-it
    gemma_folders = sorted(results_dir.glob("google_gemma-4-31b-it_JEE_ADVANCED_2026_*"), key=lambda x: x.name, reverse=True)
    
    # We want the ones that have the most questions (likely the ones we patched)
    def get_best_run(folders):
        best_run = None
        max_q = -1
        for f in folders:
            summary_path = f / "summary.jsonl"
            if summary_path.exists():
                with open(summary_path, "r") as s:
                    q_count = sum(1 for line in s if line.strip())
                    if q_count > max_q:
                        max_q = q_count
                        best_run = f
        return best_run

    return get_best_run(gemini_folders), get_best_run(gemma_folders)

def aggregate_stats(summary_path):
    stats = {
        "score": 0,
        "correct": 0,
        "partial": 0,
        "wrong": 0,
        "unattempted": 0,
        "positive_marks": 0,
        "negative_marks": 0
    }
    
    if not summary_path or not summary_path.exists():
        return stats

    with open(summary_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            marks = data.get("marks_awarded", 0)
            if marks is None: marks = 0
            
            stats["score"] += marks
            if marks > 0:
                stats["positive_marks"] += marks
            elif marks < 0:
                stats["negative_marks"] += abs(marks)
                
            status = data.get("evaluation_status", "")
            if status in ["correct", "correct_full"]:
                stats["correct"] += 1
            elif status.startswith("partial_"):
                stats["partial"] += 1
            elif status in ["incorrect", "incorrect_negative"]:
                stats["wrong"] += 1
            else: # failure_api_or_parse, skipped, etc.
                stats["unattempted"] += 1
                
    return stats

def main():
    base_dir = Path("/home/yash/Downloads/JEE Advanced 2026 Benchmark")
    gemini_run, gemma_run = get_latest_results(base_dir)
    
    print(f"Using Gemini run: {gemini_run.name if gemini_run else 'None'}")
    print(f"Using Gemma run: {gemma_run.name if gemma_run else 'None'}")
    
    gemini_stats = aggregate_stats(gemini_run / "summary.jsonl" if gemini_run else None)
    gemma_stats = aggregate_stats(gemma_run / "summary.jsonl" if gemma_run else None)
    
    models = ["Gemini 3.1 Flash Lite", "Gemma 4 31b-it"]
    
    # 1. Total Scores Chart
    plt.figure(figsize=(8, 6))
    scores = [gemini_stats["score"], gemma_stats["score"]]
    plt.bar(models, scores, color=['#4285F4', '#EA4335'])
    plt.title('JEE Advanced 2026 - Total Scores', fontsize=14)
    plt.ylabel('Score')
    for i, v in enumerate(scores):
        plt.text(i, v + 2, str(v), ha='center', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(base_dir / "score_comparison.png", dpi=300)
    print("Saved score_comparison.png")

    # 2. Question Breakdown Chart
    labels = ['Correct', 'Partial', 'Wrong', 'Unattempted']
    gemini_vals = [gemini_stats["correct"], gemini_stats["partial"], gemini_stats["wrong"], gemini_stats["unattempted"]]
    gemma_vals = [gemma_stats["correct"], gemma_stats["partial"], gemma_stats["wrong"], gemma_stats["unattempted"]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, gemini_vals, width, label='Gemini 3.1 Flash Lite', color='#4285F4')
    rects2 = ax.bar(x + width/2, gemma_vals, width, label='Gemma 4 31b-it', color='#EA4335')
    
    ax.set_ylabel('Number of Questions')
    ax.set_title('Question Outcome Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(base_dir / "outcome_breakdown.png", dpi=300)
    print("Saved outcome_breakdown.png")

    # 3. Marks Breakdown (Positive vs Negative)
    labels_marks = ['Positive Marks', 'Negative Marks (Abs)']
    gemini_marks = [gemini_stats["positive_marks"], gemini_stats["negative_marks"]]
    gemma_marks = [gemma_stats["positive_marks"], gemma_stats["negative_marks"]]
    
    x = np.arange(len(labels_marks))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, gemini_marks, width, label='Gemini 3.1 Flash Lite', color='#4285F4')
    rects2 = ax.bar(x + width/2, gemma_marks, width, label='Gemma 4 31b-it', color='#EA4335')
    
    ax.set_ylabel('Marks')
    ax.set_title('Positive vs Negative Marks')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_marks)
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(base_dir / "marks_breakdown.png", dpi=300)
    print("Saved marks_breakdown.png")

if __name__ == "__main__":
    main()
