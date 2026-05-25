import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_charts():
    results_dir = Path("results")
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

        with open(summary_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                marks = data.get("marks_awarded", 0)
                total_score += marks
                total_questions += 1

        if model_name not in model_scores or total_questions > model_scores[model_name]["questions"]:
             model_scores[model_name] = {"score": total_score, "questions": total_questions}

    # Sort models by score
    sorted_models = sorted(model_scores.items(), key=lambda item: item[1]['score'], reverse=True)
    models = [item[0] for item in sorted_models]
    scores = [item[1]['score'] for item in sorted_models]
    
    # Calculate Max Possible Score (assuming 102 questions mapped)
    # This is an approximation based on typical JEE marking, but we just use the raw score for the chart.
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color=['#4285F4', '#34A853', '#FBBC05'])
    
    # Add title and labels
    plt.title('JEE Advanced 2026 Benchmark - Model Scores', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Total Score', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    
    # Add data labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, str(int(yval)), ha='center', va='bottom', fontweight='bold')
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the chart
    chart_path = "leaderboard_chart.png"
    plt.savefig(chart_path, dpi=300)
    print(f"Chart saved to {chart_path}")

if __name__ == "__main__":
    generate_charts()
