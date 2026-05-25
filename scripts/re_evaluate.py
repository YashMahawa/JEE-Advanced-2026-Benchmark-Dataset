import json
import os
import sys
from pathlib import Path
sys.path.append(str(Path("/home/yash/Downloads/JEE Advanced 2026 Benchmark")))
from src.evaluation import calculate_single_question_score_details, calculate_exam_scores

def load_metadata(path):
    metadata = {}
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            metadata[data["question_id"]] = data
    return metadata

def main():
    base_dir = Path("/home/yash/Downloads/JEE Advanced 2026 Benchmark")
    meta_path = base_dir / "images" / "metadata.jsonl"
    metadata = load_metadata(meta_path)
    
    results_dir = base_dir / "results"
    
    for folder in results_dir.glob("google_*_JEE_ADVANCED_2026_*"):
        if not folder.is_dir():
            continue
            
        pred_path = folder / "predictions.jsonl"
        if not pred_path.exists():
            continue
            
        print(f"Re-evaluating {folder.name}...")
        
        evaluated_records = []
        with open(pred_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                pred_data = json.loads(line)
                qid = pred_data["question_id"]
                if qid not in metadata:
                    continue
                    
                meta_info = metadata[qid]
                
                # Combine data for evaluation
                eval_input = {**pred_data}
                eval_input["ground_truth"] = json.loads(meta_info["correct_answer"])
                eval_input["question_type"] = meta_info["question_type"]
                eval_input["exam_name"] = meta_info["exam_name"]
                
                # Re-evaluate
                score_details = calculate_single_question_score_details(eval_input)
                
                # Combine prediction and evaluation data for summary
                summary_record = {**pred_data}
                summary_record["evaluation_status"] = score_details["evaluation_status"]
                summary_record["marks_awarded"] = score_details["marks_awarded"]
                evaluated_records.append(summary_record)
                
        # Write summary.jsonl
        summary_path = folder / "summary.jsonl"
        with open(summary_path, "w") as f:
            for rec in evaluated_records:
                f.write(json.dumps(rec) + "\n")
                
        # Write summary.md
        score_summary = calculate_exam_scores(evaluated_records)
        md_path = folder / "summary.md"
        with open(md_path, "w") as f:
            f.write(f"# Evaluation Summary: {folder.name}\n\n")
            f.write("```json\n")
            f.write(json.dumps(score_summary, indent=2) + "\n")
            f.write("```\n")

if __name__ == "__main__":
    main()