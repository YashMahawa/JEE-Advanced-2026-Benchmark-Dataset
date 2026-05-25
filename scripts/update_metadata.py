import json
import re
import subprocess
from pathlib import Path

def get_pdf_text(filepath):
    result = subprocess.run(['pdftotext', '-layout', str(filepath), '-'], capture_output=True, text=True, check=True)
    return result.stdout

def extract_answers(text):
    answers = []
    for line in text.split('\n'):
        m = re.search(r'Answer Q(\d+):\s*(.+)', line)
        if m:
            ans_raw = m.group(2).strip()
            # Handle list formatting
            if ans_raw.startswith('[') and ans_raw.endswith(']'):
                # Handle range like [3.9 TO 4.1]
                inner = ans_raw[1:-1].strip()
                if ' TO ' in inner.upper() or ' to ' in inner:
                    parts = re.split(r'\s+TO\s+|\s+to\s+', inner, flags=re.IGNORECASE)
                    ans_list = [p.strip() for p in parts if p.strip()]
                else:
                    ans_list = [inner]
            elif ' or ' in ans_raw.lower():
                 ans_list = [p.strip() for p in re.split(r'\s+or\s+', ans_raw, flags=re.IGNORECASE)]
            else:
                 # Handle strings like 'ACD' -> ['A', 'C', 'D']
                 if re.match(r'^[A-D]{1,4}$', ans_raw):
                     ans_list = list(ans_raw)
                 else:
                     ans_list = [ans_raw]
            answers.append((int(m.group(1)), ans_list))
    return answers

def main():
    base_dir = Path("/home/yash/Downloads/JEE Advanced 2026 Benchmark")
    
    # 1. Parse P1 Answers
    p1_text = get_pdf_text("/home/yash/Downloads/p1_provisional_keys.pdf")
    p1_ans_raw = extract_answers(p1_text)
    # P1 has 3 subjects, 16 questions each.
    # The PDF structure: Maths, Physics, Chemistry
    # But wait, looking at the grep earlier:
    # First question is limit x->inf... -> Maths
    # Second block Q1 is "A hollow right circular cone..." -> Physics
    # Third block Q1 is "An ideal gas..." -> Chemistry
    # Wait, earlier we saw P1 Q1 is f(x) = sqrt(x) log(x) ... -> Maths
    
    subjects_p1 = ["Maths", "Physics", "Chemistry"]
    
    p1_answers = {}
    idx = 0
    for subj in subjects_p1:
        for qnum in range(1, 17):
            if idx < len(p1_ans_raw):
                 q, ans = p1_ans_raw[idx]
                 # Make sure qnum matches
                 if q == qnum:
                     key = f"JA26P1{subj[0].upper()}{qnum:02d}"
                     p1_answers[key] = ans
                 idx += 1

    # 2. Parse P2 Answers
    p2_text = get_pdf_text("/home/yash/Downloads/p2_provisional_keys.pdf")
    p2_ans_raw = extract_answers(p2_text)
    
    # P2 first Q: vectors a, b -> Maths
    # P2 second Q1: metal wire, drift velocity -> Physics
    # P2 third Q1: molar conductivities -> Chemistry
    subjects_p2 = ["Maths", "Physics", "Chemistry"]
    
    p2_answers = {}
    idx = 0
    for subj in subjects_p2:
        for qnum in range(1, 19): # P2 has 18 questions
            if idx < len(p2_ans_raw):
                 q, ans = p2_ans_raw[idx]
                 if q == qnum:
                     key = f"JA26P2{subj[0].upper()}{qnum:02d}"
                     p2_answers[key] = ans
                 idx += 1

    all_answers = {**p1_answers, **p2_answers}

    # 3. Update metadata.jsonl
    meta_path = base_dir / "images" / "metadata.jsonl"
    out_lines = []
    with open(meta_path, "r") as f:
        for line in f:
            data = json.loads(line)
            qid = data["question_id"]
            if qid in all_answers:
                # Format exactly as the benchmark expects a json string representation of a list
                data["correct_answer"] = json.dumps(all_answers[qid])
            out_lines.append(json.dumps(data))

    with open(meta_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
        
    print(f"Updated metadata.jsonl with {len(all_answers)} answers.")

if __name__ == "__main__":
    main()