---
license: mit
language:
  - en
task_categories:
  - visual-question-answering
  - image-text-to-text
  - question-answering
pretty_name: JEE Advanced 2026 LLM Benchmark Dataset
size_categories:
  - n<1K
tags:
  - education
  - science
  - india
  - competitive-exams
  - llm-benchmark
  - jee-advanced
configs:
  - config_name: default
    data_files:
      - split: test
        path: "images/**"
    drop_labels: true
dataset_info:
  features:
    - name: image
      dtype: image
    - name: question_id
      dtype: string
    - name: exam_name
      dtype: string
    - name: exam_year
      dtype: int32
    - name: subject
      dtype: string
    - name: question_type
      dtype: string
    - name: correct_answer
      dtype: string
    - name: paper_id
      dtype: int64
  splits:
    - name: test
      num_examples: 102
---
# JEE Advanced 2026 LLM Benchmark Dataset

This repository is formatted to match the Hugging Face `Reja1/jee-neet-benchmark` image benchmark layout.

The dataset contains JEE Advanced 2026 question images under `images/JEE_ADVANCED_2026/` and metadata in `images/metadata.jsonl`. The official answer key is not available yet, so every metadata row currently stores `correct_answer` as the JSON string `[]`. After the official key is released, update only `correct_answer` and re-score existing model outputs.

## Current Data

- JEE Advanced 2026 Paper 1: 48 question images across Physics, Chemistry, and Math
- JEE Advanced 2026 Paper 2: 54 question rows across Physics, Chemistry, and Math
- Total: 102 metadata rows

Paper 2 contains shared stem images for some numerical questions. Those stem images are duplicated into per-question filenames so the benchmark stays one image per question, which is the same loading model used by the reference dataset.

## File Format

- Images: `images/JEE_ADVANCED_2026/JEE_ADVANCED_2026_P{paper}_{SUBJECT}_{question}.png`
- Metadata: `images/metadata.jsonl`
- Runner config: `configs/benchmark_config.yaml`
- Results: `results/<provider_model_exam_timestamp>/`

Each metadata row follows this schema:

```json
{"file_name":"JEE_ADVANCED_2026/JEE_ADVANCED_2026_P1_PHYSICS_01.png","question_id":"JA26P1P01","exam_name":"JEE_ADVANCED","exam_year":2026,"subject":"Physics","question_type":"MCQ_SINGLE_CORRECT","correct_answer":"[]","paper_id":1}
```

## Model Outputs Before Official Answer Key

Use `results/provisional_model_outputs_JEE_ADVANCED_2026/` for answer collection before scoring is possible.

- `predictions.jsonl` stores the full model response in `raw_response`.
- `summary.jsonl` stores the extracted final answer in `predicted_answer`.
- `evaluation_status` remains `pending_official_answer_key` until the official key is added.

The benchmark runner copied from `Reja1/jee-neet-benchmark` already writes full responses to `predictions.jsonl` and extracted answers to `summary.jsonl` during real runs.

## Running

```bash
uv sync
uv run python src/benchmark_runner.py --config configs/benchmark_config.yaml --model "openai/gpt-5.5" --exam_name JEE_ADVANCED --exam_year 2026
```

After the official answer key is released:

```bash
uv run python src/benchmark_runner.py --score-only results/<result_dir>
```
