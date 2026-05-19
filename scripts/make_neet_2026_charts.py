"""Generate publication-quality charts for the NEET 2026 AI benchmark.

Reads live from `results/*NEET_2026*/summary.{md,jsonl}` and `predictions.jsonl`
so the charts always reflect the current run set. Outputs PNGs into
`charts/neet_2026/`.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Display label and brand color per OpenRouter model id.
# New models get a default fallback color; add an entry for nicer styling.
MODEL_META = {
    "google/gemini-3-flash-preview": ("Gemini 3 Flash", "#4285F4"),
    "google/gemini-3.1-pro-preview": ("Gemini 3.1 Pro", "#1A73E8"),
    "openai/gpt-5.5": ("GPT-5.5", "#10A37F"),
    "openai/gpt-5.4": ("GPT-5.4", "#7CC4A8"),
    "qwen/qwen3-vl-235b-a22b-thinking": ("Qwen3-VL 235B Thinking", "#615CED"),
    "anthropic/claude-opus-4.7": ("Claude Opus 4.7", "#C9622D"),
    "anthropic/claude-sonnet-4.6": ("Claude Sonnet 4.6", "#D97757"),
    "anthropic/claude-haiku-4.5": ("Claude Haiku 4.5", "#E8B496"),
    "z-ai/glm-4.6v": ("GLM-4.6V", "#FFB000"),
    "x-ai/grok-4.20": ("Grok 4.20", "#1DA1F2"),
    "x-ai/grok-4.3": ("Grok 4.3", "#000000"),
    "mistralai/mistral-medium-3-5": ("Mistral Medium 3.5", "#FA520F"),
    "moonshotai/kimi-k2.5": ("Kimi K2.5", "#0066FF"),
}
DEFAULT_COLOR = "#888888"

NEET_SUBJECTS = ["Physics", "Chemistry", "Biology"]
NEET_SUBJECT_MAX = {"Physics": 180, "Chemistry": 180, "Biology": 360}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})


def parse_run(run_dir: Path) -> dict | None:
    summary_md = run_dir / "summary.md"
    summary_jsonl = run_dir / "summary.jsonl"
    predictions_jsonl = run_dir / "predictions.jsonl"
    if not (summary_md.exists() and summary_jsonl.exists() and predictions_jsonl.exists()):
        return None

    md = summary_md.read_text()
    score_m = re.search(r"\*\*Overall Score:\*\* \*\*(\d+)\*\* / \*\*(\d+)\*\*", md)
    cost_m = re.search(r"\*\*Total Cost:\*\* \$(\S+)", md)
    if not score_m:
        return None

    subject_by_qid = {}
    with predictions_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            subject_by_qid[r["question_id"]] = r.get("subject", "Unknown")

    correct = incorrect = failed = 0
    subj_score = defaultdict(int)
    subj_correct = defaultdict(int)
    subj_total = defaultdict(int)
    with summary_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            status = r["evaluation_status"]
            if status == "correct":
                correct += 1
            elif status == "incorrect":
                incorrect += 1
            else:
                failed += 1
            sub = subject_by_qid.get(r["question_id"], "Unknown")
            subj_total[sub] += 1
            subj_score[sub] += r.get("marks_awarded", 0)
            if status == "correct":
                subj_correct[sub] += 1

    # Recover model id from dir name: provider_model_..._NEET_2026_TIMESTAMP
    name = run_dir.name.split("_NEET_2026")[0]
    # First underscore separates provider from rest of model slug.
    model_id = name.replace("_", "/", 1)

    return {
        "model": model_id,
        "score": int(score_m.group(1)),
        "total": int(score_m.group(2)),
        "correct": correct,
        "incorrect": incorrect,
        "failed": failed,
        "cost": float(cost_m.group(1)) if cost_m else 0.0,
        "subjects": {s: subj_score.get(s, 0) for s in NEET_SUBJECTS},
        "run_dir": str(run_dir),
    }


def load_results(results_dir: Path) -> list[dict]:
    runs = []
    for d in sorted(results_dir.glob("*NEET_2026*")):
        if not d.is_dir():
            continue
        parsed = parse_run(d)
        if parsed:
            runs.append(parsed)
    # Keep most recent run per model (in case of reruns)
    by_model = {}
    for r in runs:
        prev = by_model.get(r["model"])
        if prev is None or r["run_dir"] > prev["run_dir"]:
            by_model[r["model"]] = r
    return sorted(by_model.values(), key=lambda x: -x["score"])


def label_for(model_id: str) -> str:
    return MODEL_META.get(model_id, (model_id, DEFAULT_COLOR))[0]


def color_for(model_id: str) -> str:
    return MODEL_META.get(model_id, (model_id, DEFAULT_COLOR))[1]


def chart_main_leaderboard(data: list[dict], out_path: Path):
    data = sorted(data, key=lambda x: x["score"])
    labels = [label_for(d["model"]) for d in data]
    scores = [d["score"] for d in data]
    colors = [color_for(d["model"]) for d in data]
    pcts = [s / 720 * 100 for s in scores]

    fig, ax = plt.subplots(figsize=(11, max(5, 0.85 * len(data) + 1.5)), dpi=200)
    fig.patch.set_facecolor("white")
    bars = ax.barh(labels, scores, color=colors, edgecolor="white", linewidth=1.5, height=0.72)

    ax.axvline(715, color="#E63946", linestyle="--", linewidth=1.4, alpha=0.8, zorder=0)
    ax.text(715, len(labels) - 0.35, "  AIR-1 cutoff zone", color="#E63946",
            fontsize=10, fontweight="bold", va="bottom")

    for bar, score, pct in zip(bars, scores, pcts):
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
                f"{score}  ({pct:.1f}%)", va="center", fontsize=11, fontweight="bold",
                color="#222")

    ax.set_xlim(0, 800)
    ax.set_xlabel("Score (out of 720)", fontsize=12, fontweight="bold")
    ax.set_title("NEET 2026 — Frontier AI Models Take India's Hardest Medical Exam",
                 fontsize=15, pad=14)
    ax.text(0, len(labels) + 0.35,
            "180 vision questions • zero shot • image input • exam date: 3 May 2026",
            fontsize=10.5, color="#666", style="italic")
    ax.tick_params(axis="y", labelsize=11.5)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.22, linestyle="-", linewidth=0.6)

    fig.text(0.99, 0.01, "huggingface.co/datasets/Reja1/jee-neet-benchmark", ha="right",
             fontsize=8.5, color="#999", style="italic")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")
    plt.close()


def chart_cost_vs_accuracy(data: list[dict], out_path: Path):
    # Per-model label offsets (dx_log_factor, dy_pct, ha) to keep labels readable.
    # dx_factor < 1 → label sits LEFT of point; > 1 → RIGHT. ha matches.
    LABEL_OFFSETS = {
        "google/gemini-3-flash-preview":     (1.07, +1.30, "left"),   # top of cluster, label up-right
        "google/gemini-3.1-pro-preview":     (0.93, -0.20, "right"),  # rightmost point, label LEFT
        "openai/gpt-5.5":                    (1.07, +1.30, "left"),   # label up-right (clear of Flash)
        "openai/gpt-5.4":                    (1.07, +1.20, "left"),
        "qwen/qwen3-vl-235b-a22b-thinking":  (0.93, +0.10, "right"),  # label LEFT (clear of Grok)
        "anthropic/claude-sonnet-4.6":       (1.07, -1.30, "left"),   # label down-right (clear of Grok)
        "anthropic/claude-haiku-4.5":        (1.07, +1.30, "left"),
        "z-ai/glm-4.6v":                     (1.07, +1.30, "left"),
        "x-ai/grok-4.3":                     (1.07, -1.40, "left"),   # label down-right (clear of Qwen + cluster)
    }
    DEFAULT_OFFSET = (1.07, +0.80, "left")

    fig, ax = plt.subplots(figsize=(13, 8), dpi=200)
    fig.patch.set_facecolor("white")

    points = [(d, d["score"] / 720 * 100) for d in data if d["cost"] > 0]

    # Pareto frontier: cheapest cost for each accuracy level (upper envelope).
    pareto_pts = sorted([(d["cost"], pct) for d, pct in points])
    frontier, best = [], -1
    for c, p in pareto_pts:
        if p > best:
            frontier.append((c, p))
            best = p
    if frontier:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "--", color="#888", linewidth=1.5, alpha=0.55, zorder=1,
                label="Pareto frontier")
        ax.fill_between(fx, fy, [102] * len(fx), color="#88CC88", alpha=0.06, zorder=0)

    on_frontier = set(frontier)
    for d, pct in points:
        is_pareto = (d["cost"], pct) in on_frontier
        ax.scatter(d["cost"], pct, s=360, color=color_for(d["model"]),
                   edgecolors="#222" if is_pareto else "white",
                   linewidths=2.2 if is_pareto else 2,
                   zorder=4, alpha=0.97)

        dx_factor, dy, ha = LABEL_OFFSETS.get(d["model"], DEFAULT_OFFSET)
        label_x = d["cost"] * dx_factor
        ax.annotate(
            f"{label_for(d['model'])}\n${d['cost']:.2f}",
            (d["cost"], pct),
            xytext=(label_x, pct + dy),
            fontsize=10, fontweight="bold", color="#222", ha=ha,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="none", alpha=0.85),
            zorder=5,
        )

    ax.set_xscale("log")
    # Explicit log-spaced ticks with $ formatting so the cost axis is readable.
    tick_locs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    ax.xaxis.set_major_locator(mticker.FixedLocator(tick_locs))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:.2f}" if x < 1 else f"${x:.1f}"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    costs = [d["cost"] for d, _ in points]
    ax.set_xlim(min(costs) * 0.7, max(costs) * 1.5)

    ax.set_xlabel("Total cost for full exam (USD, log scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("NEET 2026 — Cost vs Accuracy (upper-left = best value)",
                 fontsize=14, pad=14)
    pcts = [pct for _, pct in points]
    ax.set_ylim(min(pcts) - 5, 102)
    ax.grid(which="major", axis="both", alpha=0.28, linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    fig.text(0.99, 0.01, "huggingface.co/datasets/Reja1/jee-neet-benchmark",
             ha="right", fontsize=8.5, color="#999", style="italic")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")
    plt.close()


def chart_subject_heatmap(data: list[dict], out_path: Path):
    data = sorted(data, key=lambda x: -x["score"])
    matrix = np.array([
        [d["subjects"].get(s, 0) / NEET_SUBJECT_MAX[s] * 100 for s in NEET_SUBJECTS]
        for d in data
    ])
    labels = [label_for(d["model"]) for d in data]

    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.8 * len(data) + 1.5)), dpi=200)
    fig.patch.set_facecolor("white")

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=40, vmax=100, aspect="auto")
    ax.set_xticks(range(len(NEET_SUBJECTS)))
    ax.set_xticklabels(NEET_SUBJECTS, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xticks(np.arange(-0.5, len(NEET_SUBJECTS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if val < 65 else "#222"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    ax.set_title("Per-Subject Accuracy — NEET 2026", fontsize=14, pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("% correct", fontsize=10)

    fig.text(0.99, 0.01, "huggingface.co/datasets/Reja1/jee-neet-benchmark",
             ha="right", fontsize=8.5, color="#999", style="italic")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--output-dir", default="charts/neet_2026")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_results(Path(args.results_dir))
    if not data:
        raise SystemExit(f"No NEET 2026 runs found under {args.results_dir}")
    print(f"loaded {len(data)} runs from {args.results_dir}")
    for d in data:
        print(f"  {label_for(d['model']):30s} {d['score']:3d}/720  ${d['cost']:.4f}")

    chart_main_leaderboard(data, out_dir / "01_leaderboard.png")
    chart_cost_vs_accuracy(data, out_dir / "02_cost_vs_accuracy.png")
    chart_subject_heatmap(data, out_dir / "03_subject_heatmap.png")
    print(f"\ncharts written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
