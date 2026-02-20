"""
Check pipeline decisions against expected labels.

Usage: uv run python scripts/evaluate.py
"""

import json
from pathlib import Path


def evaluate():
    decisions = json.loads(Path("outputs/task2_decisions.json").read_text())
    ground_truth = json.loads(Path("data/ground_truth.json").read_text())

    print(f"\n{'='*50}")
    print(f" Evaluation: {len(ground_truth)} articles")
    print(f"{'='*50}\n")

    correct = 0
    for i, (gt, pred) in enumerate(zip(ground_truth, decisions), 1):
        expected = gt["expected_decision"]
        got = pred["decision"]
        match = expected == got
        correct += match
        print(f"Article {i}: expected {expected}, got {got}")

    total = len(ground_truth)
    accuracy = correct / total if total else 0
    print(f"\n  Result: {correct}/{total} correct ({accuracy:.0%})\n")


if __name__ == "__main__":
    evaluate()