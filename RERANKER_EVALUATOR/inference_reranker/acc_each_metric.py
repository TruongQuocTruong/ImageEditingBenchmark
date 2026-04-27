import json
import argparse
from collections import defaultdict

METRIC_LABELS = {
    "aesthetic": "Aesthetic",
    "realism": "Realism",
    "fidelity": "Fidelity",
    "background consistency": "BC",
    "foreground consistency": "FC",
    "structure consistency": "SC",
}

DISPLAY_ORDER = [
    "aesthetic",
    "realism",
    "fidelity",
    "background consistency",
    "foreground consistency",
    "structure consistency",
]


def get_metric(item_id: str) -> str:
    normalized = item_id.replace("_", " ")
    for metric in METRIC_LABELS:
        if normalized.startswith(metric):
            return metric
    return "unknown"


def compute_accuracies(json_path: str) -> dict:
    with open(json_path) as f:
        data = json.load(f)

    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in data["details"]:
        metric = get_metric(item["id"])
        counts[metric]["total"] += 1
        if item["match"]:
            counts[metric]["correct"] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description="Compute per-metric accuracy from inference JSON.")
    parser.add_argument("json_path", help="Path to the inference result JSON file")
    parser.add_argument(
        "--mode",
        choices=["normal", "ratio"],
        default="ratio",
        help="Overall calculation mode: 'normal' = simple mean of per-metric accuracies, "
             "'ratio' = weighted by sample count (total_correct / total_samples)",
    )
    args = parser.parse_args()

    counts = compute_accuracies(args.json_path)

    total_correct = 0
    total_all = 0
    metric_accs = []

    for metric in DISPLAY_ORDER:
        label = METRIC_LABELS[metric]
        c = counts.get(metric, {"correct": 0, "total": 0})
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0.0
        print(f"- {label}: {acc:.3f}  ({c['correct']}/{c['total']})")
        total_correct += c["correct"]
        total_all += c["total"]
        metric_accs.append(acc)

    if args.mode == "ratio":
        overall = sum(
            acc * counts[m]["total"] / total_all
            for acc, m in zip(metric_accs, DISPLAY_ORDER)
            if total_all > 0
        )
        print(f"Overall: {overall:.3f}")
    else:
        overall = sum(metric_accs) / len(metric_accs) if metric_accs else 0.0
        print(f"Overall: {overall:.3f}")


if __name__ == "__main__":
    main()
