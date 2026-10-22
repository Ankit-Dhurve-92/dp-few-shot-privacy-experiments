
import json
import re
from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

SEED = 42
NUM_RECORDS = 100
RESULTS = Path("privacy_analysis/results")
DP_FILE = RESULTS / "agnews_dp_synthetic.json"
BASELINE_FILE = RESULTS / "agnews_baseline_synthetic.json"


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_texts(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [
        normalize(row.get("text", row.get("generated_text", "")))
        for row in rows
    ]


def audit(generated_texts, members, nonmembers):
    def score(record):
        for output in generated_texts:
            if len(record) >= 50 and record in output:
                return 1.0
            if len(output) >= 50 and output in record:
                return 1.0
        return 0.0

    member_scores = [score(x) for x in members]
    nonmember_scores = [score(x) for x in nonmembers]
    labels = [1] * len(member_scores) + [0] * len(nonmember_scores)
    scores = member_scores + nonmember_scores

    return {
        "auc": roc_auc_score(labels, scores),
        "member_matches": int(sum(member_scores)),
        "nonmember_matches": int(sum(nonmember_scores)),
        "generated_count": len(generated_texts),
    }


def main():
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {BASELINE_FILE}. Generate and save baseline outputs first."
        )
    if not DP_FILE.exists():
        raise FileNotFoundError(f"Missing {DP_FILE}")

    dataset = load_dataset("ag_news")
    train = dataset["train"].shuffle(seed=SEED)
    members = [
        normalize(x["text"])
        for x in train.select(range(NUM_RECORDS))
    ]
    nonmembers = [
        normalize(x["text"])
        for x in dataset["test"].shuffle(seed=SEED).select(
            range(NUM_RECORDS)
        )
    ]

    baseline = audit(load_texts(BASELINE_FILE), members, nonmembers)
    dp = audit(load_texts(DP_FILE), members, nonmembers)

    print("\n=== AGNews leakage audit ===")
    for name, result in [("Baseline", baseline), ("Inference-time DP", dp)]:
        print(f"\n{name}")
        print(f"Generated examples: {result['generated_count']}")
        print(f"Member matches: {result['member_matches']}/{NUM_RECORDS}")
        print(f"Non-member matches: {result['nonmember_matches']}/{NUM_RECORDS}")
        print(f"ROC-AUC: {result['auc']:.4f}")

    print("\nCaution: exact/containment audit only; not a comprehensive MIA or DP proof.")


if __name__ == "__main__":
    main()
