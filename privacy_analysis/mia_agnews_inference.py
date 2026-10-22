
import json
import re
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import roc_auc_score

SEED = 42
NUM_MEMBERS = 100
NUM_NONMEMBERS = 100
GENERATED_FILE = Path("privacy_analysis/results/agnews_dp_synthetic.json")


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def main():
    with open(GENERATED_FILE, encoding="utf-8") as f:
        generated = json.load(f)

    dataset = load_dataset("ag_news")
    shuffled = dataset["train"].shuffle(seed=SEED)

    members = [
        normalize(x["text"])
        for x in shuffled.select(range(NUM_MEMBERS))
    ]
    nonmembers = [
        normalize(x["text"])
        for x in dataset["test"].shuffle(seed=SEED).select(
            range(NUM_NONMEMBERS)
        )
    ]

    generated_texts = [
        normalize(x.get("text", x.get("generated_text", "")))
        for x in generated
    ]

    def leakage_score(text):
        # Fraction of generated examples containing this record,
        # or this record containing a substantial generated passage.
        if not text:
            return 0.0

        best = 0.0
        for output in generated_texts:
            if not output:
                continue
            if len(text) >= 50 and text in output:
                best = max(best, 1.0)
            elif len(output) >= 50 and output in text:
                best = max(best, 1.0)
        return best

    member_scores = [leakage_score(x) for x in members]
    nonmember_scores = [leakage_score(x) for x in nonmembers]

    y_true = [1] * len(member_scores) + [0] * len(nonmember_scores)
    scores = member_scores + nonmember_scores

    auc = roc_auc_score(y_true, scores)
    print("AGNews inference-output leakage audit")
    print(f"Generated examples: {len(generated_texts)}")
    print(f"Members checked: {len(member_scores)}")
    print(f"Non-members checked: {len(nonmember_scores)}")
    print(f"Member exact/containment matches: {sum(member_scores)}")
    print(f"Non-member exact/containment matches: {sum(nonmember_scores)}")
    print(f"ROC-AUC: {auc:.4f}")
    print("Note: this is a simple exact/containment leakage audit,")
    print("not a comprehensive membership-inference attack or DP proof.")


if __name__ == "__main__":
    main()
