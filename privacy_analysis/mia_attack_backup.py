
import csv
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score

MODEL_NAME = "outputs/gpt2_mia/model"
NUM_SAMPLES = 100
MAX_LENGTH = 128
SEED = 42

RESULTS_DIR = Path("privacy_analysis/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_loss(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    return outputs.loss.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print("Loading DBPedia...")
    dataset = load_dataset("dbpedia_14")
    train_split = dataset["train"].shuffle(seed=SEED)

    members = train_split.select(range(NUM_SAMPLES))
    non_members = train_split.select(
        range(NUM_SAMPLES, NUM_SAMPLES * 2)
    )

    results = []

    for split_name, samples, is_member in [
        ("member", members, 1),
        ("non_member", non_members, 0),
    ]:
        for i, sample in enumerate(samples):
            loss = calculate_loss(
                sample["content"], tokenizer, model, device
            )

            results.append({
                "split": split_name,
                "is_member": is_member,
                "loss": loss,
                "attack_score": -loss,
            })

            if (i + 1) % 10 == 0:
                print(f"{split_name}: {i + 1}/{NUM_SAMPLES}")

    labels = [row["is_member"] for row in results]
    scores = [row["attack_score"] for row in results]
    auc = roc_auc_score(labels, scores)

    csv_path = RESULTS_DIR / "mia_dp_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "is_member", "loss", "attack_score"],
        )
        writer.writeheader()
        writer.writerows(results)

    member_losses = [
        row["loss"] for row in results if row["is_member"] == 1
    ]
    non_member_losses = [
        row["loss"] for row in results if row["is_member"] == 0
    ]

    print("\n===== MIA DP MODEL RESULTS =====")
    print(f"Members: {len(member_losses)}")
    print(f"Non-members: {len(non_member_losses)}")
    print(f"Member mean loss: {sum(member_losses) / len(member_losses):.4f}")
    print(
        f"Non-member mean loss: "
        f"{sum(non_member_losses) / len(non_member_losses):.4f}"
    )
    print(f"ROC-AUC: {auc:.4f}")
    print("Random-guess reference: 0.5000")
    print(f"Per-example scores saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
