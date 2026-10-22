import csv
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score

MODELS = {
    "Baseline": "outputs/gpt2_mia/model",
    "DP-SGD": "outputs/gpt2_mia_dp/model",
}

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
    inputs = {k: v.to(device) for k, v in inputs.items()}

    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    return outputs.loss.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = load_dataset("dbpedia_14")
    shuffled = dataset["train"].shuffle(seed=SEED)

    members = shuffled.select(range(NUM_SAMPLES))
    non_members = shuffled.select(range(NUM_SAMPLES, NUM_SAMPLES * 2))

    all_results = []

    for model_name, model_path in MODELS.items():
        print(f"\nEvaluating {model_name}: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model.eval()

        scores = []
        labels = []

        for split_name, samples, is_member in [
            ("member", members, 1),
            ("non_member", non_members, 0),
        ]:
            for i, sample in enumerate(samples):
                loss = calculate_loss(
                    sample["content"], tokenizer, model, device
                )
                scores.append(-loss)
                labels.append(is_member)

                all_results.append({
                    "model": model_name,
                    "split": split_name,
                    "is_member": is_member,
                    "loss": loss,
                    "attack_score": -loss,
                })

                if (i + 1) % 10 == 0:
                    print(f"  {split_name}: {i + 1}/{NUM_SAMPLES}")

        auc = roc_auc_score(labels, scores)
        print(f"{model_name} ROC-AUC: {auc:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = RESULTS_DIR / "mia_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "split", "is_member", "loss", "attack_score"
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nDetailed scores saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
