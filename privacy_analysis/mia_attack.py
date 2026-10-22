
import csv
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score

MODEL_NAME = "outputs/gpt2_mia/model"
NUM_SAMPLES = 100
MAX_LENGTH = 128
SEED = 42

# Standard deviation of Gaussian noise added to logits.
# This is an experimental noise scale, NOT an epsilon guarantee.
NOISE_STD = 0.1

RESULTS_DIR = Path("privacy_analysis/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_loss(text, tokenizer, model, device, noise_std=0.0):
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
        outputs = model(**inputs)
        logits = outputs.logits

        # Next-token prediction: logits at position t predict token t+1.
        shift_logits = logits[:, :-1, :].clone()
        shift_labels = labels[:, 1:]

        if noise_std > 0:
            noise = torch.randn_like(shift_logits) * noise_std
            shift_logits = shift_logits + noise

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

    return float(loss.item())


def run_attack(model, tokenizer, device, members, nonmembers, noise_std):
    rows = []

    for split_name, samples, is_member in [
        ("member", members, 1),
        ("non_member", nonmembers, 0),
    ]:
        for i, sample in enumerate(samples):
            loss = calculate_loss(
                sample["content"],
                tokenizer,
                model,
                device,
                noise_std=noise_std,
            )
            rows.append({
                "split": split_name,
                "is_member": is_member,
                "loss": loss,
                "attack_score": -loss,
            })

            if (i + 1) % 10 == 0:
                print(
                    f"noise_std={noise_std}: "
                    f"{split_name}: {i + 1}/{len(samples)}"
                )

    labels = [r["is_member"] for r in rows]
    scores = [r["attack_score"] for r in rows]
    auc = roc_auc_score(labels, scores)

    member_losses = [r["loss"] for r in rows if r["is_member"] == 1]
    nonmember_losses = [r["loss"] for r in rows if r["is_member"] == 0]

    return rows, auc, float(np.mean(member_losses)), float(np.mean(nonmember_losses))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = load_dataset("dbpedia_14")
    train_split = dataset["train"].shuffle(seed=SEED)
    members = train_split.select(range(NUM_SAMPLES))
    nonmembers = train_split.select(
        range(NUM_SAMPLES, NUM_SAMPLES * 2)
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    all_results = []
    noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0]
    for noise_std in noise_levels:
        name = "baseline_no_added_noise" if noise_std == 0 else f"noise_{noise_std}"
        print(f"\n===== {name} (noise_std={noise_std}) =====")
        rows, auc, member_loss, nonmember_loss = run_attack(
            model, tokenizer, device, members, nonmembers, noise_std
        )
        print(f"Member mean loss: {member_loss:.4f}")
        print(f"Non-member mean loss: {nonmember_loss:.4f}")
        print(f"ROC-AUC: {auc:.4f}")

        for row in rows:
            row["condition"] = name
        all_results.extend(rows)

    output_file = RESULTS_DIR / "mia_inference_noise_comparison.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition", "split", "is_member", "loss", "attack_score"
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSaved results to: {output_file.resolve()}")
    print(
        "\nCaution: this tests noisy per-token logits on a fixed model. "
        "It does not establish formal differential privacy."
    )


if __name__ == "__main__":
    main()
