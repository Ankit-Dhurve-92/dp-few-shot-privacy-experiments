
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from opacus import PrivacyEngine

MODEL_NAME = "gpt2"
SEED = 42
NUM_MEMBERS = 100
NUM_NON_MEMBERS = 100
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 1
TARGET_EPSILON = 8.0
TARGET_DELTA = 1e-5
MAX_GRAD_NORM = 1.0

OUTPUT_DIR = Path("outputs/gpt2_mia_dp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading DBPedia...")
    dataset = load_dataset("dbpedia_14")
    train_split = dataset["train"].shuffle(seed=SEED)

    members = train_split.select(range(NUM_MEMBERS))
    non_members = train_split.select(
        range(NUM_MEMBERS, NUM_MEMBERS + NUM_NON_MEMBERS)
    )

    print(f"Training members: {len(members)}")
    print(f"Held-out non-members: {len(non_members)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["content"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        encoded["labels"] = [
            [
                token if mask == 1 else -100
                for token, mask in zip(ids, attention_mask)
            ]
            for ids, attention_mask in zip(
                encoded["input_ids"], encoded["attention_mask"]
            )
        ]
        return encoded

    print("Tokenizing training members...")
    train_data = members.map(
        tokenize_batch,
        batched=True,
        remove_columns=members.column_names,
    )

    train_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=default_data_collator,
    )

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    privacy_engine = PrivacyEngine()

    print("Preparing DP-SGD...")
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=TARGET_EPSILON,
        target_delta=TARGET_DELTA,
        epochs=EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
        grad_sample_mode="hooks",
    )

    model.train()
    print("Starting differentially private training...")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad()
            batch_size, seq_len = batch["input_ids"].shape
            position_ids = torch.arange(
                seq_len, device=device
            ).unsqueeze(0).expand(batch_size, seq_len)
            outputs = model(**batch, position_ids=position_ids)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            if steps % 5 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS}, "
                    f"Step {steps}, Loss: {loss.item():.4f}"
                )

        print(f"Epoch average loss: {total_loss / max(steps, 1):.4f}")

    epsilon = privacy_engine.get_epsilon(TARGET_DELTA)

    # Save the trained model without Opacus wrappers.
    model_to_save = model._module
    model_to_save.save_pretrained(OUTPUT_DIR / "model")
    tokenizer.save_pretrained(OUTPUT_DIR / "model")

    metadata = {
        "model_name": MODEL_NAME,
        "dataset": "dbpedia_14",
        "seed": SEED,
        "num_members": NUM_MEMBERS,
        "num_non_members": NUM_NON_MEMBERS,
        "member_split": "first 100 rows of shuffled train split",
        "non_member_split": "next 100 rows of same shuffled train split",
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "target_epsilon": TARGET_EPSILON,
        "achieved_epsilon": epsilon,
        "delta": TARGET_DELTA,
        "max_grad_norm": MAX_GRAD_NORM,
        "noise_multiplier": float(optimizer.noise_multiplier),
        "privacy_method": "DP-SGD using Opacus",
    }

    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAchieved privacy budget: epsilon={epsilon:.4f}, delta={TARGET_DELTA}")
    print(f"DP model and metadata saved to: {OUTPUT_DIR.resolve()}")
    print("DP training complete.")


if __name__ == "__main__":
    main()
