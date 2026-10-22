
from datasets import load_dataset
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "outputs/gpt2_mia/model"
OUTPUT_DIR = Path("privacy_analysis/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_PROMPTS = 20
MAX_NEW_TOKENS = 80
SEED = 42

PROMPTS = [
    "The history of science",
    "The city of",
    "The company was founded",
    "The main purpose of education",
    "The development of technology",
    "The history of India",
    "A famous scientist",
    "The capital city",
    "The research paper explains",
    "The natural environment",
    "The government announced",
    "The university was established",
    "The discovery of",
    "The main characteristics of",
    "The organization works",
    "The ancient civilization",
    "The new technology",
    "The scientific study",
    "The population of",
    "The important contribution",
]


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading fine-tuned GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    
    print("Loading training members for comparison...")
    dataset = load_dataset("dbpedia_14")
    train_split = dataset["train"].shuffle(seed=SEED)
    members = train_split.select(range(100))

    member_texts = [
        " ".join(sample["content"].lower().split())
        for sample in members
    ]

    

    results = []

    print("Generating text...")
    for i, prompt in enumerate(PROMPTS[:NUM_PROMPTS], start=1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
        })

        print(f"\n[{i}/{NUM_PROMPTS}] Prompt: {prompt}")
        print(f"Generated: {generated_text}")

    output_file = OUTPUT_DIR / "data_extraction_baseline.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} generations to: {output_file.resolve()}")



    # Compare generated text with training members
    exact_matches = 0

    for result in results:
        generated = " ".join(
            result["generated_text"].lower().split()
        )

        for member in member_texts:
            if len(member) >= 50 and len(generated) >= 50:
                if member in generated or generated in member:
                    exact_matches += 1
                    break

    summary = {
        "num_prompts": len(results),
        "member_examples_checked": len(member_texts),
        "exact_or_containment_matches": exact_matches,
    }

    summary_file = OUTPUT_DIR / "data_extraction_summary.json"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===== DATA EXTRACTION BASELINE =====")
    print(f"Prompts tested: {len(results)}")
    print(f"Member examples checked: {len(member_texts)}")
    print(f"Candidate matches: {exact_matches}")
    print(f"Summary saved to: {summary_file.resolve()}")


if __name__ == "__main__":
    main()
    

