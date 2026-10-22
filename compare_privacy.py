import asyncio
import os
import sys
from datasets import load_dataset
import numpy as np

torch_lib = os.path.abspath(
    os.path.join(
        os.path.dirname(sys.executable),
        "..", "Lib", "site-packages", "torch", "lib"
    )
)
_dll_handle = os.add_dll_directory(torch_lib)

import torch
from src.dp_few_shot_generation.local_gpt2 import LocalGPT2
from src.dp_few_shot_generation.run_exp_agnews import generate_with_private_prompts

async def generate(trainset, lm, sigma):
    return await generate_with_private_prompts(
        trainset=trainset,
        num_private_train=4,
        num_private_train_splits=2,
        instruction="Generate a short news article.\n",
        public_train_prompt="",
        stop_tokens=set(),
        test_input=2,
        lm=lm,
        noise_rng=np.random.RandomState(42),
        sigma=sigma,
        labels=["World", "Sports", "Business", "Technology"],
        top_p=1.0,
        no_public_token=True,
        subsample_per_token=False,
        sample_same_label_prompts=False,
        gen_seed=42,
        max_tokens=10,
        normalize_max=False,
    )

async def main():
    print("Loading dataset...")
    data = load_dataset("ag_news", split="train[:20]")
    original = data.select(range(4))
    modified = data.select([1, 2, 3, 4])
    lm = LocalGPT2()

    for sigma in [0, 0.1, 0.5, 1]:
        print(f"\n===== SIGMA {sigma} =====")
        a = await generate(original, lm, sigma)
        b = await generate(modified, lm, sigma)
        print("Original:", lm.encoding.decode(a))
        print("Modified:", lm.encoding.decode(b))
        print("Identical:", a == b)
        print("Matching positions:", sum(x == y for x, y in zip(a, b)), "/ 10")

if __name__ == "__main__":
    asyncio.run(main())
