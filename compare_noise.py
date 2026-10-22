
import os
import sys

# Fix Windows PyTorch DLL loading
torch_lib = os.path.abspath(
    os.path.join(
        os.path.dirname(sys.executable),
        "..", "Lib", "site-packages", "torch", "lib"
    )
)
_dll_handle = os.add_dll_directory(torch_lib)

import torch
import asyncio
import numpy as np
from datasets import load_dataset

from src.dp_few_shot_generation.local_gpt2 import LocalGPT2
from src.dp_few_shot_generation.run_exp_agnews import (
    generate_with_private_prompts,
)


async def main():
    print("Loading AG News dataset...", flush=True)
    trainset = load_dataset("ag_news", split="train[:20]")

    lm = LocalGPT2()
    labels = ["World", "Sports", "Business", "Technology"]

    
    for sigma in [0, 0.1, 0.5, 1]:
        print(f"\n{'=' * 35}")
        print(f"SIGMA = {sigma}")
        print("=" * 35)

        for seed in range(5):
            result = await generate_with_private_prompts(
                trainset=trainset,
                num_private_train=4,
                num_private_train_splits=2,
                instruction="Generate a short news article.\n",
                public_train_prompt="",
                stop_tokens=set(),
                test_input=2,
                lm=lm,
                noise_rng=np.random.RandomState(seed),
                sigma=sigma,
                labels=labels,
                top_p=1.0,
                no_public_token=True,
                subsample_per_token=False,
                sample_same_label_prompts=False,
                gen_seed=seed,
                max_tokens=10,
                normalize_max=False,
            )

            text = lm.encoding.decode(result)
            print(f"Seed {seed}: {text}")


if __name__ == "__main__":
    asyncio.run(main())
