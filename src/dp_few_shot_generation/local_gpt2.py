import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import AsyncGenerator

import torch
import tiktoken
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmapi.lm import CompletionsSettings, SampledToken


@dataclass
class LocalToken:
    text: str
    bytes: bytes
    logprob: float
    token_id: int


class LocalGPT2:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.device = next(self.model.parameters()).device

        # GPT-2 uses the same token IDs as the gpt2 tiktoken encoding.
        self.encoding = tiktoken.get_encoding("gpt2")

    def _encode_prompt(self, prompt):
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt)
        return list(prompt)

    def _complete_one(self, prompt, settings):
        settings = settings or CompletionsSettings()
        input_ids = self._encode_prompt(prompt)

        
        if settings.echo:
            if not input_ids:
                return []

            x = torch.tensor(
                [input_ids], dtype=torch.long, device=self.device
            )

            with torch.no_grad():
                logits = self.model(x).logits[0].float()
                log_probs = torch.log_softmax(logits, dim=-1)

            results = []
            for pos in range(1, len(input_ids)):
                token_id = input_ids[pos]
                token_text = self.tokenizer.decode([token_id])

                token = LocalToken(
                    text=token_text,
                    bytes=token_text.encode("utf-8", errors="replace"),
                    logprob=float(log_probs[pos - 1, token_id].item()),
                    token_id=token_id,
                )
                results.append(
                    SampledToken(token=token, top_choices=(token,))
                )

            return results

        if not input_ids:
            input_ids = [self.tokenizer.eos_token_id]

        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(x).logits[0, -1].float()

        if settings.logit_bias:
            for token_id, bias in settings.logit_bias.items():
                if 0 <= token_id < logits.numel():
                    logits[token_id] += float(bias)

        temperature = settings.temperature
        if temperature is not None and temperature > 0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        top_k = settings.logprobs or 0
        if top_k > 0:
            k = min(top_k, logits.numel())
            top_ids = torch.topk(logits, k).indices.tolist()
        else:
            top_ids = []
        token_id = int(torch.argmax(logits).item())
        token_logprob = float(log_probs[token_id].item())

        token = LocalToken(
            text=self.tokenizer.decode([token_id]),
            bytes=self.tokenizer.decode(
                [token_id], clean_up_tokenization_spaces=False
            ).encode("utf-8", errors="replace"),
            logprob=token_logprob,
            token_id=token_id,
        )

        choices = tuple(
            LocalToken(
                text=self.tokenizer.decode([i]),
                bytes=self.tokenizer.decode(
                    [i], clean_up_tokenization_spaces=False
                ).encode("utf-8", errors="replace"),
                logprob=float(log_probs[i].item()),
                token_id=i,
            )
            for i in top_ids
        )

        return [SampledToken(token=token, top_choices=choices)]

    async def completions(self, prompt, settings=None):
        if isinstance(prompt, Sequence) and prompt and isinstance(prompt[0], (list, tuple)):
            prompts = prompt
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
            prompts = prompt
        else:
            prompts = [prompt]

        return [
            self._complete_one(p, settings)
            for p in prompts
        ]

    async def streaming_completions(
        self, prompt, settings=None
    ) -> Sequence[AsyncGenerator[SampledToken, None]]:
        raise NotImplementedError("Streaming is not implemented yet")
