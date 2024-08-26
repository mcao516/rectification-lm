#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import torch
import random
import torch.nn as nn

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinNewTokensLengthLogitsProcessor,
    LogitsProcessor,
)
from transformers import set_seed

from lm_with_value_heads import CausalLMWithValueHeadsInference

set_seed(1)


class SecurityLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for LM rectification. See [Systematic Rectification of Language
    Models via Dead-end Analysis](https://arxiv.org/abs/2302.14003) for more information.

    Args:
        eps (`float`, *optional*, defaults to 0):
            The threshold removes any action with a value of 1 + Q(s, a) below it.
        p_min (`float`, *optional*, defaults to 0):
            The minimum value that used to clamp 1 + Q(s, a).
        p_max (`float`, *optional*, defaults to 1):
            The maximum value that used to clamp 1 + Q(s, a).
    """

    def __init__(
        self,
        rectlm,
        eps: float = 0.0,
        p_min: float = 1e-7,
        p_max: float = 1.0,
        pad_token: int = 50256,
    ):
        self.eps = eps
        self.p_min = p_min
        self.p_max = p_max
        self.rectlm = rectlm
        self.pad_token = pad_token

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
                search or log softmax for each vocabulary token when using beam search

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.
        """
        attn_mask = (input_ids != self.pad_token).long()
        security_scores = self.rectlm(input_ids, attention_mask=attn_mask).logits[:, -1, :]
        lm_probs = nn.functional.softmax(scores, dim=-1)

        security_scores = torch.clamp(security_scores, max=0.0, min=-1.0)
        secure_probs = 1 + security_scores - self.eps
        secure_probs = torch.clamp(secure_probs, max=self.p_max, min=self.p_min)

        for _ in range(3):
            mask = lm_probs > secure_probs
            lm_probs[mask] = secure_probs[mask]

            # re-normalize
            lm_probs = lm_probs / lm_probs.sum(dim=1, keepdim=True)

        # Perform the log-sum-exp operation
        lse = torch.logsumexp(lm_probs, dim=1).unsqueeze(-1)
        logits = torch.log(lm_probs) + lse

        return logits


def model_sample(
    model,
    tokenizer,
    security_model,
    input_prompt,
    min_new_tokens = 10,
    max_new_tokens = 20,
    eps = 0.0,
    p_min = 1e-7,
    p_max = 1.0,
    do_sample = False,
    num_return_sequences = 1,
):
    model_inputs = tokenizer(input_prompt, return_tensors="pt", padding='longest').to("cuda")
    input_ids, _ = model_inputs["input_ids"], model_inputs["attention_mask"]

    input_ids_length = input_ids.shape[-1]
    logits_processor = LogitsProcessorList(
        [
            MinNewTokensLengthLogitsProcessor(
                input_ids_length,
                min_new_tokens=min_new_tokens,
                eos_token_id=model.generation_config.eos_token_id,
                device=model.device),
            SecurityLogitsProcessor(security_model, eps=eps, p_min=p_min, p_max=p_max),
        ]
    )

    outputs = model.generate(
        **model_inputs,
        generation_config=model.generation_config,
        logits_processor=logits_processor,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        temperature=1.0,
        repetition_penalty=1.0,
    )

    return tokenizer.batch_decode(outputs[:, input_ids_length:], skip_special_tokens=True)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load language model
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.lm_name_or_path,
        # torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=None)
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    model.to(device)

    # load value function Q
    control_model = CausalLMWithValueHeadsInference.from_pretrained(args.q_model_path)
    control_model.to(device)
    # control_model.half()
    control_model.eval()

    # load evaluation dataset
    prompts = []
    with open(args.prompts_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            prompts.append(sample)

    if args.num_samples_for_eval > 0:
        eval_data = random.sample(prompts, min(len(prompts), args.num_samples_for_eval))
    else:
        eval_data = prompts

    print("{} prompts sampled for evaluation.".format(len(eval_data)))

    total_batches = (len(eval_data) + args.batch_size - 1) // args.batch_size    
    for i in tqdm(range(total_batches)):
        start_idx, end_idx = i * args.batch_size, min((i + 1) * args.batch_size, len(eval_data))
        batch = eval_data[start_idx:end_idx]
    
        batch_gens = model_sample(
            model,
            tokenizer,
            control_model,
            [d['prompt']['text'] for d in batch],
            eps=args.epsilon,
            do_sample=args.do_sample,
            num_return_sequences=args.num_returns
        )
    
        for i, b in enumerate(batch):
            b['generations'] = [{"text": g} for g in batch_gens[i * args.num_returns: (i+1) * args.num_returns]]
    
    with open(args.save_path, "w") as jsonl_file:
        for item in eval_data:
            json_line = json.dumps(item)
            jsonl_file.write(json_line + "\n")

    print("Generations saved at: ", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build reward model training data.")

    parser.add_argument(
        "--lm_name_or_path",
        type=str,
        default="gpt2-large",
        help="The base language model used for sampling generations."
    )
    parser.add_argument(
        "--q_model_path",
        type=str,
        default="models/",
        help="Trained Q model saving path."
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default="nontoxic_prompts-10k.jsonl",
        help="Prompts used for evaluation."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="generations.jsonl",
        help="Path for saving generations."
    )
    parser.add_argument(
        "--num_returns",
        type=int,
        default=25,
        help="Number of responses sampled per prompt."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Inference batch size."
    )
    parser.add_argument(
        "--num_samples_for_eval",
        type=int,
        default=-1,
        help="Number of samples for evaluation."
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Epsilon controls the threshold mechanism. \
        Generally, a larger epsilon imposes stricter safety conditions. \
        Thus, the larger the epsilon, the less toxic the generated content will be."
    )

    args = parser.parse_args()
    main(args)
