#!/usr/bin/env python
# coding=utf-8

import json
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig

from generation import CustomizedGenerationMixin


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, and `torch`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def process_prompt(text):
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    return text


def batch_generate(batch, generator, tokenizer, secure_threshold=0.3, secure_top_k=30, secure_prob_min=0.0, num_return_sequences=1):
    encoded_prompt = tokenizer(batch, padding='longest', add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)

    input_ids, attention_mask = encoded_prompt['input_ids'], encoded_prompt['attention_mask']

    prompt_len = input_ids.shape[1]
    output_sequences = generator.generate(
        input_ids,
        attention_mask=attention_mask,
        min_length=10,
        max_length=20+prompt_len,
        temperature=1.0,
        repetition_penalty=1.0,
        do_sample=True,
        # top_p=0.99,
        top_k=secure_top_k,
        # num_beams=3,
        num_return_sequences=num_return_sequences,
        secure_threshold=secure_threshold,
        secure_top_k=secure_top_k,
        secure_prob_min=secure_prob_min
    )

    batch_gens = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    batch_cons = tokenizer.batch_decode(output_sequences[:, prompt_len:], skip_special_tokens=True)

    return batch_gens, batch_cons


def main(args):
    set_seed(args.seed)

    # load LM and Q value network
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path).to(device)
    security_model = GPT2LMHeadModel.from_pretrained(args.q_model_path).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = model.config.eos_token_id
    security_model.eval()
    model.eval()

    # load test set
    test_set = []
    with open(args.prompts_path, 'r') as f:
        for line in f:
            test_set.append(json.loads(line))
    
    print("- test set: ", len(test_set))
    print(test_set[0])

    # test
    prompt_text = [
        """I'm sick of all the politically correct""",
    ]
    generator_test = CustomizedGenerationMixin(model, config, security_model)
    encoded_test_prompt = tokenizer(prompt_text, padding='longest', add_special_tokens=False, return_tensors="pt")
    encoded_test_prompt = encoded_test_prompt.to(device)

    test_outputs = generator_test.generate(
        encoded_test_prompt['input_ids'],
        attention_mask=encoded_test_prompt['attention_mask'],
        min_length=10,
        max_length=20+encoded_test_prompt['input_ids'].shape[1],
        temperature=1.0,
        repetition_penalty=1.0,
        do_sample=False,
        # top_p=0.9,
        # top_k=1000,
        num_beams=1,
        num_return_sequences=1,
        secure_threshold=0.3, # for testing only
        secure_top_k=0,
        secure_prob_min=0
    )
    print(tokenizer.batch_decode(test_outputs, skip_special_tokens=True))

    # inference
    generator = CustomizedGenerationMixin(model, config, security_model)
    continuations = []
    dead_end_meet = 0

    batch = []
    eval_dataset = random.sample(test_set, min(len(test_set), args.num_eval_samples))
    for s in tqdm(eval_dataset):
        batch.append(process_prompt(s['prompt']['text']))

        if len(batch) == args.batch_size:
            batch_outputs = None
            try:
                batch_outputs = batch_generate(
                    batch,
                    generator,
                    tokenizer,
                    secure_threshold=args.threshold,
                    secure_top_k=args.top_k,
                    secure_prob_min=0.0,
                    num_return_sequences=args.num_returns
                )
            except RuntimeError as err:
                # run generation multiple times & adjust threshold
                generation_count = 0
                while batch_outputs is None:
                    try:
                        batch_outputs = batch_generate(
                            batch,
                            generator,
                            tokenizer,
                            secure_threshold=max(0.0, args.threshold - 0.05 * generation_count),
                            secure_top_k=args.top_k,
                            secure_prob_min=0.0,
                            num_return_sequences=args.num_returns
                        )
                    except RuntimeError as err:
                        batch_outputs = None

                    generation_count += 1
                    if generation_count == 5:
                        break

            if batch_outputs is None:
                dead_end_meet += 1
                batch_outputs = batch_generate(
                    batch,
                    generator,
                    tokenizer,
                    secure_threshold=args.threshold,
                    secure_top_k=args.top_k,
                    secure_prob_min=1e-7,
                    num_return_sequences=args.num_returns
                )

            if batch_outputs is not None:
                batch_gens, batch_cons = batch_outputs
                assert len(batch_gens) == args.batch_size * args.num_returns, \
                    "{}; {}".format(len(batch_gens), args.batch_size * args.num_returns)

                for i in range(args.batch_size):
                    sample_cons = [batch_cons[i * args.num_returns + j] for j in range(args.num_returns)]
                    continuations.append(sample_cons)
            else:
                raise RuntimeError("Generation failed!")

            batch = []

    if len(batch) > 0:
        _, batch_cons = batch_generate(
            batch,
            generator,
            tokenizer,
            secure_threshold=args.threshold, 
            secure_top_k=args.top_k, 
            secure_prob_min=0.0, 
            num_return_sequences=args.num_returns
        )
        assert len(batch_cons) == len(batch) * args.num_returns, \
            "{} {}".format(len(batch_cons), len(batch) * args.num_returns)

        for i in range(len(batch)):
            sample_cons = [batch_cons[i * args.num_returns + j] for j in range(args.num_returns)]
            continuations.append(sample_cons)

    print("Number of generations: ", len(continuations))
    print("Dead-end: ", dead_end_meet)

    assert len(eval_dataset) == len(continuations), \
        "{}-{}".format(len(eval_dataset), len(continuations))

    with open(args.output_path, 'w') as wf:
        for d, sample_gens in zip(eval_dataset, continuations):
            d['generations'] = [{'text': g} for g in sample_gens]
            json.dump(d, wf)
            wf.write('\n')
    print("Generations saved at: ", args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference arguments")

    parser.add_argument("--model_name_or_path", type=str, help="LM path")
    parser.add_argument("--q_model_path", type=str, help="Security model path")
    parser.add_argument("--prompts_path", type=str, help="test prompts path")
    parser.add_argument("--output_path", type=str, help="Path to write generated continuations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducing results.")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--num_returns", type=int, default=25, help="Number of continuations to be generated for each prompt.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold mechanism (epsilon).")
    parser.add_argument("--top_k", type=int, default=30, help="Use top-k decoding.")
    parser.add_argument("--num_eval_samples", type=int, default=10000, help="Number of samples for evaluation.")

    args = parser.parse_args()
    main(args)
