#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import torch
import random

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.to(device)

    return model, tokenizer


def load_prompts(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def generate(
    model,
    tokenizer,
    prompt_text,
    return_prompt=False,
    max_len=20,
    num_returns=10,
    temperature=1.0,
    top_p=1.0,
):
    encoded_prompt = tokenizer(prompt_text, padding='longest', return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)

    input_ids = encoded_prompt['input_ids']
    attention_mask = encoded_prompt['attention_mask']

    prompt_len = input_ids.shape[1]
    output_sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_len,
        temperature=temperature,
        repetition_penalty=1.0,
        do_sample=True,
        top_p=top_p,
        num_return_sequences=num_returns,
        pad_token_id=tokenizer.eos_token_id
    )

    if return_prompt:
        return tokenizer.batch_decode(
            output_sequences,
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True
        )
    else:
        return tokenizer.batch_decode(
            output_sequences[:, prompt_len:],
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True
        )


def process_prompt(text):
    return text


def main(args):
    model, tokenizer = load_model(args.model_name_or_path)

    # load prompts
    data = load_prompts(args.data_file)
    if args.sample_size != -1 and args.sample_size < len(data):
        data = random.sample(data, args.sample_size)
    print('- Total data sampled: ', len(data))

    # generate continuations conditioned on the prompts
    total_length = len(data)
    generations = []
    for start_index in tqdm(range(0, total_length, args.batch_size)):
        end_index = start_index + args.batch_size
        # If the end index goes beyond the length of the data, set it to the length of the data
        if end_index > total_length:
            end_index = total_length

        batch = [process_prompt(s['prompt']['text']) for s in data[start_index:end_index]]
        batch_gens = generate(
            model,
            tokenizer,
            batch,
            max_len=20,
            num_returns=args.num_returns,
            temperature=args.temperature
        )
        assert len(batch_gens) == len(batch) * args.num_returns

        for i in range(len(batch)):
            generations.append([batch_gens[i * args.num_returns + j] for j in range(args.num_returns)])
    
    assert len(generations) == len(data)

    for d, gens in zip(data, generations):
        assert len(gens) == args.num_returns
        d["generations"] = [{"text": g_text} for g_text in gens]

    with open(args.save_file, 'w') as wf:
        for d in data:
            json.dump(d, wf)
            wf.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='huggingface/gpt2'
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default='realtoxicityprompts-new/rescored/prompts.jsonl'
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default='realtoxicityprompts-new/rescored/generations/prompted/prompts_gens_gpt2-xl.jsonl'
    )
    parser.add_argument(
        "--num_returns",
        type=int,
        default=10
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0
    )

    args = parser.parse_args()
    main(args)

