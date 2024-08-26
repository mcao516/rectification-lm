#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import json
import torch
import random
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer

random.seed(0)


def cut_comment(tokenizer, text, max_length=20):
    tok_ids = tokenizer.encode(text)
    return tokenizer.decode(torch.tensor(tok_ids[:max_length]))


def main(args):
    train = pd.read_csv(args.file)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompts = []
    pbar = tqdm(total=len(train.comment_text))
    for i, (text, label) in enumerate(zip(train.comment_text, train.target)):
        pbar.update(1)

        if label < 0.5 and random.random() <= args.drop_rate:
            continue

        if args.max_len > 0:
            try:
                text = cut_comment(tokenizer, text, max_length=args.max_len)
            except Exception as e:
                print(e)
                continue

        prompts.append({'prompt': {'text': text}})
            
    pbar.close()

    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))

    with open(args.save_path, 'w') as wf:
        for d in prompts:
            json.dump(d, wf)
            wf.write('\n')

    print("Total {} prompts saved at: {}".format(len(prompts), args.save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build reward model training data.")

    parser.add_argument("-f", "--file", type=str, required=True, help="Input file path")
    parser.add_argument("-s", "--save_path", type=str, required=True, help="Save path")
    parser.add_argument(
        "--drop_rate",
        type=float,
        default=0.9,
        help="Percentage of non-toxic prompts discarded"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=20,
        help="Prompt max length"
    )

    args = parser.parse_args()
    main(args)