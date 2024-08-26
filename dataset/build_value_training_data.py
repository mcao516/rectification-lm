#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import json
import random

from tqdm import tqdm

random.seed(0)


def sort_generation_by_toxicity(generation, toxicity_type='toxicity'):
    generation['generations'].sort(reverse=True, key=lambda x: x[toxicity_type])


def is_clean_prompt(sample, toxicity_type):
    for g in sample['generations']:
        if g[toxicity_type] > 0.5:
            return False
    return True


def main(args):
    prompts = []
    with open(args.file, 'r') as f:
        prompts = [json.loads(line) for line in f]

    print("{} samples loaded".format(len(prompts)))

    # build dataset
    train, val = [], []
    for g in tqdm(prompts):
        # drop some non-toxic prompts to deal with the imbalance issue
        if random.random() < args.drop_rate and is_clean_prompt(g, args.toxicity_type):
            continue

        dataset = train if random.random() < 0.9 else val
        
        samples = []
        if args.mode == "most":
            sort_generation_by_toxicity(g, toxicity_type=args.toxicity_type)
            
            samples.append(g['generations'][0])
            samples.append(g['generations'][-1])
        elif args.mode == "sample":
            samples.extend(random.sample(g['generations'], args.sample_num))
        elif args.mode == "all":
            for s in g['generations']:
                samples.append(s)
        else:
            raise Exception("Unknown mode!")
        
        prompt_text = g['prompt']['text']
        for s in samples:
            reward_value = -1.0 if s[args.toxicity_type] >= 0.5 else 0.0
            dataset.append({
                'text': prompt_text + s['text'],
                'reward': [reward_value]
            })

    print("Training trajectories: {}".format(len(train)))
    print("Validation trajectories: {}".format(len(val)))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(os.path.join(args.save_dir, "train.json"), 'w') as wf:
        for d in train:
            json.dump(d, wf)
            wf.write('\n')

    with open(os.path.join(args.save_dir, "val.json"), 'w') as wf:
        for d in val:
            json.dump(d, wf)
            wf.write('\n')

    print("Files saved at: ", args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build value function training data.")

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Jsonl file that contains prompts and scored generations."
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        required=True,
        help="Saving directory."
    )
    parser.add_argument(
        "--drop_rate",
        type=float,
        default=0.5,
        help="Percentage of non-toxic prompts discarded."
    )
    parser.add_argument(
        "--toxicity_type",
        type=str,
        default="toxicity",
        help="Type of toxicity for filtering."
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=6,
        help="Number of responses sampled per example."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "sample", "most"],
        default="most",
        help="Sampling strategy."
    )

    args = parser.parse_args()
    main(args)