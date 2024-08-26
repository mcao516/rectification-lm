#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import json
import random
import pandas as pd


def main(args):
    # set random seed for results reproducing
    random.seed(0)

    train = pd.read_csv(args.file)
    train_datast, val_dataset = [], []
    for i, (text, label) in enumerate(zip(train.comment_text, train.toxic)):
        # discarded 30% non-toxic comments
        if label == 0 and random.random() <= args.drop_rate:
            continue

        # split training and validation 9:1
        if random.random() <= 0.1:
            val_dataset.append({'text': text, 'label': label})
        else:
            train_datast.append({'text': text, 'label': label})

    print("# Training samples: {}".format(len(train_datast)))
    print("# Validation samples: {}".format(len(val_dataset)))

    with open(os.path.join(args.save_dir, 'rm_train.json'), 'w') as wf:
        for d in train_datast:
            json.dump(d, wf)
            wf.write('\n')

    with open(os.path.join(args.save_dir, 'rm_valid.json'), 'w') as wf:
        for d in val_dataset:
            json.dump(d, wf)
            wf.write('\n')

    print("Files saved at: ", args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build reward model training data.")

    parser.add_argument("-f", "--file", type=str, required=True, help="Input file path")
    parser.add_argument("-s", "--save_dir", type=str, required=True, help="Save directory")
    parser.add_argument(
        "--drop_rate",
        type=float,
        default=0.3,
        help="Percentage of non-toxic prompts discarded"
    )

    args = parser.parse_args()
    main(args)