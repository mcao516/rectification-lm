# coding=utf-8
import argparse
import json
import numpy as np

from tqdm import tqdm


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def distinctness(responses, num_sample):
    generations_batch = list(chunks(responses, num_sample))
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for generations in tqdm(generations_batch, total=len(generations_batch), desc='Evaluating diversity'):
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(' ')
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i + 1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)

    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)


def main(args):
    # load generations
    generations = []
    with open(args.file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            assert "prompt" in sample and "generations" in sample
            generations.append(sample)

    all_gens = []
    for item in generations:
        all_gens.extend(
            [t["text"] for t in item["generations"]]
        )

    print(distinctness(all_gens, args.num_return))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Jsonl file that contains model generations."
    )
    parser.add_argument(
        '--num_return',
        type=int,
        default=25,
        help='Number of returns for each sample.'
    )

    args = parser.parse_args() 
    main(args)