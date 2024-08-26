# Systematic Rectification of Language Models via Dead-end Analysis

This repository contains code necessary to replicate the training and evaluation for our ICLR 2023 paper "[Systematic Rectification of Language Models via Dead-end Analysis](https://openreview.net/forum?id=k8_yVW3Wqln&referrer=%5Bthe%20profile%20of%20Meng%20Cao%5D(%2Fprofile%3Fid%3D~Meng_Cao3))" by [Meng Cao](https://mcao516.github.io/), [Mehdi Fatemi](https://scholar.google.com/citations?user=X9_mSpYAAAAJ&hl=en), [Jackie CK Cheung](https://www.cs.mcgill.ca/~jcheung/) and [Samira Shabanian](https://scholar.google.ca/citations?user=CHkNfSMAAAAJ&hl=en).

## Requirements and Installation

* Python >= 3.8
* [PyTorch](http://pytorch.org/) >= 1.7.1
* [transformers](https://huggingface.co/docs/transformers/index) >= 4.22.0
* [accelerate](https://huggingface.co/docs/accelerate/index) >= 0.12.0
* [scikit-learn](https://scikit-learn.org/stable/install.html) >= 1.0.2
<!-- * [kaggle](https://www.kaggle.com/docs/api) -->

## Running the Code

<!-- We trained the value function network $Q_D$ using the dataset from the [Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview). For evaluation, we employed the [RealToxicityPrompts](https://allenai.org/data/real-toxicity-prompts) dataset. More details can be found in Section 4.4 of the paper. -->
Begin by creating a training dataset according to the guidelines provided in the [dataset](dataset/) folder. Alternatively, you can download our pre-processed training data from [Google drive](https://drive.google.com/drive/folders/1JcssUe-mXkDoZq0Pkj6rZkN-3EgagaLl?usp=sharing). After obtaining the `train.json` and `val.json` files, execute the command below to start training:

## Training

``` bash
OUTPUT_DIR=checkpoint/
TRAIN_FILE=dataset/qd_dataset/processed/train.json
VALID_FILE=dataset/qd_dataset/processed/val.json

accelerate launch --config_file accelerator_config.yaml src/train.py \
    --overwrite_cache true \
    --gamma 1.0 \
    --num_train_epochs 3 \
    --q_max 0.0 --q_min -1.0 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --preprocessing_num_workers 16 \
    --num_warmup_steps 500 \
    --polyak_update_lr 0.1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --train_file $TRAIN_FILE \
    --validation_file $VALID_FILE \
    --model_name_or_path gpt2 \
    --output_dir $OUTPUT_DIR \
    --checkpointing_steps "epoch";

```

Our checkpoint can be found [here](https://drive.google.com/drive/folders/1JcssUe-mXkDoZq0Pkj6rZkN-3EgagaLl?usp=sharing).

## Evaluation

We use [Real Toxicity Prompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts) and [Perspective API](https://perspectiveapi.com/) for evaluating language model toxicity. To access the API, follow the instructions [here](https://developers.perspectiveapi.com/s/docs-enable-the-api?language=en_US). You can download `prompts-nontoxic-10k.jsonl` from [here](https://drive.google.com/drive/folders/1JcssUe-mXkDoZq0Pkj6rZkN-3EgagaLl?usp=sharing).

```bash
Q_PATH=checkpoint/
SAVE_PATH=generations.jsonl

python src/inference.py \
    --lm_name_or_path "gpt2-large" \
    --q_model_path $Q_PATH \
    --prompts_path dataset/prompts-nontoxic-10k.jsonl \
    --save_path $SAVE_PATH \
    --do_sample \
    --batch_size 3 \
    --num_samples_for_eval 5000 \
    --epsilon 0.1;


echo "Run Perspective API for toxicity evaluation..."
python evaluation/run_perspective_api.py $SAVE_PATH \
    --num_return 25 \
    --num_thread 8 \
    --perspective_api_key $PERSPECTIVE_API_KEY \
    --save_scores --score_saving_path "eval_scores.jsonl";
```

## Citation

Please cite as:

``` bibtex
@inproceedings{
cao2023systematic,
title={Systematic Rectification of Language Models via Dead-end Analysis},
author={Meng Cao and Mehdi Fatemi and Jackie CK Cheung and Samira Shabanian},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=k8_yVW3Wqln}
}
```
