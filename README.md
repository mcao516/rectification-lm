# Systematic Rectification of Language Models via Dead-end Analysis
This repository contains code necessary to replicate the training and evaluation for our ICLR 2023 paper "[Systematic Rectification of Language Models via Dead-end Analysis](https://openreview.net/forum?id=k8_yVW3Wqln&referrer=%5Bthe%20profile%20of%20Meng%20Cao%5D(%2Fprofile%3Fid%3D~Meng_Cao3))" by [Meng Cao](https://mcao516.github.io/), [Mehdi Fatemi](https://scholar.google.com/citations?user=X9_mSpYAAAAJ&hl=en), [Jackie CK Cheung](https://www.cs.mcgill.ca/~jcheung/) and [Samira Shabanian](https://scholar.google.ca/citations?user=CHkNfSMAAAAJ&hl=en).

# Requirements and Installation
* Python version >= 3.8
* [PyTorch](http://pytorch.org/) version >= 1.7.1
* [transformers](https://huggingface.co/docs/transformers/index) >= 4.22.0
* [accelerate](https://huggingface.co/docs/accelerate/index) >= 0.12.0

# Running the Code
To reproduce the results in the paper, you need to first download the [RealToxicityPrompts](https://allenai.org/data/real-toxicity-prompts) dataset.

## Training

``` bash
OUTPUT_DIR=./models
TRAIN_FILE=./dataset/train.json
VALID_FILE=./dataset/val.json

accelerate launch --config_file training_config.yaml train_detox.py \
    --overwrite_cache true \
    --gamma 1.0 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 32 \
    --preprocessing_num_workers 16 \
    --num_warmup_steps 500 \
    --polyak_update_lr 0.5 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --train_file $TRAIN_FILE \
    --validation_file $VALID_FILE \
    --model_name_or_path gpt2 \
    --output_dir $OUTPUT_DIR;
```

## Inference

```bash
MODEL_NAME_OR_PATH=./models/huggingface/gpt2-large
Q_MODEL_PATH=./models
PROMPTS_PATH=./dataset/prompts/nontoxic_prompts-10k.jsonl
OUTPUT_PATH=outputs.jsonl

python decoding.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --q_model_path $Q_MODEL_PATH \
    --prompts_path $PROMPTS_PATH \
    --output_path $OUTPUT_PATH \
    --seed 0 \
    --batch_size 1 \
    --num_returns 25 \
    --threshold 0.4 \
    --top_k 30;
```
