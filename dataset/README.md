# Training Dataset Generation

This guide details the steps to train a reward model and generate a training dataset as outlined in Section 4.4 of the [paper](https://openreview.net/forum?id=k8_yVW3Wqln).

## Reward Model Data Preparation

We employ the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) dataset for reward model training.
First, install the Kaggle package if you haven't already:

```bash
pip install kaggle
```

Next, create a directory for the dataset, download the dataset by running the kaggle command, and unzip the downloaded files:

```bash
mkdir rm_dataset
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
unzip jigsaw-toxic-comment-classification-challenge.zip -d rm_dataset/
unzip rm_dataset/train.csv.zip -d rm_dataset/
rm jigsaw-toxic-comment-classification-challenge.zip
```

Next, generate the training data for the reward model using the following commands:

```bash
SAVE_DIR=rm_dataset/processed/
mkdir $SAVE_DIR

python build_reward_training_data.py -f rm_dataset/train.csv -s $SAVE_DIR --drop_rate 0.3;
```

After preparing the dataset, proceed to train the reward model. Make sure to replace `YOUR_REWARD_MODEL_SAVE_DIR` with your actual save directory path where the trained model will be stored:

```bash
OUTPUT_DIR=YOUR_REWARD_MODEL_SAVE_DIR

python train_reward_model.py \
  --train_file rm_dataset/processed/rm_train.json \
  --validation_file rm_dataset/processed/rm_valid.json \
  --model_name_or_path bert-large-uncased \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir $OUTPUT_DIR;
```

## Value Function $Q_D$ Data Preparation

### Step 1: Download the Dataset

Begin by downloading the [Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview) dataset. Use the following commands to create a directory for the dataset and download it:

```bash
mkdir qd_dataset
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
unzip jigsaw-unintended-bias-in-toxicity-classification.zip -d qd_dataset/
rm jigsaw-unintended-bias-in-toxicity-classification.zip
```

After unzipping, there will be a `train.csv` file that will be used to create training prompts.

### Step 2: Generate Prompts

Execute the `build_value_training_prompts.py` to generate prompts:

```bash
RAW_JIGSAW_FILE=qd_dataset/train.csv
SAVE_PATH=qd_dataset/processed/prompts.jsonl

python build_value_training_prompts.py -f $RAW_JIGSAW_FILE -s $SAVE_PATH --drop_rate 0.9;
```

### Step 3: Generate Responses

Since we are doing offline RL training, we need to generate responses from the language model first. This process might take a few hours, depending on your model size and dataset size.

```bash
MODEL_PATH=gpt2-large
PROMPT_FILE=qd_dataset/processed/prompts.jsonl
SAVE_FILE=qd_dataset/processed/prompts_and_gens.jsonl

python generate_continuations.py \
    --model_name_or_path $MODEL_PATH \
    --data_file $PROMPT_FILE \
    --save_file $SAVE_FILE \
    --num_returns 25 \
    --batch_size 8 \
    --sample_size -1;
```

Then, score the generated responses using the trained reward model with the following commands:

```bash
REWARD_MODEL_PATH=YOUR_REWARD_MODEL_SAVE_DIR
GENS_FILE=qd_dataset/processed/prompts_and_gens.jsonl

python BERT_reward.py \
    --model_name_or_path $REWARD_MODEL_PATH \
    --data_file $GENS_FILE \
    --toxicity_type "BERT_RM" \
    --batch_size 512;
```

### Step 4: Compile Training Data

Finally, compile the training data by running the following commands:

```bash
FILE_PATH=qd_dataset/processed/prompts_and_gens.jsonl
SAVE_DIR=qd_dataset/processed/

python build_value_training_data.py \
  -f $FILE_PATH -s $SAVE_DIR \
  --toxicity_type "BERT_RM" \
  --mode "most";
```

After running this script, a `train.jsonl` and `val.jsonl` file will be saved under `SAVE_DIR`. Now, you can run the training script in `src` using the created dataset.
