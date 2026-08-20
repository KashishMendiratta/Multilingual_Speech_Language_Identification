from datetime import datetime
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Current time: {current_time_str}")

import os
import numpy as np
import torch
from typing import Any, Dict, List

from datasets import load_dataset, Audio

from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    AutoConfig,
    TrainingArguments,
    Trainer
)

import evaluate

print("Check if GPU available:")
print("torch.cuda.is_available():", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ===============================
# MODEL
# ===============================

model_id = "facebook/mms-300m"

feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id,
    do_normalize=True,
    return_attention_mask=True
)

# ===============================
# DATASET
# ===============================

print("Loading dataset...")
dataset = load_dataset("badrex/nnti-dataset-full")

train_ds = dataset["train"].shuffle(seed=42)
valid_ds = dataset["validation"].shuffle(seed=42)

train_ds = train_ds.cast_column(
    "audio_filepath",
    Audio(sampling_rate=16000)
)

valid_ds = valid_ds.cast_column(
    "audio_filepath",
    Audio(sampling_rate=16000)
)

LABELS = train_ds.unique("language")
str_to_int = {s: i for i, s in enumerate(LABELS)}
int_to_str = {i: s for s, i in str_to_int.items()}

print("Languages:", LABELS)

input_features_key = "input_values"
max_duration = 7

# ===============================
# PREPROCESS
# ===============================

def preprocess_function(examples):

    audio_arrays = [x["array"] for x in examples["audio_filepath"]]

    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        truncation=True,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        return_attention_mask=True,
    )

    inputs["label"] = [str_to_int[x] for x in examples["language"]]

    inputs[input_features_key] = [
        np.array(x) for x in inputs[input_features_key]
    ]

    return inputs


keep_cols = ["speaker_id", "language"]

print("Preprocessing dataset...")

train_ds_encoded = train_ds.map(
    preprocess_function,
    remove_columns=[c for c in train_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=32,
    num_proc=6
)

valid_ds_encoded = valid_ds.map(
    preprocess_function,
    remove_columns=[c for c in valid_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=32,
    num_proc=6
)

# ===============================
# MODEL CONFIG
# ===============================

config = AutoConfig.from_pretrained(model_id)

config.num_labels = len(int_to_str)
config.label2id = str_to_int
config.id2label = int_to_str

slid_model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    config=config
)

# ===============================
# COLLATOR
# ===============================

class AudioDataCollator:

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, features: List[Dict[str, Any]]):

        batch = {
            input_features_key: [f[input_features_key] for f in features],
            "attention_mask": [f["attention_mask"] for f in features]
        }

        batch = self.feature_extractor.pad(
            batch,
            padding=True,
            return_tensors="pt"
        )

        batch["labels"] = torch.tensor(
            [f["label"] for f in features],
            dtype=torch.long
        )

        return batch


data_collator = AudioDataCollator(feature_extractor)

# ===============================
# TRAINING
# ===============================

batch_size = 8
gradient_accumulation_steps = 4
num_train_epochs = 6
lr = 2e-5

training_args = TrainingArguments(

    output_dir="./results",

    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,

    gradient_accumulation_steps=gradient_accumulation_steps,

    num_train_epochs=num_train_epochs,

    learning_rate=lr,
    weight_decay=0.01,
    warmup_ratio=0.1,

    logging_steps=50,

    evaluation_strategy="steps",
    eval_steps=200,

    save_strategy="steps",
    save_steps=200,

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="accuracy",
    greater_is_better=True,

    fp16=True,

    dataloader_num_workers=4,

    report_to="none"
)

# ===============================
# METRICS
# ===============================

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):

    predictions = np.argmax(eval_pred.predictions, axis=1)

    return accuracy_metric.compute(
        predictions=predictions,
        references=eval_pred.label_ids
    )

# ===============================
# TRAINER
# ===============================

trainer = Trainer(
    model=slid_model,
    args=training_args,
    train_dataset=train_ds_encoded,
    eval_dataset=valid_ds_encoded,
    tokenizer=feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("Training starting...")

trainer.train()

print("Final evaluation...")

trainer.evaluate()

# ===============================
# SAVE MODEL
# ===============================

save_dir = "./indic-SLID/inprogress"

os.makedirs(save_dir, exist_ok=True)

slid_model.save_pretrained(save_dir)

print("Training finished.")