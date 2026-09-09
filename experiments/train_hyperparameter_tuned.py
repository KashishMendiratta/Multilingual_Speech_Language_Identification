import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn
import numpy as np
import evaluate
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset, Audio, ClassLabel
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)

BASE_MODEL = "facebook/mms-300m"
HF_DATASET = "badrex/nnti-dataset-full"
AUDIO_FIELD = "audio_filepath"
LABEL_FIELD = "language"
CLIP_SECONDS = 7

print("Loading dataset...")
data = load_dataset(HF_DATASET)

for split in ["train", "validation"]:
    data[split] = data[split].cast_column(AUDIO_FIELD, Audio(sampling_rate=16000))

label_names = sorted(data["train"].unique(LABEL_FIELD))
for split in ["train", "validation"]:
    data[split] = data[split].cast_column(LABEL_FIELD, ClassLabel(names=label_names))

train_ds = data["train"]
valid_ds = data["validation"]

id2label = dict(enumerate(train_ds.features[LABEL_FIELD].names))
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

processor = AutoFeatureExtractor.from_pretrained(
    BASE_MODEL,
    do_normalize=True,
    return_attention_mask=True,
)

sr = processor.sampling_rate
max_len = int(sr * CLIP_SECONDS)


def preprocess_batch(batch):
    waves = [a["array"].astype(np.float32) for a in batch[AUDIO_FIELD]]

    out = processor(
        waves,
        sampling_rate=sr,
        max_length=max_len,
        truncation=True,
        return_attention_mask=True,
    )
    out["labels"] = batch[LABEL_FIELD]
    out["length"] = [len(x) for x in out["input_values"]]
    return out


print("Preprocessing...")
train_encoded = train_ds.map(
    preprocess_batch,
    batched=True,
    remove_columns=train_ds.column_names,
)

valid_encoded = valid_ds.map(
    preprocess_batch,
    batched=True,
    remove_columns=valid_ds.column_names,
)


@dataclass
class AudioCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_inputs = [
            {
                "input_values": f["input_values"],
                "attention_mask": f["attention_mask"],
            }
            for f in features
        ]
        labels = [f["labels"] for f in features]

        batch = self.processor.pad(
            audio_inputs,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


collator = AudioCollator(processor=processor)
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)


def model_init():
    config = AutoConfig.from_pretrained(BASE_MODEL)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = id2label

    model = AutoModelForAudioClassification.from_pretrained(
        BASE_MODEL,
        config=config,
        ignore_mismatched_sizes=True,
    )

    model.freeze_feature_encoder()

    if hasattr(model, "projector"):
        nn.init.normal_(model.projector.weight, mean=0.0, std=0.02)
        if model.projector.bias is not None:
            nn.init.zeros_(model.projector.bias)

    if hasattr(model, "classifier"):
        nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)

    return model


base_args = TrainingArguments(
    output_dir="./mms-300m-hparam-search",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    remove_unused_columns=True,
)


trainer = Trainer(
    model_init=model_init,
    args=base_args,
    train_dataset=train_encoded,
    eval_dataset=valid_encoded,
    processing_class=processor,
    data_collator=collator,
    compute_metrics=compute_metrics,
)


def hp_space_optuna(trial):
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 2e-5, 3e-5, 5e-5]),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [8, 12, 16]),
        "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.03, 0.05, 0.1]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.05]),
        "per_device_eval_batch_size": 32,
    }


def compute_objective(metrics):
    return metrics["eval_loss"]


print("Starting hyperparameter search...")
best_trial = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    hp_space=hp_space_optuna,
    compute_objective=compute_objective,
    n_trials=8,
)

print("Best trial:")
print(best_trial)

best_params = best_trial.hyperparameters

final_args = TrainingArguments(
    output_dir="./mms-300m-nnti-best-tuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    remove_unused_columns=True,
    learning_rate=best_params["learning_rate"],
    per_device_train_batch_size=best_params["per_device_train_batch_size"],
    per_device_eval_batch_size=best_params["per_device_eval_batch_size"],
    gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
    num_train_epochs=int(best_params["num_train_epochs"]),
    warmup_ratio=best_params["warmup_ratio"],
    weight_decay=best_params["weight_decay"],
)

final_trainer = Trainer(
    model=model_init(),
    args=final_args,
    train_dataset=train_encoded,
    eval_dataset=valid_encoded,
    processing_class=processor,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_trainer.model.to(device)

print("Training final model with best hyperparameters...")
final_trainer.train()

save_dir = "./mms-300m-nnti-final-best"
final_trainer.save_model(save_dir)
print(f"Saved to {save_dir}")

plot_dir = "./evaluation_train_mms-300m-nnti-final-best"
os.makedirs(plot_dir, exist_ok=True)

logs = final_trainer.state.log_history

train_loss = [
    (x["step"], x["loss"])
    for x in logs
    if "loss" in x and "eval_loss" not in x
]
eval_logs = [x for x in logs if "eval_loss" in x]

eval_epochs = [x["epoch"] for x in eval_logs]
eval_losses = [x["eval_loss"] for x in eval_logs]
eval_acc = [x.get("eval_accuracy") for x in eval_logs]

fig, ax = plt.subplots(figsize=(8, 5))
if train_loss:
    steps, losses = zip(*train_loss)
    ax.plot(steps, losses, label="Train loss", alpha=0.7)
ax.plot(eval_epochs, eval_losses, marker="o", label="Validation loss")
ax.set_xlabel("Step / Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training and Validation Loss")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "loss_curve.png"), dpi=150)
plt.close(fig)

if any(v is not None for v in eval_acc):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_epochs, eval_acc, marker="o", label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "accuracy_curve.png"), dpi=150)
    plt.close(fig)
