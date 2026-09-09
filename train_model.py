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

from speech_lid.augmentation import maybe_augment


BASE_MODEL = "facebook/mms-300m"
HF_DATASET = "badrex/nnti-dataset-full"
AUDIO_FIELD = "audio_filepath"
LABEL_FIELD = "language"
CLIP_SECONDS = 7

# only these languages will be augmented for selective augmentation based on previous analysis
AUGMENT_LANGS = {
    "hindi",
    "urdu",
    "tamil",
    "malayalam",
    "manipuri",
    "punjabi",
    "nepali",
}


print("Loading dataset...")
raw_data = load_dataset(HF_DATASET)

for split_name in ["train", "validation"]:
    raw_data[split_name] = raw_data[split_name].cast_column(
        AUDIO_FIELD,
        Audio(sampling_rate=16000),
    )

label_names = sorted(raw_data["train"].unique(LABEL_FIELD))
for split_name in ["train", "validation"]:
    raw_data[split_name] = raw_data[split_name].cast_column(
        LABEL_FIELD,
        ClassLabel(names=label_names),
    )

train_ds = raw_data["train"]
valid_ds = raw_data["validation"]

id2label = dict(enumerate(train_ds.features[LABEL_FIELD].names))
label2id = {name: idx for idx, name in id2label.items()}
n_classes = len(id2label)

feature_processor = AutoFeatureExtractor.from_pretrained(
    BASE_MODEL,
    do_normalize=True,
    return_attention_mask=True,
)

sample_rate = feature_processor.sampling_rate
max_audio_len = int(sample_rate * CLIP_SECONDS)


def prepare_train_batch(batch):
    waveforms = []
    lang_ids = batch[LABEL_FIELD]

    for audio_obj, lang_id in zip(batch[AUDIO_FIELD], lang_ids):
        wav = audio_obj["array"].astype(np.float32)
        lang_name = id2label[int(lang_id)]
        wav = maybe_augment(wav, lang_name, AUGMENT_LANGS)
        waveforms.append(wav)

    processed = feature_processor(
        waveforms,
        sampling_rate=sample_rate,
        max_length=max_audio_len,
        truncation=True,
        return_attention_mask=True,
    )

    processed["labels"] = lang_ids
    processed["length"] = [len(x) for x in processed["input_values"]]
    return processed


def prepare_eval_batch(batch):
    waveforms = [audio_obj["array"].astype(np.float32) for audio_obj in batch[AUDIO_FIELD]]

    processed = feature_processor(
        waveforms,
        sampling_rate=sample_rate,
        max_length=max_audio_len,
        truncation=True,
        return_attention_mask=True,
    )

    processed["labels"] = batch[LABEL_FIELD]
    processed["length"] = [len(x) for x in processed["input_values"]]
    return processed


print("Preprocessing train/validation splits...")
train_encoded = train_ds.map(
    prepare_train_batch,
    batched=True,
    remove_columns=train_ds.column_names,
)

valid_encoded = valid_ds.map(
    prepare_eval_batch,
    batched=True,
    remove_columns=valid_ds.column_names,
)

print("Example label:", train_encoded[0]["labels"])
print("Max label id:", max(train_encoded["labels"]))
print("Total classes:", n_classes)


@dataclass
class AudioBatchCollator:
    processor: Any

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_part = [
            {
                "input_values": sample["input_values"],
                "attention_mask": sample["attention_mask"],
            }
            for sample in samples
        ]
        targets = [sample["labels"] for sample in samples]

        batch = self.processor.pad(
            audio_part,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(targets, dtype=torch.long)
        return batch


collator = AudioBatchCollator(processor=feature_processor)

print("Building model...")
cfg = AutoConfig.from_pretrained(BASE_MODEL)
cfg.num_labels = n_classes
cfg.label2id = label2id
cfg.id2label = id2label

model = AutoModelForAudioClassification.from_pretrained(
    BASE_MODEL,
    config=cfg,
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

print(f"Head reset done. Expected initial loss ~ {np.log(n_classes):.2f}")

metric_acc = evaluate.load("accuracy")


def metric_fn(eval_pred):
    logits, gold = eval_pred
    pred_ids = np.argmax(logits, axis=-1)
    return metric_acc.compute(predictions=pred_ids, references=gold)


train_args = TrainingArguments(
    output_dir="./mms-300m-nnti-finetuned",
    save_strategy="steps",
    save_steps=600,
    eval_strategy="steps",
    eval_steps=300,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    num_train_epochs=20,
    warmup_steps=100,
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_encoded,
    eval_dataset=valid_encoded,
    processing_class=feature_processor,
    data_collator=collator,
    compute_metrics=metric_fn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for mini_batch in trainer.get_train_dataloader():
    mini_batch = {k: v.to(device) for k, v in mini_batch.items()}
    with torch.no_grad():
        out = model(**mini_batch)
    print("Initial loss:", out.loss.item())
    break


print("Starting training...")
trainer.train()
print("Training finished. Best checkpoint has been loaded.")

save_path = "./mms-300m-nnti-final-best"
trainer.save_model(save_path)
print(f"\nSaved best model to {save_path}")


plot_dir = "./evaluation_train_mms-300m-nnti-final-best"
os.makedirs(plot_dir, exist_ok=True)

logs = trainer.state.log_history

train_loss_log = [
    (entry["step"], entry["loss"])
    for entry in logs
    if "loss" in entry and "eval_loss" not in entry
]

eval_log = [entry for entry in logs if "eval_loss" in entry]

eval_epochs = [entry["epoch"] for entry in eval_log]
eval_losses = [entry["eval_loss"] for entry in eval_log]
eval_accs = [entry.get("eval_accuracy") for entry in eval_log]

fig, ax = plt.subplots(figsize=(8, 5))

if train_loss_log:
    step_ids, train_losses = zip(*train_loss_log)
    ax.plot(step_ids, train_losses, label="Train loss", alpha=0.7)

ax.plot(eval_epochs, eval_losses, marker="o", label="Validation loss")
ax.set_xlabel("Step / Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training and Validation Loss")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "loss_curve.png"), dpi=150)
plt.close(fig)

print(f"Saved loss plot to {plot_dir}/loss_curve.png")

if any(score is not None for score in eval_accs):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_epochs, eval_accs, marker="o", label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "accuracy_curve.png"), dpi=150)
    plt.close(fig)

    print(f"Saved accuracy plot to {plot_dir}/accuracy_curve.png")
