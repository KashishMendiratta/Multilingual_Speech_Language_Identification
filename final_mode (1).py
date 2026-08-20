import os

os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.cache/torchinductor")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

import torch
import numpy as np
import evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Dict, List
from datasets import load_dataset, Audio, ClassLabel
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoConfig,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import classification_report

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_ID = "facebook/mms-300m"
DATASET_ID = "badrex/nnti-dataset-full"
AUDIO_COLUMN = "audio_filepath"      
LABEL_COLUMN = "language"      
MAX_DURATION_SECONDS = 10.0 

# ==========================================
# 2. LOAD DATASET (USE EXISTING SPLITS)
# ==========================================
print("Loading dataset...")
full_dataset = load_dataset(DATASET_ID)

# Cast to 16kHz
full_dataset["train"] = full_dataset["train"].cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
full_dataset["validation"] = full_dataset["validation"].cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))

# --- Extract unique labels and cast to ClassLabel ---
unique_labels = sorted(full_dataset["train"].unique(LABEL_COLUMN))
full_dataset["train"] = full_dataset["train"].cast_column(LABEL_COLUMN, ClassLabel(names=unique_labels))
full_dataset["validation"] = full_dataset["validation"].cast_column(LABEL_COLUMN, ClassLabel(names=unique_labels))
# -----------------------------------------------------

dataset = {
    "train": full_dataset["train"],
    "validation": full_dataset["validation"],
}

print(f"Dataset loaded with existing splits:")
print(f"Train samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")

# Extract labels and create ID mappings (This will now work perfectly!)
labels = dataset["train"].features[LABEL_COLUMN].names
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
num_labels = len(labels)

# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================
print("Initializing feature extractor...")
feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_ID,
    do_normalize=True,
    return_attention_mask=True,
)
target_sampling_rate = feature_extractor.sampling_rate
max_length = int(target_sampling_rate * MAX_DURATION_SECONDS)

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples[AUDIO_COLUMN]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=target_sampling_rate,
        max_length=max_length,
        truncation=True,
        return_attention_mask=True,
    )
    inputs["labels"] = examples[LABEL_COLUMN]
    # Store length for group_by_length
    inputs["length"] = [len(x) for x in inputs["input_values"]]
    return inputs

print("Preprocessing datasets...")
encoded_train = dataset["train"].map(preprocess_function, remove_columns=dataset["train"].column_names, batched=True)
encoded_val = dataset["validation"].map(preprocess_function, remove_columns=dataset["validation"].column_names, batched=True)

print("Label example:", encoded_train[0]["labels"])
print("Max label:", max(encoded_train["labels"]))
print("Num labels:", num_labels)

# ==========================================
# 4. CUSTOM DATA COLLATOR
# ==========================================
@dataclass
class DataCollatorForAudioClassification:
    feature_extractor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [
            {
                "input_values": feature["input_values"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        label_features = [feature["labels"] for feature in features]

        batch = self.feature_extractor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(label_features, dtype=torch.int64)
        return batch

data_collator = DataCollatorForAudioClassification(feature_extractor=feature_extractor)

# ==========================================
# 5. MODEL INITIALIZATION
# ==========================================
print("Initializing model...")
config = AutoConfig.from_pretrained(MODEL_ID)
config.num_labels = num_labels
config.label2id = label2id
config.id2label = id2label

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_ID,
    config=config,
    ignore_mismatched_sizes=True,
)
model.freeze_feature_encoder()

# ---- FIX: Reinitialize classifier head with small weights ----
# The projector and classifier are randomly initialized when
# ignore_mismatched_sizes=True. Default init produces large weights
# causing extreme logits and very high initial loss (~24+).
# Reinitializing with small std brings initial loss to ~ln(num_classes).
import torch.nn as nn
if hasattr(model, 'projector'):
    nn.init.normal_(model.projector.weight, mean=0.0, std=0.02)
    if model.projector.bias is not None:
        nn.init.zeros_(model.projector.bias)
if hasattr(model, 'classifier'):
    nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
    if model.classifier.bias is not None:
        nn.init.zeros_(model.classifier.bias)
print(f"Classifier head reinitialized (std=0.02). Expected initial loss: ~{np.log(num_labels):.2f}")
# ---------------------------------------------------------------

import evaluate, numpy as np
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# ---- FIX: Correct inflated training loss logging ----
# In accelerate-based Trainer, the raw (un-scaled) loss is accumulated
# but divided only by global_steps (not micro-steps), inflating the
# displayed loss by gradient_accumulation_steps. This subclass fixes it.
# ------------------------------------------------------

# ==========================================
# 6. TRAINING ARGUMENTS & TRAINER
# ==========================================
training_args = TrainingArguments(
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
    warmup_steps=100,             # <--- Add this instead (100 is a safe default)
    logging_steps=50,
    save_total_limit=1,
    # Track lowest validation loss to pick the best model at the end
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss", 
    greater_is_better=False,          
    
    fp16=False,                 
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_val,         
    processing_class=feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for batch in trainer.get_train_dataloader():
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    print("Initial loss:", outputs.loss.item())
    break


# ==========================================
# 7. EXECUTION (FINE-TUNING)
# ==========================================
print("Starting training...")
trainer.train()

print("Training complete! Best model (based on validation loss) has been automatically loaded.")

# ==========================================
# 8. SAVE MODEL
# ==========================================
trainer.save_model("./mms-300m-nnti-final-best")
print("\nFinal Best Model saved to ./mms-300m-nnti-final-best")

# ==========================================
# 9. SAVE TRAINING / VALIDATION GRAPHS
# ==========================================
EVAL_DIR = "./evaluation_train_mms-300m-nnti-final-best"
os.makedirs(EVAL_DIR, exist_ok=True)

history = trainer.state.log_history

# Separate train-step logs and per-epoch eval logs
train_loss_steps = [(e["step"], e["loss"]) for e in history if "loss" in e and "eval_loss" not in e]
eval_logs = [e for e in history if "eval_loss" in e]

eval_epochs    = [e["epoch"]        for e in eval_logs]
eval_losses    = [e["eval_loss"]    for e in eval_logs]
eval_accuracy  = [e.get("eval_accuracy", None) for e in eval_logs]

# --- Loss curve ---
fig, ax = plt.subplots(figsize=(8, 5))
if train_loss_steps:
    steps, t_losses = zip(*train_loss_steps)
    ax.plot(steps, t_losses, label="Train loss (step)", alpha=0.6)
ax.plot(eval_epochs, eval_losses, marker="o", label="Val loss (epoch)")
ax.set_xlabel("Step / Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training & Validation Loss")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(EVAL_DIR, "loss_curve.png"), dpi=150)
plt.close(fig)
print(f"Loss curve saved to {EVAL_DIR}/loss_curve.png")

# --- Accuracy curve ---
if any(v is not None for v in eval_accuracy):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_epochs, eval_accuracy, marker="o", color="tab:green", label="Val accuracy (epoch)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "accuracy_curve.png"), dpi=150)
    plt.close(fig)
    print(f"Accuracy curve saved to {EVAL_DIR}/accuracy_curve.png")