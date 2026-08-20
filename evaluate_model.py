import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "./mms-300m-nnti-final-best"
DATASET_ID = "badrex/nnti-dataset-full"
AUDIO_COLUMN = "audio_filepath"
LABEL_COLUMN = "language"
MAX_DURATION_SECONDS = 10
OUTPUT_DIR = "./evaluation_final_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# LOAD MODEL
# ===============================
print("Loading model...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

labels = list(model.config.label2id.keys())
label2id = model.config.label2id
id2label = model.config.id2label

# ===============================
# LOAD DATASET
# ===============================
print("Loading dataset...")
dataset = load_dataset(DATASET_ID)

dataset["validation"] = dataset["validation"].cast_column(
    AUDIO_COLUMN, Audio(sampling_rate=16000)
)

target_sampling_rate = feature_extractor.sampling_rate
max_length = int(target_sampling_rate * MAX_DURATION_SECONDS)

# ===============================
# FEATURE EXTRACTION
# ===============================
def preprocess(example):
    audio = example[AUDIO_COLUMN]["array"]

    inputs = feature_extractor(
        audio,
        sampling_rate=target_sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    return inputs

# ===============================
# INFERENCE
# ===============================
all_preds = []
all_labels = []
embeddings = []

print("Running inference...")

with torch.no_grad():
    for example in dataset["validation"]:

        inputs = preprocess(example)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).cpu().numpy()[0]

        all_preds.append(pred)
        all_labels.append(label2id[example[LABEL_COLUMN]])

        # extract embedding before classifier
        hidden = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
        embeddings.append(hidden[0])

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
embeddings = np.array(embeddings)

# ===============================
# CONFUSION MATRIX
# ===============================
print("Generating confusion matrix...")

cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=False
)

plt.title("Confusion Matrix - Improved Model")
plt.tight_layout()

conf_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(conf_path, dpi=200)
plt.close()

print(f"Confusion matrix saved to {conf_path}")

# ===============================
# TSNE
# ===============================
print("Generating t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)

emb_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    emb_2d[:, 0],
    emb_2d[:, 1],
    c=all_labels,
    cmap="tab20",
    alpha=0.7
)

plt.title("t-SNE of Language Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.tight_layout()

tsne_path = os.path.join(OUTPUT_DIR, "tsne_embeddings.png")
plt.savefig(tsne_path, dpi=200)
plt.close()

print(f"t-SNE saved to {tsne_path}")

print("Evaluation complete.")