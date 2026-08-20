# Spoken Language Identification (NNTI Project)

This project implements a spoken language identification system for 22 Indian languages using a pretrained speech transformer model.

The model is based on **facebook/mms-300m** and is fine-tuned on the **badrex/nnti-dataset-full** dataset.


# Project Structure

code/
    train_model.py
    evaluate_model.py
    requirements.txt
    README.md
report/
    main.tex
    main.pdf
    figures/


---

# Installation

Create a Python environment and install dependencies:

pip install -r requirements.txt

# Training

Run the training script:
    python train_model.py

The trained model will be saved to:
./mms-300m-nnti-final-best

# Evaluation

Generate evaluation plots:
    python evaluate_model.py

This script produces:
- Confusion matrix
- t-SNE visualization of embeddings

Outputs are saved to:
./evaluation_final_model/

# Dataset

Dataset used:
badrex/nnti-dataset-full
Available on HuggingFace:
https://huggingface.co/datasets/badrex/nnti-dataset-full

The dataset contains speech recordings for 22 Indian languages, with approximately 400 samples per language and five speakers per language, creating a challenging speaker bias scenario.

# Model

Pretrained model:
facebook/mms-300m

A multilingual speech model based on the Wav2Vec2 transformer architecture.

## Key Features

- Pretrained speech transformer
- Fine-tuning for language classification
- Feature encoder freezing to reduce overfitting
- Custom audio preprocessing pipeline
- Confusion matrix analysis
- t-SNE visualization of learned embeddings

# Authors
Muhammad Saqib (7075880)
Kashish Mendiratta (7022904)
