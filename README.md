# Rice Pest Identification System

## Overview
This project uses transfer learning (pretrained ResNet18) to identify 14 species of rice pests from images, built as part of the AI GPU Summer Internship (Presidency University x NVIDIA AI CoE).

## Problem Statement
Farmers often struggle to correctly identify crop pests, leading to delayed or incorrect pesticide use. This project predicts the pest species from an uploaded image and provides an actionable recommendation through a simple, user-friendly interface.

## Project Architecture
This project is structured as two independent components:

1. **Backend / Training Notebook** (`rice_pest_identification.ipynb`) — handles dataset loading, preprocessing, model training (transfer learning), evaluation, and saves the trained model artifacts (`rice_pest_model.pth`, `class_names.json`).
2. **Frontend / Inference Notebook** (`frontend_app.ipynb`) — a standalone notebook containing **no training code**. It downloads the trained model artifacts directly from this repository, loads them into a fresh ResNet18 architecture, and serves an interactive Gradio-based web interface for real-time predictions.

This separation mirrors real-world ML system design, where model training and model serving are handled independently.

## Dataset
- Source: IP102 dataset (Kaggle)
- Scope: Rice superclass only, 14 pest classes
- Train: 5043 images | Val: 843 images | Test: 2531 images

## Approach
- Pretrained ResNet18 (ImageNet weights)
- Transfer learning: initially trained only the final FC layer, then fine-tuned `layer4` + FC for improved accuracy
- Framework: PyTorch

## Results
- Initial (FC-only training): 50.06% test accuracy
- After fine-tuning layer4: **62.70% test accuracy**
- See `confusion_matrix.png` and `training_curves.png` for detailed performance analysis
- Notable finding: the three plant hopper species (brown, small brown, white-backed) are frequently confused with one another due to genuine visual similarity

## Demo (Frontend)
Built with Gradio (`gr.Blocks`, custom HTML/CSS) — upload a rice pest image, get:
- Predicted pest species
- Color-coded confidence badge and visual confidence bar
- Recommended action

Run `frontend_app.ipynb` independently in Colab — it downloads the trained model directly from this repo, no retraining required.

## Files
- `rice_pest_identification.ipynb` — backend: data loading, training, evaluation
- `frontend_app.ipynb` — frontend: standalone Gradio demo, loads trained model only
- `rice_pest_model.pth` — trained model weights (fine-tuned, 62.70% test accuracy)
- `class_names.json` — class label mapping
- `training_curves.png` — loss/accuracy curves
- `confusion_matrix.png` — per-class performance breakdown

## Limitations & Future Work
This is a proof-of-concept prototype. Real-world deployment for farmer use would require:
- A permanently hosted API/mobile app (not a temporary Colab/Gradio demo link)
- Offline/low-bandwidth support for rural connectivity
- Further training data or augmentation to resolve confusion between visually similar species (e.g., the three plant hopper types)
- Testing was primarily conducted on the IP102 dataset's own test images; predictions on uncurated real-world/internet images showed reduced accuracy due to domain shift, an expected limitation for a prototype trained on a single benchmark dataset
