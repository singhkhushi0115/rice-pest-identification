# Rice Pest Identification System

## Overview
This project uses transfer learning (pretrained ResNet18) to identify 14 species of rice pests from images, built as part of the AI GPU Summer Internship (Presidency University x NVIDIA AI CoE).

## Problem Statement
Farmers often struggle to correctly identify crop pests, leading to delayed or incorrect pesticide use. This project predicts the pest species from an uploaded image and provides an actionable recommendation.

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

## Demo
Built with Gradio — upload a rice pest image, get:
- Predicted pest species
- Confidence score
- Recommended action

## Files
- `rice_pest_identification.ipynb` — full pipeline (data loading, training, evaluation, demo)
- `rice_pest_model.pth` — trained model weights
- `class_names.json` — class label mapping
- `training_curves.png` — loss/accuracy curves
- `confusion_matrix.png` — per-class performance breakdown

## Limitations & Future Work
This is a proof-of-concept prototype. Real-world deployment for farmer use would require:
- A permanently hosted API/mobile app (not a temporary Colab demo link)
- Offline/low-bandwidth support for rural connectivity
- Further training data to resolve confusion between visually similar species (e.g., the three plant hopper types)
