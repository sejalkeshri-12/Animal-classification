# Animal Image Classification 🐶🐱🐮

This project classifies animal images using a Convolutional Neural Network built with PyTorch.

## Dataset Structure
dataset/
├── train/
│   ├── cat/
│   ├── dog/
│   └── horse/
└── val/
    ├── cat/
    ├── dog/
    └── horse/

## Installation
pip install -r requirements.txt

## Training
python animal_classifier.py

## Inference
python infer.py --model best_model.pth --image test_images/cat.jpg

## Output
Predicted class: Cat (Confidence: 92%)
