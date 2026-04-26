# Skin Type Classifier

An image classification app that detects skin type from face photos.

## Live Demo
[Click here to try the app](https://muzakkir-cmd-skin-type-classifier.hf.space/?__theme=system&deep_link=kEwQTyNBfj4)

## Problem
Identifying skin type helps people choose the right skincare products. 
This app classifies skin as dry, normal or oily from a photo.

## Dataset
- Source: Kaggle — Oily Dry and Normal Skin Types Dataset
- 2756 training images across 3 classes
- Classes: Dry, Normal, Oily

## Approach
- Transfer learning with ResNet34 using fastai
- 5 epochs of fine tuning
- Data augmentation to handle class imbalance

## Key Finding
Skin type classification is genuinely difficult even for humans because 
the visual differences between skin types are subtle. The model reflects 
this real world challenge.

## Disclaimer
This app is for educational purposes only. Not a substitute for 
professional medical advice.

## Technologies Used
- Python
- fastai
- PyTorch
- Gradio
- Hugging Face Spaces
