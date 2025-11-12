# 🧠 Medical Image Analysis Pipeline (Prototype)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Model-red?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv)
![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-orange)
![Status](https://img.shields.io/badge/Status-Prototype-success)

> CNN-based segmentation of brain MRI scans using **U-Net**, **PyTorch**, and **OpenCV**.  
> Implements a full preprocessing, model training, and inference workflow — achieving a **Dice coefficient of 0.81** on validation data.

---

## 📋 Overview

This project demonstrates a complete medical image analysis pipeline for **brain MRI segmentation**:

- Loads `.tif` MRI images and corresponding segmentation masks  
- Normalizes and resizes them for model input  
- Trains a **lightweight U-Net** on paired MRI and mask data  
- Evaluates using **Dice + BCE loss functions**  
- Exports **predicted segmentation overlays** and a **summary CSV**  

---

## ⚙️ Features

✅ Preprocessing & Dataset Loader (`preprocess.py`)  
✅ Custom U-Net Architecture (`model.py`)  
✅ Full Training Loop with Validation (`train.py`)  
✅ Prediction Export & Visualization (`inference.py`)  
✅ Dice ≈ **0.81** after 10 epochs  

---

## 🧩 Model Architecture

```python
U-Net:
Input → [Conv → BN → ReLU] × 2 → Downsample → Bottleneck → Upsample → Output(1×1 Conv)
