# Diabetic Retinopathy Classification with CABNet

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning implementation for the automated classification of Diabetic Retinopathy (DR) severity from retinal fundus images. This project utilizes a **CABNet** (Channel Attention Block Network) inspired architecture, combining a **MobileNetV2** backbone with custom attention mechanisms to achieve high performance with computational efficiency.

## 📌 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Inference](#inference)
- [References](#references)

## 🔍 Overview
Diabetic Retinopathy is a leading cause of blindness. Early detection is crucial. This project provides an end-to-end pipeline for:
1.  Organizing grading datasets (like DDR).
2.  Training a lightweight yet powerful attention-based CNN.
3.  Performing inference to grade images into 5 severity levels:
    -   0: No DR
    -   1: Mild
    -   2: Moderate
    -   3: Severe
    -   4: Proliferative

## ✨ Key Features
*   **Efficient Backbone**: Uses **MobileNetV2** pre-trained on ImageNet for rapid convergence and low resource usage.
*   **Attention Mechanisms**:
    *   **CAB (Channel Attention Block)**: Focuses on "what" is meaningful in the features.
    *   **GAB (Global Attention Block)**: Focuses on "where" the informative regions are.
*   **Production Ready**: Includes scripts for easy inference and data organization.

## 🧠 Model Architecture
The architecture consists of:
1.  **Feature Extractor**: MobileNetV2 layers (up to the final feature map).
2.  **Attention Modules**:
    -   `CAB`: Reweights channel importance.
    -   `GAB`: Spatially focuses on relevant retinal lesions.
3.  **Classifier**: Global Average Pooling followed by a dense layer for 5-class classification.

## 📂 Dataset
This project is designed to work with standard DR datasets (e.g., DDR, IDRiD, APTOS). The code expects a CSV file containing image filenames and their corresponding diagnosis labels (0-4).

Structure details:
-   `data/`: Root directory for datasets.
-   `organize_data.py`: Script to split data into `train/` and `val/` folders based on class labels.

## ⚙️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Sabyasachi03/CABNet.git
    cd CABNet
    ```

2.  **Install Dependencies**
    Ensure you have Python installed. Install the required packages:
    ```bash
    pip install torch torchvision torchaudio pandas pillow tqdm
    ```

## 🚀 Usage

### Data Preparation
To organize your raw dataset (images + CSV) into a format suitable for training:
1.  Place your images and `DR_grading.csv` in the `data/` directory.
2.  Run the organization script:
    ```bash
    python organize_data.py
    ```
    *This will create `train` and `val` directories with class-specific subfolders.*

### Training
Open the Jupyter Notebook `DR_CABNet_DDR.ipynb` to start training. The notebook covers:
-   Data loading and augmentation.
-   Model initialization.
-   Training loop with validation.
-   Saving the best model weights.

### Inference
To predict the DR level of a single image using a trained model:
```bash
python check_model.py path/to/your/image.jpg
```

**Output Example:**
```text
Loading model...
Processing image...
Running inference...

Prediction Result:
==================
Class Index: 2
Confidence:  0.9854
DR Level:    Moderate (DR2)
```

## 📚 References
This implementation is inspired by research into attention mechanisms for medical image analysis, specifically "CABNet: Category Attention Block for Imbalanced Diabetic Retinopathy Grading".

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
