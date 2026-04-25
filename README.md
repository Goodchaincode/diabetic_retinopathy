# 👁️ Diabetic Retinopathy Detection: A Hybrid Deep Learning Approach

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24%2B-yellow)
![Gradio](https://img.shields.io/badge/Interface-Gradio-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview
Diabetic Retinopathy (DR) is a leading cause of blindness worldwide. This project provides an automated, clinical-grade screening tool to classify the severity of DR from retinal fundus images. 

Unlike standard end-to-end classification models, this project utilizes a **Hybrid Architecture** that combines the powerful spatial feature extraction capabilities of Deep Learning (ResNet50) with the robust mathematical classification boundaries of Support Vector Machines (SVM).

## 🚀 Key Features
* **Advanced Preprocessing:** Implements **Ben Graham's Preprocessing** (Gaussian blur and weighted subtraction) to normalize lighting, significantly enhance blood vessel visibility, and highlight microaneurysms.
* **Hybrid Architecture:** Uses a pre-trained **ResNet50** to extract high-dimensional features, which are then passed into an **SVM** for optimal hyperplane classification.
* **Explainable AI (XAI):** Built-in visualization tools to highlight the specific retinal regions (lesions/hemorrhages) that trigger the model's decision.
* **Interactive Web UI:** Deployed using **Gradio** for real-time inference, allowing users to upload images and get instant predictions.

## 🧠 Core Implementation: Preprocessing
The secret to the high accuracy of this model lies in the preprocessing phase rather than just the network depth. Here is the core implementation used to process the raw fundus images before feeding them to the ResNet50 extractor:

```python
import cv2
import numpy as np

def load_ben_color(image_path, target_size=(224, 224), sigmaX=10):
    """
    Applies Ben Graham's preprocessing technique to enhance retinal images.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1. Crop to remove excess black borders
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > 7
    image = image[np.ix_(mask.any(1), mask.any(0))]
    
    # 2. Resize to standard dimensions
    image = cv2.resize(image, target_size)
    
    # 3. Apply Gaussian Blur and weighted subtraction to enhance lesions
    blurred = cv2.GaussianBlur(image, (0,0), sigmaX)
    image = cv2.addWeighted(image, 4, blurred, -4, 128)
    
    return image
