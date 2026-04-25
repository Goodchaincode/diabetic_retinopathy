# diabetic_retinopathy
Detects Diabetic Retinopathy from Ultra-wide Field (UWF) Imaging


1. Data Collection
  The system utilizes the APTOS 2019 Blindness Detection dataset which includes 3,662 labeled retinal fundus images.
  Images are categorized into five diagnostic classes: 0: No DR, 1: Mild, 2: Moderate, 3: Severe, and 4: Proliferative DR.
  The model collects deep-level data by extracting 2048-D feature vectors (ResNet50) or 1280-D vectors (EfficientNetV2) from the raw image pixels.
.
2. Data Processing
    Data is normalized by resizing all input images to a standard 224x224 pixels to ensure consistency.
    The system implements Ben Graham’s preprocessing, applying Gaussian Blurring and color space conversions to highlight clinical features like hemorrhages.
  SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the dataset, ensuring rare severe cases are given equal weightage during training.
  Data Augmentation (horizontal flips and rotations) is used to synthetically increase the dataset size and improve model robustness.
                                         3. Model Design
  Deep Learning (Feature Extractor): Employs EfficientNetV2-S (and ResNet50) with ImageNet weights to identify complex medical textures and lesions.
  Machine Learning (Classifier): Uses a Support Vector Machine (SVM) with Linear or RBF kernels to perform the final diagnostic classification on the extracted features.
  Transfer Learning: Adapts pre-trained deep learning models by removing the final layers to focus specifically on retinal feature extraction.
                                        4. Algorithm Flow
  User Input: A retinal fundus scan is uploaded through the Gradio interface.
  Preprocessing: The image undergoes resizing, Gaussian filtering, and normalization.
  Feature Mapping: The processed image is passed through the CNN backbone to generate a numerical vector.
  Classification: The SVM model analyzes the vector to predict the DR stage and calculate confidence scores.
  Output: The system displays the severity label and a visual result to the user.
                                      5. Deployment
  Backend: Powered by Python using TensorFlow/Keras for deep learning and Scikit-learn for machine learning logic.
  Frontend: Built with Gradio for a real-time, web-based clinical dashboard.
  Platform: Hosted on Google Colab to leverage GPU acceleration for high-speed image processing.
  Model Persistence: The trained SVM weights are serialized using Joblib for efficient loading and deployment.



TECHNOLOGY STACK
  Programming Language: Python 
  Libraries: scikit-learn, pandas, NumPy, TensorFlow, Gradio, imbalanced-learn 
  Frontend Tools: Gradio (Web-based Interface) 
  Database: Google Drive (used for saving .npy feature matrices and .pkl model files) 
  Environment: Google Colab 
SYSTEM ARCHITECTURE
Modules:
    1. Image Acquisition & Preprocessing Module: Handles image uploads and applies Ben Graham’s preprocessing (Gaussian Blurring and resizing).
    2. CNN Feature Extraction Module: Utilizes pre-trained EfficientNetV2/ResNet50 to convert retinal images into high-dimensional feature vectors.
    3. SVM Classification Module: Performs the final diagnosis and calculates confidence probabilities for the five stages of Diabetic Retinopathy.
    4. Gradio Diagnostic Dashboard: A web-based interface for real-time visualization of retinal scans and diagnostic results.
Architecture Workflow:
    • User uploads fundus image → Preprocessing enhances retinal features → CNN extracts deep feature vectors → SVM predicts disease severity stage → Dashboard displays diagnosis and confidence score.

IMPLEMENTATION
Algorithm (Simplified):
Python
def predict_dr_severity(input_image, cnn_model, svm_model):
    # 1. Enhance image features
    processed_img = apply_ben_graham_preprocessing(input_image) 
    # 2. Extract deep medical features
    feature_vector = cnn_model.predict(processed_img) 
    # 3. Classify and get confidence
    prediction = svm_model.predict(feature_vector)
    confidence = svm_model.predict_proba(feature_vector)
    return prediction, confidence

    Code Impletation
1. Image Preprocessing and Feature Extraction
2. Training with SVM and Data Balancing (SMOTE)
3. Web Interface Implementation (Gradio)


Features:
    • Hybrid AI Framework: Combines Deep Learning (CNN) for feature detection with Machine Learning (SVM) for robust classification.
    • Clinical Feature Enhancement: Uses Gaussian filters to make microaneurysms and hemorrhages more visible to the model.
    • Imbalance Correction: Integrated SMOTE and data augmentation to ensure high accuracy even for rare, severe cases.

    • Real-time Inference: Provides instantaneous screening results via a shareable web-based URL.
