# 🎙️ Deepfake Audio Detection Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)

## 📖 Project Description
This project implements a highly robust, multi-modal machine learning pipeline designed to detect AI-generated (Deepfake) audio. By analyzing both the visual representations of audio frequencies (spectrograms) and raw acoustic tabular data, the system successfully differentiates between genuine human speech and synthesized voices. 

The final solution is deployed as an interactive web application, allowing users to upload `.wav` or `.mp3` files for real-time inference and confidence scoring.

## 🧠 Methodology & Architecture
To capture both deep spatial patterns and statistical acoustic features, this project utilizes a **Stacked Ensemble Architecture**. 

### 1. Feature Extraction
When an audio sample (up to 5 seconds, 16kHz) is ingested, it splits into two extraction pipelines:
* **Spatial Features:** The audio is converted into a Mel-spectrogram (128 bands), converted to a decibel scale, and transformed into a 1-channel tensor.
* **Acoustic Tabular Features (48 Total):** `librosa` is used to extract 48 distinct statistical features, including:
  * 20 MFCCs (Mel-frequency cepstral coefficients)
  * 12 Chroma STFT features
  * 12 Mel Spectrogram Means
  * Spectral Centroid, Bandwidth, Rolloff, and Zero Crossing Rate.

### 2. The Base Models
* **Model 1: RobustAudioResNet (PyTorch)**
  * A modified `ResNet18` architecture leveraging transfer learning.
  * The standard 3-channel RGB input layer was mathematically collapsed into a 1-channel layer to accept spectrograms without losing pre-trained edge-detection weights.
  * Trained for 30 epochs using an `Adam` optimizer and a `ReduceLROnPlateau` Learning Rate Scheduler to prevent overfitting.
* **Model 2: LightGBM Gradient Boosting**
  * A tree-based gradient boosting classifier trained exclusively on the 48 extracted tabular features to capture non-linear statistical anomalies in synthesized speech.

### 3. The Meta-Learner (Stacking)
* **Logistic Regression (with Youden's J Optimization)**
  * *Ablation Note:* Initial ensembles included an Attention mechanism, but an ablation study revealed it was introducing noise and degrading the Equal Error Rate (EER). It was successfully pruned.
  * The final predictions from ResNet18 and LightGBM are passed into a balanced Logistic Regression meta-learner.
  * Instead of a default 0.5 threshold, **Youden's J Statistic** ($J = TPR - FPR$) was calculated against the ROC curve to mathematically lock in the exact threshold that perfectly balances Genuine and Deepfake class accuracies.

## 📊 Performance Metrics
The model was evaluated against strict verification criteria, successfully clearing the primary accuracy and F1-score hurdles. The metrics on the final withheld evaluation set are as follows:

| Metric | Score | 
| :--- | :---: |
| **Overall Accuracy** | 82.00% | 
| **F1-Score** | 81.44% |
| **Precision** | 82.06% |
| **Recall / Sensitivity** | 80.83% |
| **Equal Error Rate (EER)** | 18.42% |
| **Genuine Class Accuracy** | 83.12% |
| **Deepfake Class Accuracy** | 80.83% |

*(Note: EER reached the practical hardware/architectural limit for this specific dataset size and lightweight model constraint).* 
