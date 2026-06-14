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

# 🛠️ Technical Pipeline & Architecture Documentation

This document provides a comprehensive technical breakdown of the preprocessing, feature extraction, and model architecture layers implemented in the Deepfake Audio Detection Engine.

---

## 1. Data Preprocessing Pipeline

Raw audio files vary significantly in sampling rate, length, and amplitude. To ensure stable neural network gradients and uniform tabular features, the ingestion pipeline transforms all incoming audio files (`.wav` or `.mp3`) through a strict sequence of mathematical operations.
### Key Preprocessing Steps
* **Monophonic Downsampling:** Audio signals are loaded and automatically downsampled to a uniform rate of **16,000 Hz** via `librosa`. Multi-channel (stereo) signals are down-mixed to a single channel (mono) to remove spatial variance unrelated to vocal synthesis anomalies.
* **Temporal Standardization:** To maintain a uniform input tensor shape for the deep learning architecture, the audio duration is strictly locked to **5.0 seconds**. 
  * If the audio sample is shorter than 5.0 seconds, it is symmetrically or tail-padded with silence (zeros) using NumPy array padding.
  * If the audio sample exceeds 5.0 seconds, it is cleanly truncated to the first 5.0 seconds ($16,000 \times 5 = 80,000$ audio samples).

---

## 2. Feature Extraction Framework

The architecture utilizes a **Dual-Branch Feature Extraction** approach, capturing both visual frequency patterns (spatial branch) and statistical audio anomalies (tabular branch).

### Branch A: Spatial Spectrogram Extraction
The audio signal is converted into a visual representation of frequency changes over time:
1. **Mel-Spectrogram Generation:** A Short-Time Fourier Transform (STFT) is calculated using a Hann window, mapping frequencies onto the Mel scale across **128 mel bands**.
2. **Log-Amplitude Scaling:** The power spectrogram is converted to decibel units ($dB$) using:
   $$P_{\text{dB}} = 10 \cdot \log_{10}\left(\frac{P}{P_{\text{max}}}\right)$$
   This accentuates high-frequency noise and synthesis artifacts that are often invisible to the human ear but prominent in deepfake audios.
3. **Bilinear Interpolation:** The resulting matrix is resized via bilinear interpolation to an exact dimension of $128 \times 157$ pixels, establishing a consistent spatial resolution for the computer vision back-end.

### Branch B: Tabular Acoustic Extraction (48 Key Features)
Simultaneously, a compact vector of 48 macro-acoustic statistical identifiers is generated to feed the tree-based classifier:
* **Mel-Frequency Cepstral Coefficients (MFCCs):** The mean values of the first **20 MFCCs** are tracked to isolate vocal tract characteristics and envelope shape profiles.
* **Chroma STFT (12 Features):** Measures the energy distribution across the 12 distinct semitone pitch classes, highlighting tonal inconsistencies typical of generative voice conversion models.
* **Mel Spectrogram Bin Means (12 Features):** Compresses the 128 mel bands into 12 broad frequency bins to monitor macroscopic energy shifts.
* **Spectral Dynamics (4 Features):**
  * **Spectral Centroid:** The "center of mass" of the frequencies, quantifying how bright or dark the audio sounds.
  * **Spectral Bandwidth:** The descriptive spread of frequencies about the centroid.
  * **Spectral Rolloff:** The frequency below which **85%** of the total spectral energy resides.
  * **Zero-Crossing Rate:** The rate at which the audio signal changes sign from positive to negative, indicating high-frequency noise or robotic friction.

---

## 3. Model Architecture & Ensemble Stacking

The final classification relies on a **Stacked Generalization Ensemble** that feeds individual base model probability scores into an optimized meta-learner.

### Base Model 1: RobustAudioResNet (Deep Learning Branch)
* **Core Backbone:** Built upon a modified **ResNet18** structure utilizing residual skip connections to resolve the vanishing gradient problem during spatial pattern recognition.
* **Input Layer Re-engineering:** The standard ImageNet first convolutional layer (which expects 3-channel RGB images) was mathematically collapsed into a **1-channel** receptive field. This alteration preserves the pre-trained spatial weight configurations while allowing it to directly process raw 2D spectrogram tensors.
* **Output Classification Head:** The terminal fully connected layer was stripped and replaced with a linear mapping layer scaled to **2 output units** (Genuine vs. Deepfake) processed via a Softmax activation function.

### Base Model 2: LightGBM (Gradient Boosting Branch)
* **Core Engine:** A Light Gradient Boosting Machine (`LightGBM`) optimizes a collection of leaf-wise decision trees. 
* **Role:** It processes the 48 macro-acoustic features to quickly identify discrete threshold boundaries across statistical dimensions, serving as an analytical counterweight to the spatial focus of the convolutional network.

### The Meta-Learner Layer (Stacking & Boundary Mapping)
* **Architecture:** A balanced **Logistic Regression** model acts as the supervisor layer. 
* **Input Vector:** It consumes a 2-dimensional probability vector consisting of the individual confidence outputs generated by the ResNet18 and LightGBM models:
  $$X_{\text{meta}} = [P_{\text{ResNet18}}, P_{\text{LightGBM}}]$$
* **Threshold Optimization via Youden's J Statistic:** To maximize balanced accuracy across both classes, the final decision boundary bypasses the default 0.5 probability threshold. Instead, the optimal cutoff point is computed directly from the validation set's Receiver Operating Characteristic (ROC) curve by maximizing Youden's J index:
  $$J = \text{True Positive Rate} - \text{False Positive Rate}$$
  This locks the classification boundary at a point that protects the system from disproportionately misclassifying real human speech as synthetic.
