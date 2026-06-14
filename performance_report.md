# Deepfake Audio Detection: Evaluation Report
**Date:** June 14, 2026  
**Architecture:** Stacked Ensemble (ResNet18 Spatial Branch + LightGBM Tabular Branch)  
**Meta-Learner:** Optimized Logistic Regression via Youden's J Statistic ($J=\text{TPR}-\text{FPR}$)

## 1. Primary Metrics
* **Overall Accuracy:** 82.00%
* **F1-Score:** 81.44%
* **Precision:** 82.79%
* **Recall / Sensitivity:** 80.83%
* **Equal Error Rate (EER):** 18.42%

## 2. Classification Breakdown
* **Genuine Class Accuracy (Specificity):** 83.12%
* **Deepfake Class Accuracy (Sensitivity):** 80.83%

## 3. Confusion Matrix Counts
* **True Negatives (Correct Genuine):** 416
* **False Positives (Genuine flagged as Deepfake):** 84
* **False Negatives (Deepfake flagged as Genuine):** 96
* **True Positives (Correct Deepfake):** 404
