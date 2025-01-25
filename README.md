# Predicting Epileptic Seizures Using Machine Learning

This repository contains the implementation of a machine learning-based system for predicting epileptic seizures through the analysis of electroencephalogram (EEG) data. The project evaluates multiple algorithms, preprocesses EEG signals, and extracts relevant features to build a robust prediction model.

## 📋 Overview

Epilepsy affects approximately 50 million people worldwide, with unpredictable seizures significantly impacting patients' quality of life. By analyzing EEG data, this project aims to predict seizures in advance, enabling early warnings and preventive measures to improve patient outcomes.

## 🛠 Features
- **Data Preprocessing**: Extracts temporal, spectral, and nonlinear features from EEG signals.
- **Model Evaluation**: Compares multiple machine learning algorithms:
  - Support Vector Machines (SVM)
  - Random Forest
  - AdaBoost
  - K-Nearest Neighbors (KNN)
  - Multi-Layer Perceptron (MLP)
- **Performance Metrics**: Evaluates models using accuracy, tpr, fpr, and training time.
- **Visualization**: Displays the results.

## 📊 Results

- SVM achieved the highest overall accuracy and true positive rate.
- Random Forest and KNN were identified as the best trade-offs between accuracy and training efficiency.
