# 🌱 Smart Crop Recommendation System

## 📌 Project Overview
This project is a **Multi-class Machine Learning Classification** system designed to recommend the most suitable crop to grow based on soil and weather parameters. The model analyzes 7 key agricultural features to predict one of 22 different crops, helping farmers make data-driven decisions.

## 📊 Dataset Details
The dataset contains **2200 instances** (100 samples for each of the 22 crops) with the following 7 features:
* **N**: Ratio of Nitrogen content in soil
* **P**: Ratio of Phosphorous content in soil
* **K**: Ratio of Potassium content in soil
* **Temperature**: Temperature in degree Celsius
* **Humidity**: Relative humidity in percentage
* **pH**: Soil pH value
* **Rainfall**: Rainfall in mm

**Target Variable:** `label` (22 unique crops including Rice, Maize, Chickpea, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee).

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib
* **Models Used:** Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Support Vector Machine (SVM)

## 🚀 Machine Learning Pipeline
The project is structured into a modular 5-step Jupyter Notebook pipeline:

1. **`1_data_exploration.ipynb`**: Data cleaning, structural inspection, and exploratory data analysis (EDA) using distribution and correlation heatmaps.
2. **`2_feature_engineering.ipynb`**: Feature scaling using `StandardScaler` and feature selection using Mutual Information and Tree-based importance. Artifacts are saved using `joblib`.
3. **`3_modeling.ipynb`**: Training 5 different baseline models and saving the trained objects.
4. **`4_evaluation.ipynb`**: Evaluating models using Accuracy, Precision, Recall, F1-Score (Macro & Micro), and visual Confusion Matrices to select the best model.
5. **`5_error_analysis.ipynb`**: Deep dive into the best model's performance, plotting the final confusion matrix and extracting Feature Importance.

## 📂 Project Structure
```text
crop_project/
│
├── data/
│   └── Crop_recommendation.csv
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_feature_engineering.ipynb
│   ├── 3_modeling.ipynb
│   ├── 4_evaluation.ipynb
│   └── 5_error_analysis.ipynb
├── models/
│   └── (Saved .pkl files: Models, Scaler, Selected Features)
├── reports/
│   └── analysis_report.md
└── README.md
