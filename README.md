# 🌱 Smart Crop Recommendation System

A **multi-class Machine Learning classification** system that recommends the most suitable crop to grow based on soil and climatic conditions. The project follows a rigorous, modular notebook pipeline — from exploratory analysis through ablation studies, robustness testing, and SHAP-based model interpretability.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-✓-blue)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-✓-green)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)](https://shap.readthedocs.io/)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Details](#-dataset-details)
- [Tech Stack](#️-tech-stack)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [License](#-license)

---

## 🔍 Project Overview

Farmers often struggle to decide which crop is best suited for their land given varying soil nutrient profiles and weather patterns. This project tackles that challenge by training **7 classification models** on agricultural data and selecting the best performer through cross-validated ablation studies and robustness testing.

### Highlights

- **22-class classification** across major Indian crops.
- **Ablation study** comparing *All Features*, *Soil-Only*, and *Climate-Only* feature groups.
- **Robustness evaluation** via Gaussian noise injection to test real-world sensor tolerance.
- **Explainability** through SHAP values (global & local) and human-readable Decision Tree rules.

---

## 📊 Dataset Details

**Source:** [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

| Property     | Value            |
|:------------ |:---------------- |
| Instances    | 2,200            |
| Features     | 7 (numeric)      |
| Classes      | 22 crop types    |
| Balance      | 100 samples/crop |

### Features

| Feature         | Description                           | Unit    |
|:--------------- |:------------------------------------- |:------- |
| `N`             | Ratio of Nitrogen content in soil     | —       |
| `P`             | Ratio of Phosphorous content in soil  | —       |
| `K`             | Ratio of Potassium content in soil    | —       |
| `temperature`   | Ambient temperature                   | °C      |
| `humidity`      | Relative humidity                     | %       |
| `ph`            | Soil pH value                         | 0–14    |
| `rainfall`      | Annual rainfall                       | mm      |

### Target Crops (22)

`apple` · `banana` · `blackgram` · `chickpea` · `coconut` · `coffee` · `cotton` · `grapes` · `jute` · `kidneybeans` · `lentil` · `maize` · `mango` · `mothbeans` · `mungbean` · `muskmelon` · `orange` · `papaya` · `pigeonpeas` · `pomegranate` · `rice` · `watermelon`

---

## 🛠️ Tech Stack

| Category           | Tools                                                          |
|:------------------ |:-------------------------------------------------------------- |
| **Language**       | Python 3.10+                                                   |
| **Data**           | Pandas, NumPy                                                  |
| **ML Models**      | Scikit-learn, XGBoost, LightGBM                                |
| **Visualization**  | Matplotlib, Seaborn                                            |
| **Explainability** | SHAP                                                           |
| **Serialization**  | Joblib                                                         |
| **Environment**    | Jupyter Notebook                                               |

---

## 🚀 Machine Learning Pipeline

The project is structured as a **modular 5-notebook pipeline**, where each stage produces artifacts consumed by subsequent stages:

```
 ┌─────────────────────┐     ┌──────────────────────────┐     ┌─────────────────────┐
 │  1. Data Exploration │ ──▶ │  2. Feature Engineering  │ ──▶ │    3. Modeling       │
 │  EDA & Cleaning      │     │  Scaling & Ablation Sets │     │  7 Models + CV      │
 └─────────────────────┘     └──────────────────────────┘     └─────────┬───────────┘
                                                                        │
                              ┌──────────────────────────┐     ┌────────▼───────────┐
                              │  5. Error Analysis       │ ◀── │    4. Evaluation    │
                              │  SHAP & Decision Rules   │     │  Robustness & F1   │
                              └──────────────────────────┘     └────────────────────┘
```

### Notebook Details

| # | Notebook | Description |
|:-:|:---------|:------------|
| 1 | **`1_data_exploration.ipynb`** | Load & clean data, structural inspection, target distribution, statistical profiling, correlation heatmaps, and feature distribution analysis. |
| 2 | **`2_feature_engineering.ipynb`** | Feature importance reporting (Random Forest–based), define ablation study feature sets (*all_features*, *soil_only*, *climate_only*), and save processed artifacts. |
| 3 | **`3_modeling.ipynb`** | Encode targets for gradient boosting models, train-test split, define a model zoo with conditional pipelines (StandardScaler applied per model), run ablation study with stratified 5-fold CV, and save final trained models. |
| 4 | **`4_evaluation.ipynb`** | Load all models, inject Gaussian noise for robustness testing, evaluate clean vs. noisy F1 scores, rank models by robustness, and plot confusion matrices. |
| 5 | **`5_error_analysis.ipynb`** | Extract human-readable Decision Tree rules, compute SHAP values for the best model, produce global summary bar plots and local (per-crop) waterfall explanations. |

### Models Trained

| Model                | Type              |
|:-------------------- |:----------------- |
| Logistic Regression  | Linear            |
| K-Nearest Neighbors  | Instance-based    |
| Support Vector Machine (SVM) | Kernel-based |
| Decision Tree        | Tree-based        |
| Random Forest        | Ensemble (Bagging)|
| XGBoost              | Ensemble (Boosting)|
| LightGBM             | Ensemble (Boosting)|

---

## 📈 Key Results

### Cross-Validation Accuracy (All Features — 5-Fold CV)

| Model               | CV Mean Accuracy |
|:-------------------- |:----------------:|
| **Random Forest**   | **99.43%** 🏆   |
| XGBoost             | 99.15%           |
| LightGBM            | 99.15%           |
| Decision Tree       | 98.52%           |
| SVM                 | 97.90%           |
| Logistic Regression | 96.82%           |
| KNN                 | 96.53%           |

### Ablation Study Summary

| Feature Group    | Best Model     | Best CV Accuracy |
|:---------------- |:-------------- |:----------------:|
| All Features     | Random Forest  | 99.43%           |
| Climate Only     | Random Forest  | 91.25%           |
| Soil Only        | Random Forest  | 80.97%           |

> **Insight:** Climate features contribute more predictive power than soil features alone, but combining both groups yields the highest accuracy.

### Robustness to Noise (Gaussian Perturbation)

| Model               | Clean F1 | Noisy F1 | Drop    |
|:-------------------- |:--------:|:--------:|:-------:|
| **SVM**             | 98.40%   | 98.40%   | **0.00%** 🛡️ |
| Random Forest        | 99.55%   | 98.64%   | 0.90%   |
| KNN                  | 97.93%   | 97.48%   | 0.45%   |
| Logistic Regression  | 97.25%   | 96.56%   | 0.68%   |
| Decision Tree        | 97.94%   | 96.10%   | 1.84%   |
| LightGBM             | 98.86%   | 94.77%   | 4.09%   |
| XGBoost              | 99.31%   | 94.30%   | 5.01%   |

> **Insight:** While Random Forest achieves the highest clean accuracy, SVM is the most robust to sensor noise — an important consideration for deployment in noisy agricultural IoT environments.

---

## 📂 Project Structure

```text
crop_project/
│
├── dataset/                          # Raw and processed data
│   ├── Crop_recommendation.csv       # Original dataset (2,200 × 8)
│   ├── X_processed.csv               # Scaled feature matrix
│   ├── y_processed.csv               # Encoded target vector
│   ├── X_test.csv                    # Holdout test features
│   └── y_test.csv                    # Holdout test targets
│
├── notebooks/                        # Modular ML pipeline
│   ├── 1_data_exploration.ipynb      # EDA & data profiling
│   ├── 2_feature_engineering.ipynb   # Feature importance & ablation sets
│   ├── 3_modeling.ipynb              # Model training & cross-validation
│   ├── 4_evaluation.ipynb            # Robustness testing & evaluation
│   └── 5_error_analysis.ipynb        # SHAP explainability & decision rules
│
├── models/                           # Serialized model artifacts (.pkl)
│   ├── RandomForest.pkl              # 🏆 Best accuracy model
│   ├── SVM.pkl                       # 🛡️ Most robust model
│   ├── XGBoost.pkl
│   ├── LightGBM.pkl
│   ├── DecisionTree.pkl
│   ├── KNN.pkl
│   ├── LogisticRegression.pkl
│   ├── scaler.pkl                    # StandardScaler artifact
│   ├── label_encoder.pkl             # LabelEncoder for target
│   ├── selected_features.pkl         # Final feature list
│   └── ablation_groups.pkl           # Feature group definitions
│
├── reports/                          # Generated analysis outputs
│   ├── confusion_matrix_RandomForest.png
│   ├── shap_global_summary.png       # Global SHAP importance
│   ├── shap_rice_explanation.png     # Local SHAP for Rice class
│   ├── cv_ablation_results.csv       # Cross-validation ablation data
│   ├── robustness_results.csv        # Clean vs. noisy F1 scores
│   └── agronomist_rules.txt          # Decision Tree rules (text)
│
├── README.md
└── .venv/                            # Python virtual environment
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/iiiyahiya111/Crop_Recommendation_Model.git
cd Crop_Recommendation_Model

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn shap joblib jupyter
```

---

## 💡 Usage

### Run the Full Pipeline

Launch Jupyter and run the notebooks **in order** (1 → 5):

```bash
jupyter notebook notebooks/
```

### Quick Prediction (Example)

```python
import joblib
import numpy as np

# Load artifacts
model   = joblib.load("models/RandomForest.pkl")
scaler  = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Sample input: [N, P, K, temperature, humidity, ph, rainfall]
sample = np.array([[90, 42, 43, 20.87, 82.00, 6.50, 202.93]])

# Predict
scaled = scaler.transform(sample)
prediction = model.predict(scaled)
crop_name = encoder.inverse_transform(prediction)

print(f"Recommended Crop: {crop_name[0]}")
# Output: Recommended Crop: rice
```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">
  Made with 🌾 by <a href="https://github.com/iiiyahiya111">Yahiya</a>
</p>
