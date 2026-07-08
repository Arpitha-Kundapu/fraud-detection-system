# End-to-End Fraud Detection System

An end-to-end Machine Learning system for real-time credit card fraud detection using Python, Scikit-learn, FastAPI, Docker, and cloud deployment.

---

Welcome to the **End-to-End Fraud Detection System** workspace. This repository contains the foundational setup and exploratory data analysis (EDA) for detecting credit card fraud using machine learning.

## Phase 1: Setup & EDA

### 1. Project Objective
The main goal of this system is to identify fraudulent credit card transactions in real-time. In this phase (Phase 1), we established a professional project structure, set up the development environment, generated a highly realistic synthetic dataset, and performed a comprehensive Exploratory Data Analysis (EDA) to guide our preprocessing and model design decisions.

---

### 2. Dataset Description
The dataset used in this project represents transactions made by credit cards. It is designed to mimic the standard Kaggle Credit Card Fraud Detection dataset, which contains transactions made by European cardholders in September 2013.

*   **Total Transactions**: 30,000
*   **Legitimate Transactions (Class 0)**: 29,949 (99.83% of the dataset)
*   **Fraudulent Transactions (Class 1)**: 51 (0.17% of the dataset)
*   **Severe Class Imbalance**: Fraud is a rare event, which represents the primary challenge of this project.
*   **Anonymized Features (`V1` to `V28`)**: Due to confidentiality issues, the original features (e.g., location, card details) are not provided. Instead, they are represented by 28 numerical features obtained via Principal Component Analysis (PCA). These features are scaled and centered around 0.
*   **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
*   **Amount**: Transaction dollar amount.
*   **Class**: Target variable, where `1` represents fraud and `0` represents legitimate.

---

### 3. Folder Structure
The repository is organized following professional, production-grade machine learning folder patterns:

```text
├── .gitignore               # Standard Python Git ignore patterns
├── .env.example             # Template for API and model environment variables
├── README.md                # Project overview and run instructions for Phase 1
├── requirements.txt         # Package dependencies for ML and API serving
├── setup.py                 # Setuptools config for editable installation of local src module
├── api/                     # API serving layer
│   ├── __init__.py
│   └── app.py               # FastAPI application script with serving endpoints (placeholder)
├── data/                    # Local storage for raw and processed datasets (ignored by git)
│   ├── raw/
│   │   └── creditcard.csv   # Synthetic Credit Card Fraud dataset
│   └── processed/
├── documentation/           # In-depth architectural and setup manuals
│   ├── architecture.md
│   └── setup_guide.md
├── notebooks/               # Jupyter/VS Code notebook directory for EDA
│   └── eda.ipynb            # Interactive Exploratory Data Analysis notebook
└── src/                     # Core system modules (packaged locally)
    ├── __init__.py
    ├── data_pipeline.py     # Ingestion, cleaning, and preprocessing utilities (placeholder)
    ├── feature_engineering.py # Categorical encoding, scaling, and time features (placeholder)
    ├── model_pipeline.py    # Model training, evaluation, and loading utilities (placeholder)
    └── utils.py             # Reproducibility seeding and logger configuration
```

---

### 4. Technologies Used
*   **Programming Language**: Python 3.13
*   **Data Analysis & Manipulation**: `pandas`, `numpy`
*   **Data Visualization**: `matplotlib`, `seaborn`
*   **Machine Learning (Core Libraries)**: `scikit-learn`, `xgboost`, `lightgbm`
*   **API Serving & Backend**: `fastapi`, `uvicorn`, `pydantic`
*   **Configuration & Logging**: `python-dotenv`, `loguru`
*   **Environment Isolation**: `venv` (Python virtual environment)

---

### 5. EDA Performed
Within `notebooks/eda.ipynb`, we conducted the following diagnostic steps:
1.  **First Five Rows Preview**: Inspected column formatting and index styling.
2.  **Dataset Shape Check**: Verified total sample size (30,000) and column counts (31).
3.  **Column Names & Data Types Check**: Confirmed all 31 features are numeric (`float64` or `int64`).
4.  **Missing Values Scan**: Scanned every column to confirm there are zero null/missing values.
5.  **Duplicate Entries Check**: Inspected duplicate rows to ensure data cleanliness.
6.  **Descriptive Statistics**: Evaluated feature means, standard deviations, and range distributions.
7.  **Class Distribution Plotting**: Generated countplots on normal and logarithmic scales to study class imbalance.
8.  **Amount Distribution Plotting**: Created comparative histograms and overlaid median thresholds.
9.  **Time Density Plotting**: Generated KDE density charts comparing transaction frequency over day/night cycles.
10. **PCA Boxplots Analysis**: Plotted distributions of V-features (e.g., V10, V17) to find highly discriminative signals.
11. **Correlation Matrix Heatmap**: Visualized feature correlation coefficients to verify PCA orthogonality.

---

### 6. Key Observations
*   **Class Imbalance**: Fraud represents only **0.17%** of transactions. Consequently, standard **Accuracy is a deficient metric**. We must optimize models using **Precision, Recall, F1-Score, and AUPRC**.
*   **Scale Discrepancies**: Features `V1` through `V28` are already scaled. However, the `Amount` feature is unscaled, right-skewed, and ranges up to $15,000. It must be standard-scaled or log-transformed during preprocessing to prevent algorithm bias.
*   **Cyclic Behavior**: Legitimate transactions decrease during night hours and peak during daytime hours. Fraudulent transactions do not show this day/night cyclical behavior and remain active even during normal sleep windows.
*   **Discriminative Features**: Specific features (specifically V10, V12, V14, and V17) exhibit stark separation boundaries between legitimate and fraudulent transactions, indicating high feature importance.
*   **No Multicollinearity**: The correlation heatmap confirmed that all 28 PCA-derived features are completely independent (orthogonal) to one another.

---

### 7. Next Phase: Phase 2 (Preprocessing and Modeling)
Building upon our Phase 1 findings, the next phase will encompass:
1.  **Data Preprocessing**: Implementing robust scaling for the `Amount` and `Time` features.
2.  **Imbalance Mitigation**: Training models using class weighting or resampling algorithms (such as SMOTE).
3.  **Model Training**: Implementing and comparing classifiers (Logistic Regression, Random Forest, and XGBoost).
4.  **Model Evaluation**: Tuning hyperparameters and evaluating performance curves (Precision-Recall AUC).
5.  **API Deployment**: Connecting the trained model pipeline to our FastAPI REST endpoint.
