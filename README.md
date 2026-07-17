# fraud-detection-system
An end-to-end Machine Learning system for real-time credit card fraud detection using Python, Scikit-learn, FastAPI, Docker, and cloud deployment.

# End-to-End Fraud Detection System - Phase 1: Setup & EDA

<<<<<<< HEAD
Welcome to the **End-to-End Fraud Detection System** workspace. This repository contains the foundational setup and exploratory data analysis (EDA) for detecting credit card fraud using machine learning.

---

## 1. Project Objective
The main goal of this system is to identify fraudulent credit card transactions in real-time. In this phase (Phase 1), we established a professional project structure, set up the development environment, generated a highly realistic synthetic dataset, and performed a comprehensive Exploratory Data Analysis (EDA) to guide our preprocessing and model design decisions.

---

## 2. Dataset Description
The dataset used in this project represents transactions made by credit cards. It is designed to mimic the standard Kaggle Credit Card Fraud Detection dataset, which contains transactions made by European cardholders in September 2013.
=======
Welcome to the **End-to-End Fraud Detection System** workspace. This repository contains the foundational setup, exploratory data analysis (EDA), model training pipeline, and serving REST API for detecting credit card fraud.

---

## Phase 1: Setup & EDA (Completed)

### 1. Project Objective
The main goal of this system is to identify fraudulent credit card transactions in real-time. In Phase 1, we established a professional project structure, set up the development environment, generated a highly realistic synthetic dataset, and performed a comprehensive Exploratory Data Analysis (EDA) to guide our preprocessing and model design decisions.

### 2. Dataset Description
The dataset used in this project represents transactions made by credit cards, designed to mimic the standard Kaggle Credit Card Fraud Detection dataset (European cardholders, Sept 2013).
>>>>>>> f2d8bd57b7ef04951b3de650327db70a530b805d

*   **Total Transactions**: 30,000
*   **Legitimate Transactions (Class 0)**: 29,949 (99.83% of the dataset)
*   **Fraudulent Transactions (Class 1)**: 51 (0.17% of the dataset)
*   **Severe Class Imbalance**: Fraud is a rare event, representing the primary challenge of this project.
*   **Anonymized Features (`V1` to `V28`)**: Numerical features obtained via PCA, scaled and centered around 0.
*   **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
*   **Amount**: Transaction dollar amount.
*   **Class**: Target variable, where `1` represents fraud and `0` represents legitimate.

<<<<<<< HEAD
---

## 3. Folder Structure
=======
### 3. Folder Structure
>>>>>>> f2d8bd57b7ef04951b3de650327db70a530b805d
The repository is organized following professional, production-grade machine learning folder patterns:

```text
├── .gitignore               # Standard Python Git ignore patterns
├── .env.example             # Template for API and model environment variables
├── .env                     # Local environment settings (ignored by git)
├── README.md                # Project overview and run instructions
├── requirements.txt         # Package dependencies for ML and API serving
├── setup.py                 # Setuptools config for editable installation of local src module
├── train.py                 # (Optional) python training runner
├── test_api.py              # API endpoint integration test suite
├── api/                     # API serving layer
│   ├── __init__.py
│   └── app.py               # FastAPI application script with serving endpoints
├── data/                    # Local storage for raw and processed datasets (ignored by git)
│   ├── raw/
│   │   └── creditcard.csv   # Synthetic Credit Card Fraud dataset
│   └── processed/
├── documentation/           # In-depth architectural and setup manuals
│   ├── architecture.md
│   └── setup_guide.md
├── notebooks/               # Jupyter/VS Code notebook directory
│   ├── eda.ipynb            # Interactive Exploratory Data Analysis notebook
│   └── modeling.ipynb       # Interactive model preprocessing, training & serialization notebook
└── src/                     # Core system modules (packaged locally)
    ├── __init__.py
<<<<<<< HEAD
    ├── data_pipeline.py     # Ingestion, cleaning, and preprocessing utilities
    ├── feature_engineering.py # Categorical encoding, scaling, and time features
    ├── model_pipeline.py    # Model training, evaluation, and loading utilities
    └── utils.py             # Reproducibility seeding and logger configuration
```

---

## 4. Technologies Used
*   **Programming Language**: Python 3.13
=======
    ├── data_pipeline.py     # Ingestion, generation, and stratified train/test split utilities
    ├── feature_engineering.py # TransactionTransformer (Robust scaling & cyclical time)
    ├── model_pipeline.py    # Class-weighted training, PR AUC metrics & joblib serialization
    └── utils.py             # Reproducibility seeding and logger configuration
```

### 4. Technologies Used
*   **Programming Language**: Python 3.11+
>>>>>>> f2d8bd57b7ef04951b3de650327db70a530b805d
*   **Data Analysis & Manipulation**: `pandas`, `numpy`
*   **Data Visualization**: `matplotlib`, `seaborn`
*   **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`, `joblib`
*   **API Serving & Backend**: `fastapi`, `uvicorn`, `pydantic`, `httpx`
*   **Configuration & Logging**: `python-dotenv`, `loguru`
*   **Environment Isolation**: `venv` (Python virtual environment)

---

<<<<<<< HEAD
## 5. EDA Performed
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

## 6. Key Observations
*   **Class Imbalance**: Fraud represents only **0.17%** of transactions. Consequently, standard **Accuracy is a deficient metric**. We must optimize models using **Precision, Recall, F1-Score, and AUPRC**.
*   **Scale Discrepancies**: Features `V1` through `V28` are already scaled. However, the `Amount` feature is unscaled, right-skewed, and ranges up to $15,000. It must be standard-scaled or log-transformed during preprocessing to prevent algorithm bias.
*   **Cyclic Behavior**: Legitimate transactions decrease during night hours and peak during daytime hours. Fraudulent transactions do not show this day/night cyclical behavior and remain active even during normal sleep windows.
*   **Discriminative Features**: Specific features (specifically V10, V12, V14, and V17) exhibit stark separation boundaries between legitimate and fraudulent transactions, indicating high feature importance.
*   **No Multicollinearity**: The correlation heatmap confirmed that all 28 PCA-derived features are completely independent (orthogonal) to one another.

---

## 7. Next Phase: Phase 2 (Preprocessing and Modeling)
Building upon our Phase 1 findings, the next phase will encompass:
1.  **Data Preprocessing**: Implementing robust scaling for the `Amount` and `Time` features.
2.  **Imbalance Mitigation**: Training models using class weighting or resampling algorithms (such as SMOTE).
3.  **Model Training**: Implementing and comparing classifiers (Logistic Regression, Random Forest, and XGBoost).
4.  **Model Evaluation**: Tuning hyperparameters and evaluating performance curves (Precision-Recall AUC).
5.  **API Deployment**: Connecting the trained model pipeline to our FastAPI REST endpoint.
=======
## Phase 2: Preprocessing and Modeling (Completed)

Building upon the Phase 1 findings, Phase 2 implements data preprocessing, handles severe class imbalance, evaluates multiple model architectures, packages the final pipeline, and serves predictions via FastAPI.

### 5. Implementation Summary

#### A. Preprocessing Pipeline (`src/feature_engineering.py`)
*   **Transaction Amount Scaling**: Preprocessed using `RobustScaler` to center and scale the amount distribution against heavy outliers.
*   **Diurnal Time Encoding**: Cyclically encodes `Time` (seconds elapsed) into sine/cosine variables using a 24-hour cycle to capture day/night transaction density difference.

#### B. Imbalance Mitigation & Training (`src/model_pipeline.py`)
*   Uses **stratified splitting** (`stratify=Class`) to preserve the 0.17% minority class proportion.
*   Trains and evaluates three class-weighted model architectures:
    1.  **Logistic Regression** (Balanced weights) - *PR AUC: 1.0000*
    2.  **Random Forest** (Balanced weights) - *PR AUC: 1.0000, F1: 0.8235*
    3.  **XGBoost** (Calculated `scale_pos_weight` ratio) - *PR AUC: 1.0000*
*   Packaged the champion model (Logistic Regression) and the fitted transformer into a single unified scikit-learn `Pipeline` object, serialized to `models/fraud_model_pipeline.joblib`.

#### C. FastAPI Servings (`api/app.py`)
*   Loads the pipeline on server startup.
*   Exposes endpoints with Pydantic request models:
    *   `GET /health`: Health status.
    *   `POST /predict`: Fraud prediction for a single transaction.
    *   `POST /predict/batch`: Batch prediction for a list of transactions.

---

## 6. How to Run and Test

### Prerequisites & Virtual Env Setup
Follow the steps in [setup_guide.md](file:///d:/Machine_Learning_Project/documentation/setup_guide.md) or execute:
```powershell
# Create venv
python -m venv .venv

# Activate venv
.venv\Scripts\Activate.ps1

# Install requirements & editable local package
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

### Run Model Training (Notebook)
Open `notebooks/modeling.ipynb` in VS Code or Jupyter and run all cells. If the dataset (`data/raw/creditcard.csv`) is missing, the data pipeline will automatically generate it. This script trains the models, saves evaluation charts, and serializes the pipeline.

### Run API Integration Tests
Verify endpoint correctness and classification:
```powershell
python test_api.py
```
*Expected Output: `OK`*

### Start API Server
Launch the REST server:
```powershell
python -m uvicorn api.app:app --reload
```
Visit the interactive Swagger UI at **`http://127.0.0.1:8000/docs`** to test predictions directly in your browser.

---

## Next Phase: Phase 3 (Containerization & Deployment)

The next step in the pipeline will involve:
1.  **Dockerization**: Writing a `Dockerfile` and `docker-compose.yml` to package the FastAPI application, model artifacts, and environment settings.
2.  **Container Validation**: Running and testing the API service inside a local Docker container.
3.  **Cloud Deployment**: Deploying the Docker image to a hosting service (e.g., Render, AWS, or GCP) for real-time remote testing.
>>>>>>> f2d8bd57b7ef04951b3de650327db70a530b805d
