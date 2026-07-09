# End-to-End Fraud Detection System

An end-to-end Machine Learning system for real-time credit card fraud detection using Python, Scikit-learn, FastAPI, Docker, and cloud deployment.

---

Welcome to the **End-to-End Fraud Detection System** workspace. This repository contains the foundational setup, exploratory data analysis (EDA), model training pipeline, and serving REST API for detecting credit card fraud.

---

## Phase 1: Setup & EDA (Completed)

### 1. Project Objective
The main goal of this system is to identify fraudulent credit card transactions in real-time. In Phase 1, we established a professional project structure, set up the development environment, generated a highly realistic synthetic dataset, and performed a comprehensive Exploratory Data Analysis (EDA) to guide our preprocessing and model design decisions.

### 2. Dataset Description
The dataset used in this project represents transactions made by credit cards, designed to mimic the standard Kaggle Credit Card Fraud Detection dataset (European cardholders, Sept 2013).

*   **Total Transactions**: 30,000
*   **Legitimate Transactions (Class 0)**: 29,949 (99.83% of the dataset)
*   **Fraudulent Transactions (Class 1)**: 51 (0.17% of the dataset)
*   **Severe Class Imbalance**: Fraud is a rare event, representing the primary challenge of this project.
*   **Anonymized Features (`V1` to `V28`)**: Numerical features obtained via PCA, scaled and centered around 0.
*   **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
*   **Amount**: Transaction dollar amount.
*   **Class**: Target variable, where `1` represents fraud and `0` represents legitimate.

### 3. Folder Structure
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
    ├── data_pipeline.py     # Ingestion, generation, and stratified train/test split utilities
    ├── feature_engineering.py # TransactionTransformer (Robust scaling & cyclical time)
    ├── model_pipeline.py    # Class-weighted training, PR AUC metrics & joblib serialization
    └── utils.py             # Reproducibility seeding and logger configuration
```

### 4. Technologies Used
*   **Programming Language**: Python 3.11+
*   **Data Analysis & Manipulation**: `pandas`, `numpy`
*   **Data Visualization**: `matplotlib`, `seaborn`
*   **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`, `joblib`
*   **API Serving & Backend**: `fastapi`, `uvicorn`, `pydantic`, `httpx`
*   **Configuration & Logging**: `python-dotenv`, `loguru`
*   **Environment Isolation**: `venv` (Python virtual environment)

---

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
