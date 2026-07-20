# Credit Card Fraud Detection System

An end-to-end Machine Learning system for real-time credit card fraud detection using Python, Scikit-learn, FastAPI, Docker, and cloud deployment.

---

## Project Roadmap

The development of this system is structured into six sequential phases:
The development of this system is structured into six sequential phases:
1. **Phase 1: Environment Setup & Exploratory Data Analysis (EDA)** (Completed)
2. **Phase 2: Data Preprocessing & Handling Class Imbalance** (Completed)
3. **Phase 3: Model Training & Evaluation** (Completed)
4. **Phase 4: REST API Serving with FastAPI** (Completed)
5. **Phase 5: Containerization with Docker** (Future Phase)
6. **Phase 6: Cloud Deployment & Monitoring** (Future Phase)

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

### 3. Key EDA Observations
*   **Class Imbalance**: Fraud represents only **0.17%** of transactions. Consequently, standard **Accuracy is a deficient metric**. We must optimize models using **Precision, Recall, F1-Score, and AUPRC**.
*   **Scale Discrepancies**: Features `V1` through `V28` are already scaled. However, the `Amount` feature is unscaled, right-skewed, and ranges up to $15,000. It must be scaled using robust scalers during preprocessing to prevent algorithm bias.
*   **Cyclic Behavior**: Legitimate transactions decrease during night hours and peak during daytime hours. Fraudulent transactions do not show this day/night cyclical behavior and remain active even during normal sleep windows.
*   **Discriminative Features**: Specific features (specifically V10, V12, V14, and V17) exhibit stark separation boundaries between legitimate and fraudulent transactions, indicating high feature importance.

---

## Phase 2: Data Preprocessing & Handling Class Imbalance (Completed)

Building upon the Phase 1 findings, Phase 2 implements data preprocessing and handles severe class imbalance:

### 1. Preprocessing Pipeline (`src/feature_engineering.py`)
*   **Transaction Amount Scaling**: Preprocessed using `RobustScaler` to center and scale the amount distribution against heavy outliers.
*   **Diurnal Time Encoding**: Cyclically encodes `Time` (seconds elapsed) into sine/cosine variables using a 24-hour cycle to capture day/night transaction density difference.

### 2. Imbalance Mitigation (`src/data_pipeline.py`)
*   Uses **stratified splitting** (`stratify=Class`) to preserve the 0.17% minority class proportion across training (80%) and testing (20%) splits.
*   Provides `apply_smote()` synthetic oversampling using `imbalanced-learn` alongside class weighting (`class_weight="balanced"` and `scale_pos_weight`).

---

## Phase 3: Model Training & Evaluation (Completed)

In this phase, we implement full model training, hyperparameter tuning, evaluation curves, and pipeline serialization:
*   **Models Trained**: Baseline Logistic Regression, Tuned Logistic Regression (GridSearchCV), Random Forest, and XGBoost.
*   **Metrics Evaluated**: Precision, Recall, F1-Score, ROC-AUC, and Precision-Recall AUC (PR-AUC).
*   **Visualizations Generated**: Multi-panel evaluation plots saved in `reports/figures/` (`confusion_matrices.png`, `roc_curves.png`, `precision_recall_curves.png`, `feature_importances.png`).
*   **Model Serialization**: Best model selected by PR-AUC and serialized to `models/fraud_model_pipeline.joblib`.

---

## Phase 4: REST API Serving with FastAPI (Completed)

Phase 4 wraps the trained model pipeline into a production-ready asynchronous REST API using **FastAPI**, **Pydantic**, and **Uvicorn**:

### 1. API Endpoints
*   `GET /health`: Health check verifying server status, model loaded state, and model version.
*   `POST /predict`: Real-time prediction endpoint for a single transaction input (JSON payload with `Time`, `V1`..`V28`, `Amount`). Returns `is_fraud`, `probability`, and `model_version`.
*   `POST /predict/batch`: High-throughput batch prediction endpoint processing a list of transactions in a single vectorized DataFrame call.

### 2. Validation & Lifespan Architecture
*   Pydantic schema validation for strict type and structure enforcement on input features.
*   FastAPI `@asynccontextmanager` `lifespan` handler to efficiently load the serialized model pipeline into memory once at application startup.
*   Auto-generated interactive Swagger UI documentation at `http://127.0.0.1:8000/docs`.

---

## Folder Structure

The repository is organized following professional, production-grade machine learning folder patterns:

```text
├── .gitignore               # Standard Python Git ignore patterns
├── .env.example             # Template for API and model environment variables
├── .env                     # Local environment settings (ignored by git)
├── README.md                # Project overview and run instructions
├── requirements.txt         # Package dependencies for ML and API serving
├── setup.py                 # Setuptools config for editable installation of local src module
├── train.py                 # Phase 3 training runner (Step 1 implementation)
├── test_api.py              # API endpoint integration test suite (Future Phase)
├── api/                     # API serving layer (Future Phase)
│   ├── __init__.py
│   └── app.py               # FastAPI application script with serving endpoints
├── data/                    # Local storage for raw and processed datasets (ignored by git)
│   ├── raw/
│   │   └── creditcard.csv   # Synthetic Credit Card Fraud dataset
│   └── processed/
│       ├── train_processed.csv # Preprocessed training split
│       └── test_processed.csv  # Preprocessed testing split
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

---

## Technologies Used
*   **Programming Language**: Python 3.11+
*   **Data Analysis & Manipulation**: `pandas`, `numpy`
*   **Data Visualization**: `matplotlib`, `seaborn`
*   **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`, `joblib`
*   **Backend & Servings**: `fastapi`, `uvicorn`, `pydantic` (Future Phases)
*   **Configuration & Logging**: `python-dotenv`, `loguru`
*   **Environment Isolation**: `venv` (Python virtual environment)

---

## How to Run and Test

### Setup Environment
```powershell
# Create venv
python -m venv .venv

# Activate venv
.venv\Scripts\Activate.ps1

# Install requirements & editable local package
pip install -r requirements.txt
pip install -e .
```

### 1. Run Model Training & Pipeline Serialization
Train all models, evaluate metrics, generate figures, and save the model pipeline:
```powershell
python train.py
```

### 2. Launch FastAPI REST API Server (Phase 4)
Start the local server with Uvicorn:
```powershell
uvicorn api.app:app --reload --port 8000
```
- Access Interactive Swagger UI Docs: http://127.0.0.1:8000/docs
- Health check endpoint: http://127.0.0.1:8000/health

### 3. Run API Test Suite
Run the automated integration test suite:
```powershell
python test_api.py
```
