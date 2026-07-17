# Credit Card Fraud Detection System

An end-to-end Machine Learning system for real-time credit card fraud detection using Python, Scikit-learn, FastAPI, Docker, and cloud deployment.

---

## Project Roadmap

The development of this system is structured into six sequential phases:
1. **Phase 1: Environment Setup & Exploratory Data Analysis (EDA)** (Completed)
2. **Phase 2: Data Preprocessing & Handling Class Imbalance** (Completed)
3. **Phase 3: Model Training & Evaluation** (Active - Step 1 Completed)
4. **Phase 4: API serving with FastAPI** (Future Phase)
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

---

## Phase 3: Model Training & Evaluation (Active - Step 1 Completed)

In this phase, we implement model training, hyperparameter tuning, validation curves, and model serialization.

### Step 1: Data Ingestion, Splitting, and Preprocessing (Completed)
We run the offline data pipeline to load, split, and preprocess the dataset, saving the processed splits as CSV files for training:
*   **Train Size**: 24,000 samples (23,959 Legitimate, 41 Fraudulent)
*   **Test Size**: 6,000 samples (5,990 Legitimate, 10 Fraudulent)
*   **Output Files**: Saved to `data/processed/train_processed.csv` and `data/processed/test_processed.csv`

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

### Run Phase 3 Step 1 Data Processing
To run the ingestion, splitting, preprocessing, and class distribution verification:
```powershell
python train.py
```
This generates `train_processed.csv` and `test_processed.csv` inside `data/processed/`.
