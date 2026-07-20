import os
import numpy as np  # pyright: ignore [reportMissingImports]
import pandas as pd  # pyright: ignore [reportMissingImports]
from sklearn.model_selection import train_test_split  # pyright: ignore [reportMissingImports]
from loguru import logger  # pyright: ignore [reportMissingImports]

def generate_synthetic_data(filepath: str, seed: int = 42) -> pd.DataFrame:
    """
    Generates a highly realistic synthetic credit card transactions dataset
    that mimics the characteristics observed in Phase 1 (EDA).
    
    Total samples: 30,000
    Legitimate (Class 0): 29,949
    Fraudulent (Class 1): 51
    """
    logger.info(f"Generating synthetic dataset with seed {seed}...")
    
    # Initialize random number generator
    rng = np.random.default_rng(seed)
    
    n_legit = 29949
    n_fraud = 51
    n_total = n_legit + n_fraud
    
    # 1. Generate Class label
    classes = np.array([0] * n_legit + [1] * n_fraud)
    
    # 2. Generate Time (seconds over 2 days = 172800 seconds)
    # Legitimate transactions: peak during the day, trough at night
    legit_times = []
    while len(legit_times) < n_legit:
        # Candidate time in seconds
        candidate = rng.uniform(0, 172800)
        hour = (candidate % 86400) / 3600.0
        # Diurnal probability curve (peaks at 15:00, trough at 3:00)
        p = (np.cos(2 * np.pi * (hour - 15) / 24) + 1.2) / 2.2
        if rng.uniform(0, 1) < p:
            legit_times.append(candidate)
            
    # Fraudulent transactions: uniform over the entire period (flat)
    fraud_times = rng.uniform(0, 172800, n_fraud)
    times = np.concatenate([legit_times, fraud_times])
    
    # 3. Generate V1 to V28 features (PCA components)
    # Most features are orthogonal/independent N(0, 1)
    v_features = rng.normal(0, 1, size=(n_total, 28))
    
    # Apply discriminative shift for Fraudulent class (Class 1) in V10, V12, V14, V17
    # (these indices in 0-based array are: V10->9, V12->11, V14->13, V17->16)
    fraud_indices = np.arange(n_legit, n_total)
    v_features[fraud_indices, 9] = rng.normal(-3.5, 1.5, n_fraud)  # V10
    v_features[fraud_indices, 11] = rng.normal(-4.0, 1.5, n_fraud) # V12
    v_features[fraud_indices, 13] = rng.normal(-4.5, 1.5, n_fraud) # V14
    v_features[fraud_indices, 16] = rng.normal(-3.8, 1.5, n_fraud) # V17
    
    # 4. Generate Amount feature (log-normal, right-skewed)
    # Legitimate transactions: median around $33, max up to $1,000
    legit_amounts = rng.lognormal(mean=3.5, sigma=1.0, size=n_legit)
    # Fraudulent transactions: median around $150, max up to $15,000 (clipped)
    fraud_amounts = rng.lognormal(mean=5.0, sigma=1.2, size=n_fraud)
    fraud_amounts = np.clip(fraud_amounts, 0.01, 15000.0)
    amounts = np.concatenate([legit_amounts, fraud_amounts])
    
    # Create DataFrame
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data = np.column_stack([times, v_features, amounts, classes])
    df = pd.DataFrame(data, columns=columns)
    
    # Ensure Class is integer
    df["Class"] = df["Class"].astype(int)
    
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # Save to disk
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Synthetic dataset of {n_total} records written successfully to {filepath}")
    return df

def load_raw_data(filepath: str = "data/raw/creditcard.csv", seed: int = 42) -> pd.DataFrame:
    """
    Loads raw transaction data from CSV. If the file is missing,
    automatically invokes generator to create the synthetic data.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Raw data file not found at {filepath}. Generating synthetic data...")
        return generate_synthetic_data(filepath, seed=seed)
    
    logger.info(f"Loading raw data from {filepath}...")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded dataset with shape {df.shape}")
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs train/test splitting using stratified splitting to preserve class imbalance ratios.
    """
    logger.info(f"Splitting data with test_size={test_size} and stratify=Class...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["Class"]
    )
    logger.info(f"Split complete. Train size: {train_df.shape[0]}, Test size: {test_df.shape[0]}")
    return train_df, test_df

def apply_smote(X_train, y_train, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Applies Synthetic Minority Oversampling Technique (SMOTE) on the training set only
    to synthetically balance the fraud class ratio.
    """
    from imblearn.over_sampling import SMOTE  # pyright: ignore [reportMissingImports]
    logger.info("Applying SMOTE oversampling on training data...")
    smote = SMOTE(random_state=seed)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"SMOTE oversampling complete. Resampled shape: {X_resampled.shape}, Fraud samples: {(y_resampled == 1).sum()}")
    return X_resampled, y_resampled
