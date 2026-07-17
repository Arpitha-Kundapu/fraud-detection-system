import numpy as np  # pyright: ignore [reportMissingImports]
import pandas as pd  # pyright: ignore [reportMissingImports]
from sklearn.base import BaseEstimator, TransformerMixin  # pyright: ignore [reportMissingImports]
from sklearn.preprocessing import RobustScaler  # pyright: ignore [reportMissingImports]
from loguru import logger  # pyright: ignore [reportMissingImports]

class TransactionTransformer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for scaling the transaction 'Amount'
    using RobustScaler and cyclically encoding 'Time' using sine/cosine functions.
    """
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        
    def fit(self, X, y=None):
        """
        Fits the RobustScaler on the transaction 'Amount' feature.
        """
        logger.info("Fitting TransactionTransformer...")
        
        # Convert to DataFrame if inputs are numpy arrays
        if isinstance(X, np.ndarray):
            columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
            X = pd.DataFrame(X, columns=columns)
            
        self.feature_names_in_ = list(X.columns)
        
        # Fit scaler on Amount
        amount_data = X[["Amount"]]
        self.scaler.fit(amount_data)
        
        # Compute output feature names
        v_cols = [f"V{i}" for i in range(1, 29)]
        self.feature_names_out_ = v_cols + ["Amount_scaled", "sin_time", "cos_time"]
        
        logger.info("TransactionTransformer fitted successfully.")
        return self
        
    def transform(self, X):
        """
        Applies transformations to scale Amount and cyclically encode Time.
        Returns a DataFrame containing V1..V28, Amount_scaled, sin_time, and cos_time.
        """
        # Convert to DataFrame if inputs are numpy arrays
        if isinstance(X, np.ndarray):
            columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
            X = pd.DataFrame(X, columns=columns)
            
        X_clean = X.copy()
        
        # 1. Scale Amount
        scaled_amount = self.scaler.transform(X_clean[["Amount"]])
        
        # 2. Cyclical Time Encoding
        # Convert seconds to hours of the day (0-23.99)
        hours = (X_clean["Time"] % 86400) / 3600.0
        sin_time = np.sin(2 * np.pi * hours / 24.0)
        cos_time = np.cos(2 * np.pi * hours / 24.0)
        
        # 3. Construct the output feature set
        v_cols = [f"V{i}" for i in range(1, 29)]
        X_v = X_clean[v_cols].copy()
        
        # Combine
        X_v["Amount_scaled"] = scaled_amount.flatten()
        X_v["sin_time"] = sin_time
        X_v["cos_time"] = cos_time
        
        return X_v
        
    def get_feature_names_out(self, input_features=None):
        """Returns output feature names."""
        return self.feature_names_out_
